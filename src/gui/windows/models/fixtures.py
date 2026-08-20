import ast
import json
import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import QDate, QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QDialog, QLabel, QComboBox, QDateEdit, QFileDialog, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget
from src.database.model import ModelDatabase
from src.gui.utils.taskrunner import TaskRunnerDialog
from src.gui.widgets.comboboxes import CheckableComboBox
from src.gui.widgets.tables import ExcelTable, StylizedTable
from src.network.fixtures.footystats.scraper import FootyStatsScraper
from src.network.fixtures.sporttery import SportteryMobileClient, latest_had_odds
from src.network.fixtures.utils import match_fixture_teams
from src.network.leagues.league import League
from src.preprocessing.utils.inputs import construct_inputs_by_fixture
from src.preprocessing.utils.target import TargetType, class_to_score
from src.services.daily_sporttery import identify_league


CURRENT_TEAM_EXTRAS = {
    '英超': ['Coventry'],
}


class FixtureFetchWorker(QObject):
    """Fetch fixtures outside the GUI thread so Selenium cannot freeze the window."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fixture_url: str, date_str: str, iso_date: str, league_id: str):
        super().__init__()
        self._fixture_url = fixture_url
        self._date_str = date_str
        self._iso_date = iso_date
        self._league_id = league_id

    @staticmethod
    def _field(row: dict, *names, default=''):
        for name in names:
            value = row.get(name)
            if value not in (None, ''):
                return value
        return default

    def _sporttery_fixtures(self) -> pd.DataFrame:
        """Read selected-date fixtures from the official lottery endpoint."""
        aliases = json.loads(
            Path('storage/network/sporttery_team_aliases.json').read_text(encoding='utf-8')
        )
        rows = []
        for match in SportteryMobileClient(timeout=12.0, retries=2).selling_matches():
            league_name = str(self._field(
                match, 'leagueAllName', 'leagueName', 'leagueAbbName',
            ))
            if identify_league(league_name) != self._league_id:
                continue
            if str(self._field(match, 'matchDate', 'matchDateTime', 'startTime'))[:10] != self._iso_date:
                continue

            home_cn = str(self._field(
                match, 'homeTeamAllName', 'homeTeamName', 'homeTeamAbbName',
            ))
            away_cn = str(self._field(
                match, 'awayTeamAllName', 'awayTeamName', 'awayTeamAbbName',
            ))
            home = aliases.get(self._league_id, {}).get(home_cn)
            away = aliases.get(self._league_id, {}).get(away_cn)
            odds = latest_had_odds(match)
            if home and away and odds:
                rows.append({
                    'Home': home,
                    'Away': away,
                    '1': odds['H'],
                    'X': odds['D'],
                    '2': odds['A'],
                })
        return pd.DataFrame(rows, columns=['Home', 'Away', '1', 'X', '2'])

    def run(self):
        scraper = None
        try:
            official_df = self._sporttery_fixtures()
            if not official_df.empty:
                self.finished.emit(official_df)
                return

            # Keep the original league page as a secondary source when the
            # official current-selling list is temporarily unavailable.
            scraper = FootyStatsScraper()
            if not scraper.load_page(self._fixture_url):
                self.failed.emit(
                    f'{self._iso_date} 没有可用的官方在售赛程，备用网站也无法访问。'
                )
                return
            df = scraper.parse_fixture_table(date_str=self._date_str)
            if df is None:
                self.failed.emit('该日期在数据网站上没有可用赛程，请选择实际比赛日期。')
                return
            self.finished.emit(df)
        except Exception as error:
            self.failed.emit(f'赛程获取失败：{error}')
        finally:
            if scraper is not None:
                try:
                    scraper.quit()
                except Exception:
                    pass


class FixturesDialog(QDialog):
    """ Fixtures dialog which downloads the upcoming league's fixture and makes predictions. """

    def __init__(
            self,
            df: pd.DataFrame,
            model_db: ModelDatabase,
            league: League,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._df = df.reset_index(drop=True)
        self._model_db = model_db
        self._league = league

        self._model_ids = model_db.get_model_ids()
        self._model_configs = {}
        self._title = 'Fixtures Dialog'
        self._width = 800
        self._height = 450

        # Declare placeholders.
        self._y_prob = None
        self._y_pred = None
        self._percentiles = None
        self._odds = None
        self._odd_mask = None
        self._index = None

        self._target_types = {
            '胜平负（主胜/平/客胜）': TargetType.RESULT,
            '大小球 2.5': TargetType.OVER_UNDER,
            '半场胜平负': TargetType.HALF_RESULT,
            '半全场（9种结果）': TargetType.HALF_FULL,
            '准确比分（含6+）': TargetType.SCORE,
        }
        self._historical_teams = set(df['Home'].dropna()) | set(df['Away'].dropna())
        extra_teams = set(CURRENT_TEAM_EXTRAS.get(model_db.league_id, []))
        self._home_teams = sorted(set(df['Home'].dropna()) | extra_teams)
        self._away_teams = sorted(set(df['Away'].dropna()) | extra_teams)
        self._result_model_ids = []
        self._uo_model_ids = []
        self._half_result_model_ids = []
        self._half_full_model_ids = []
        self._score_model_ids = []
        for model_id in self._model_ids:
            if '早期模型' in model_id:
                continue
            config = model_db.load_model_config(model_id=model_id)
            if not config:
                continue
            self._model_configs[model_id] = config
            if config.get('target_type') == TargetType.RESULT:
                self._result_model_ids.append(model_id)
            elif config.get('target_type') == TargetType.OVER_UNDER:
                self._uo_model_ids.append(model_id)
            elif config.get('target_type') == TargetType.HALF_RESULT:
                self._half_result_model_ids.append(model_id)
            elif config.get('target_type') == TargetType.HALF_FULL:
                self._half_full_model_ids.append(model_id)
            elif config.get('target_type') == TargetType.SCORE:
                if '让球胜负' not in model_id:
                    self._score_model_ids.append(model_id)

        for model_ids in (
                self._result_model_ids,
                self._uo_model_ids,
                self._half_result_model_ids,
                self._half_full_model_ids,
                self._score_model_ids,
        ):
            model_ids.sort(key=self._model_quality_key, reverse=True)

        # Declare UI Placeholders.
        self._calendar = None
        self._combo_model = None
        self._combo_target = None
        self._export_btn = None
        self._combo_filters = None
        self._table = None
        self._fetch_thread = None
        self._fetch_worker = None
        self._status_label = None
        self._fetch_btn = None

        self._initialize_window()
        self._add_widgets()

    def _model_quality_key(self, model_id: str) -> tuple:
        """Put the best independently tested model first in each picker."""
        tuning = self._model_configs.get(model_id, {}).get('train', {}).get('tuning', {})
        accuracy = tuning.get('test_accuracy')
        samples = tuning.get('test_samples', tuning.get('test_sample_count', 0))
        try:
            accuracy = float(accuracy)
        except (TypeError, ValueError):
            accuracy = -1.0
        try:
            samples = int(samples or 0)
        except (TypeError, ValueError):
            samples = 0
        return accuracy >= 0.0, accuracy, samples, model_id

    def exec(self):
        if len(self._model_ids) == 0:
            QMessageBox.critical(
                self,
                'No Existing Models.',
                'There are no existing models to predict fixtures.',
                QMessageBox.StandardButton.Ok
            )
            return QDialog.Rejected

        super().exec()

    def _initialize_window(self):
        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

    def closeEvent(self, event):
        if self._fetch_thread is not None and self._fetch_thread.isRunning():
            QMessageBox.information(self, '正在获取赛程', '后台正在获取赛程，请等待完成后再关闭窗口。')
            event.ignore()
            return
        super().closeEvent(event)

    def _add_widgets(self):
        root = QVBoxLayout(self)

        # --- Date/Model initialization ---
        model_hbox = QHBoxLayout()
        model_hbox.addStretch(1)

        td = timedelta(days=180)
        today = date.today()
        q_today = QDate(today.year, today.month, today.day)
        q_min = QDate((today - td).year, (today - td).month, (today - td).day)
        q_max = QDate((today + td).year, (today + td).month, (today + td).day)
        self._calendar = QDateEdit(self)
        self._calendar.setCalendarPopup(True)
        self._calendar.setDate(q_today)
        self._calendar.setDateRange(q_min, q_max)
        self._calendar.setDisplayFormat('yyyy-MM-dd')
        self._calendar.dateChanged.connect(lambda qdate: QTimer.singleShot(50, lambda: self._on_date_change(qdate)))
        model_hbox.addWidget(QLabel('Fixture Date: '))
        model_hbox.addWidget(self._calendar)
        self._fetch_btn = QPushButton('获取赛程')
        self._fetch_btn.setFixedWidth(90)
        self._fetch_btn.clicked.connect(lambda: self._on_date_change(self._calendar.date()))
        model_hbox.addWidget(self._fetch_btn)

        self._combo_target = QComboBox()
        self._combo_target.setFixedWidth(120)
        for target, target_type in self._target_types.items():
            self._combo_target.addItem(target, target_type)
        self._combo_target.setCurrentIndex(-1)
        self._combo_target.setEnabled(False)
        self._combo_target.currentIndexChanged.connect(self._on_target_change)
        model_hbox.addWidget(QLabel('预测类型：'))
        model_hbox.addWidget(self._combo_target)

        self._combo_model = QComboBox()
        self._combo_model.setFixedWidth(220)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.setEnabled(False)
        self._combo_model.currentIndexChanged.connect(self._on_model_change)
        model_hbox.addWidget(QLabel('模型选择：'))
        model_hbox.addWidget(self._combo_model)
        model_hbox.addStretch(1)
        root.addLayout(model_hbox)

        self._status_label = QLabel('请选择比赛日期，程序将从网络获取当天赛程。')
        self._status_label.setStyleSheet('color: #777;')
        root.addWidget(self._status_label)

        filters_hbox = QHBoxLayout()
        filters_hbox.addStretch(1)
        self._combo_filters = CheckableComboBox()
        self._combo_filters.setFixedWidth(180)
        self._combo_filters.setEnabled(False)
        self._combo_filters.checkedItemsChanged.connect(self._on_filters_change)
        filters_hbox.addWidget(QLabel('Filters: '))
        filters_hbox.addWidget(self._combo_filters)
        filters_hbox.addStretch(1)
        root.addLayout(filters_hbox)

        export_hbox = QHBoxLayout()
        export_hbox.addStretch(1)
        self._export_btn = QPushButton('Export')
        self._export_btn.setFixedWidth(100)
        self._export_btn.setFixedHeight(30)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        export_hbox.addWidget(self._export_btn)
        export_hbox.addStretch(1)
        root.addLayout(export_hbox)

        empty_row = ['']*10
        table_df = pd.DataFrame({
            'Home': empty_row,
            'Away': empty_row,
            '1': empty_row,
            'X': empty_row,
            '2': empty_row,
            'Predicted': empty_row,
            'Prob(1)': empty_row,
            'Prob(X)': empty_row,
            'Prob(2)': empty_row,
            'Prob(U)': empty_row,
            'Prob(O)': empty_row
        })
        self._table = ExcelTable(
            parent=self,
            df=table_df,
            readonly=False,
            supports_sorting=False,
            supports_query_search=True,
            supports_deletion=True
        )
        self._table = StylizedTable().stylize_table(table=self._table, options_dict={0: self._home_teams, 1: self._away_teams})
        self._table.setColumnWidth(0, 150)
        self._table.setColumnWidth(1, 150)
        self._table.hide_columns(columns=['Prob(1)', 'Prob(X)', 'Prob(2)', 'Prob(U)', 'Prob(O)'], hide=True)
        root.addWidget(self._table)

    def _on_date_change(self, qdate: QDate):
        # Fetching date.
        date_str = qdate.toPyDate().strftime('%b %d').replace(' 0', ' ')

        if self._fetch_thread is not None and self._fetch_thread.isRunning():
            return

        self._calendar.setEnabled(False)
        self._fetch_btn.setEnabled(False)
        self._combo_target.setEnabled(False)
        self._combo_model.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._status_label.setText(f'正在后台获取 {qdate.toString("yyyy-MM-dd")} 的赛程，请稍候…')

        self._fetch_thread = QThread(self)
        self._fetch_worker = FixtureFetchWorker(
            self._league.fixture,
            date_str,
            qdate.toString('yyyy-MM-dd'),
            self._league.league_id,
        )
        self._fetch_worker.moveToThread(self._fetch_thread)
        self._fetch_thread.started.connect(self._fetch_worker.run)
        self._fetch_worker.finished.connect(self._on_fixture_fetched)
        self._fetch_worker.failed.connect(self._on_fixture_fetch_failed)
        self._fetch_worker.finished.connect(self._fetch_thread.quit)
        self._fetch_worker.failed.connect(self._fetch_thread.quit)
        self._fetch_thread.finished.connect(self._fetch_finished)
        self._fetch_thread.start()

    def _fetch_finished(self):
        self._calendar.setEnabled(True)
        self._fetch_btn.setEnabled(True)
        if self._fetch_worker is not None:
            self._fetch_worker.deleteLater()
        if self._fetch_thread is not None:
            self._fetch_thread.deleteLater()
        self._fetch_worker = None
        self._fetch_thread = None

    def _on_fixture_fetch_failed(self, message: str):
        self._status_label.setText(message)
        QMessageBox.critical(self, '赛程获取失败', message)

    def _on_fixture_fetched(self, df: pd.DataFrame):
        self._status_label.setText(f'已获取 {len(df)} 场比赛，请选择预测目标和模型。')

        # Matching fixtures.
        fixtures_df = match_fixture_teams(parsed_teams_df=df, league_df=self._df)

        # Keep promoted/new clubs even when the selected league history has no
        # rows for them yet.  Input construction will use neutral league
        # medians; only genuinely malformed fixtures are dropped.
        valid_mask = (
            fixtures_df['Home'].notna()
            & fixtures_df['Away'].notna()
            & fixtures_df['Home'].ne('')
            & fixtures_df['Away'].ne('')
            & fixtures_df['Home'].ne(fixtures_df['Away'])
        )
        rows_dropped = valid_mask.sum() != fixtures_df.shape[0]
        fixtures_df = fixtures_df[valid_mask].reset_index(drop=True)

        neutral_teams = sorted(
            (set(fixtures_df['Home']) | set(fixtures_df['Away'])) - self._historical_teams
        )
        if neutral_teams:
            names = '、'.join(neutral_teams)
            self._status_label.setText(
                f'已获取 {len(fixtures_df)} 场；新球队 {names} 将使用联赛中位统计，置信度较低。'
            )

        # Add self._fixtures_df to table.
        columns = ['Home', 'Away', '1', 'X', '2']
        self._table.clearContents()
        self._table.modify_columns(columns=columns, data=fixtures_df[columns].to_numpy().tolist())

        # Erase History.
        self._y_prob = None
        self._y_pred = None
        self._percentiles = None
        self._odds = None
        self._odd_mask = None

        # Enable target.
        self._combo_target.setEnabled(True)
        self._combo_target.blockSignals(True)
        self._combo_target.setCurrentIndex(-1)
        self._combo_target.blockSignals(False)

        # Clear & Disable models and filters.
        self._combo_model.blockSignals(True)
        self._combo_model.clear()
        self._combo_model.setEnabled(False)
        self._combo_model.blockSignals(False)
        self._combo_filters.blockSignals(True)
        self._combo_filters.clear()
        self._combo_filters.setEnabled(False)
        self._combo_filters.blockSignals(False)

        # Disable export button.
        self._export_btn.setEnabled(False)

        if rows_dropped:
            QMessageBox.information(self, 'Insufficient Data', 'Some matches have been dropped due to insufficient historical data.')

    def _on_target_change(self):
        """ Adds model ids based on the selected target. """

        # Erasing history.
        self._y_prob = None
        self._y_pred = None
        self._percentiles = None

        empty_cols = ['']*10
        self._table.modify_columns(
            columns=['Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)', 'Prob(U)', 'Prob(O)'],
            data=[empty_cols, empty_cols, empty_cols, empty_cols, empty_cols, empty_cols]
        )

        # Disable model, filter, buttons.
        self._combo_filters.blockSignals(True)
        self._combo_filters.clear()
        self._combo_filters.setEnabled(False)
        self._combo_filters.blockSignals(False)
        self._export_btn.setEnabled(False)

        # Setting models and columns.
        target_type = self._combo_target.currentData()

        if target_type == TargetType.RESULT:
            model_ids = self._result_model_ids
            self._table.hide_columns(columns=['Prob(1)', 'Prob(X)', 'Prob(2)'], hide=False)
            self._table.hide_columns(columns=['Prob(U)', 'Prob(O)'], hide=True)

            empty_cols = ['']*10
            self._table.modify_columns(columns=['Predicted', 'Prob(U)', 'Prob(O)'], data=[empty_cols, empty_cols, empty_cols])
        elif target_type == TargetType.OVER_UNDER:
            model_ids = self._uo_model_ids
            self._table.hide_columns(columns=['Prob(1)', 'Prob(X)', 'Prob(2)'], hide=True)
            self._table.hide_columns(columns=['Prob(U)', 'Prob(O)'], hide=False)
        elif target_type == TargetType.HALF_RESULT:
            model_ids = self._half_result_model_ids
            self._table.hide_columns(columns=['Prob(1)', 'Prob(X)', 'Prob(2)'], hide=False)
            self._table.hide_columns(columns=['Prob(U)', 'Prob(O)'], hide=True)
        elif target_type == TargetType.HALF_FULL:
            model_ids = self._half_full_model_ids
            self._table.hide_columns(columns=['Prob(1)', 'Prob(X)', 'Prob(2)', 'Prob(U)', 'Prob(O)'], hide=True)
        elif target_type == TargetType.SCORE:
            model_ids = self._score_model_ids
            self._table.hide_columns(columns=['Prob(1)', 'Prob(X)', 'Prob(2)', 'Prob(U)', 'Prob(O)'], hide=True)
        else:
            raise ValueError(f'Undefined targets: "{target_type}"')

        # Adding model ids.
        self._combo_model.setEnabled(True)
        self._combo_model.blockSignals(True)
        self._combo_model.clear()
        for model_id in model_ids:
            self._combo_model.addItem(model_id)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.blockSignals(False)
        if model_ids:
            best = self._model_configs.get(model_ids[0], {}).get('train', {}).get('tuning', {})
            accuracy = best.get('test_accuracy')
            detail = ''
            if isinstance(accuracy, (int, float)):
                detail = f'（独立测试 {accuracy:.1%}）'
            self._status_label.setText(
                f'请选择模型；已按独立测试表现排序，第一项是当前最优模型{detail}。'
            )
        else:
            self._status_label.setText('该预测类型没有完整可用的已训练模型。')

    def _on_model_change(self):
        fixture_df = self._read_fixture()

        if fixture_df is None:
            self._combo_model.blockSignals(True)
            self._combo_model.setCurrentIndex(-1)
            self._combo_model.blockSignals(False)
            self._combo_filters.blockSignals(True)
            self._combo_filters.clear()
            self._combo_filters.setEnabled(False)
            self._combo_filters.blockSignals(False)
            return

        self._index = fixture_df.index.to_numpy()
        self._odds = fixture_df[['1', 'X', '2']]

        # Prepare the odd mask, which is fixed.
        self._prepare_odd_mask(df=fixture_df)

        model_id = self._combo_model.currentText()
        if not model_id:
            return
        runner = TaskRunnerDialog(
            title='赛程预测',
            info=f'正在载入 {model_id} 并计算比赛概率…',
            task_fn=lambda: self._predict_fixtures(fixture_df, model_id),
            parent=self,
        )
        result = runner.run()
        if runner.error_message is not None or result is None:
            QMessageBox.critical(
                self, '赛程预测失败', runner.error_message or '模型没有返回结果。',
            )
            self._combo_model.blockSignals(True)
            self._combo_model.setCurrentIndex(-1)
            self._combo_model.blockSignals(False)
            return
        model_config, self._y_prob, score_classes = result
        self._y_pred = self._y_prob.argmax(axis=1)

        # Load filters.
        self._combo_filters.blockSignals(True)
        self._combo_filters.clear()
        self._combo_filters.setEnabled(True)
        if 'eval' in model_config and 'percentiles' in model_config['eval']:
            self._combo_filters.addItem(f'--- Select Filters ---')

            self._percentiles = model_config['eval']['percentiles']
            for key in self._percentiles.keys():
                self._combo_filters.addItem(f'{key}')
        self._combo_filters.blockSignals(False)

        # Add data to table.
        target_type = self._combo_target.currentData()

        if target_type == TargetType.RESULT:
            mapper = np.array(['H', 'D', 'A'])
            columns = ['Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)']
            mapped_y_pred = mapper.take(self._y_pred)
            data = np.hstack([np.expand_dims(mapped_y_pred, axis=-1), self._y_prob])
        elif target_type == TargetType.OVER_UNDER:
            mapper = np.array(['U', 'O'])
            columns = ['Predicted', 'Prob(U)', 'Prob(O)']
            mapped_y_pred = mapper.take(self._y_pred)
            data = np.hstack([np.expand_dims(mapped_y_pred, axis=-1), self._y_prob])
        elif target_type == TargetType.HALF_RESULT:
            mapper = np.array(['H', 'D', 'A'])
            columns = ['Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)']
            mapped_y_pred = mapper.take(self._y_pred)
            threshold = model_config.get('train', {}).get('tuning', {}).get('selective_threshold')
            if threshold is not None:
                mapped_y_pred = mapped_y_pred.astype(object)
                mapped_y_pred[self._y_prob.max(axis=1) < float(threshold)] = '暂不预测'
            data = np.hstack([np.expand_dims(mapped_y_pred, axis=-1), self._y_prob])
        elif target_type == TargetType.HALF_FULL:
            mapper = np.array(['胜/胜', '胜/平', '胜/负', '平/胜', '平/平', '平/负', '负/胜', '负/平', '负/负'])
            columns = ['Predicted']
            data = np.expand_dims(mapper.take(self._y_pred), axis=-1)
        elif target_type == TargetType.SCORE:
            predicted_classes = score_classes.take(self._y_pred)
            columns = ['Predicted']
            data = np.expand_dims([class_to_score(value) for value in predicted_classes], axis=-1)
        else:
            raise ValueError(f'Undefined target type: "{target_type}"')
        self._table.modify_columns(columns=columns, data=data, rows=fixture_df.index.tolist())
        self._highlight_matches()

        # Enable export button.
        self._export_btn.setEnabled(True)

    def _predict_fixtures(self, fixture_df: pd.DataFrame, model_id: str):
        """Pure worker task used to keep model loading off the GUI thread."""
        model, model_config = self._model_db.load_model(model_id=model_id)
        if model is None or model_config is None:
            raise RuntimeError(f'模型文件缺失：{model_id}')
        inputs = construct_inputs_by_fixture(df=self._df, fixture_df=fixture_df)
        probabilities = model.predict_proba(df=inputs).round(4)
        score_classes = np.asarray(getattr(model.classifier, 'classes_', ()))
        return model_config, probabilities, score_classes

    def _on_filters_change(self):
        self._highlight_matches()

    def _read_fixture(self) -> Optional[pd.DataFrame]:
        """ Reads the fixture and validates the values. """

        data = []
        indices = []
        for row in range(10):
            home_item = self._table.item(row, 0)
            home = home_item.text().strip() if home_item else ""
            away_item = self._table.item(row, 1)
            away = away_item.text().strip() if away_item else ""

            if home == "" and away == "":
                continue
            elif home == "":
                QMessageBox.critical(self, 'Home Missing', f'Home team missing at row {row}.')
                return None
            elif away == "":
                QMessageBox.critical(self, 'Away Missing', f'Away team missing at row {row}.')
                return None
            elif home == away:
                QMessageBox.critical(self, 'Same Teams', f'Found matches with a single team at row {row}.')
                return None

            try:
                odd_1 = float(self._table.item(row, 2).text().strip())
                odd_x = float(self._table.item(row, 3).text().strip())
                odd_2 = float(self._table.item(row, 4).text().strip())
            except (TypeError, ValueError, AttributeError):
                QMessageBox.critical(self, 'Invalid Odds', f'Found invalid odd values or missing at row {row}.')
                return None
            else:
                if odd_1 < 1.01 or odd_x < 1.01 or odd_2 < 1.01:
                    QMessageBox.critical(self, 'Invalid Odds', f'Found odds < 1.01 at row {row}.')
                    return None

                data.append([home, away, odd_1, odd_x, odd_2])
                indices.append(row)

        fixtures_df = pd.DataFrame(data=data, columns=['Home', 'Away', '1', 'X', '2'], index=indices)
        return fixtures_df

    def _prepare_odd_mask(self, df: pd.DataFrame):
        """ Prepares the standard odd mask for this league, based on the selected odds. """

        odd_1_filter = self._league.odd_1_range
        if odd_1_filter is not None:
            min_odd, max_odd = odd_1_filter
            self._odd_mask = ((df['1'] >= min_odd) & (df['1'] <= max_odd))
        else:
            self._odd_mask = np.array([1]*df.shape[0], dtype=bool)

        odd_x_filter = self._league.odd_x_range
        if odd_x_filter is not None:
            min_odd, max_odd = odd_x_filter
            odd_x_mask = ((df['X'] >= min_odd) & (df['X'] <= max_odd))
            self._odd_mask = self._odd_mask & odd_x_mask

        odd_2_filter = self._league.odd_2_range
        if odd_2_filter is not None:
            min_odd, max_odd = odd_2_filter
            odd_2_mask = ((df['2'] >= min_odd) & (df['2'] <= max_odd))
            self._odd_mask = self._odd_mask & odd_2_mask

    def _highlight_matches(self):
        """ Filters and highlights the matches using the specified league odds and the selected filters. """

        mask = self._odd_mask
        selected_filters = self._combo_filters.getSelectedTexts()

        if selected_filters and selected_filters[0] == '--- Select Filters ---':
            selected_filters = selected_filters[1:]
        if selected_filters:
            target_type = self._combo_target.currentData()
            all_filter_mask = np.zeros(shape=(mask.shape[0],), dtype=bool)
            for filter_id in selected_filters:
                # Filter odds.
                if filter_id != 'None':
                    odd, low, high = ast.literal_eval(filter_id)
                    odd_df = self._odds[odd]
                    odd_mask = (low <= odd_df) & (odd_df <= high)
                else:
                    odd_mask = np.ones(shape=(mask.shape[0],), dtype=bool)

                # Filter percentiles.
                prob_percentiles = self._percentiles[filter_id] if filter_id == 'None' else self._percentiles[ast.literal_eval(filter_id)]

                if target_type in {TargetType.RESULT, TargetType.HALF_RESULT}:
                    thresholds = np.float32([prob_percentiles['1'][1], prob_percentiles['X'][1], prob_percentiles['2'][1]])
                else:
                    thresholds = np.float32([prob_percentiles['U'][1], prob_percentiles['O'][1]])

                percentile_mask = np.all(self._y_prob >= thresholds, axis=1)
                filter_mask = odd_mask & percentile_mask
                all_filter_mask = all_filter_mask | filter_mask
            mask = mask & all_filter_mask

        highlight_ids = self._index[mask].tolist()

        if len(highlight_ids) > 0:
            self._table.highlight_rows(row_ids=highlight_ids)
        else:
            self._table.clear_selection()

    def _export(self):
        # Fetch the selected items.
        highlight_ids = sorted({index.row() for index in self._table.selectedIndexes()})

        if len(highlight_ids) == 0:
            QMessageBox.information(self, 'None Selected', 'Select the matches (rows) you want to export.')
            return

        # Export the selected items.
        data = []
        target_type = self._combo_target.currentData()
        for row in range(10):
            if row in highlight_ids:
                home_item = self._table.item(row, 0)
                home = home_item.text().strip() if home_item else ""
                away_item = self._table.item(row, 1)
                away = away_item.text().strip() if away_item else ""
                odd_1 = self._table.item(row, 2).text().strip()
                odd_x = self._table.item(row, 3).text().strip()
                odd_2 = self._table.item(row, 4).text().strip()
                predicted = self._table.item(row, 5).text().strip()
                data_row = [home, away, odd_1, odd_x, odd_2, predicted]

                if target_type in {TargetType.RESULT, TargetType.HALF_RESULT}:
                    data_row.extend([
                        float(self._table.item(row, 6).text().strip()),
                        float(self._table.item(row, 7).text().strip()),
                        float(self._table.item(row, 8).text().strip())
                    ])
                elif target_type == TargetType.OVER_UNDER:
                    data_row.extend([
                        float(self._table.item(row, 6).text().strip()),
                        float(self._table.item(row, 7).text().strip())
                    ])

                data.append(data_row)

        if target_type in {TargetType.RESULT, TargetType.HALF_RESULT}:
            df = pd.DataFrame(data=data, columns=['Home Team', 'Away Team', '1', 'X', '2', 'Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)'])
        elif target_type == TargetType.OVER_UNDER:
            df = pd.DataFrame(data=data, columns=['Home Team', 'Away Team', '1', 'X', '2', 'Predicted', 'Prob(U)', 'Prob(O)'])
        else:
            df = pd.DataFrame(data=data, columns=['Home Team', 'Away Team', '1', 'X', '2', 'Predicted'])

        default_filepath = f'{self._league.league_id}-fixures.csv'
        path, _ = QFileDialog.getSaveFileName(self, 'Export to CSV', default_filepath, 'CSV Files (*.csv)')

        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'

        file_exists = os.path.exists(path)
        try:
            if not file_exists:
                df.to_csv(path, mode='w', header=True, index=False)
            else:
                df.to_csv(path, mode='a', header=False, index=False)

            QMessageBox.information(self, 'Success', 'Export Completed!')
        except Exception as e:
            QMessageBox.critical(self, 'Export Failed', f'Could not export data.\n\nError:\n{e}')
