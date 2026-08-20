import pandas as pd
from datetime import date
from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QLabel, QLineEdit, QComboBox, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout
from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.gui.utils.taskrunner import TaskRunnerDialog
from src.gui.widgets.tables import SimpleTableDialog
from src.gui.i18n import team_en, team_zh
from src.preprocessing.utils.target import TargetType, class_to_score
from src.preprocessing.utils.inputs import construct_inputs_by_teams


CURRENT_TEAM_EXTRAS = {
    '英超': ['Coventry'],
}


class PredictorDialog(QDialog):
    """ Predictor dialog which predicts the outcome of a match. """

    def __init__(
            self,
            df: pd.DataFrame,
            model_db: ModelDatabase,
            league_db: Optional[LeagueDatabase] = None,
    ):
        super().__init__()

        self._league_db = league_db
        self._model_db = model_db
        self._df = pd.DataFrame()
        self._model_ids = []
        self._model_configs = {}
        self._title = '手动比赛预测'
        self._width = 500
        self._height = 230

        self._historical_teams = set()
        self._home_teams = []
        self._away_teams = []
        self._result_dict = {0: 'H', 1: 'D', 2: 'A'}
        self._result_uo_dict = {0: 'U', 1: 'O'}
        self._half_full_dict = {
            0: '胜/胜', 1: '胜/平', 2: '胜/负',
            3: '平/胜', 4: '平/平', 5: '平/负',
            6: '负/胜', 7: '负/平', 8: '负/负',
        }

        # Declare placeholders.
        self._target_types = {
            '胜平负（主胜/平/客胜）': TargetType.RESULT,
            '大小球 2.5': TargetType.OVER_UNDER,
            '半场胜平负': TargetType.HALF_RESULT,
            '半全场（9种结果）': TargetType.HALF_FULL,
            '准确比分（含6+）': TargetType.SCORE,
        }
        self._result_model_ids = []
        self._uo_model_ids = []
        self._half_result_model_ids = []
        self._half_full_model_ids = []
        self._score_model_ids = []
        self._set_prediction_context(df, model_db)

        # Declare UI Placeholders.
        self._combo_league = None
        self._combo_model = None
        self._combo_target = None
        self._combo_home = None
        self._combo_away = None
        self._edit_home_odd = None
        self._edit_draw_odd = None
        self._edit_away_odd = None
        self._predict_btn = None
        self._quality_label = None
        self._quality_base_text = '选择模型后显示最近比赛的真实测试表现。'

        self._initialize_window()
        self._add_widgets()

    def _initialize_window(self):
        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

    def _add_widgets(self):
        root = QVBoxLayout(self)
        root.addSpacing(15)

        if self._league_db is not None:
            league_ids = [
                league_id for league_id in self._league_db.get_league_ids()
                if self._model_db.index.get(league_id)
            ]
            league_hbox = QHBoxLayout()
            league_hbox.addStretch(1)
            self._combo_league = QComboBox()
            self._combo_league.setFixedWidth(220)
            for league_id in league_ids:
                self._combo_league.addItem(league_id, league_id)
            current = self._combo_league.findData(self._model_db.league_id)
            self._combo_league.setCurrentIndex(max(0, current))
            league_hbox.addWidget(QLabel('已训练联赛：'))
            league_hbox.addWidget(self._combo_league)
            league_hbox.addStretch(1)
            root.addLayout(league_hbox)

        # --- Model initialization ---
        model_hbox = QHBoxLayout()
        model_hbox.addStretch(1)
        self._combo_target = QComboBox()
        self._combo_target.setFixedWidth(120)
        for target, target_type in self._target_types.items():
            self._combo_target.addItem(target, target_type)
        self._combo_target.setCurrentIndex(-1)
        self._combo_target.currentIndexChanged.connect(self._on_target_change)
        model_hbox.addWidget(QLabel('预测类型：'))
        model_hbox.addWidget(self._combo_target)

        self._combo_model = QComboBox()
        self._combo_model.setFixedWidth(220)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.currentIndexChanged.connect(self._on_model_change)
        model_hbox.addWidget(QLabel('模型选择：'))
        model_hbox.addWidget(self._combo_model)
        model_hbox.addStretch(1)
        root.addLayout(model_hbox)

        self._quality_label = QLabel(self._quality_base_text)
        self._quality_label.setStyleSheet('color: #666;')
        root.addWidget(self._quality_label)

        teams_hbox = QHBoxLayout()
        teams_hbox.addStretch(1)
        self._combo_home = QComboBox()
        self._combo_home.setEditable(True)
        self._combo_home.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._combo_home.setFixedWidth(150)
        self._combo_home.lineEdit().setPlaceholderText('输入或搜索主队')
        self._combo_home.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._combo_home.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self._combo_home.currentTextChanged.connect(self._refresh_quality_label)
        teams_hbox.addWidget(QLabel(text='Home Team'))
        teams_hbox.addWidget(self._combo_home)
        teams_hbox.addWidget(QLabel(' vs '))

        self._combo_away = QComboBox()
        self._combo_away.setEditable(True)
        self._combo_away.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._combo_away.setFixedWidth(150)
        self._combo_away.lineEdit().setPlaceholderText('输入或搜索客队')
        self._combo_away.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._combo_away.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self._combo_away.currentTextChanged.connect(self._refresh_quality_label)
        teams_hbox.addWidget(QLabel(text='Away Team'))
        teams_hbox.addWidget(self._combo_away)
        teams_hbox.addStretch(1)
        root.addLayout(teams_hbox)
        self._populate_team_combos()

        odds_hbox = QHBoxLayout()
        odds_hbox.addStretch(1)
        self._edit_home_odd = QLineEdit(text='1.00')
        self._edit_home_odd.setFixedWidth(60)
        self._edit_home_odd.setPlaceholderText('Home Odd...')
        odds_hbox.addWidget(QLabel('1:'))
        odds_hbox.addWidget(self._edit_home_odd)

        self._edit_draw_odd = QLineEdit(text='1.00')
        self._edit_draw_odd.setFixedWidth(60)
        self._edit_draw_odd.setPlaceholderText('Home Odd...')
        odds_hbox.addWidget(QLabel('X:'))
        odds_hbox.addWidget(self._edit_draw_odd)

        self._edit_away_odd = QLineEdit(text='1.00')
        self._edit_away_odd.setFixedWidth(60)
        self._edit_away_odd.setPlaceholderText('Away Odd...')
        odds_hbox.addWidget(QLabel('2:'))
        odds_hbox.addWidget(self._edit_away_odd)
        odds_hbox.addStretch(1)
        root.addLayout(odds_hbox)

        self._predict_btn = QPushButton('Predict')
        self._predict_btn.setFixedWidth(100)
        self._predict_btn.setFixedHeight(30)
        self._predict_btn.clicked.connect(self._predict)
        self._predict_btn.setEnabled(False)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 10, 0, 0)
        btn_row.addStretch(1)
        btn_row.addWidget(self._predict_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)
        root.addStretch(1)

        if self._combo_league is not None:
            self._combo_league.currentIndexChanged.connect(self._on_league_change)
        self._combo_target.setCurrentIndex(0)

    def exec(self):
        if len(self._model_ids) == 0:
            QMessageBox.critical(
                self,
                'No Existing Models.',
                'There are no existing models to predict.',
                QMessageBox.StandardButton.Ok
            )
            return QDialog.Rejected

        super().exec()

    @staticmethod
    def _selected_team(combo: QComboBox) -> str:
        text = combo.currentText().strip()
        index = combo.currentIndex()
        if index >= 0 and text == combo.itemText(index):
            return str(combo.itemData(index) or text)
        return team_en(text)

    def _set_prediction_context(self, df: pd.DataFrame, model_db: ModelDatabase):
        """Switch the dialog to one trained league without rebuilding it."""
        self._df = df.reset_index(drop=True)
        self._model_db = model_db
        self._model_ids = model_db.get_model_ids()
        self._historical_teams = (
            set(self._df['Home'].dropna()) | set(self._df['Away'].dropna())
        )
        extra_teams = set(CURRENT_TEAM_EXTRAS.get(model_db.league_id, []))
        # A club can play either role. Separate Home/Away lists hid clubs that
        # only appeared in one column of a filtered or newly synced dataset.
        all_teams = sorted(self._historical_teams | extra_teams)
        self._home_teams = all_teams
        self._away_teams = all_teams

        self._result_model_ids = []
        self._uo_model_ids = []
        self._half_result_model_ids = []
        self._half_full_model_ids = []
        self._score_model_ids = []
        self._model_configs = {}
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
            elif config.get('target_type') == TargetType.SCORE and '让球胜负' not in model_id:
                self._score_model_ids.append(model_id)

        for model_ids in (
                self._result_model_ids,
                self._uo_model_ids,
                self._half_result_model_ids,
                self._half_full_model_ids,
                self._score_model_ids,
        ):
            model_ids.sort(key=self._model_quality_key, reverse=True)

    def _model_quality_key(self, model_id: str) -> tuple:
        """Rank saved models by honest holdout quality, never training accuracy."""
        config = self._model_configs.get(model_id, {})
        tuning = config.get('train', {}).get('tuning', {})
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
        # A measured holdout always outranks a model with no audit. Accuracy is
        # primary; sample count breaks ties in favor of the more stable result.
        return accuracy >= 0.0, accuracy, samples, model_id

    def _populate_team_combos(self):
        for combo, teams in (
                (self._combo_home, self._home_teams),
                (self._combo_away, self._away_teams),
        ):
            combo.blockSignals(True)
            combo.clear()
            for team in teams:
                combo.addItem(team_zh(team), team)
            combo.setCurrentIndex(-1)
            combo.blockSignals(False)

    def _on_league_change(self):
        league_id = self._combo_league.currentData()
        if not league_id or league_id == self._model_db.league_id:
            return

        runner = TaskRunnerDialog(
            title='切换联赛',
            info=f'正在载入 {league_id} 的比赛和模型…',
            task_fn=lambda: (
                self._league_db.load_league(league_id),
                ModelDatabase(league_id),
            ),
            parent=self,
        )
        result = runner.run()
        if runner.error_message is not None or not result or result[0] is None:
            QMessageBox.critical(
                self, '联赛载入失败', runner.error_message or f'{league_id} 没有可用数据。',
            )
            return
        self._set_prediction_context(*result)
        self._populate_team_combos()
        self._on_target_change()
        self._set_quality_text('请选择模型和两支球队。')

    def _set_quality_text(self, text: str):
        self._quality_base_text = text
        self._refresh_quality_label()

    def _refresh_quality_label(self):
        if self._quality_label is None:
            return
        selected = {
            self._selected_team(self._combo_home),
            self._selected_team(self._combo_away),
        }
        neutral = sorted(team for team in selected if team and team not in self._historical_teams)
        suffix = ''
        if neutral:
            shown = '、'.join(team_zh(team) for team in neutral)
            suffix = f' 新球队（{shown}）暂无本联赛历史，将使用联赛中位统计，置信度较低。'
        self._quality_label.setText(self._quality_base_text + suffix)

    def _on_target_change(self):
        """ Adds model ids based on the selected target. """

        # Disable Predict button.
        self._predict_btn.setEnabled(False)

        # Clearing model ids.
        self._combo_model.blockSignals(True)
        self._combo_model.clear()

        # Setting model. percentiles.
        target_type = self._combo_target.currentData()

        if target_type == TargetType.RESULT:
            model_ids = self._result_model_ids
        elif target_type == TargetType.OVER_UNDER:
            model_ids = self._uo_model_ids
        elif target_type == TargetType.HALF_RESULT:
            model_ids = self._half_result_model_ids
        elif target_type == TargetType.HALF_FULL:
            model_ids = self._half_full_model_ids
        elif target_type == TargetType.SCORE:
            model_ids = self._score_model_ids
        else:
            raise ValueError(f'Undefined targets: "{target_type}"')

        # Adding model ids.
        for model_id in model_ids:
            self._combo_model.addItem(model_id)
        self._combo_model.setCurrentIndex(0 if model_ids else -1)
        self._combo_model.blockSignals(False)
        self._on_model_change()

    def _on_model_change(self):
        model_id = self._combo_model.currentText()
        self._predict_btn.setEnabled(bool(model_id))
        if not model_id:
            self._set_quality_text('选择模型后显示最近比赛的真实测试表现。')
            return
        config = self._model_configs.get(model_id)
        if not config:
            self._predict_btn.setEnabled(False)
            self._set_quality_text('模型文件不完整，请重新同步或训练。')
            return
        tuning = config.get('train', {}).get('tuning', {})
        test_accuracy = tuning.get('test_accuracy')
        if isinstance(test_accuracy, (int, float)):
            majority_baseline = tuning.get('majority_baseline')
            text = (
                f"最近比赛独立测试：全部场次命中率 {test_accuracy:.1%}"
            )
            if isinstance(majority_baseline, (int, float)):
                text += f"；简单基线 {majority_baseline:.1%}"
            if config['target_type'] in {TargetType.SCORE, TargetType.HALF_FULL}:
                top3 = tuning.get('top3_accuracy')
                top5 = tuning.get('top5_accuracy')
                if isinstance(top3, (int, float)):
                    text += f"；Top-3 {top3:.1%}"
                if isinstance(top5, (int, float)):
                    text += f"；Top-5 {top5:.1%}"
            if isinstance(tuning.get('selective_accuracy'), (int, float)):
                text += (
                    f"；高置信度出手命中率 {tuning['selective_accuracy']:.1%}"
                )
                if isinstance(tuning.get('coverage'), (int, float)):
                    text += f"；覆盖 {tuning['coverage']:.1%} 场次"
                if 'selective_samples' in tuning:
                    text += f"（{tuning['selective_samples']} 场）"
                if tuning.get('selective_validated') is False:
                    text += '。注意：高置信度门槛未在验证集达到稳定目标'
            if isinstance(majority_baseline, (int, float)) and test_accuracy <= majority_baseline:
                text += '。注意：首选结果未显示出稳定优势，请仅参考候选概率。'
            elif (
                    isinstance(majority_baseline, (int, float))
                    and tuning.get('mcnemar_p_value_vs_baseline') is not None
                    and tuning.get('mcnemar_p_value_vs_baseline', 0.0) >= 0.05
            ):
                text += '。注意：相对简单基线的优势尚未通过稳定性检验。'
            self._set_quality_text(text)
        else:
            self._set_quality_text('该模型没有独立调参测试记录。')

    def _predict(self):
        if not self._validate_inputs():
            return

        # Constructing model input.
        raw_match_df = pd.DataFrame({
            'Date': [date.today().strftime(format='%Y-%m-%d')],
            'Home': [self._selected_team(self._combo_home)],
            'Away': [self._selected_team(self._combo_away)],
            '1': [float(self._edit_home_odd.text().strip())],
            'X': [float(self._edit_draw_odd.text().strip())],
            '2': [float(self._edit_away_odd.text().strip())]
        })
        model_id = self._combo_model.currentText()
        runner = TaskRunnerDialog(
            title='比赛预测',
            info='正在载入模型并计算概率…',
            task_fn=lambda: self._predict_match(raw_match_df, model_id),
            parent=self,
        )
        result = runner.run()
        if runner.error_message is not None or result is None:
            QMessageBox.critical(self, '预测失败', runner.error_message or '模型没有返回结果。')
            return
        match_df, y_pred, y_prob, model_config, score_classes = result

        # Show table dialog.
        match_df = match_df[['Date', 'Season', 'Week', 'Home', 'Away', '1', 'X', '2']]

        # Adding predictions to Dataframe.
        target_type = self._combo_target.currentData()

        if target_type == TargetType.RESULT:
            match_df['Predicted'] = self._result_dict[y_pred[0]]
            match_df['Prob(1)'] = y_prob[0]
            match_df['Prob(X)'] = y_prob[1]
            match_df['Prob(2)'] = y_prob[2]
        elif target_type == TargetType.OVER_UNDER:
            match_df['Predicted'] = self._result_uo_dict[y_pred[0]]
            match_df['Prob(U)'] = y_prob[0]
            match_df['Prob(O)'] = y_prob[1]
        elif target_type == TargetType.HALF_RESULT:
            threshold = model_config.get('train', {}).get('tuning', {}).get('selective_threshold')
            confident = threshold is None or float(y_prob.max()) >= float(threshold)
            match_df['Predicted'] = (
                self._result_dict[y_pred[0]]
                if confident
                else '低置信度：暂不预测'
            )
            match_df['半场主胜概率'] = y_prob[0]
            match_df['半场平局概率'] = y_prob[1]
            match_df['半场客胜概率'] = y_prob[2]
        elif target_type == TargetType.HALF_FULL:
            match_df['Predicted'] = self._half_full_dict[y_pred[0]]
            labels = list(self._half_full_dict.values())
            for index, label in enumerate(labels):
                match_df[f'概率({label})'] = y_prob[index]
        elif target_type == TargetType.SCORE:
            ranked = sorted(
                zip(score_classes, y_prob),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            match_df['Predicted'] = class_to_score(ranked[0][0])
            match_df['首选比分概率'] = ranked[0][1]
            for rank, (target_class, probability) in enumerate(ranked[1:], start=1):
                match_df[f'备选比分{rank}'] = class_to_score(target_class)
                match_df[f'备选{rank}概率'] = probability
        else:
            raise ValueError(f'Undefined target type: "{target_type}".')

        SimpleTableDialog(df=match_df, parent=self, title='Prediction', readonly=False).show()

    def _predict_match(self, raw_match_df: pd.DataFrame, model_id: str):
        """Pure worker task: no Qt widgets are read or modified here."""
        match_df = construct_inputs_by_teams(df=self._df, match_df=raw_match_df)
        model, model_config = self._model_db.load_model(model_id=model_id)
        if model is None or model_config is None:
            raise RuntimeError(f'模型文件缺失：{model_id}')
        probabilities = model.predict_proba(df=match_df)
        predictions = probabilities.argmax(axis=1)
        score_classes = getattr(model.classifier, 'classes_', ())
        return (
            match_df,
            predictions,
            probabilities[0].round(4),
            model_config,
            score_classes,
        )

    def _validate_inputs(self) -> bool:
        home = self._selected_team(self._combo_home)
        away = self._selected_team(self._combo_away)
        if not home or not away:
            QMessageBox.critical(self, 'Missing Teams', 'Please select both home and away teams.')
            return False
        if home == away:
            QMessageBox.critical(self, 'Same Teams', 'Home and away teams must be different.')
            return False

        odds = [
            self._edit_home_odd.text(),
            self._edit_draw_odd.text(),
            self._edit_away_odd.text(),
        ]
        for odd in odds:
            try:
                odd = float(odd.strip())
            except ValueError:
                QMessageBox.critical(self, 'Invalid Odds', f'Odd "{odd}" is not numeric.')
                return False
            else:
                if odd <= 1.0:
                    QMessageBox.critical(self, 'Invalid Odds', f'Odds cannot be less than 1.00, found "{odd}".')
                    return False
        return True
