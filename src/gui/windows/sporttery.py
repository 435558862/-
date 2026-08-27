import json
import os
from queue import Empty, Queue
import re
import sys
from threading import Thread
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QStyle, QStyledItemDelegate, QVBoxLayout, QWidget,
)

from src.database.model import ModelDatabase
from src.gui.utils.taskrunner import TaskRunnerDialog
from src.gui.widgets.tables import ExcelTable
from src.network.fixtures.sporttery import SportteryMobileClient
from src.services.daily_learning import review_and_learn
from src.services.daily_sporttery import (
    LEAGUE_ALIASES, _sort_by_match_number, backfill_missing_simulations,
    run_daily_sporttery,
)
from src.services.market_trends import build_trend_rows, live_snapshot_from_match, summarize_trend
from src.services.odds_tracking import (
    format_market_flow, read_odds_series, record_odds_snapshots,
    record_official_history,
)
from src.services.yesterday_review import load_yesterday_hit_report


REPORT_ROOT = Path('storage/jingcai/reports')
PREDICTION_PATH = REPORT_ROOT / '最新竞彩预测.csv'
LEARNING_STATUS_PATH = Path('storage/jingcai/learning/status.json')


def prediction_export_root() -> Path:
    """Return a writable, platform-native prediction export directory."""
    configured = os.environ.get('PROPHITBET_EXPORT_DIR', '').strip()
    if configured:
        return Path(configured).expanduser()
    if sys.platform in {'darwin', 'win32'}:
        return Path.home() / 'Desktop'
    wsl_desktop = Path('/mnt/c/Users/Administrator/Desktop')
    if wsl_desktop.is_dir():
        return wsl_desktop
    return Path.home()


def display_export_path(path: Path) -> str:
    """Format a native path for the completion message."""
    text = str(path)
    if sys.platform.startswith('linux') and text.startswith('/mnt/c/'):
        return text.replace('/mnt/c', 'C:', 1).replace('/', '\\')
    return text
ALL_MODELS = '__all__'
DEDICATED_MODELS = '__dedicated__'
GENERIC_MODELS = '__generic__'
SIMULATION_MODELS = '__simulation__'
INDEPENDENT_SIMULATION_SOURCE = '历史攻防双泊松蒙特卡洛'
DEDICATED_LEAGUE_COLUMN = '专用模型联赛'
REQUIRED_DEDICATED_MODELS = ('胜平负模型', '大小球模型', '比分模型', '半全场模型')


class _ComboPopupDelegate(QStyledItemDelegate):
    """Paint combo rows explicitly so desktop themes cannot force black selection."""

    def paint(self, painter, option, index):
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        background = '#2f79bd' if selected else '#dcebf8' if hovered else '#ffffff'
        foreground = '#ffffff' if selected else '#202020'
        painter.save()
        painter.fillRect(option.rect, QColor(background))
        painter.setPen(QColor(foreground))
        painter.drawText(
            option.rect.adjusted(6, 0, -4, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            str(index.data(Qt.ItemDataRole.DisplayRole) or ''),
        )
        painter.restore()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(25)
        return size


class _WindowDragBar(QWidget):
    """In-window drag and close controls for platforms that hide dialog chrome."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._drag_offset = None
        self.setObjectName('windowDragBar')
        self.setFixedHeight(30)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 1, 3, 1)
        title_label = QLabel(title, self)
        title_label.setObjectName('windowDragTitle')
        layout.addWidget(title_label)
        layout.addStretch(1)
        close_button = QPushButton('✕', self)
        close_button.setObjectName('windowCloseButton')
        close_button.setToolTip('关闭')
        close_button.setFixedSize(28, 24)
        close_button.clicked.connect(lambda: self.window().close())
        layout.addWidget(close_button)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.window().move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)


class YesterdayHitDetailsDialog(QDialog):
    """A compact, cache-only view of yesterday's honest settlement details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        frame, summary = load_yesterday_hit_report()
        title = (
            f'{summary["date"]} 最近已结算命中（昨日待补）'
            if summary.get('is_fallback') else f'{summary["date"]} 昨日命中明细'
        )
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(1280, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(7, 7, 7, 7)
        root.setSpacing(5)
        headline = QLabel(summary['headline'])
        headline.setObjectName('yesterdayHeadline')
        headline.setWordWrap(True)
        root.addWidget(headline)
        if summary['patterns']:
            patterns = QLabel(f'规律：{summary["patterns"]}')
            patterns.setObjectName('yesterdayPatterns')
            patterns.setWordWrap(True)
            root.addWidget(patterns)
        table = ExcelTable(
            parent=self,
            df=frame,
            readonly=True,
            supports_sorting=True,
            supports_query_search=False,
        )
        self._highlight_hit_rows(table, frame)
        root.addWidget(table)
        self.setStyleSheet('''
            QLabel#yesterdayHeadline {
                color: #202020; font-size: 12px; font-weight: 600;
                padding: 4px 5px; background: #eef5fb;
                border: 1px solid #c8d9e8; border-radius: 3px;
            }
            QLabel#yesterdayPatterns {
                color: #404040; font-size: 12px; padding: 2px 5px;
            }
            QTableWidget {
                color: #202020; background: #ffffff;
                alternate-background-color: #f6f8fa;
                gridline-color: #cfd4d8; font-size: 12px;
            }
            QHeaderView::section {
                color: #202020; background: #e8edf1;
                border: 0; border-right: 1px solid #bdc5cb;
                border-bottom: 1px solid #9da5ab;
                padding: 2px 4px; font-size: 12px; font-weight: 600;
            }
        ''')

    @staticmethod
    def _highlight_hit_rows(table: ExcelTable, frame: pd.DataFrame) -> None:
        """Highlight only the market cells that actually hit."""
        foreground = QBrush(QColor('#c62828'))
        background = QBrush(QColor('#fff1f1'))
        hit_marks = ('（命中）', '（首中）', '（次中）', '（次1中）', '（次2中）',
                     '（冷中）', '（进中）')
        for row_index in range(table.rowCount()):
            for column_index in range(table.columnCount()):
                item = table.item(row_index, column_index)
                if item is not None and any(mark in item.text() for mark in hit_marks):
                    item.setForeground(foreground)
                    item.setBackground(background)
                    item.setToolTip('该项命中')


class _MarketTrendChart(QWidget):
    """Small dependency-free H/D/A implied-probability chart."""

    COLORS = {'H': '#ef5350', 'D': '#2f80d0', 'A': '#18a66a'}
    LABELS = {'H': '胜', 'D': '平', 'A': '负'}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self.setMinimumHeight(300)

    def set_rows(self, rows):
        self._rows = list(rows or [])
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#ffffff'))
        left, top, right, bottom = 48, 28, 18, 34
        width = max(1, self.width() - left - right)
        height = max(1, self.height() - top - bottom)
        painter.setFont(QFont('', 9))
        values = [
            float(row[key])
            for row in self._rows for key in ('H', 'D', 'A')
            if row.get(key) is not None
        ]
        minimum, maximum = 0.0, 1.0
        if values:
            span = max(values) - min(values)
            padding = max(0.012, span * 0.12)
            minimum = max(0.0, min(values) - padding)
            maximum = min(1.0, max(values) + padding)
            if maximum - minimum < 0.04:
                middle = (maximum + minimum) / 2
                minimum, maximum = max(0.0, middle - 0.02), min(1.0, middle + 0.02)
        value_span = max(0.0001, maximum - minimum)
        for step in range(6):
            value = minimum + value_span * step / 5
            y = top + height * (1.0 - (value - minimum) / value_span)
            painter.setPen(QPen(QColor('#dce3e9'), 1))
            painter.drawLine(left, int(y), left + width, int(y))
            painter.setPen(QColor('#66717a'))
            painter.drawText(4, int(y) - 8, 40, 16, Qt.AlignmentFlag.AlignRight,
                             f'{value:.0%}')
        legend_x = left
        for key in ('H', 'D', 'A'):
            painter.setPen(QPen(QColor(self.COLORS[key]), 3))
            painter.drawLine(legend_x, 12, legend_x + 18, 12)
            painter.setPen(QColor('#303840'))
            painter.drawText(legend_x + 23, 4, 30, 17,
                             Qt.AlignmentFlag.AlignLeft, self.LABELS[key])
            legend_x += 62
        if not self._rows:
            painter.setPen(QColor('#6b747c'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             '正在等待实时赔率…')
            return
        denominator = max(1, len(self._rows) - 1)
        for key in ('H', 'D', 'A'):
            path = QPainterPath()
            for index, row in enumerate(self._rows):
                x = left + width * index / denominator
                y = top + height * (1.0 - (float(row[key]) - minimum) / value_span)
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(QColor(self.COLORS[key]), 2))
            painter.drawPath(path)
            painter.setBrush(QColor(self.COLORS[key]))
            for point_index, point_row in enumerate(self._rows):
                point_x = left + width * point_index / denominator
                point_y = top + height * (
                    1.0 - (float(point_row[key]) - minimum) / value_span
                )
                painter.drawEllipse(int(point_x) - 3, int(point_y) - 3, 6, 6)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        label_indexes = sorted({0, len(self._rows) // 2, len(self._rows) - 1})
        painter.setPen(QColor('#66717a'))
        for index in label_indexes:
            raw_label = str(self._rows[index].get('label') or '')
            label = (
                f'初盘 {raw_label[-5:]}'
                if self._rows[index].get('is_opening') else raw_label[-8:]
            )
            x = left + width * index / denominator
            painter.drawText(int(x) - 38, top + height + 8, 76, 18,
                             Qt.AlignmentFlag.AlignCenter, label)


class MarketTrendDialog(QDialog):
    """Live H/D/A trend chart polling the official mobile odds feed."""

    def __init__(self, predictions: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle('实时市场走势图')
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(1080, 640)
        self._series = {
            key: list(rows) for key, rows in read_odds_series(
                max_rows_per_match=300, keep_opening=True,
            ).items()
        }
        self._live_queue = Queue()
        self._live_running = False
        self._did_select_live = False
        self._table = None

        root = QVBoxLayout(self)
        root.setContentsMargins(7, 7, 7, 7)
        root.setSpacing(5)
        controls = QHBoxLayout()
        controls.addWidget(QLabel('比赛'))
        self._match_selector = QComboBox()
        self._match_selector.setMinimumWidth(430)
        controls.addWidget(self._match_selector)
        controls.addWidget(QLabel('区间'))
        self._range_selector = QComboBox()
        for label, value in (('6小时', 6), ('12小时', 12), ('24小时', 24),
                             ('72小时', 72), ('全部', None)):
            self._range_selector.addItem(label, value)
        self._range_selector.setCurrentIndex(2)
        controls.addWidget(self._range_selector)
        controls.addStretch(1)
        root.addLayout(controls)

        self._live_status = QLabel('实时数据源：竞彩网官方移动端｜正在连接…')
        self._live_status.setObjectName('liveMarketStatus')
        self._live_status.setWordWrap(True)
        root.addWidget(self._live_status)
        self._summary = QLabel()
        self._summary.setObjectName('marketTrendSummary')
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)
        self._chart = _MarketTrendChart(self)
        root.addWidget(self._chart, 3)
        self._table_container = QWidget(self)
        self._table_layout = QVBoxLayout(self._table_container)
        self._table_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._table_container, 2)

        seen = set()
        for _, row in predictions.iterrows():
            value = row.get('比赛ID')
            try:
                match_id = str(int(value))
            except (TypeError, ValueError):
                match_id = str(value or '').strip()
            if not match_id or match_id in seen:
                continue
            seen.add(match_id)
            label = (
                f'{row.get("赛事编号", "")}  {row.get("联赛", "")}  '
                f'{row.get("主队", "")} vs {row.get("客队", "")}'
            )
            self._match_selector.addItem(label.strip(), match_id)
        if self._match_selector.count() == 0:
            for match_id, rows in self._series.items():
                latest = rows[-1] if rows else {}
                label = (
                    f'{latest.get("match_num", "")}  {latest.get("league", "")}  '
                    f'{latest.get("home", "")} vs {latest.get("away", "")}'
                )
                self._match_selector.addItem(label.strip(), match_id)

        self._match_selector.currentIndexChanged.connect(self._render)
        self._range_selector.currentIndexChanged.connect(self._render)
        self.setStyleSheet('''
            QLabel#marketTrendSummary {
                color: #1f2b34; background: #edf5fb;
                border: 1px solid #c2d8e8; border-radius: 3px;
                padding: 6px; font-size: 12px; font-weight: 600;
            }
            QComboBox {
                color: #202020; background: #ffffff;
                border: 1px solid #aeb5bb; border-radius: 3px;
                padding: 2px 20px 2px 5px;
            }
        ''')
        self._render()

        self._live_poll_timer = QTimer(self)
        self._live_poll_timer.setInterval(60_000)
        self._live_poll_timer.timeout.connect(self._start_live_poll)
        self._live_poll_timer.start()
        self._live_result_timer = QTimer(self)
        self._live_result_timer.setInterval(250)
        self._live_result_timer.timeout.connect(self._collect_live_poll)
        self._live_result_timer.start()
        QTimer.singleShot(0, self._start_live_poll)

    def _start_live_poll(self):
        if self._live_running:
            return
        self._live_running = True
        self._live_status.setText(
            '实时数据源：竞彩网官方移动端｜正在刷新…｜每60秒自动采集'
        )
        selected_id = str(self._match_selector.currentData() or '')
        Thread(
            target=self._fetch_live_market, args=(selected_id,), daemon=True,
        ).start()

    def _fetch_live_market(self, selected_id=''):
        captured_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
        try:
            client = SportteryMobileClient(
                timeout=8.0, retries=1,
            )
            matches = client.selling_matches()
            history = (
                client.fixed_bonus_history(selected_id) if selected_id else {}
            )
            self._live_queue.put(('ok', captured_at, (matches, history, selected_id)))
        except Exception as error:
            self._live_queue.put(('error', captured_at, str(error)))

    def _collect_live_poll(self):
        try:
            state, captured_at, payload = self._live_queue.get_nowait()
        except Empty:
            return
        self._live_running = False
        shown_time = datetime.fromisoformat(
            captured_at.replace('Z', '+00:00'),
        ).astimezone().strftime('%H:%M:%S')
        if state == 'error':
            self._live_status.setText(
                f'实时数据暂不可用｜{shown_time}｜{str(payload)[:120]}'
            )
            return
        matches, official_history, history_match_id = payload
        persisted = record_odds_snapshots(
            matches, captured_at=captured_at,
        )
        history_added = record_official_history(
            history_match_id, official_history,
        ) if official_history and history_match_id else 0
        # Reload after both writes so the chart starts with the true official
        # opening row and includes all historical handicap prices immediately.
        self._series = {
            key: list(rows) for key, rows in read_odds_series(
                max_rows_per_match=300, keep_opening=True,
            ).items()
        }
        live_ids = set()
        first_live_id = None
        selector_ids = {
            str(self._match_selector.itemData(index) or '')
            for index in range(self._match_selector.count())
        }
        appended = 0
        unchanged = 0
        for raw in matches:
            snapshot = live_snapshot_from_match(raw, captured_at)
            if snapshot is None:
                continue
            match_id = snapshot['match_id']
            live_ids.add(match_id)
            if first_live_id is None:
                first_live_id = match_id
            if match_id not in selector_ids:
                match_number = (
                    raw.get('matchNumStr') or raw.get('matchNum') or ''
                )
                league = raw.get('leagueAllName') or raw.get('leagueAbbName') or ''
                home = raw.get('homeTeamAllName') or raw.get('homeTeamAbbName') or ''
                away = raw.get('awayTeamAllName') or raw.get('awayTeamAbbName') or ''
                label = f'{match_number}  {league}  {home} vs {away}'.strip()
                self._match_selector.addItem(label, match_id)
                selector_ids.add(match_id)
            rows = self._series.setdefault(match_id, [])
            if rows and self._same_market_snapshot(rows[-1], snapshot):
                unchanged += 1
            else:
                rows.append(snapshot)
                self._series[match_id] = rows[-300:]
                appended += 1
        selected_id = str(self._match_selector.currentData() or '')
        if not self._did_select_live and first_live_id:
            if selected_id not in live_ids:
                for index in range(self._match_selector.count()):
                    if str(self._match_selector.itemData(index) or '') == first_live_id:
                        self._match_selector.setCurrentIndex(index)
                        selected_id = first_live_id
                        break
            self._did_select_live = True
        selected_note = (
            '当前比赛已更新' if selected_id in live_ids
            else '当前比赛已停售或官方暂未返回'
        )
        self._live_status.setText(
            f'实时数据源：竞彩网官方移动端｜{shown_time}｜'
            f'补录官方初盘/变盘{history_added}点、入库{persisted}场、'
            f'新增走势点{appended}场、未变{unchanged}场｜'
            f'{selected_note}｜每60秒自动采集'
        )
        self._render()

    @staticmethod
    def _same_market_snapshot(left: dict, right: dict) -> bool:
        """Do not draw artificial horizontal segments for unchanged odds."""
        keys = ('had', 'hhad', 'ttg')
        return all((left.get(key) or {}) == (right.get(key) or {}) for key in keys)

    def closeEvent(self, event):
        self._live_poll_timer.stop()
        self._live_result_timer.stop()
        super().closeEvent(event)

    def _render(self):
        match_id = self._match_selector.currentData()
        hours = self._range_selector.currentData()
        rows = build_trend_rows(match_id, self._series, hours=hours)
        summary = summarize_trend(rows)
        probabilities = summary.get('latest_probabilities') or {}
        probability_text = ' / '.join(
            f'{label}{probabilities.get(key, 0):.1%}'
            for key, label in (('H', '胜'), ('D', '平'), ('A', '负'))
        ) if probabilities else '--'
        self._summary.setText(
            f'综合结论：{summary["conclusion"]}　｜　'
            f'{summary["handicap"]}　｜　{summary["total_goals"]}　｜　'
            f'稳定性：{summary["stability"]}　｜　实时曲线点：{summary["observations"]}个\n'
            f'当前胜平负：{probability_text}　｜　'
            f'初盘基准：{rows[0]["label"] if rows else "--"}'
        )
        self._chart.set_rows(rows)
        while self._table_layout.count():
            item = self._table_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        frame = pd.DataFrame([{
            '记录时间': row['label'],
            '胜赔': f'{row["had_H"]:.2f}',
            '平赔': f'{row["had_D"]:.2f}',
            '负赔': f'{row["had_A"]:.2f}',
            '胜概率': f'{row["H"]:.1%}', '平概率': f'{row["D"]:.1%}',
            '负概率': f'{row["A"]:.1%}',
            '让球线': '' if row['hhad_line'] is None else f'{row["hhad_line"]:g}',
            '让球胜赔': '' if row['hhad_H'] is None else f'{row["hhad_H"]:.2f}',
            '让球平赔': '' if row['hhad_D'] is None else f'{row["hhad_D"]:.2f}',
            '让球负赔': '' if row['hhad_A'] is None else f'{row["hhad_A"]:.2f}',
            '大球': '' if row['over'] is None else f'{row["over"]:.1%}',
            '小球': '' if row['under'] is None else f'{row["under"]:.1%}',
        } for row in rows])
        self._table = ExcelTable(
            parent=self, df=frame, readonly=True, supports_sorting=False,
            supports_query_search=False,
        )
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        widths = (190, 55, 55, 55, 65, 65, 65, 62, 76, 76, 76, 72, 72)
        for column_index, width in enumerate(widths):
            if column_index < self._table.columnCount():
                self._table.setColumnWidth(column_index, width)
        self._table.verticalHeader().setDefaultSectionSize(26)
        self._table_layout.addWidget(self._table)


def _safe_filename(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', '-', str(value)).strip(' .-') or '当前模型'


def write_predictions_xlsx(frame: pd.DataFrame, path: Path):
    """Write a Windows-ready workbook with readable headers and filters."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        frame.to_excel(writer, sheet_name='竞彩预测', index=False)
        sheet = writer.sheets['竞彩预测']
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(color='FFFFFF', bold=True)
            cell.fill = PatternFill('solid', fgColor='1F4E78')
            cell.alignment = Alignment(horizontal='center', vertical='center')
        for index, column in enumerate(frame.columns, start=1):
            values = [str(column), *(str(value) for value in frame[column].fillna('').head(500))]
            sheet.column_dimensions[get_column_letter(index)].width = min(
                36, max(10, max(map(len, values)) + 2),
            )
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical='center')


def trained_dedicated_leagues() -> list[str]:
    """List leagues with the complete model set used by daily predictions."""
    if not LEAGUE_ALIASES:
        return []
    try:
        probe = ModelDatabase(next(iter(LEAGUE_ALIASES)))
        index = probe.index
    except Exception:
        # A damaged/temporarily unavailable index must not prevent the daily
        # prediction window from opening; report-derived options still work.
        return []
    result = []
    for league in LEAGUE_ALIASES:
        model_ids = set(index.get(league, {}))
        required = {f'{league}{suffix}' for suffix in REQUIRED_DEDICATED_MODELS}
        if required.issubset(model_ids):
            result.append(league)
    return result


def _prediction_model_scopes(df: pd.DataFrame) -> pd.Series:
    """Return the dedicated league used by every row, or an empty scope."""
    if DEDICATED_LEAGUE_COLUMN in df.columns:
        return df[DEDICATED_LEAGUE_COLUMN].fillna('').astype(str)
    # Backward compatibility for reports created before model scope was saved.
    basis = df.get('预测依据', pd.Series('', index=df.index)).fillna('').astype(str)
    leagues = df.get('联赛', pd.Series('', index=df.index)).fillna('').astype(str)
    return leagues.where(basis.eq('历史数据训练模型'), '')


def _upcoming_predictions(
        predictions: pd.DataFrame,
        now: datetime | None = None,
) -> pd.DataFrame:
    """Keep only fixtures that have not kicked off; retain malformed dates visibly."""
    if predictions.empty or '比赛时间' not in predictions.columns:
        return predictions.copy()
    kickoff = pd.to_datetime(predictions['比赛时间'], errors='coerce')
    return predictions.loc[
        kickoff.isna() | kickoff.gt(now or datetime.now())
    ].copy()


def _score_recommendation_mask(predictions: pd.DataFrame) -> pd.Series:
    """Rows whose audited top-score probability reaches the 12% gate."""
    status = predictions.get(
        '比分推荐状态', pd.Series('', index=predictions.index),
    ).fillna('').astype(str)
    probability = pd.to_numeric(
        predictions.get(
            '原始最高概率比分概率', pd.Series(float('nan'), index=predictions.index),
        ),
        errors='coerce',
    )
    return status.eq('推荐') | (status.eq('') & probability.ge(0.12))


def _daily_priority_aspects(predictions: pd.DataFrame) -> pd.Series:
    """Pick at most one strongest row per day and market for red emphasis."""
    aspects = pd.Series([[] for _ in range(len(predictions))], index=predictions.index)
    if predictions.empty:
        return aspects
    days = predictions.get(
        '比赛时间', pd.Series('', index=predictions.index),
    ).fillna('').astype(str).str.slice(0, 10)
    days = days.where(days.str.fullmatch(r'\d{4}-\d{2}-\d{2}'), '全部')
    gate = predictions.get(
        '盘口门控', pd.Series('', index=predictions.index),
    ).fillna('').astype(str)
    stable = ~gate.str.contains('冲突|震荡|不稳定', regex=True)

    def numbers(column: str) -> pd.Series:
        return pd.to_numeric(
            predictions.get(column, pd.Series(float('nan'), index=predictions.index)),
            errors='coerce',
        )

    advice = predictions.get(
        '建议状态', pd.Series('', index=predictions.index),
    ).fillna('').astype(str)
    candidates = {
        '胜负': (
            advice.isin(('精选主推', '高置信主推'))
            & numbers('胜平负首选概率').ge(0.625) & stable,
            numbers('胜平负首选概率'),
        ),
        '让球': (
            numbers('让球首选概率').ge(0.60)
            & numbers('让球最大概率优势').fillna(0).ge(0.03) & stable,
            numbers('让球首选概率'),
        ),
        '大小球': (
            numbers('大小球首选概率').ge(0.60) & stable,
            numbers('大小球首选概率'),
        ),
        '半全场': (
            numbers('半全场首选概率').ge(0.35) & stable,
            numbers('半全场首选概率'),
        ),
        '比分': (
            _score_recommendation_mask(predictions) & stable,
            numbers('原始最高概率比分概率'),
        ),
    }
    for _, group_indices in days.groupby(days, sort=False).groups.items():
        group_indices = list(group_indices)
        for label, (eligible, strength) in candidates.items():
            available = [index for index in group_indices if bool(eligible.loc[index])]
            if not available:
                continue
            best = strength.loc[available].idxmax()
            aspects.at[best] = [*aspects.at[best], label]
    return aspects


def filter_predictions_by_model(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Strictly isolate one dedicated league or the generic/market rows."""
    if df.empty or model_key in ('', ALL_MODELS):
        return df.copy()
    if model_key == SIMULATION_MODELS:
        simulations = pd.to_numeric(
            df.get('模拟次数', pd.Series(0, index=df.index)), errors='coerce',
        ).fillna(0)
        result = df.get(
            '模拟胜负', pd.Series('', index=df.index),
        ).fillna('').astype(str).str.strip()
        return df.loc[simulations.gt(0) | result.ne('')].copy()
    scopes = _prediction_model_scopes(df)
    if model_key == DEDICATED_MODELS:
        mask = scopes.ne('')
    elif model_key == GENERIC_MODELS:
        mask = scopes.eq('')
    else:
        mask = scopes.eq(model_key)
    return df.loc[mask].reset_index(drop=True)


class SportteryPredictionsDialog(QDialog):
    """Synchronize and display today's official Sporttery model predictions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('今日竞彩概率分析')
        # Force a regular native top-level window. On macOS a parented QDialog
        # can otherwise be presented as a sheet without usable title-bar
        # controls, leaving no close button or area from which to drag it.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(1250, 650)
        self._predictions = pd.DataFrame()
        self._table_container = QWidget()
        self._table_layout = QVBoxLayout(self._table_container)
        self._table_layout.setContentsMargins(0, 0, 0, 0)
        self._summary = QLabel()
        self._model_selector = QComboBox()
        self._date_selector = QComboBox()
        self._advice_selector = QComboBox()
        self._yesterday_dialog = None
        self._market_dialog = None
        self._odds_pulse_running = False
        self._odds_pulse_queue = Queue()
        self._trained_leagues = trained_dedicated_leagues()
        self._build_ui()
        self._load_saved_reports()
        self._odds_pulse_timer = QTimer(self)
        self._odds_pulse_timer.setInterval(5 * 60_000)
        self._odds_pulse_timer.timeout.connect(self._start_odds_pulse)
        self._odds_pulse_timer.start()
        self._odds_pulse_result_timer = QTimer(self)
        self._odds_pulse_result_timer.setInterval(300)
        self._odds_pulse_result_timer.timeout.connect(self._collect_odds_pulse)
        self._odds_pulse_result_timer.start()
        QTimer.singleShot(12_000, self._start_odds_pulse)

    def _start_odds_pulse(self):
        """Persist actual market changes while the prediction window is open."""
        if self._odds_pulse_running:
            return
        self._odds_pulse_running = True
        Thread(target=self._fetch_odds_pulse, daemon=True).start()

    def _fetch_odds_pulse(self):
        try:
            matches = SportteryMobileClient(timeout=8.0, retries=1).selling_matches()
            saved = record_odds_snapshots(matches)
            self._odds_pulse_queue.put(('ok', len(matches), saved))
        except Exception as error:
            self._odds_pulse_queue.put(('error', 0, str(error)))

    def _collect_odds_pulse(self):
        try:
            state, count, payload = self._odds_pulse_queue.get_nowait()
        except Empty:
            return
        self._odds_pulse_running = False
        if state == 'ok':
            self._summary.setToolTip(
                f'后台盘口采集：{count}场，新增真实变化 {payload} 条。'
            )

    def closeEvent(self, event):
        self._odds_pulse_timer.stop()
        self._odds_pulse_result_timer.stop()
        super().closeEvent(event)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(5)

        root.addWidget(_WindowDragBar(self.windowTitle(), self))

        action_bar = QHBoxLayout()
        action_bar.setSpacing(5)
        sync_button = QPushButton('同步赔率并更新分析')
        sync_button.setToolTip('重新获取官方最新赔率；建议在安全截止前20～60分钟执行')
        sync_button.clicked.connect(self._sync)
        review_button = QPushButton('补赛果并复盘')
        review_button.clicked.connect(self._review_history)
        yesterday_button = QPushButton('昨日命中复盘')
        yesterday_button.setToolTip('查看昨日已结算场次的逐项命中明细和简短规律')
        yesterday_button.clicked.connect(self._show_yesterday_hits)
        market_button = QPushButton('市场走势')
        market_button.setObjectName('marketTrendButton')
        market_button.setToolTip('查看欧赔概率、让球和大小球的初盘到临盘走势')
        market_button.clicked.connect(self._show_market_trends)
        export_button = QPushButton('导出 Excel')
        export_button.clicked.connect(self._export)
        for button, width in (
                (sync_button, 136), (review_button, 102),
                (yesterday_button, 100), (market_button, 80),
                (export_button, 78)):
            button.setFixedSize(width, 25)
            action_bar.addWidget(button)
        action_bar.addStretch(1)
        self._summary.setObjectName('summaryLabel')
        self._summary.setMinimumWidth(195)
        action_bar.addWidget(self._summary)
        root.addLayout(action_bar)

        filter_panel = QWidget(self)
        filter_panel.setObjectName('filterPanel')
        filters = QHBoxLayout(filter_panel)
        filters.setContentsMargins(4, 3, 4, 3)
        filters.setSpacing(5)
        filters.addWidget(QLabel('模型'))
        self._model_selector.setMinimumWidth(220)
        self._model_selector.setFixedHeight(24)
        self._model_selector.currentIndexChanged.connect(lambda _: self._render())
        filters.addWidget(self._model_selector)
        filters.addSpacing(6)
        filters.addWidget(QLabel('比赛日期'))
        self._date_selector.setMinimumWidth(165)
        self._date_selector.setFixedHeight(24)
        self._date_selector.currentIndexChanged.connect(lambda _: self._render())
        filters.addWidget(self._date_selector)
        filters.addStretch(1)
        root.addWidget(filter_panel)
        risk_notice = QLabel(
            '概率不是赛果保证；优先查看样本量、覆盖率、阵容状态与模型分歧。'
        )
        risk_notice.setObjectName('riskNotice')
        risk_notice.setToolTip(
            '比分和半全场属于高方差市场；历史命中率只描述指定样本与时间窗口。'
        )
        root.addWidget(risk_notice)
        root.addWidget(self._table_container)

        self.setStyleSheet('''
            QWidget#windowDragBar {
                background: #e8edf1; border: 1px solid #b8c0c7;
                border-radius: 3px;
            }
            QLabel#windowDragTitle {
                color: #202020; font-size: 13px; font-weight: 600;
            }
            QPushButton#windowCloseButton {
                color: #303030; background: transparent;
                border: 0; font-size: 16px; font-weight: 600; padding: 0;
            }
            QPushButton#windowCloseButton:hover {
                color: #ffffff; background: #d93025; border-radius: 3px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:0.55 #f5f6f7, stop:1 #e2e5e8);
                color: #202020; border: 1px solid #aeb5bb;
                border-bottom: 2px solid #8f979e; border-radius: 3px;
                padding: 1px 6px 2px 6px; font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #dceaf7);
                border-color: #7199bd; border-bottom-color: #557d9f;
            }
            QPushButton:pressed {
                background: #d9e4ee; border: 1px solid #7f8b94;
                padding-top: 2px; padding-bottom: 1px;
            }
            QWidget#filterPanel {
                border-top: 1px solid #d7d7d7; border-bottom: 1px solid #d7d7d7;
            }
            QComboBox {
                color: #202020;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #e7eaed);
                border: 1px solid #aeb5bb; border-bottom: 2px solid #90989f;
                border-radius: 3px; padding: 0 22px 0 5px; font-size: 12px;
            }
            QComboBox:hover { border-color: #6f9dcc; }
            QComboBox::drop-down {
                width: 18px; border-left: 1px solid #b8bec4;
                background: #e8ebee;
            }
            QComboBox QAbstractItemView {
                color: #202020; background: #ffffff;
                border: 1px solid #90989f; outline: 0;
                selection-background-color: #2f79bd;
                selection-color: #ffffff;
                font-size: 12px;
            }
            QLabel#summaryLabel {
                color: #404040; padding: 3px 5px; font-size: 12px;
            }
            QLabel#riskNotice {
                color: #7a4b00; background: #fff8e6;
                border: 1px solid #ead7a5; border-radius: 3px;
                padding: 4px 7px; font-size: 12px;
            }
            QTableWidget {
                color: #202020; background: #ffffff;
                alternate-background-color: #f6f8fa;
                gridline-color: #cfd4d8; font-size: 12px;
            }
            QHeaderView::section {
                color: #202020;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #dfe3e6);
                border: 0; border-right: 1px solid #b8bec3;
                border-bottom: 1px solid #969da3;
                padding: 2px 4px; font-size: 12px; font-weight: 600;
            }
        ''')

        # Combo popups are separate Qt views. The application theme can style
        # them after the dialog stylesheet is applied, so pin their palette on
        # the popup itself to prevent black-on-black/white-on-white selections.
        for combo in (self._model_selector, self._date_selector):
            popup = combo.view()
            palette = popup.palette()
            palette.setColor(QPalette.ColorRole.Base, QColor('#ffffff'))
            palette.setColor(QPalette.ColorRole.Window, QColor('#ffffff'))
            palette.setColor(QPalette.ColorRole.Text, QColor('#202020'))
            palette.setColor(QPalette.ColorRole.WindowText, QColor('#202020'))
            palette.setColor(QPalette.ColorRole.Highlight, QColor('#2f79bd'))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor('#ffffff'))
            popup.setPalette(palette)
            popup.setItemDelegate(_ComboPopupDelegate(popup))
            popup.setStyleSheet('''
                QListView { background: #ffffff; color: #202020;
                    border: 1px solid #90989f; outline: 0; }
                QListView::item { background: #ffffff; color: #202020;
                    min-height: 24px; padding: 1px 5px; }
                QListView::item:hover { background: #dcebf8; color: #202020; }
                QListView::item:selected { background: #2f79bd; color: #ffffff; }
            ''')

    @staticmethod
    def _display_predictions(df: pd.DataFrame) -> pd.DataFrame:
        # Keep the user-facing ticket view compact. Detailed model/market
        # probabilities remain in the internal report but are not repeated here.
        display = df.copy()

        def score_with_probability(
                score: str,
                probability: str,
                labels: dict[str, str] | None = None,
        ) -> pd.Series:
            values = display.get(score, pd.Series('', index=display.index)).fillna('').astype(str)
            probabilities = display.get(
                probability, pd.Series(float('nan'), index=display.index),
            )
            return pd.Series([
                (labels or {}).get(value, value) if pd.isna(prob)
                else f'{(labels or {}).get(value, value)}（{float(prob):.1%}）'
                for value, prob in zip(values, probabilities)
            ], index=display.index)

        def upset_direction(row: pd.Series) -> str:
            score = str(row.get('比分爆冷') or row.get('爆冷比分') or '')
            match = re.fullmatch(r'(\d+)\+?-(\d+)\+?', score)
            if match is None:
                return ''
            home, away = map(int, match.groups())
            if home > away:
                label, column = '主胜冷门', '模型主胜概率'
            elif home == away:
                label, column = '平局冷门', '模型平局概率'
            else:
                label, column = '客胜冷门', '模型客胜概率'
            probability = row.get(column)
            return label if pd.isna(probability) else f'{label}（{float(probability):.1%}）'

        def odds_pair(row: pd.Series, opening: str, current: str) -> str:
            def fmt(value):
                try:
                    return f'{float(value):.2f}' if pd.notna(value) else ''
                except (TypeError, ValueError):
                    return ''
            first, latest = fmt(row.get(opening)), fmt(row.get(current))
            if not first and not latest:
                return ''
            if not first:
                return f'现 {latest}'
            if not latest:
                return f'初 {first}'
            return f'初 {first} → 现 {latest}'

        def had_opening_current(row: pd.Series) -> str:
            pairs = [
                odds_pair(row, '首次采集胜奖金', '官方胜奖金'),
                odds_pair(row, '首次采集平奖金', '官方平奖金'),
                odds_pair(row, '首次采集负奖金', '官方负奖金'),
            ]
            return '｜'.join(pairs) if any(pairs) else '暂无指数'

        def hhad_opening_current(row: pd.Series) -> str:
            line_first = row.get('首次采集让球数')
            line_latest = row.get('官方让球数')
            def line(value):
                try:
                    return f'{float(value):g}' if pd.notna(value) else ''
                except (TypeError, ValueError):
                    return ''
            first, latest = line(line_first), line(line_latest)
            values = [
                odds_pair(row, '首次采集让胜奖金', '官方让胜奖金'),
                odds_pair(row, '首次采集让平奖金', '官方让平奖金'),
                odds_pair(row, '首次采集让负奖金', '官方让负奖金'),
            ]
            if not any(values) and not first and not latest:
                return '暂无指数'
            handicap_line = f'线 初 {first} → 现 {latest}' if first and latest else ''
            return '｜'.join(part for part in (handicap_line, *values) if part)

        def market_reference(row: pd.Series) -> str:
            accuracy = row.get('同阈值历史命中率')
            samples = row.get('筛选回测样本数')
            coverage = row.get('同阈值历史覆盖率')
            period = str(row.get('筛选回测期间') or '').strip()
            if pd.isna(accuracy):
                return '未提供可比回测'
            result = f'历史同档 {float(accuracy):.1%}'
            if pd.notna(coverage):
                result += f'｜覆盖 {float(coverage):.1%}'
            if pd.notna(samples):
                result += f'｜{int(samples)}场'
            if period and period.lower() != 'nan':
                result += f'｜{period}'
            return result

        def evidence_status(row: pd.Series) -> str:
            samples = pd.to_numeric(row.get('筛选回测样本数'), errors='coerce')
            concerns = []
            if pd.isna(samples):
                concerns.append('回测未披露')
            elif samples < 30:
                concerns.append('小样本')
            lineup = str(row.get('首发状态') or '')
            if lineup and lineup != '已确认':
                concerns.append('首发未确认')
            if str(row.get('模拟差异') or '').strip():
                concerns.append('模型分歧')
            gate = str(row.get('盘口门控') or '')
            if any(word in gate for word in ('冲突', '震荡', '不稳定')):
                concerns.append('市场信号不稳')
            return '需谨慎：' + '、'.join(concerns) if concerns else '证据项已披露'

        display['胜负首选'] = score_with_probability('胜平负首选', '胜平负首选概率')
        handicap_labels = {'胜': '让胜', '平': '让平', '负': '让负'}
        display['让球首选/次选'] = (
            score_with_probability('让球首选', '让球首选概率', handicap_labels) + '/'
            + score_with_probability('让球次选', '让球次选概率', handicap_labels)
        )
        display['大小球首选'] = score_with_probability('大小球首选', '大小球首选概率')
        display['半全场首选/次选'] = (
            score_with_probability('半全场首选', '半全场首选概率') + '/'
            + score_with_probability('半全场次选', '半全场次选概率')
        )
        def combined_scores(row: pd.Series) -> str:
            values = []
            for column in (
                '首选比分', '次选比分', '第三比分',
                '比分爆冷', '大小球进取比分',
            ):
                value = row.get(column)
                if pd.notna(value) and str(value).strip() and str(value).strip() not in values:
                    values.append(str(value).strip())
            return ' / '.join(values)

        display['比分情景（Top3/反向/高进球）'] = display.apply(
            combined_scores, axis=1,
        )
        display['置信度'] = display.get(
            '置信等级', pd.Series('', index=display.index),
        ).fillna('').astype(str)
        display['胜平负指数（首次采集→当前）'] = display.apply(had_opening_current, axis=1)
        display['让球指数（首次采集→当前）'] = display.apply(hhad_opening_current, axis=1)
        display['市场概率档参考'] = display.apply(market_reference, axis=1)
        display['分析依据'] = display.get(
            '预测依据', pd.Series('', index=display.index),
        ).fillna('').astype(str)
        display['胜负模型'] = display.get(
            '胜负模型类别', display.get(
                '模型类别', pd.Series('', index=display.index),
            ),
        ).fillna('').astype(str)
        display['模拟半全场'] = display.get(
            '模拟半全场', pd.Series('', index=display.index),
        ).fillna('').astype(str).map(
            lambda value: ' / '.join(
                part.strip() for part in value.split('/')[:2] if part.strip()
            ),
        )

        def simulation_difference(row: pd.Series) -> str:
            def first_choice(value: object) -> str:
                text = '' if pd.isna(value) else str(value).strip()
                return re.split(r'[\s/｜]', text, maxsplit=1)[0]

            def comparison(label: str, primary: str, simulated: str) -> str:
                if not primary or not simulated or primary == simulated:
                    return ''
                return f'{label} {primary}→{simulated}✕'

            parts = []
            primary_result = first_choice(row.get('胜平负首选'))[:1]
            simulated_result = first_choice(row.get('模拟胜负'))[:1]
            parts.append(comparison('胜负', primary_result, simulated_result))

            primary_handicap = first_choice(row.get('让球首选'))
            if primary_handicap in ('胜', '平', '负'):
                primary_handicap = f'让{primary_handicap}'
            simulated_handicap = first_choice(row.get('模拟让球'))
            parts.append(comparison('让球', primary_handicap, simulated_handicap))

            primary_total = first_choice(row.get('大小球首选'))
            simulated_total = first_choice(row.get('模拟总进球'))
            if primary_total and simulated_total:
                primary_direction = (
                    '大' if primary_total.startswith('大') else
                    '小' if primary_total.startswith('小') else ''
                )
                simulated_direction = (
                    '大' if simulated_total.startswith('4球以上') else
                    '小' if simulated_total.startswith(('0-1球', '1球以内')) else ''
                )
                if (
                    primary_direction and simulated_direction
                    and primary_direction != simulated_direction
                ):
                    parts.append(f'进球 {primary_total}→{simulated_total}✕')

            primary_score = str(row.get('首选比分') or '').strip()
            simulated_match = re.match(
                r'(\d+\+?-\d+\+?)', str(row.get('模拟Top3比分') or '').strip(),
            )
            simulated_score = simulated_match.group(1) if simulated_match else ''
            parts.append(comparison('比分', primary_score, simulated_score))

            primary_half_full = first_choice(row.get('半全场首选'))
            simulated_half_full = first_choice(row.get('模拟半全场'))
            parts.append(comparison('半全场', primary_half_full, simulated_half_full))
            return '｜'.join(part for part in parts if part)

        display['模拟差异'] = display.apply(simulation_difference, axis=1)
        display['证据状态'] = display.apply(evidence_status, axis=1)
        odds_series = read_odds_series()
        match_ids = display.get('比赛ID', pd.Series('', index=display.index))
        calculated_flow = [format_market_flow(mid, series=odds_series) for mid in match_ids]
        stored_gate = display.get('盘口门控', pd.Series('', index=display.index)).fillna('')
        display['盘口流向'] = [
            flow if flow not in ('', '待积累') else (gate or flow)
            for gate, flow in zip(stored_gate, calculated_flow)
        ]
        remaining = pd.to_numeric(
            display.get('距参考截止分钟', pd.Series(float('nan'), index=display.index)),
            errors='coerce',
        )
        display['距参考截止'] = remaining.map(
            lambda value: '' if pd.isna(value) else (
                f'约{float(value):.0f}分钟' if value >= 0 else '已过参考线'
            )
        )
        display['对阵'] = (
            display.get('主队', pd.Series('', index=display.index)).fillna('').astype(str)
            + ' vs '
            + display.get('客队', pd.Series('', index=display.index)).fillna('').astype(str)
        )

        def compact_direction(row: pd.Series) -> str:
            parts = []
            result = str(row.get('胜负首选') or '').strip()
            handicap = str(row.get('让球首选/次选') or '').strip()
            total = str(row.get('大小球首选') or '').strip()
            if result:
                parts.append(f'胜负 {result}')
            if handicap and handicap != '/':
                parts.append(f'让球 {handicap.split("/", 1)[0]}')
            if total:
                parts.append(f'进球 {total}')
            return '｜'.join(parts) or '数据待补'

        def compact_risk(row: pd.Series) -> str:
            parts = []
            disagreement = str(row.get('模拟差异') or '').strip()
            evidence = str(row.get('证据状态') or '').strip()
            lineup = str(row.get('阵容分析') or '').strip()
            if disagreement:
                parts.append(disagreement)
            if evidence.startswith('需谨慎'):
                parts.append(evidence.removeprefix('需谨慎：'))
            if lineup and lineup not in ('首发已确认', '阵容暂无明显影响'):
                parts.append(lineup)
            return '｜'.join(dict.fromkeys(parts)) or '正常'

        display['综合方向'] = display.apply(compact_direction, axis=1)
        display['风险提示'] = display.apply(compact_risk, axis=1)
        display['让球'] = display['让球首选/次选']
        display['大小球'] = display['大小球首选']
        display['半全场'] = display['半全场首选/次选']
        display['比分'] = display['比分情景（Top3/反向/高进球）']
        preferred = [
            '赛事编号', '比赛时间', '联赛', '对阵', '距参考截止',
            '综合方向', '盘口流向', '让球', '大小球', '半全场', '比分',
            '风险提示',
        ]
        shown = display[[column for column in preferred if column in display.columns]].copy()
        for column in shown.columns:
            if (
                ('概率' in column or '优势' in column)
                and pd.api.types.is_numeric_dtype(shown[column])
            ):
                shown[column] = shown[column].map(
                    lambda value: '' if pd.isna(value) else f'{float(value):.1%}',
                )
        return shown

    def _load_saved_reports(self):
        def read_report(path: Path, columns: list[str]) -> pd.DataFrame:
            if not path.exists() or path.stat().st_size == 0:
                return pd.DataFrame(columns=columns)
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=columns)

        self._predictions = _sort_by_match_number(backfill_missing_simulations(
            read_report(PREDICTION_PATH, ['赛事编号', '联赛', '主队', '客队']),
        ))
        self._refresh_model_selector()
        self._refresh_date_selector()
        self._refresh_advice_selector()
        self._render()

    def _refresh_model_selector(self):
        had_selection = self._model_selector.count() > 0
        current = self._model_selector.currentData() if had_selection else None
        scopes = _prediction_model_scopes(self._predictions)
        reported_leagues = [scope for scope in scopes.unique() if scope]
        leagues = list(dict.fromkeys(self._trained_leagues + sorted(reported_leagues)))
        dedicated_count = int(scopes.ne('').sum())
        generic_count = int(scopes.eq('').sum())
        counts = scopes.value_counts().to_dict()
        self._model_selector.blockSignals(True)
        self._model_selector.clear()
        self._model_selector.addItem(
            f'全部模型结果（{len(self._predictions)}场，含不同模型）', ALL_MODELS,
        )
        self._model_selector.addItem(
            f'全部专用模型（{dedicated_count}场）', DEDICATED_MODELS,
        )
        self._model_selector.addItem(
            f'通用/市场模型（{generic_count}场）', GENERIC_MODELS,
        )
        simulated = pd.to_numeric(
            self._predictions.get(
                '模拟次数', pd.Series(0, index=self._predictions.index),
            ), errors='coerce',
        ).fillna(0).gt(0)
        self._model_selector.addItem(
            f'蒙特卡洛模拟（{int(simulated.sum())}场）', SIMULATION_MODELS,
        )
        for league in leagues:
            self._model_selector.addItem(
                f'{league}专用模型（{int(counts.get(league, 0))}场）', league,
            )
        if current is None:
            # Opening the window must show the complete report.  Defaulting to
            # a model subset made valid rows look as if they had not loaded.
            current = ALL_MODELS
        index = self._model_selector.findData(current)
        self._model_selector.setCurrentIndex(index if index >= 0 else 0)
        self._model_selector.blockSignals(False)

    def _visible_predictions(self) -> pd.DataFrame:
        visible = filter_predictions_by_model(
            self._predictions,
            self._model_selector.currentData() or ALL_MODELS,
        )
        # This is a betting/prediction view, never a history viewer. Ticket
        # labels such as 周六 may kick off after midnight on Sunday; filtering
        # by the real kickoff timestamp prevents those settled rows reappearing
        # when a date is selected.
        visible = _upcoming_predictions(visible)
        selected_date = self._date_selector.currentData()
        if selected_date and '比赛时间' in visible.columns:
            dates = visible['比赛时间'].fillna('').astype(str).str.slice(0, 10)
            visible = visible.loc[dates.eq(str(selected_date))]
        return _sort_by_match_number(visible)

    def _refresh_advice_selector(self):
        current = self._advice_selector.currentData()
        advice = self._predictions.get(
            '建议状态', pd.Series('', index=self._predictions.index),
        ).fillna('').astype(str)
        selected = int(advice.eq('精选主推').sum())
        high = int(advice.eq('高置信主推').sum())
        self._advice_selector.blockSignals(True)
        self._advice_selector.clear()
        self._advice_selector.addItem(f'达到筛选阈值（{selected + high}场）', '正式推荐')
        self._advice_selector.addItem(f'A档概率样本（{selected}场）', '精选主推')
        self._advice_selector.addItem(f'B档概率样本（{high}场）', '高置信主推')
        self._advice_selector.addItem(f'全部场次（{len(self._predictions)}场）', '')
        if current is None or (current == '正式推荐' and selected + high == 0):
            # Never open on an empty recommendation view when the report itself
            # contains fixtures.  Users can still select each recommendation
            # grade explicitly when such rows exist.
            current = ''
        index = self._advice_selector.findData(current)
        self._advice_selector.setCurrentIndex(index if index >= 0 else 0)
        self._advice_selector.blockSignals(False)

    def _refresh_date_selector(self):
        current = self._date_selector.currentData()
        upcoming = _upcoming_predictions(self._predictions)
        if '比赛时间' in upcoming.columns:
            dates = upcoming['比赛时间'].fillna('').astype(str).str.slice(0, 10)
            dates = sorted(value for value in dates.unique() if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value))
        else:
            dates = []
        self._date_selector.blockSignals(True)
        self._date_selector.clear()
        upcoming_count = len(_upcoming_predictions(self._predictions))
        self._date_selector.addItem(f'尚未开赛（{upcoming_count}场）', '')
        for value in dates:
            count = int(upcoming['比赛时间'].astype(str).str.startswith(value).sum())
            weekday = '一二三四五六日'[date.fromisoformat(value).weekday()]
            self._date_selector.addItem(f'{value} 周{weekday}（{count}场）', value)
        index = self._date_selector.findData(current)
        self._date_selector.setCurrentIndex(index if index >= 0 else 0)
        self._date_selector.blockSignals(False)

    def _render(self):
        visible = self._visible_predictions()
        while self._table_layout.count():
            item = self._table_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        display_frame = self._display_predictions(visible)
        priorities = _daily_priority_aspects(visible).reset_index(drop=True)
        if '综合方向' in display_frame.columns:
            for row_index, labels in priorities.items():
                if labels:
                    display_frame.at[row_index, '综合方向'] = (
                        f'★重点：{"/".join(labels)}｜'
                        f'{display_frame.at[row_index, "综合方向"]}'
                    )
        table = ExcelTable(
            parent=self,
            df=display_frame,
            readonly=True,
            supports_sorting=True,
        )
        for row_index, labels in priorities.items():
            if not labels:
                continue
            for column_index in range(table.columnCount()):
                item = table.item(row_index, column_index)
                if item is None:
                    continue
                item.setForeground(QBrush(QColor('#c62828')))
                item.setBackground(QBrush(QColor('#fff1f1')))
                font = QFont(item.font())
                font.setBold(True)
                item.setFont(font)
        difference_column = '模拟差异'
        if difference_column in display_frame.columns:
            column_index = display_frame.columns.get_loc(difference_column)
            for row_index, value in enumerate(display_frame[difference_column].fillna('')):
                item = table.item(row_index, column_index)
                if item is None:
                    continue
                text = str(value)
                if '✕' in text:
                    item.setForeground(QBrush(QColor('#c62828')))
                    item.setBackground(QBrush(QColor('#fff1f1')))
                elif '✓' in text:
                    item.setForeground(QBrush(QColor('#137333')))
                    item.setBackground(QBrush(QColor('#eef8f0')))
                font = QFont(item.font())
                font.setBold(True)
                item.setFont(font)
        table.horizontalHeader().setSectionsMovable(True)
        table.horizontalHeader().setFixedHeight(26)
        table.verticalHeader().setDefaultSectionSize(25)
        table.verticalHeader().setMinimumSectionSize(23)
        self._table_layout.addWidget(table)
        if visible.empty and not self._predictions.empty:
            self._summary.setText(
                f'当前显示 0 场  ·  全部 {len(self._predictions)} 场'
            )
        else:
            self._summary.setText(
                f'当前显示 {len(visible)} 场  ·  全部 {len(self._predictions)} 场'
            )

    def _sync(self):
        runner = TaskRunnerDialog(
            title='同步竞猜数据',
            info='正在获取官方场次并生成预测…',
            task_fn=run_daily_sporttery,
            parent=self,
        )
        result = runner.run()
        if runner.error_message is not None or result is None:
            QMessageBox.critical(
                self, '同步失败', runner.error_message or '同步任务没有返回数据，请稍后重试。',
            )
            return
        self._predictions, _ = result
        self._refresh_model_selector()
        self._refresh_date_selector()
        self._refresh_advice_selector()
        self._render()
        learning_message = ''
        if LEARNING_STATUS_PATH.exists():
            try:
                learning = json.loads(LEARNING_STATUS_PATH.read_text(encoding='utf-8'))
                learning_message = (
                    f'\n每日复盘：本次结算 {int(learning.get("newly_settled") or 0)} 场，'
                    f'累计 {int(learning.get("settled_samples") or 0)} 场，'
                    f'等待官方赛果 {int(learning.get("pending_samples") or 0)} 场；'
                    f'{learning.get("model_status", "积累样本中")}。'
                )
            except (OSError, ValueError, TypeError):
                pass
        QMessageBox.information(
            self, '同步完成',
            f'已生成 {len(self._predictions)} 场预测。'
            f'{learning_message}',
        )

    def _review_history(self):
        runner = TaskRunnerDialog(
            title='补同步历史赛果',
            info='正在扫描遗漏场次、获取官方赛果并执行复盘…',
            task_fn=lambda: review_and_learn(full_backfill=True),
            parent=self,
        )
        result = runner.run()
        if runner.error_message is not None:
            QMessageBox.critical(self, '补同步失败', runner.error_message)
            return
        self._render()
        accuracy = result.get('result_accuracy')
        accuracy_text = f'{float(accuracy):.1%}' if accuracy is not None else '--'
        message = (
            f'本次补结算 {int(result.get("newly_settled") or 0)} 场，'
            f'累计复盘 {int(result.get("settled_samples") or 0)} 场，'
            f'等待官方赛果 {int(result.get("pending_samples") or 0)} 场，'
            f'胜平负历史命中率 {accuracy_text}（累计已结算样本）。\n'
            f'本次补入官方市场样本 {int(result.get("new_official_history") or 0)} 场，'
            f'通用训练样本累计 {int(result.get("total_training_samples") or 0)} 场。\n'
            f'模型状态：{result.get("model_status", "积累样本中")}。\n'
            f'模型迭代：已审计 {int(result.get("evolution_attempts") or 0)} 次，'
            f'当前采用第 {int(result.get("champion_generation") or 0)} 代。'
        )
        selection_rows = (result.get('selection_profile') or {}).get('rows') or []
        recommended = [
            row for row in selection_rows
            if row.get('grade') in ('精选主推', '高置信主推')
        ]
        if recommended:
            message += '\n历史筛选阈值：' + '；'.join(
                f'{"A档" if row["grade"] == "精选主推" else "B档"}'
                f'≥{float(row["threshold"]):.1%}'
                f'（历史同档{float(row["accuracy"]):.1%}）'
                for row in recommended
            )
        model_lines = []
        for name, audit in result.get('accuracy_by_model', {}).items():
            model_lines.append(
                f'{name}：{int(audit.get("samples") or 0)}场，'
                f'模型 {float(audit.get("accuracy") or 0):.1%}，'
                f'市场基线 {float(audit.get("market_accuracy") or 0):.1%}，'
                f'{audit.get("status", "继续观察")}'
            )
        if model_lines:
            message += '\n\n分模型最近实战：\n' + '\n'.join(model_lines)
        if result.get('review_error'):
            QMessageBox.warning(
                self, '部分赛果暂未补齐',
                f'{message}\n\n'
                f'官方赛果服务暂时无响应，已保留现有复盘数据。'
                f'等待补齐的 {int(result.get("pending_samples") or 0)} 场将在下次自动重试；'
                f'不影响当前预测和已训练模型。',
            )
        else:
            QMessageBox.information(self, '补同步完成', message)

    def _show_yesterday_hits(self):
        # Reading the local settlement cache is intentionally instant and never
        # starts a network request. Use the existing review button when official
        # results still need to be synchronized.
        if self._yesterday_dialog is not None:
            self._yesterday_dialog.close()
            self._yesterday_dialog.deleteLater()
        self._yesterday_dialog = YesterdayHitDetailsDialog(self)
        self._yesterday_dialog.show()
        self._yesterday_dialog.raise_()
        self._yesterday_dialog.activateWindow()

    def _show_market_trends(self):
        visible = self._visible_predictions()
        if visible.empty and self._predictions.empty:
            QMessageBox.information(self, '没有数据', '请先同步竞猜数据。')
            return
        source = visible if not visible.empty else self._predictions
        if self._market_dialog is not None:
            self._market_dialog.close()
            self._market_dialog.deleteLater()
        self._market_dialog = MarketTrendDialog(source, self)
        self._market_dialog.show()
        self._market_dialog.raise_()
        self._market_dialog.activateWindow()

    def _export(self):
        visible = self._visible_predictions()
        if visible.empty:
            QMessageBox.information(self, '没有数据', '请先同步竞猜数据。')
            return
        selected = self._model_selector.currentText().split('（', 1)[0]
        exported_at = datetime.now()
        timestamp = exported_at.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
        path = (
            prediction_export_root()
            / f'{timestamp}-{_safe_filename(selected)}.xlsx'
        )
        try:
            write_predictions_xlsx(self._display_predictions(visible), path)
        except PermissionError:
            QMessageBox.critical(
                self, '导出失败', 'Excel 文件正在被占用，请关闭对应文件后重试。',
            )
            return
        except Exception as error:
            QMessageBox.critical(self, '导出失败', f'无法生成 Excel：{error}')
            return
        QMessageBox.information(
            self, '导出完成', f'已生成 Excel 文件：\n{display_export_path(path)}',
        )
