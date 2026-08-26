import pandas as pd
from PySide6.QtWidgets import QApplication, QDialog

from src.gui.i18n import translate_widget
from src.gui.widgets.tables import DataFrameTable, DataFrameTableModel
from src.gui.windows.models import fixtures
from src.gui.windows.models.predictor import PredictorDialog
from src.gui.windows import sporttery as sporttery_window
from src.gui.windows.sporttery import (
    DEDICATED_MODELS, GENERIC_MODELS, filter_predictions_by_model,
    write_predictions_xlsx,
)
from src.network.fixtures.utils import match_fixture_teams
from src.preprocessing.utils.inputs import construct_inputs_by_teams


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_large_readonly_table_uses_lazy_dataframe_model():
    _app()
    frame = pd.DataFrame({
        'Home': [f'Home {index}' for index in range(8000)],
        'Away': [f'Away {index}' for index in range(8000)],
        '1': [2.0] * 8000,
    })
    table = DataFrameTable(parent=None, df=frame)

    assert isinstance(table.model(), DataFrameTableModel)
    assert table.model().rowCount() == 8000
    hits = table.compute_matches('Home 7999', exact_match=True, selected_scope_index=1)
    assert len(hits) == 1
    assert hits[0].column() == 0
    assert hits[0].data() == 'Home 7999'


def test_fixture_worker_maps_official_sporttery_rows(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def selling_matches(self):
            return [{
                'matchDate': '2026-08-11',
                'leagueAllName': '瑞典超级联赛',
                'homeTeamAllName': '天狼星',
                'awayTeamAllName': '布鲁马波卡纳',
                'had': {'h': '1.20', 'd': '5.65', 'a': '8.40'},
            }]

    monkeypatch.setattr(fixtures, 'SportteryMobileClient', FakeClient)
    worker = fixtures.FixtureFetchWorker(
        fixture_url='https://invalid.example',
        date_str='Aug 11',
        iso_date='2026-08-11',
        league_id='瑞超',
    )

    frame = worker._sporttery_fixtures()
    assert frame.to_dict('records') == [{
        'Home': 'Sirius',
        'Away': 'Brommapojkarna',
        '1': 1.2,
        'X': 5.65,
        '2': 8.4,
    }]


def test_translation_keeps_native_window_title_ascii():
    _app()
    dialog = QDialog()
    dialog.setWindowTitle('Fixtures Dialog')
    translate_widget(dialog)
    assert dialog.windowTitle() == 'Fixtures Dialog'
    assert dialog.windowTitle().isascii()


def test_new_team_uses_neutral_league_statistics_without_renaming():
    history = pd.DataFrame([
        {
            'Date': '2026-05-24', 'Season': 2025, 'Week': 38,
            'Home': 'Arsenal', 'Away': 'Chelsea', '1': 1.8, 'X': 3.5, '2': 4.5,
            'HPoints': 2.4, 'APoints': 1.5,
        },
        {
            'Date': '2026-05-17', 'Season': 2025, 'Week': 37,
            'Home': 'Liverpool', 'Away': 'Arsenal', '1': 2.0, 'X': 3.4, '2': 3.8,
            'HPoints': 2.0, 'APoints': 1.7,
        },
    ])
    fixture = pd.DataFrame([{
        'Home': 'Coventry', 'Away': 'Arsenal', '1': 3.1, 'X': 3.2, '2': 2.4,
    }])

    built = construct_inputs_by_teams(history, fixture)

    assert built.at[0, 'Home'] == 'Coventry'
    assert built.at[0, 'Away'] == 'Arsenal'
    assert built.at[0, 'HPoints'] == 2.2
    assert not built.isna().any().any()


def test_unknown_fixture_team_is_not_fuzzily_replaced_by_another_club():
    fixtures_df = pd.DataFrame([{
        'Home': 'Coventry', 'Away': 'Arsenal', '1': 3.1, 'X': 3.2, '2': 2.4,
    }])
    history = pd.DataFrame({
        'Home': ['Everton', 'Arsenal'],
        'Away': ['Chelsea', 'Liverpool'],
    })

    matched = match_fixture_teams(fixtures_df, history)

    assert matched.at[0, 'Home'] == 'Coventry'
    assert matched.at[0, 'Away'] == 'Arsenal'


def test_manual_prediction_allows_every_historical_team_in_either_role():
    class FakeModelDatabase:
        league_id = '英超'

        @staticmethod
        def get_model_ids():
            return []

    _app()
    history = pd.DataFrame({
        'Home': ['Arsenal', 'Liverpool'],
        'Away': ['Chelsea', 'Everton'],
    })
    dialog = PredictorDialog(history, FakeModelDatabase())

    expected = {'Arsenal', 'Liverpool', 'Chelsea', 'Everton', 'Coventry'}
    assert set(dialog._home_teams) == expected
    assert set(dialog._away_teams) == expected
    dialog.close()


def test_manual_prediction_ranks_models_by_holdout_not_training_accuracy():
    class ModelContext:
        _model_quality_key = PredictorDialog._model_quality_key

    dialog = ModelContext()
    dialog._model_configs = {
        '训练看起来很高': {
            'train': {'results': {'fit': {'Accuracy': 0.99}}},
        },
        '独立测试更好': {
            'train': {'tuning': {'test_accuracy': 0.56, 'test_samples': 400}},
        },
        '独立测试较低': {
            'train': {'tuning': {'test_accuracy': 0.52, 'test_samples': 800}},
        },
    }
    ranked = sorted(dialog._model_configs, key=dialog._model_quality_key, reverse=True)

    assert ranked == ['独立测试更好', '独立测试较低', '训练看起来很高']


def test_fixture_model_picker_uses_holdout_ranking():
    class ModelContext:
        _model_quality_key = fixtures.FixturesDialog._model_quality_key

    dialog = ModelContext()
    dialog._model_configs = {
        '无独立测试': {'train': {'results': {'fit': {'Accuracy': 1.0}}}},
        '稳定模型': {'train': {'tuning': {'test_accuracy': 0.55, 'test_samples': 600}}},
        '较弱模型': {'train': {'tuning': {'test_accuracy': 0.51, 'test_samples': 900}}},
    }

    ranked = sorted(dialog._model_configs, key=dialog._model_quality_key, reverse=True)

    assert ranked == ['稳定模型', '较弱模型', '无独立测试']


def test_sporttery_dedicated_model_filter_never_mixes_leagues():
    predictions = pd.DataFrame([
        {'赛事编号': '周六001', '联赛': '英超', '专用模型联赛': '英超'},
        {'赛事编号': '周六002', '联赛': '西甲', '专用模型联赛': '西甲'},
        {'赛事编号': '周六003', '联赛': '欧冠', '专用模型联赛': ''},
    ])

    english = filter_predictions_by_model(predictions, '英超')
    dedicated = filter_predictions_by_model(predictions, DEDICATED_MODELS)
    generic = filter_predictions_by_model(predictions, GENERIC_MODELS)

    assert english['赛事编号'].tolist() == ['周六001']
    assert dedicated['赛事编号'].tolist() == ['周六001', '周六002']
    assert generic['赛事编号'].tolist() == ['周六003']


def test_sporttery_selector_lists_only_complete_dedicated_model_sets(monkeypatch):
    class FakeModelDatabase:
        def __init__(self, _league):
            self.index = {
                '英超': {
                    '英超胜平负模型': {}, '英超大小球模型': {},
                    '英超比分模型': {}, '英超半全场模型': {},
                },
                '西甲': {
                    '西甲胜平负模型': {}, '西甲比分模型': {},
                },
            }

    monkeypatch.setattr(sporttery_window, 'ModelDatabase', FakeModelDatabase)

    assert sporttery_window.trained_dedicated_leagues() == ['英超']


def test_sporttery_excel_export_is_a_real_formatted_workbook(tmp_path):
    from openpyxl import load_workbook

    path = tmp_path / '2026-08-15-英超专用模型.xlsx'
    write_predictions_xlsx(pd.DataFrame([{
        '赛事编号': '周六001', '联赛': '英超', '胜负首选': '胜（55.0%）',
    }]), path)

    workbook = load_workbook(path)
    sheet = workbook['竞彩预测']
    assert sheet.freeze_panes == 'A2'
    assert sheet.auto_filter.ref == 'A1:C2'
    assert sheet['A1'].value == '赛事编号'
    assert sheet['A2'].value == '周六001'


def test_sporttery_table_shows_decision_probabilities_and_audit_reference():
    predictions = pd.DataFrame([{
        '赛事编号': '周六009', '联赛': '测试联赛', '主队': '主队', '客队': '客队',
        '胜负模型类别': '市场基线', '建议状态': '精选主推', '置信等级': '高',
        '模型主胜概率': 0.70, '模型平局概率': 0.18, '模型客胜概率': 0.12,
        '胜平负首选': '胜', '胜平负首选概率': 0.70,
        '同阈值历史命中率': 0.762, '同阈值历史覆盖率': 0.137,
        '筛选回测样本数': 137, '预测依据': '官方赔率市场基线',
    }])

    display = sporttery_window.SportteryPredictionsDialog._display_predictions(predictions)

    assert '胜平负概率' not in display.columns
    assert '官方销售状态' not in display.columns
    assert display.loc[0, '市场概率档参考'] == '同档命中 76.2%｜覆盖 13.7%｜137场'
    assert display.loc[0, '分析依据'] == '官方赔率市场基线'
