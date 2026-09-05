import pandas as pd
import pytest

from src.gui.windows import sporttery


def candidate():
    return {
        '赛事编号': '周六001', '比赛时间': '2099-09-05 18:00',
        '主队': '主队', '客队': '客队', '官方让球数': -1,
        '首次采集让球数': -1, '让球首选': '胜', '让球首选概率': .64,
        '模型让胜概率': .64, '模型让平概率': .20, '模型让负概率': .16,
        '首次采集让胜奖金': 2.0, '首次采集让平奖金': 3.2, '首次采集让负奖金': 3.4,
        '官方让胜奖金': 2.0, '官方让平奖金': 3.2, '官方让负奖金': 3.4,
        '模拟让球': '让胜 65.0%', '让球建议状态': '精选主推',
        '比分模型状态': '英超专用模型启用',
    }


@pytest.fixture(autouse=True)
def isolate_history(monkeypatch):
    monkeypatch.setattr(sporttery, 'read_odds_series', lambda **kw: {})
    monkeypatch.setattr(sporttery, 'historical_calibration', lambda *a, **kw: (None, 0))


@pytest.mark.parametrize('conflict', ['False', False, float('nan'), None])
def test_false_or_missing_lineup_flag_does_not_delete_candidate(conflict):
    row = candidate()
    row['阵容方向冲突'] = conflict
    result = sporttery.build_daily_recommendations(pd.DataFrame([row]))
    assert len(result) == 1
    assert result.iloc[0]['推荐等级'] == '核心重点'


@pytest.mark.parametrize('change', ['market', 'monte', 'negative_ev'])
def test_legacy_handicap_label_cannot_promote_observations(change):
    row = candidate()
    if change == 'market':
        row['比分模型状态'] = '弱模型已禁用，回退市场基线'
    elif change == 'monte':
        row['模拟让球'] = '让负 65.0%'
    else:
        row['官方让胜奖金'] = row['首次采集让胜奖金'] = 1.2
    result = sporttery.build_daily_recommendations(pd.DataFrame([row]))
    assert len(result) == 1
    item = result.iloc[0]
    assert item['推荐等级'] not in {'核心重点', '可买优选'}
    assert not item['重点选项'].startswith('★')
    assert item['建议仓位'] == '不投注'


def test_market_favorite_disagreement_stays_visible_as_observation():
    row = candidate()
    row['盘口门控'] = '盘口流向平·与模型胜冲突'
    row['官方让平奖金'] = row['首次采集让平奖金'] = 1.6
    result = sporttery.build_daily_recommendations(pd.DataFrame([row]))
    assert len(result) == 1
    assert result.iloc[0]['推荐等级'] == '盘口分歧观察'
    assert result.iloc[0]['建议仓位'] == '不投注'


def test_market_evidence_affects_ranking_within_grade(monkeypatch):
    monkeypatch.setattr(sporttery, 'read_odds_series', lambda **kw: {'1': [1], '2': [2]})
    monkeypatch.setattr(sporttery, 'assess_market', lambda rows, *a, **kw: {
        'state': True, 'text': '走势核验', 'score': .04 if rows == [2] else 0})
    first = dict(candidate(), 比赛ID=1)
    second = dict(candidate(), 比赛ID=2, 赛事编号='周六002')
    result = sporttery.build_daily_recommendations(pd.DataFrame([first, second]))
    assert result.iloc[0]['赛事编号'] == '周六002'
