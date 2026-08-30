import pandas as pd
from datetime import datetime
from pathlib import Path

import src.gui.windows.sporttery as sporttery_module

from src.gui.windows.sporttery import (
    SportteryPredictionsDialog, _score_recommendation_mask, _upcoming_predictions,
)


def test_default_view_hides_kicked_off_matches_but_keeps_future_rows():
    predictions = pd.DataFrame([
        {'赛事编号': '周日001', '比赛时间': '2026-08-23 09:30'},
        {'赛事编号': '周日002', '比赛时间': '2026-08-23 21:30'},
    ])
    visible = _upcoming_predictions(
        predictions, now=datetime(2026, 8, 23, 12, 0),
    )
    assert visible['赛事编号'].tolist() == ['周日002']


def test_date_filter_cannot_restore_kicked_off_ticket_rows():
    predictions = pd.DataFrame([
        {'赛事编号': '周六028', '比赛时间': '2026-08-23 09:30'},
        {'赛事编号': '周日001', '比赛时间': '2026-08-23 20:00'},
    ])
    visible = _upcoming_predictions(
        predictions, now=datetime(2026, 8, 23, 19, 0),
    )
    same_date = visible.loc[
        visible['比赛时间'].astype(str).str.startswith('2026-08-23')
    ]
    assert same_date['赛事编号'].tolist() == ['周日001']


def test_handicap_display_hides_full_distribution_and_keeps_ranked_picks():
    predictions = pd.DataFrame([{
        '赛事编号': '周二002',
        '官方让球数': -1,
        '模型让胜概率': 0.128105,
        '模型让平概率': 0.212275,
        '模型让负概率': 0.659620,
        '让球首选': '负',
        '让球首选概率': 0.659620,
        '让球次选': '平',
        '让球次选概率': 0.212275,
    }])

    display = SportteryPredictionsDialog._display_predictions(predictions)

    assert '让球胜/平/负' not in display.columns
    assert display.loc[0, '让球'] == '让负（66.0%）/让平（21.2%）'


def test_dedicated_half_full_predictions_are_visible_in_ticket_table():
    predictions = pd.DataFrame([{
        '赛事编号': '周四007', '专用模型联赛': '西甲',
        '半全场首选': '胜胜', '半全场首选概率': 0.274497,
        '半全场次选': '平胜', '半全场次选概率': 0.174302,
    }])

    display = SportteryPredictionsDialog._display_predictions(predictions)

    assert display.loc[0, '半全场'] == '胜胜（27.4%）/平胜（17.4%）'


def test_single_display_combines_all_scores_without_percentages():
    predictions = pd.DataFrame([{
        '赛事编号': '周二003',
        '联赛': '欧洲冠军联赛',
        '主队': '主队A',
        '客队': '客队B',
        '胜平负首选': '胜',
        '胜平负首选概率': 0.60,
        '置信等级': '中',
        '模型主胜概率': 0.60,
        '模型平局概率': 0.20,
        '模型客胜概率': 0.20,
        '首选比分': '2-1',
        '首选比分概率': 0.12,
        '次选比分': '1-0',
        '次选比分概率': 0.10,
        '第三比分': '2-0',
        '比分爆冷': '1-2',
        '爆冷比分概率': 0.08,
        '大小球进取比分': '3-1',
    }])

    display = SportteryPredictionsDialog._display_predictions(predictions)

    assert '置信度' not in display.columns
    assert display.loc[0, '综合方向'].startswith('胜负 胜（60.0%）')
    scores = display.loc[0, '比分']
    assert scores == '2-1 / 1-0 / 2-0 / 1-2 / 3-1'
    assert '%' not in scores


def test_all_scores_remain_visible_even_when_confidence_is_low():
    predictions = pd.DataFrame([{
        '赛事编号': '周日001', '比分推荐状态': '可信度不足',
        '首选比分': '1-1', '次选比分': '1-0', '第三比分': '2-1',
    }])
    display = SportteryPredictionsDialog._display_predictions(predictions)
    assert display.loc[0, '比分'] == '1-1 / 1-0 / 2-1'


def test_only_audited_score_recommendations_are_marked():
    predictions = pd.DataFrame([
        {'比分推荐状态': '推荐', '原始最高概率比分概率': 0.13},
        {'比分推荐状态': '可信度不足', '原始最高概率比分概率': 0.11},
        {'比分推荐状态': '', '原始最高概率比分概率': 0.12},
    ])
    assert _score_recommendation_mask(predictions).tolist() == [True, False, True]
def test_yesterday_score_review_compares_only_visible_priority_pick(monkeypatch, tmp_path):
    details = pd.DataFrame([{
        '赛事编号': '周六004', '完场比分': '3-1',
        '比分（首/次1/次2/冷/进）': '1-1/1-0/2-1/0-0/2-2 → 3-1（未中）',
    }])
    monkeypatch.setattr(
        sporttery_module, 'load_yesterday_hit_report',
        lambda: (details, {'date': '2026-08-29'}),
    )
    pd.DataFrame([{
        '比赛ID': 4, '比赛时间': '2026-08-29 18:30',
        '赛事编号': '周六004', '联赛': '韩职', '主队': '蔚山现代',
        '客队': '金泉尚武', '首选比分': '1-1',
    }]).to_csv(tmp_path / '2026-08-29-竞彩预测.csv', index=False)
    monkeypatch.setattr(sporttery_module, 'REPORT_ROOT', tmp_path)
    monkeypatch.setattr(
        sporttery_module, '_load_daily_recommendation_snapshot',
        lambda day: pd.DataFrame([{
            '比赛日期': day,
            '赛事编号': '周六004', '联赛': '韩职',
            '对阵': '蔚山现代 vs 金泉尚武', '推荐玩法': '比分',
            '重点选项': '★ 1-1', '模型概率': '13.4%',
        }]),
    )

    review, review_date = sporttery_module.build_yesterday_recommendation_review()

    assert review_date == '2026-08-29'
    assert review.loc[0, '昨日推荐'] == '★ 1-1'
    assert review.loc[0, '复盘结果'] == '1-1 → 3-1（未中）'
    assert review.loc[0, '命中状态'] == '✕ 未中'


def test_daily_recommendation_keeps_delayed_match_on_ticket_card_date(monkeypatch):
    predictions = pd.DataFrame([{
        '赛事编号': '周六018', '比赛时间': '2026-08-30 23:00',
        '联赛': '挪威超级联赛', '主队': '维京', '客队': '奥勒松',
        '大小球首选': '大于2.5球', '大小球首选概率': 0.749,
    }])
    monkeypatch.setattr(
        sporttery_module, '_daily_priority_aspects',
        lambda frame: pd.Series([['大小球']], index=frame.index),
    )

    result = sporttery_module.build_daily_recommendations(
        predictions, future_only=False,
    )

    assert result.loc[0, '比赛日期'] == '2026-08-29'
    assert result.loc[0, '模型概率'] == '74.9%'


def test_daily_recommendation_snapshot_is_frozen_and_upserted(monkeypatch, tmp_path):
    monkeypatch.setattr(sporttery_module, 'DAILY_RECOMMENDATION_ROOT', tmp_path)
    first = pd.DataFrame([{
        '比赛日期': '2026-08-29', '赛事编号': '周六001',
        '推荐玩法': '胜负', '重点选项': '★ 胜', '模型概率': '60.0%',
    }])
    later = pd.DataFrame([{
        '比赛日期': '2026-08-29', '赛事编号': '周六002',
        '推荐玩法': '比分', '重点选项': '★ 1-1', '模型概率': '13.0%',
    }])
    sporttery_module._save_daily_recommendation_snapshot(first)
    sporttery_module._save_daily_recommendation_snapshot(later)

    frozen = sporttery_module._load_daily_recommendation_snapshot('2026-08-29')

    assert frozen['赛事编号'].tolist() == ['周六001', '周六002']
