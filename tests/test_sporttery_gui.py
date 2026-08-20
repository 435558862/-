import pandas as pd

from src.gui.windows.sporttery import SportteryPredictionsDialog


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
    assert display.loc[0, '让球首选/次选'] == '让负（66.0%）/让平（21.2%）'


def test_single_display_combines_confidence_scores_and_upset_result():
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
        '比分爆冷': '1-2',
        '爆冷比分概率': 0.08,
    }])

    display = SportteryPredictionsDialog._display_predictions(predictions)

    assert display.loc[0, '置信度'] == '中'
    assert display.loc[0, '胜负首选'] == '胜（60.0%）'
    assert display.loc[0, '比分首选'] == '2-1（12.0%）'
    assert display.loc[0, '比分次选'] == '1-0（10.0%）'
    assert display.loc[0, '爆冷方向'] == '客胜冷门（20.0%）'
    assert display.loc[0, '冷门比分'] == '1-2（8.0%）'
