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
    assert display.loc[0, '综合方向'].startswith('胜（60.0%）')
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
        '盘口门控': '稳定', '胜平负首选': '胜', '胜平负首选概率': .68,
        '模型主胜概率': .68, '模型平局概率': .20, '模型客胜概率': .12,
        '首次采集胜奖金': 1.90, '首次采集平奖金': 3.40,
        '首次采集负奖金': 4.20, '官方胜奖金': 1.80,
        '官方平奖金': 3.50, '官方负奖金': 4.40,
        '模拟胜负': '胜 69.0%',
    }])
    result = sporttery_module.build_daily_recommendations(
        predictions, future_only=False,
    )

    assert result.loc[0, '比赛日期'] == '2026-08-29'
    assert result.loc[0, '推荐玩法'] == '胜平负'
    assert result.loc[0, '重点选项'] == '★ 胜'
    assert result.loc[0, '正式模型概率'] == '68.0%'
    assert result.loc[0, '蒙特卡洛是否同向'] == '同向（蒙特：胜）'


def test_daily_recommendation_snapshot_preserves_every_displayed_list(monkeypatch, tmp_path):
    monkeypatch.setattr(sporttery_module, 'DAILY_RECOMMENDATION_ROOT', tmp_path)
    first = pd.DataFrame([{
        '比赛日期': '2099-08-29', '赛事编号': '周六001',
        '推荐玩法': '胜负', '重点选项': '★ 胜', '模型概率': '60.0%',
    }])
    later = pd.DataFrame([{
        '比赛日期': '2099-08-29', '赛事编号': '周六002',
        '推荐玩法': '比分', '重点选项': '★ 1-1', '模型概率': '13.0%',
    }])
    sporttery_module._save_daily_recommendation_snapshot(first)
    sporttery_module._save_daily_recommendation_snapshot(later)

    frozen = sporttery_module._load_daily_recommendation_snapshot('2099-08-29')

    assert frozen['赛事编号'].tolist() == ['周六001', '周六002']
    assert frozen['首次展示时间'].notna().all()
    latest = pd.read_csv(tmp_path / '2099-08-29.latest.csv')
    assert latest['赛事编号'].tolist() == ['周六002']


def test_yesterday_review_uses_only_latest_recommendation_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(sporttery_module, 'DAILY_RECOMMENDATION_ROOT', tmp_path)
    first = pd.DataFrame([{
        '比赛日期': '2099-08-29', '赛事编号': '周六001',
        '联赛': '测试', '对阵': '甲 vs 乙', '推荐玩法': '胜平负',
        '重点选项': '★ 胜',
    }])
    latest = pd.DataFrame([{
        '比赛日期': '2099-08-29', '赛事编号': '周六002',
        '联赛': '测试', '对阵': '丙 vs 丁', '推荐玩法': '胜平负',
        '重点选项': '★ 负',
    }])
    sporttery_module._save_daily_recommendation_snapshot(first)
    sporttery_module._save_daily_recommendation_snapshot(latest)
    monkeypatch.setattr(
        sporttery_module, 'load_yesterday_hit_report',
        lambda: (pd.DataFrame([{
            '赛事编号': '周六002', '完场比分': '0-1',
            '胜负': '胜 → 负（命中）',
        }]), {'date': '2099-08-29'}),
    )

    review, _ = sporttery_module.build_yesterday_recommendation_review()

    assert review['赛事编号'].tolist() == ['周六002']
    assert review.loc[0, '命中状态'] == '✓ 命中'


def test_previous_card_merges_still_visible_after_midnight_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(sporttery_module, 'DAILY_RECOMMENDATION_ROOT', tmp_path)
    card_day = (pd.Timestamp.today().date() - pd.Timedelta(days=1)).isoformat()
    frozen_before_midnight = pd.DataFrame([{
        '比赛日期': card_day, '赛事编号': '周六001',
        '推荐玩法': '胜平负', '重点选项': '★ 胜', '正式模型概率': '66.0%',
    }])
    newly_visible_after_midnight = pd.DataFrame([{
        '比赛日期': card_day, '赛事编号': '周六026',
        '推荐玩法': '让球胜平负', '重点选项': '★ +1球 负',
        '正式模型概率': '54.6%',
    }])

    sporttery_module._save_daily_recommendation_snapshot(frozen_before_midnight)
    sporttery_module._save_daily_recommendation_snapshot(newly_visible_after_midnight)
    frozen = sporttery_module._load_daily_recommendation_snapshot(card_day)

    assert frozen['赛事编号'].tolist() == ['周六001', '周六026']
    assert frozen.loc[frozen['赛事编号'].eq('周六026'), '重点选项'].item() == '★ +1球 负'


def test_previous_card_never_collapses_legacy_multi_play_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(sporttery_module, 'DAILY_RECOMMENDATION_ROOT', tmp_path)
    card_day = (pd.Timestamp.today().date() - pd.Timedelta(days=1)).isoformat()
    legacy = pd.DataFrame([
        {
            '比赛日期': card_day, '赛事编号': '周六018',
            '推荐玩法': '大小球', '重点选项': '★ 大于2.5球',
        },
        {
            '比赛日期': card_day, '赛事编号': '周六018',
            '推荐玩法': '半全场', '重点选项': '★ 胜胜',
        },
    ])
    today_rule_for_same_delayed_match = pd.DataFrame([{
        '比赛日期': card_day, '赛事编号': '周六018',
        '推荐玩法': '胜平负', '重点选项': '★ 胜',
    }])

    sporttery_module._save_daily_recommendation_snapshot(legacy)
    sporttery_module._save_daily_recommendation_snapshot(today_rule_for_same_delayed_match)
    frozen = sporttery_module._load_daily_recommendation_snapshot(card_day)

    assert frozen['推荐玩法'].tolist() == ['大小球', '半全场', '胜平负']


def test_yesterday_review_keeps_postponed_recommendation_pending(monkeypatch):
    frozen = pd.DataFrame([{
        '比赛日期': '2026-08-29', '赛事编号': '周六018',
        '联赛': '挪威超级联赛', '对阵': '维京 vs 奥勒松',
        '推荐玩法': '大小球', '重点选项': '★ 大于2.5球',
        '模型概率': '74.9%',
    }])
    monkeypatch.setattr(
        sporttery_module, 'load_yesterday_hit_report',
        lambda: (pd.DataFrame([{
            '赛事编号': '周六001', '完场比分': '1-0',
        }]), {'date': '2026-08-29'}),
    )
    monkeypatch.setattr(
        sporttery_module, '_load_daily_recommendation_snapshot',
        lambda day: frozen,
    )

    review, review_date = sporttery_module.build_yesterday_recommendation_review()

    assert review_date == '2026-08-29'
    assert review.loc[0, '赛事编号'] == '周六018'
    assert review.loc[0, '命中状态'] == '○ 待复盘'
    assert '延期或未完场' in review.loc[0, '复盘结果']


def test_yesterday_review_lists_frozen_rows_when_no_result_is_settled(monkeypatch):
    frozen = pd.DataFrame([{
        '比赛日期': '2026-08-29', '赛事编号': '周六018',
        '联赛': '挪威超级联赛', '对阵': '维京 vs 奥勒松',
        '推荐玩法': '胜平负', '重点选项': '★ 胜',
        '正式模型概率': '74.9%',
    }])
    monkeypatch.setattr(
        sporttery_module, 'load_yesterday_hit_report',
        lambda: (pd.DataFrame(), {'date': '2026-08-29'}),
    )
    monkeypatch.setattr(
        sporttery_module, '_load_daily_recommendation_snapshot',
        lambda day: frozen,
    )
    review, _ = sporttery_module.build_yesterday_recommendation_review()
    assert review.loc[0, '命中状态'] == '○ 待复盘'
    assert review.loc[0, '失败原因'] == '待官方赛果，不计失败'


def test_official_total_goals_preserves_earlier_displayed_over_under_audit(
        monkeypatch, tmp_path):
    monkeypatch.setattr(sporttery_module, 'DAILY_RECOMMENDATION_ROOT', tmp_path)
    legacy = pd.DataFrame([{
        '比赛日期': '2099-08-30', '赛事编号': '周日013',
        '推荐玩法': '大小球', '重点选项': '★ 大于2.5球', '模型概率': '80.2%',
    }])
    official = pd.DataFrame([{
        '比赛日期': '2099-08-30', '赛事编号': '周日013',
        '推荐玩法': '总进球', '重点选项': '★ 3球', '模型概率': '24.0%',
    }])
    sporttery_module._save_daily_recommendation_snapshot(legacy)
    sporttery_module._save_daily_recommendation_snapshot(official)

    frozen = sporttery_module._load_daily_recommendation_snapshot('2099-08-30')

    assert frozen['推荐玩法'].tolist() == ['大小球', '总进球']
    assert frozen['重点选项'].tolist() == ['★ 大于2.5球', '★ 3球']


def _half_time_combination_prediction():
    return pd.DataFrame([{
        '比赛ID': 7, '赛事编号': '周日001',
        '比赛时间': '2099-08-30 18:00', '联赛': '测试联赛',
        '主队': '主队', '客队': '客队',
        '半全场模型来源': '测试联赛专用半全场模型（已验证）',
        '半场模型来源': '测试联赛专用半场胜平负模型（已验证）',
        '半场模型高置信门槛': .55,
        '半场模型当前置信度': .65,
        '正式半场胜概率': .25, '正式半场平概率': .65,
        '正式半场负概率': .10,
        '模拟半场胜概率': .28, '模拟半场平概率': .62,
        '模拟半场负概率': .10,
        '官方半全场胜胜奖金': 4.00, '官方半全场胜平奖金': 10.00,
        '官方半全场胜负奖金': 20.00,
        '官方半全场平胜奖金': 5.20, '官方半全场平平奖金': 5.30,
        '官方半全场平负奖金': 7.00,
        '官方半全场负胜奖金': 8.00, '官方半全场负平奖金': 12.00,
        '官方半全场负负奖金': 3.50,
        '阵容方向冲突': False, '阵容预警级别': '无',
    }])


def test_half_time_combination_uses_inverse_odds_equal_return_math():
    result = sporttery_module.build_half_time_combinations(
        _half_time_combination_prediction(), future_only=False,
    )

    assert len(result) == 1
    assert result.loc[0, '目标半场'] == '平'
    expected = 1.0 / (1.0 / 5.20 + 1.0 / 5.30 + 1.0 / 7.00)
    assert abs(result.loc[0, '组合赔率'] - expected) < .001
    assert result.loc[0, '组合玩法'] == '平胜@5.20 / 平平@5.30 / 平负@7.00'
    assert result.loc[0, '半场含金量'].endswith('/100')
    assert 'EV +' in result.loc[0, '模型优势']


def test_half_combination_downgrades_when_real_history_is_overconfident(monkeypatch):
    monkeypatch.setattr(sporttery_module, 'historical_calibration', lambda *a, **kw: (.35, 400))
    result = sporttery_module.build_half_time_combinations(
        _half_time_combination_prediction(), future_only=False,
    )
    assert result.empty
    observations = sporttery_module.build_half_time_observations(
        _half_time_combination_prediction(), future_only=False,
    )
    assert len(observations) == 1
    assert '校准后50.0%' in observations.iloc[0]['观察结论']


def test_half_time_combination_rejects_market_derived_model():
    prediction = _half_time_combination_prediction()
    prediction.loc[0, '半场模型来源'] = '半全场概率聚合（非独立半场模型）'
    assert sporttery_module.build_half_time_combinations(
        prediction, future_only=False,
    ).empty
    observation = sporttery_module.build_half_time_observations(
        prediction, future_only=False,
    )
    assert len(observation) == 1
    assert observation.loc[0, '赛事编号'] == '周日001'
    assert '缺独立验证' in observation.loc[0, '观察结论']


def test_half_time_combination_requires_direct_model_confidence_threshold():
    prediction = _half_time_combination_prediction()
    prediction.loc[0, '半场模型高置信门槛'] = .70

    assert sporttery_module.build_half_time_combinations(
        prediction, future_only=False,
    ).empty
    observation = sporttery_module.build_half_time_observations(
        prediction, future_only=False,
    )
    assert '未达到半场模型高置信门槛' in observation.loc[0, '观察结论']


def test_half_time_observation_keeps_only_relative_best_per_card_day():
    prediction = _half_time_combination_prediction()
    weaker = prediction.iloc[0].copy()
    weaker['比赛ID'] = 8
    weaker['赛事编号'] = '周日002'
    weaker['正式半场胜概率'] = .33
    weaker['正式半场平概率'] = .36
    weaker['正式半场负概率'] = .31
    weaker['模拟半场胜概率'] = .33
    weaker['模拟半场平概率'] = .37
    weaker['模拟半场负概率'] = .30
    frame = pd.concat([prediction, weaker.to_frame().T], ignore_index=True)

    observation = sporttery_module.build_half_time_observations(
        frame, future_only=False,
    )

    assert len(observation) == 1
    assert observation.loc[0, '赛事编号'] == '周日001'
    assert observation.loc[0, '相对含金量'].endswith('/100')


def test_half_time_observation_snapshot_never_rewrites_first_pick(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        sporttery_module, 'HALF_TIME_OBSERVATION_ROOT', tmp_path,
    )
    first = sporttery_module.build_half_time_observations(
        _half_time_combination_prediction(), future_only=False,
    )
    sporttery_module._save_half_time_observation_snapshot(first)
    changed = first.copy()
    changed.loc[0, '赛事编号'] = '周日002'
    changed.loc[0, '目标半场'] = '胜'
    sporttery_module._save_half_time_observation_snapshot(changed)

    frozen = sporttery_module._load_half_time_observation_snapshot(
        first.loc[0, '比赛日期'],
    )

    assert frozen.loc[0, '赛事编号'] == '周日001'
    assert frozen.loc[0, '目标半场'] == '平'


def test_half_time_observation_review_matches_after_midnight_ticket_card(
        monkeypatch, tmp_path):
    observation_root = tmp_path / 'observations'
    settled_path = tmp_path / 'settled.csv'
    monkeypatch.setattr(
        sporttery_module, 'HALF_TIME_OBSERVATION_ROOT', observation_root,
    )
    monkeypatch.setattr(
        sporttery_module, 'SETTLED_PREDICTIONS_PATH', settled_path,
    )
    observation = sporttery_module.build_half_time_observations(
        _half_time_combination_prediction(), future_only=False,
    )
    observation['比赛ID'] = observation['比赛ID'].astype(str)
    observation.loc[0, '比赛ID'] = ''
    sporttery_module._save_half_time_observation_snapshot(observation)
    pd.DataFrame([{
        'match_id': 70, 'match_number': '周日001',
        'match_date': '2099-08-31 00:30',
        'actual_half_full': '平负', 'official_status': 'Payout',
    }]).to_csv(settled_path, index=False)

    review, summary = sporttery_module.build_half_time_observation_review(
        '2099-08-30',
    )

    assert review.loc[0, '半场赛果'] == '半场平'
    assert review.loc[0, '命中状态'] == '✓ 命中'
    assert summary['settled'] == 1
    assert summary['hits'] == 1
    assert summary['hit_rate'] == 1.0


def test_half_time_observation_review_never_uses_same_number_from_other_day(
        monkeypatch, tmp_path):
    observation_root = tmp_path / 'observations'
    settled_path = tmp_path / 'settled.csv'
    monkeypatch.setattr(
        sporttery_module, 'HALF_TIME_OBSERVATION_ROOT', observation_root,
    )
    monkeypatch.setattr(
        sporttery_module, 'SETTLED_PREDICTIONS_PATH', settled_path,
    )
    observation = sporttery_module.build_half_time_observations(
        _half_time_combination_prediction(), future_only=False,
    )
    observation['比赛ID'] = observation['比赛ID'].astype(str)
    observation.loc[0, '比赛ID'] = ''
    sporttery_module._save_half_time_observation_snapshot(observation)
    pd.DataFrame([{
        'match_id': 99, 'match_number': '周日001',
        'match_date': '2099-08-23 18:00',
        'actual_half_full': '平负', 'official_status': 'Payout',
    }]).to_csv(settled_path, index=False)

    review, summary = sporttery_module.build_half_time_observation_review(
        '2099-08-30',
    )

    assert review.loc[0, '命中状态'] == '○ 延期/待定'
    assert '不计失败' in review.loc[0, '复盘结果']
    assert summary['settled'] == 0
    assert summary['pending'] == 1


def test_half_time_combination_ledger_freezes_first_pick_and_settles_profit(
        monkeypatch, tmp_path):
    combination_root = tmp_path / 'combinations'
    settled_path = tmp_path / 'settled.csv'
    monkeypatch.setattr(
        sporttery_module, 'HALF_TIME_COMBINATION_ROOT', combination_root,
    )
    monkeypatch.setattr(
        sporttery_module, 'SETTLED_PREDICTIONS_PATH', settled_path,
    )
    first = sporttery_module.build_half_time_combinations(
        _half_time_combination_prediction(), future_only=False,
    )
    sporttery_module._save_half_time_combination_snapshot(first)
    changed = first.copy()
    changed.loc[0, '组合赔率'] = 9.99
    sporttery_module._save_half_time_combination_snapshot(changed)
    frozen = sporttery_module._load_half_time_combination_snapshot(
        first.loc[0, '比赛日期'],
    )
    assert abs(float(frozen.loc[0, '组合赔率']) - float(first.loc[0, '组合赔率'])) < .001

    pd.DataFrame([{
        'match_id': 7, 'match_number': '周日001',
        'match_date': '2099-08-30 18:00',
        'actual_half_full': '平负', 'official_status': 'Payout',
    }]).to_csv(settled_path, index=False)
    ledger, summary = sporttery_module.build_half_time_combination_ledger()

    assert ledger.loc[0, '结算状态'] == '✓ 命中'
    assert summary['settled'] == 1
    assert summary['hits'] == 1
    assert abs(summary['profit'] - 1000 * (float(first.loc[0, '组合赔率']) - 1)) < 1
    assert abs(summary['roi'] - (float(first.loc[0, '组合赔率']) - 1)) < .001


def test_half_time_ledger_never_matches_same_number_from_another_day(
        monkeypatch, tmp_path):
    combination_root = tmp_path / 'combinations'
    settled_path = tmp_path / 'settled.csv'
    monkeypatch.setattr(
        sporttery_module, 'HALF_TIME_COMBINATION_ROOT', combination_root,
    )
    monkeypatch.setattr(
        sporttery_module, 'SETTLED_PREDICTIONS_PATH', settled_path,
    )
    candidate = sporttery_module.build_half_time_combinations(
        _half_time_combination_prediction(), future_only=False,
    )
    candidate.loc[0, '比赛ID'] = ''
    sporttery_module._save_half_time_combination_snapshot(candidate)
    pd.DataFrame([{
        'match_id': 99, 'match_number': '周日001',
        'match_date': '2099-08-23 18:00',
        'actual_half_full': '平负', 'official_status': 'Payout',
    }]).to_csv(settled_path, index=False)

    ledger, summary = sporttery_module.build_half_time_combination_ledger()

    assert ledger.loc[0, '结算状态'] == '○ 延期/待定'
    assert summary['settled'] == 0
