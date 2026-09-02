from datetime import date

import pandas as pd

from src.services import daily_learning


class FakeResultClient:
    def settled_matches(self, begin_date, end_date):
        return [{
            'matchId': 12345,
            'matchDate': '2026-08-10',
            'matchNumStr': '周一001',
            'leagueName': '测试联赛',
            'allHomeTeam': '主队',
            'allAwayTeam': '客队',
            'sectionsNo1': '1:0',
            'sectionsNo999': '2:1',
            'matchResultStatus': '2',
            'poolStatus': 'Payout',
            'h': '1.82', 'd': '3.35', 'a': '4.10',
        }]


def test_daily_review_settles_once_and_builds_truth_dataset(tmp_path, monkeypatch):
    reports = tmp_path / 'reports'
    learning = tmp_path / 'learning'
    reports.mkdir()
    pd.DataFrame([{
        '比赛ID': 12345,
        '比赛时间': '2026-08-10',
        '赛事编号': '周一001',
        '联赛': '测试联赛',
        '主队': '主队',
        '客队': '客队',
        '官方胜奖金': 1.80,
        '官方平奖金': 3.40,
        '官方负奖金': 4.20,
        '市场去水主胜概率': 0.52,
        '市场去水平局概率': 0.27,
        '市场去水客胜概率': 0.21,
        '模型主胜概率': 0.55,
        '模型平局概率': 0.25,
        '模型客胜概率': 0.20,
        '胜平负首选': '胜',
        '建议状态': '谨慎主推',
        '预测依据': '测试模型',
        '首选比分': '2-1',
        '次选比分': '1-1',
        '第三比分': '2-0',
        '比分爆冷': '0-1',
        '大小球进取比分': '3-1',
        '大小球首选': '大于2.5球',
        '官方让球数': -1,
        '让球首选': '负',
        '让球次选': '平',
        '半全场首选': '平胜',
        '半全场次选': '胜胜',
        '模拟半全场': '负负 31.0% / 平负 22.0%',
        '模拟模型来源': '独立历史攻防蒙特卡洛',
    }]).to_csv(reports / '2026-08-09-竞彩预测.csv', index=False)

    monkeypatch.setattr(daily_learning, 'REPORT_ROOT', reports)
    monkeypatch.setattr(daily_learning, 'LEARNING_ROOT', learning)
    monkeypatch.setattr(daily_learning, 'SETTLED_PATH', learning / 'settled.csv')
    monkeypatch.setattr(
        daily_learning, 'OFFICIAL_HISTORY_PATH', learning / 'official.csv',
    )
    monkeypatch.setattr(daily_learning, 'STATUS_PATH', learning / 'status.json')
    monkeypatch.setattr(daily_learning, 'GENERIC_MODEL_PATH', tmp_path / 'generic.joblib')
    monkeypatch.setattr(daily_learning, 'AUDIT_ROOT', tmp_path / 'audits')
    daily_learning.load_generic_artifact.cache_clear()

    first = daily_learning.review_and_learn(
        today=date(2026, 8, 11), result_client=FakeResultClient(),
    )
    second = daily_learning.review_and_learn(
        today=date(2026, 8, 11), result_client=FakeResultClient(),
    )
    settled = pd.read_csv(daily_learning.SETTLED_PATH)

    assert first['newly_settled'] == 1
    assert first['settled_samples'] == 1
    assert first['new_official_history'] == 1
    assert first['official_history_samples'] == 1
    assert first['total_training_samples'] == 1
    assert first['result_accuracy'] == 1.0
    assert first['score_accuracy'] == 1.0
    assert first['over_under_accuracy'] == 1.0
    assert first['model_status'] == '积累样本中'
    assert second['newly_settled'] == 0
    assert second['new_official_history'] == 0
    assert second['settled_samples'] == 1
    assert len(settled) == 1
    assert settled.loc[0, 'actual_result_label'] == '胜'
    assert settled.loc[0, 'actual_score'] == '2-1'
    assert settled.loc[0, 'actual_handicap'] == '平'
    assert settled.loc[0, 'handicap_hit'] == 0
    assert settled.loc[0, 'handicap_second_hit'] == 1
    assert settled.loc[0, 'actual_half_full'] == '胜胜'
    assert settled.loc[0, 'half_full_hit'] == 0
    assert settled.loc[0, 'half_full_second_hit'] == 1
    assert settled.loc[0, 'monte_carlo_half_full'].startswith('负负')
    assert settled.loc[0, 'monte_carlo_source'] == '独立历史攻防蒙特卡洛'
    assert settled.loc[0, 'score_hit_any'] == 1
    assert settled.loc[0, 'score_hit_top3'] == 1
    assert settled.loc[0, 'score_hit_source'] == '首'
    assert first['score_top1_accuracy'] == 1.0
    assert first['score_top3_accuracy'] == 1.0


def test_suspended_result_is_not_used_as_training_truth():
    prediction = pd.Series({
        '_prediction_date': '2026-08-09', '_match_id': '1',
        '_source_report': 'source.csv',
    })
    result = {
        'sectionsNo999': '2:1', 'matchResultStatus': '3',
    }
    assert daily_learning._settled_record(prediction, result) is None


def test_match_dates_accept_mixed_date_and_datetime_values():
    parsed = daily_learning._match_dates(pd.Series([
        '2026-08-27', '2026-08-28 01:00', '2026-08-28T02:30:00', '',
    ]))

    assert parsed.tolist() == [
        date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 28), None,
    ]


def test_prediction_reports_keep_latest_nonempty_market_snapshot(tmp_path, monkeypatch):
    reports = tmp_path / 'reports'
    reports.mkdir()
    older = pd.DataFrame([{
        '比赛ID': 88, '比赛时间': '2026-08-29 18:30',
        '赛事编号': '周六004', '官方让球数': -1, '让球首选': '负',
        '首选比分': '1-1',
    }])
    newer = older.copy()
    newer['官方让球数'] = float('nan')
    newer['让球首选'] = ''
    newer['首选比分'] = '2-1'
    older.to_csv(reports / '2026-08-28-竞彩预测.csv', index=False)
    newer.to_csv(reports / '2026-08-29-竞彩预测.csv', index=False)
    monkeypatch.setattr(daily_learning, 'REPORT_ROOT', reports)

    row = daily_learning._prediction_reports(date(2026, 8, 30)).iloc[0]

    assert row['_prediction_date'] == '2026-08-29'
    assert row['官方让球数'] == -1
    assert row['让球首选'] == '负'
    assert row['首选比分'] == '2-1'


def test_restore_missing_settled_handicap_from_real_snapshot():
    settled = pd.DataFrame([{
        'match_id': '88', 'home_goals': 3, 'away_goals': 1,
        'handicap_line': float('nan'), 'predicted_handicap': '',
        'predicted_handicap_second': '', 'monte_carlo_handicap': '',
    }])
    predictions = pd.DataFrame([{
        '_match_id': '88', '官方让球数': -1, '让球首选': '负',
        '让球次选': '平', '模拟让球': '让负 60.0%',
    }])

    restored, changed = daily_learning._restore_missing_settled_markets(
        settled, predictions,
    )

    assert changed == 4
    assert restored.loc[0, 'actual_handicap'] == '胜'
    assert restored.loc[0, 'handicap_hit'] == 0
    assert restored.loc[0, 'handicap_second_hit'] == 0


def test_generic_challenger_uses_chronological_three_way_audit(tmp_path, monkeypatch):
    rows = []
    templates = [
        (1.40, 5.0, 8.0, 0),
        (4.00, 2.0, 4.0, 1),
        (8.00, 5.0, 1.4, 2),
    ]
    for index in range(30):
        home, draw, away, target = templates[index % len(templates)]
        rows.append({
            'match_date': f'2026-07-{index + 1:02d}',
            'prediction_date': f'2026-07-{index + 1:02d}',
            'league': f'联赛{index % 2}',
            'odds_home': home, 'odds_draw': draw, 'odds_away': away,
            'actual_result': target,
        })
    monkeypatch.setattr(daily_learning, 'MIN_TRAIN_ROWS', 30)
    monkeypatch.setattr(daily_learning, 'VALIDATION_ROWS', 6)
    monkeypatch.setattr(daily_learning, 'TEST_ROWS', 6)
    monkeypatch.setattr(daily_learning, 'RETRAIN_STEP', 5)
    monkeypatch.setattr(daily_learning, 'GENERIC_MODEL_PATH', tmp_path / 'generic.joblib')
    monkeypatch.setattr(daily_learning, 'AUDIT_ROOT', tmp_path / 'audits')
    daily_learning.load_generic_artifact.cache_clear()

    status = daily_learning._train_if_ready(pd.DataFrame(rows), {})

    assert status['last_training_attempt_rows'] == 30
    assert status['model_samples'] == 30
    assert status['model_status'].startswith('自主进化')
    assert list((tmp_path / 'audits').glob('audit-30-*.json'))


def test_learning_holdout_grows_with_history_but_is_bounded():
    assert daily_learning._holdout_rows(300, 60) == 60
    assert daily_learning._holdout_rows(4793, 60) == 479
    assert daily_learning._holdout_rows(10000, 60) == 500


def test_selection_thresholds_are_relearned_from_recent_results(tmp_path, monkeypatch):
    rows = []
    for index in range(800):
        rows.append({
            'match_date': f'2026-{1 + index // 70:02d}-{1 + index % 28:02d}',
            'prediction_date': '2026-01-01',
            'odds_home': 1.20, 'odds_draw': 5.0, 'odds_away': 8.0,
            'actual_result': 0,
        })
    for index in range(200):
        rows.append({
            'match_date': f'2026-12-{1 + index % 28:02d}',
            'prediction_date': '2026-12-01',
            'odds_home': 2.50, 'odds_draw': 3.0, 'odds_away': 2.8,
            'actual_result': index % 3,
        })
    path = tmp_path / 'selection-profile.json'
    monkeypatch.setattr(daily_learning, 'SELECTION_PROFILE_PATH', path)

    profile = daily_learning._learn_selection_profile(
        pd.DataFrame(rows), date(2026, 12, 31),
    )

    assert path.exists()
    assert profile['total_samples'] == 1000
    assert [row['grade'] for row in profile['rows']] == [
        '精选主推', '高置信主推', '观察', '跳过',
    ]
    thresholds = [row['threshold'] for row in profile['rows']]
    assert thresholds == sorted(thresholds, reverse=True)
    assert daily_learning.load_selection_profile()['total_samples'] == 1000


def test_over_under_profile_enables_only_direction_passing_time_audit(tmp_path, monkeypatch):
    rows = []
    for index in range(200):
        is_over = index % 2 == 0
        rows.append({
            'match_id': str(index), 'match_date': f'2026-08-{1 + index // 8:02d}',
            'predicted_over_under': '大于2.5球' if is_over else '小于2.5球',
            'over_under_probability': 0.62,
            'over_under_hit': (
                int(index % 8 != 0) if is_over else int(index % 6 == 1)
            ),
        })
    path = tmp_path / 'over-under.json'
    monkeypatch.setattr(daily_learning, 'OVER_UNDER_PROFILE_PATH', path)

    profile = daily_learning._learn_over_under_profile(
        pd.DataFrame(rows), date(2026, 8, 27),
    )

    directions = {row['pick']: row for row in profile['directions']}
    assert directions['大于2.5球']['enabled'] is True
    assert directions['大于2.5球']['threshold'] == 0.60
    assert directions['小于2.5球']['enabled'] is False
    assert daily_learning.load_over_under_profile()['total_samples'] == 200


def test_waiting_for_more_samples_keeps_last_audit_metrics():
    rows = pd.DataFrame([{
        'match_date': f'2026-07-{index + 1:02d}',
        'prediction_date': f'2026-07-{index + 1:02d}',
        'league': '测试联赛',
        'odds_home': 1.8, 'odds_draw': 3.2, 'odds_away': 4.0,
        'actual_result': 0,
    } for index in range(30)])
    previous = {
        'model_status': '候选未胜过基线，保留旧模型',
        'last_training_attempt_rows': 30,
        'training_evaluation_version': daily_learning.TRAINING_EVALUATION_VERSION,
        'last_training_passed': False,
        'challenger_test_accuracy': 0.51,
        'market_test_accuracy': 0.52,
    }
    original_minimum = daily_learning.MIN_TRAIN_ROWS
    original_step = daily_learning.RETRAIN_STEP
    try:
        daily_learning.MIN_TRAIN_ROWS = 30
        daily_learning.RETRAIN_STEP = 5
        status = daily_learning._train_if_ready(rows, previous)
    finally:
        daily_learning.MIN_TRAIN_ROWS = original_minimum
        daily_learning.RETRAIN_STEP = original_step
    assert status['last_training_passed'] is False
    assert status['challenger_test_accuracy'] == 0.51
    assert status['market_test_accuracy'] == 0.52


def test_low_live_accuracy_triggers_model_specific_market_fallback(tmp_path, monkeypatch):
    rows = []
    for index in range(30):
        rows.append({
            'match_date': f'2026-07-{index + 1:02d}',
            'prediction_date': f'2026-07-{index + 1:02d}',
            'model_category': '英超专用模型',
            'actual_result': 0,
            'result_hit': 0,
            'market_p_home': 0.70,
            'market_p_draw': 0.20,
            'market_p_away': 0.10,
        })

    audit = daily_learning._accuracy_by_model(pd.DataFrame(rows))
    assert audit['英超专用模型']['accuracy'] == 0.0
    assert audit['英超专用模型']['market_accuracy'] == 1.0
    assert audit['英超专用模型']['action'] == 'fallback_market'

    status_path = tmp_path / 'status.json'
    monkeypatch.setattr(daily_learning, 'STATUS_PATH', status_path)
    daily_learning._write_json(status_path, {'accuracy_by_model': audit})
    assert not daily_learning.model_result_is_allowed('英超专用模型')
    assert daily_learning.model_result_is_allowed('西甲专用模型')


def test_dedicated_model_weight_stays_shadowed_until_live_sample_gate(tmp_path, monkeypatch):
    status_path = tmp_path / 'status.json'
    monkeypatch.setattr(daily_learning, 'STATUS_PATH', status_path)
    daily_learning._write_json(status_path, {'accuracy_by_model': {
        '英超专用模型': {
            'samples': 20, 'edge_vs_market': 0.20, 'action': 'active',
        },
    }})
    assert daily_learning.model_result_blend_weight('英超专用模型') == 0.0

    daily_learning._write_json(status_path, {'accuracy_by_model': {
        '英超专用模型': {
            'samples': 40, 'edge_vs_market': 0.02, 'action': 'active',
        },
    }})
    assert abs(
        daily_learning.model_result_blend_weight('英超专用模型') - 4 / 15
    ) < 1e-12
