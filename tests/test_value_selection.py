import pandas as pd

from src.services.value_selection import evaluate_value, historical_calibration


def test_half_calibration_uses_only_verified_past_predictions(tmp_path):
    path = tmp_path / 'half.csv'
    pd.DataFrame({
        'half_p_home': [.2] * 4, 'half_p_draw': [.6] * 4,
        'half_p_away': [.2] * 4,
        'actual_half_full': ['平胜', '胜平', '平平', '平负'],
        'half_model_source': ['英超专用半场胜平负模型（已验证）'] * 3 + ['市场基线'],
        'settled_at': ['2026-09-01'] * 2 + ['2026-09-10', '2026-09-01'],
    }).to_csv(path, index=False)
    assert historical_calibration(
        '半场胜平负', .6, path=path, selection='平', as_of='2026-09-05',
    ) == (.5, 2)


def test_regular_result_core_value_uses_conservative_probability():
    decision = evaluate_value('胜平负', 0.68, 1.80)
    assert decision.grade == '核心重点'
    assert decision.conservative_probability == 0.655
    assert round(decision.conservative_ev, 3) == 0.179
    assert 0 < decision.stake_fraction <= 0.01


def test_high_probability_without_price_value_is_observation_only():
    decision = evaluate_value('胜平负', 0.70, 1.35)
    assert decision.conservative_ev < 0
    assert decision.grade == '观察'
    assert decision.stake_fraction == 0


def test_handicap_uses_its_own_stricter_gate():
    candidate = evaluate_value('让球胜平负', 0.60, 1.90)
    core = evaluate_value('让球胜平负', 0.64, 2.00)
    assert candidate.grade == '可买优选'
    assert core.grade == '核心重点'


def test_history_can_only_reduce_not_inflate_live_probability():
    lowered = evaluate_value(
        '胜平负', 0.68, 1.80,
        empirical_accuracy=0.55, empirical_samples=400,
    )
    lucky = evaluate_value(
        '胜平负', 0.68, 1.80,
        empirical_accuracy=0.90, empirical_samples=400,
    )
    assert lowered.calibrated_probability < 0.68
    assert lucky.calibrated_probability == 0.68


def test_handicap_history_is_learned_separately(tmp_path):
    path = tmp_path / 'settled.csv'
    pd.DataFrame({
        'handicap_probability': [0.60] * 100,
        'handicap_hit': [1] * 60 + [0] * 40,
        'predicted_result': ['胜'] * 100,
        'result_hit': [0] * 100,
        'model_p_home': [0.60] * 100,
        'model_p_draw': [0.20] * 100,
        'model_p_away': [0.20] * 100,
    }).to_csv(path, index=False, encoding='utf-8-sig')
    accuracy, samples = historical_calibration('让球胜平负', 0.61, path=path)
    assert samples == 100
    assert accuracy == 0.60


def test_regular_calibration_is_direction_specific_and_forward_only(tmp_path):
    path = tmp_path / 'settled.csv'
    pd.DataFrame({
        'predicted_result': ['胜'] * 100 + ['负'] * 100 + ['胜'],
        'result_hit': [1] * 70 + [0] * 30 + [1] * 20 + [0] * 80 + [1],
        'model_p_home': [0.60] * 100 + [0.20] * 100 + [0.60],
        'model_p_draw': [0.20] * 201,
        'model_p_away': [0.20] * 100 + [0.60] * 100 + [0.20],
        'settled_at': ['2026-09-01 10:00:00'] * 200 + ['2026-09-06 10:00:00'],
    }).to_csv(path, index=False, encoding='utf-8-sig')
    accuracy, samples = historical_calibration(
        '胜平负', 0.60, path=path, selection='胜', as_of='2026-09-05',
    )
    assert samples == 100
    assert accuracy == 0.70


def test_handicap_calibration_separates_direction_and_line(tmp_path):
    path = tmp_path / 'settled.csv'
    pd.DataFrame({
        'handicap_probability': [0.60] * 300,
        'handicap_hit': [1] * 65 + [0] * 35 + [1] * 20 + [0] * 80 + [1] * 90 + [0] * 10,
        'predicted_handicap': ['胜'] * 100 + ['负'] * 100 + ['胜'] * 100,
        'handicap_line': [-1] * 200 + [1] * 100,
    }).to_csv(path, index=False, encoding='utf-8-sig')
    accuracy, samples = historical_calibration(
        '让球胜平负', 0.60, path=path, selection='胜', handicap_line=-1,
    )
    assert samples == 100
    assert accuracy == 0.65
