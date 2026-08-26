import numpy as np
import pandas as pd

from src.services.draw_calibration import _apply, _metrics, _prepare


def test_draw_adjustment_preserves_total_and_side_ratio():
    base = np.array([[0.50, 0.20, 0.30]])
    result = _apply(base, np.array([0.32]), 0.50)
    assert np.isclose(result.sum(), 1.0)
    assert result[0, 1] > base[0, 1]
    assert np.isclose(result[0, 0] / result[0, 2], 5 / 3)


def test_hierarchical_priors_use_only_earlier_matches():
    rows = pd.DataFrame([
        {'match_date': f'2026-01-0{index + 1}', 'prediction_date': '2026-01-01',
         'league': '测试联赛', 'home': f'H{index}', 'away': f'A{index}',
         'odds_home': 2.0, 'odds_draw': 3.2, 'odds_away': 3.6,
         'actual_result': result}
        for index, result in enumerate([1, 1, 0])
    ])
    features, _ = _prepare(rows, prior_strength=30)
    assert np.isclose(features.loc[0, 'league_draw_prior'], 0.27)
    assert features.loc[2, 'league_draw_prior'] > features.loc[0, 'league_draw_prior']


def test_draw_audit_reports_all_required_metrics():
    target = np.array([0, 1, 2, 1])
    probability = np.array([
        [0.6, 0.2, 0.2], [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6], [0.3, 0.4, 0.3],
    ])
    metrics = _metrics(target, probability)
    assert {'accuracy', 'brier', 'log_loss', 'draw_precision', 'draw_recall'} <= metrics.keys()
