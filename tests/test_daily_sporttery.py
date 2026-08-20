import unittest
from datetime import date
from unittest.mock import patch

from src.network.fixtures.sporttery import latest_had_odds, latest_hhad_odds
from src.services.daily_sporttery import (
    _cup_market_features,
    _handicap_probabilities,
    _market_baseline_probabilities,
    _market_selection,
    _result_model_is_reliable,
    _predict_supported_match,
    _score_ranking_consistent_with_total,
    _sort_by_match_number,
    _upset_score,
    identify_league,
)


class DailySportteryTests(unittest.TestCase):

    def test_ticket_sort_uses_weekday_before_daily_sequence(self):
        import pandas as pd
        frame = pd.DataFrame([
            {'赛事编号': '周三003', '比赛时间': '2026-08-13'},
            {'赛事编号': '周二010', '比赛时间': '2026-08-12'},
            {'赛事编号': '周三001', '比赛时间': '2026-08-13'},
            {'赛事编号': '周二003', '比赛时间': '2026-08-12'},
            {'赛事编号': '星期二001', '比赛时间': '2026-08-13'},
        ])
        sorted_frame = _sort_by_match_number(frame)
        self.assertEqual(sorted_frame['赛事编号'].tolist(), [
            '星期二001', '周二003', '周二010', '周三001', '周三003',
        ])

    def test_ticket_sort_without_dates_keeps_weekdays_separate(self):
        import pandas as pd
        frame = pd.DataFrame({'赛事编号': ['周三003', '周二010', '周二003']})
        self.assertEqual(_sort_by_match_number(frame)['赛事编号'].tolist(), [
            '周二003', '周二010', '周三003',
        ])

    def test_identifies_supported_leagues_only(self):
        self.assertEqual(identify_league('英格兰超级联赛'), '英超')
        self.assertEqual(identify_league('西甲'), '西甲')
        self.assertEqual(identify_league('日本职业联赛'), '日职')
        self.assertEqual(identify_league('韩国职业联赛'), '韩职')

    def test_extracts_latest_had_fixed_bonus(self):
        value = {'oddsHistory': {'hadList': [
            {'h': '1.85', 'd': '3.20', 'a': '3.60'},
            {'h': '1.90', 'd': '3.10', 'a': '3.50'},
        ]}}
        self.assertEqual(latest_had_odds(value), {'H': 1.85, 'D': 3.2, 'A': 3.6})

    def test_missing_had_bonus_is_explicit(self):
        self.assertIsNone(latest_had_odds({'oddsHistory': {'hadList': []}}))

    def test_extracts_mobile_calculator_had(self):
        self.assertEqual(
            latest_had_odds({'had': {'h': '1.25', 'd': '5.25', 'a': '7.25'}}),
            {'H': 1.25, 'D': 5.25, 'A': 7.25},
        )

    def test_extracts_handicap_market(self):
        self.assertEqual(
            latest_hhad_odds({'hhad': {'goalLine': '-1', 'h': '2.2', 'd': '3.4', 'a': '2.7'}}),
            {'line': -1.0, 'H': 2.2, 'D': 3.4, 'A': 2.7},
        )

    def test_aggregates_score_grid_for_handicap(self):
        import numpy as np
        # 1-0 becomes a draw after a -1 home handicap; 2-0 remains a win.
        probabilities = _handicap_probabilities(
            np.array([0.4, 0.6]), np.array([7, 14]), -1,
        )
        self.assertTrue(np.allclose(probabilities, [0.6, 0.4, 0.0]))

    def test_selects_best_score_for_market_longshot_outcome(self):
        import numpy as np
        score, probability = _upset_score(
            np.array([0.10, 0.08, 0.12, 0.07]),
            np.array([7, 14, 1, 2]),  # 1-0, 2-0, 0-1, 0-2
            np.array([0.60, 0.25, 0.15]),
        )
        self.assertEqual(score, '0-1')
        self.assertAlmostEqual(probability, 0.12)

    def test_displayed_score_ranking_agrees_with_over_under_side(self):
        import numpy as np
        # Raw modal score is 2-0, but when aggregate O/U selects over, the
        # displayed shortlist must start from an over-2.5 score.
        classes = np.array([14, 15, 21, 8])  # 2-0, 2-1, 3-0, 1-1
        probabilities = np.array([0.20, 0.18, 0.12, 0.19])
        ranking = _score_ranking_consistent_with_total(
            probabilities, classes, prefer_over=True,
        )
        first_home, first_away = divmod(int(classes[ranking[0]]), 7)
        self.assertGreater(first_home + first_away, 2)
        self.assertEqual((first_home, first_away), (2, 1))

    def test_upset_score_skips_already_displayed_picks(self):
        import numpy as np
        # 1-1 is the strongest draw but is already the main pick; the upset row
        # must fall back to the next draw score instead of repeating it.
        score, probability = _upset_score(
            np.array([0.20, 0.12, 0.10, 0.08]),
            np.array([8, 24, 7, 14]),  # 1-1, 3-3, 1-0, 2-0
            np.array([0.35, 0.25, 0.40]),
            excluded=frozenset({'1-1'}),
        )
        self.assertEqual(score, '3-3')
        self.assertAlmostEqual(probability, 0.12)

    def test_market_baseline_is_normalized(self):
        baseline = _market_baseline_probabilities({'H': 2.6, 'D': 2.9, 'A': 2.5})
        for key in ('result', 'over_under', 'score', 'half_full'):
            self.assertAlmostEqual(float(baseline[key].sum()), 1.0, places=8)
        self.assertEqual(len(baseline['score']), 49)
        self.assertEqual(len(baseline['half_full']), 9)

    @patch('src.services.daily_sporttery.load_selection_profile', return_value=None)
    def test_market_selection_uses_audited_selective_thresholds(self, _profile):
        self.assertEqual(_market_selection(0.72)['grade'], '精选主推')
        self.assertEqual(_market_selection(0.63)['grade'], '高置信主推')
        self.assertEqual(_market_selection(0.57)['grade'], '观察')
        self.assertEqual(_market_selection(0.54)['grade'], '跳过')
        self.assertEqual(_market_selection(0.52)['grade'], '跳过')
        self.assertEqual(_market_selection(0.49)['grade'], '跳过')
        self.assertGreater(_market_selection(0.72)['accuracy'], 0.75)
        self.assertEqual(_market_selection(0.72)['samples'], 4816)

    def test_weak_or_tiny_result_model_fails_reliability_gate(self):
        weak = {'train': {'tuning': {'test_accuracy': 0.49, 'test_samples': 500}}}
        tiny = {'train': {'tuning': {'test_accuracy': 0.60, 'test_samples': 50}}}
        stable = {'train': {'tuning': {'test_accuracy': 0.54, 'test_samples': 500}}}
        self.assertFalse(_result_model_is_reliable(weak))
        self.assertFalse(_result_model_is_reliable(tiny))
        self.assertTrue(_result_model_is_reliable(stable))

    def test_cup_calibrator_feature_contract_is_finite(self):
        import numpy as np
        features = _cup_market_features(
            {'H': 2.1, 'D': 3.3, 'A': 3.5},
            date(2026, 8, 11), 'qualification', 1700.0, 1600.0, False,
        )
        self.assertEqual(features.shape, (1, 23))
        self.assertTrue(np.isfinite(features).all())

    @patch('src.services.daily_sporttery._load_cup_market_artifact', return_value=None)
    def test_untrained_competition_gets_marked_market_prediction(self, _artifact):
        row = _predict_supported_match(
            raw={
                'matchNumStr': '周二002', 'matchId': 2040829,
                'matchDate': '2026-08-11', 'leagueAllName': '欧洲冠军联赛',
            },
            league=None,
            home_cn='阿拉木图凯拉特', away_cn='索菲亚列夫斯基',
            home='阿拉木图凯拉特', away='索菲亚列夫斯基',
            odds={'H': 2.6, 'D': 2.9, 'A': 2.5},
            handicap_odds={'line': -1.0, 'H': 6.7, 'D': 4.0, 'A': 1.37},
            display_league='欧洲冠军联赛', fallback_reason='未训练联赛',
        )
        self.assertEqual(row['联赛'], '欧洲冠军联赛')
        self.assertIn('市场基线', row['预测依据'])
        self.assertEqual(row['专用模型联赛'], '')
        self.assertEqual(row['模型类别'], '通用/市场模型')
        self.assertEqual(row['置信等级'], '低')
        self.assertEqual(row['建议状态'], '跳过')
        self.assertIn('同阈值历史命中率', row)
        self.assertIn(row['胜平负首选'], {'胜', '平', '负'})
        self.assertTrue(row['首选比分'])
        self.assertTrue(row['半全场首选'])


if __name__ == '__main__':
    unittest.main()
