import unittest
import numpy as np
import pandas as pd
from datetime import date
from unittest.mock import patch

from src.network.fixtures.sporttery import latest_had_odds, latest_hhad_odds
from src.services.daily_sporttery import (
    _cached_league_model,
    _calibrate_draw_probability,
    _clear_prediction_model_cache,
    _cup_market_features,
    _handicap_probabilities,
    _diverse_score_ranking,
    _aggressive_upset_score,
    _implied_had_from_handicap_market,
    _implied_had_without_result_market,
    _market_baseline_probabilities,
    _monte_carlo_summary,
    _market_selection,
    _over_under_model_is_reliable,
    _result_model_is_reliable,
    _predict_supported_match,
    _score_ranking_consistent_with_total,
    _sort_by_match_number,
    _upset_score,
    identify_league,
)
from src.services.team_names import resolve_model_team


class DailySportteryTests(unittest.TestCase):

    def test_draw_calibration_uses_league_prior_without_breaking_sum(self):
        history = pd.DataFrame({'Result': ['D'] * 30 + ['H'] * 40 + ['A'] * 30})
        result = _calibrate_draw_probability(
            np.array([0.48, 0.20, 0.32]),
            np.array([0.42, 0.30, 0.28]),
            history,
        )
        self.assertGreater(result[1], 0.20)
        self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_over_under_model_must_beat_its_sealed_baseline(self):
        weak = {'train': {'tuning': {
            'test_accuracy': 0.55, 'majority_baseline': 0.59,
        }}}
        useful = {'train': {'tuning': {
            'test_accuracy': 0.57, 'majority_baseline': 0.51,
        }}}
        self.assertFalse(_over_under_model_is_reliable(weak))
        self.assertTrue(_over_under_model_is_reliable(useful))

    @patch('src.services.daily_sporttery._cached_model_database')
    def test_dedicated_model_is_loaded_once_per_prediction_batch(self, database):
        model = database.return_value.load_model.return_value
        _clear_prediction_model_cache()
        first = _cached_league_model('英超', '英超胜平负模型')
        second = _cached_league_model('英超', '英超胜平负模型')
        self.assertIs(first, second)
        database.return_value.load_model.assert_called_once_with('英超胜平负模型')
        _clear_prediction_model_cache()

    def test_official_full_and_short_names_resolve_to_dedicated_model(self):
        self.assertEqual(
            resolve_model_team('英超', ['曼彻斯特联', '曼联']), 'Man United',
        )
        self.assertEqual(
            resolve_model_team('西甲', ['维戈塞尔塔', '塞尔塔']), 'Celta',
        )
        self.assertEqual(resolve_model_team('意甲', '弗洛西诺内'), 'Frosinone')
        self.assertIsNone(resolve_model_team('英超', '完全不存在球队'))

    def test_ticket_sort_uses_real_date_then_daily_sequence(self):
        import pandas as pd
        frame = pd.DataFrame([
            {'赛事编号': '周三003', '比赛时间': '2026-08-12 20:00'},
            {'赛事编号': '周二010', '比赛时间': '2026-08-11 21:00'},
            {'赛事编号': '周三001', '比赛时间': '2026-08-12 18:00'},
            {'赛事编号': '周二003', '比赛时间': '2026-08-11 19:00'},
            {'赛事编号': '星期二001', '比赛时间': '2026-08-11 17:00'},
        ])
        sorted_frame = _sort_by_match_number(frame)
        self.assertEqual(sorted_frame['赛事编号'].tolist(), [
            '星期二001', '周二003', '周二010', '周三001', '周三003',
        ])

    def test_next_monday_never_precedes_current_weekend(self):
        import pandas as pd
        frame = pd.DataFrame([
            {'赛事编号': '周一001', '比赛时间': '2026-08-24 18:00'},
            {'赛事编号': '周日002', '比赛时间': '2026-08-23 20:00'},
            {'赛事编号': '周六003', '比赛时间': '2026-08-22 21:00'},
        ])
        self.assertEqual(_sort_by_match_number(frame)['赛事编号'].tolist(), [
            '周六003', '周日002', '周一001',
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
        self.assertIsNone(identify_league('巴西甲级联赛'))
        self.assertEqual(identify_league('西甲 皇家马德里 VS 巴塞罗那'), '西甲')

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

    def test_handicap_only_market_produces_internal_had_input(self):
        inferred = _implied_had_from_handicap_market(
            {'line': -2.0, 'H': 2.30, 'D': 3.50, 'A': 2.45},
        )
        probabilities = 1.0 / np.array([
            inferred['H'], inferred['D'], inferred['A'],
        ])
        probabilities /= probabilities.sum()
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertGreater(probabilities[0], probabilities[2])

    def test_fixture_without_had_or_hhad_still_gets_hidden_input(self):
        inferred = _implied_had_without_result_market({})
        self.assertEqual(set(inferred), {'H', 'D', 'A'})
        self.assertTrue(all(value > 1.0 for value in inferred.values()))

    @patch('src.services.daily_sporttery._load_cup_market_artifact', return_value=None)
    def test_handicap_only_row_keeps_other_prediction_outputs(self, _artifact):
        handicap = {'line': -2.0, 'H': 2.30, 'D': 3.50, 'A': 2.45}
        inferred = _implied_had_from_handicap_market(handicap)
        row = _predict_supported_match(
            raw={
                'matchNumStr': '周六020', 'matchId': 2040992,
                'matchDate': '2026-08-22', 'matchTime': '20:00:00',
                'leagueAllName': '意大利甲级联赛', 'sellStatus': '2',
            },
            league=None, home_cn='国际米兰', away_cn='蒙扎',
            home='国际米兰', away='蒙扎', odds=inferred,
            handicap_odds=handicap, regular_market_offered=False,
        )
        self.assertTrue(pd.isna(row['官方胜奖金']))
        self.assertEqual(row['官方让球数'], -2.0)
        self.assertIn(row['让球首选'], ('胜', '平', '负'))
        self.assertIn(row['大小球首选'], ('大于2.5球', '小于2.5球'))
        self.assertTrue(row['首选比分'])
        self.assertTrue(row['半全场首选'])

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

    def test_main_score_ranking_is_not_overwritten_by_over_under_direction(self):
        classes = np.array([14, 15, 21, 8])
        probabilities = np.array([0.20, 0.18, 0.12, 0.19])
        ranking = np.argsort(probabilities)[::-1][:3]
        self.assertEqual(divmod(int(classes[ranking[0]]), 7), (2, 0))

    def test_score_shortlist_covers_a_different_match_script(self):
        classes = np.array([15, 7, 14, 8, 1])  # 2-1, 1-0, 2-0, 1-1, 0-1
        probabilities = np.array([0.25, 0.20, 0.18, 0.17, 0.16])
        ranking = _diverse_score_ranking(probabilities, classes)
        self.assertEqual(classes[ranking].tolist(), [15, 7, 8])

    def test_aggressive_score_is_high_scoring_and_in_market_cold_side(self):
        classes = np.array([21, 22, 4, 5, 12])  # 3-0, 3-1, 0-4, 0-5, 1-5
        probabilities = np.array([0.08, 0.06, 0.03, 0.004, 0.02])
        probability, score = _aggressive_upset_score(
            probabilities, classes,
            np.array([0.60, 0.25, 0.15]), frozenset(),
        )
        self.assertEqual(score, '0-4')
        self.assertAlmostEqual(probability, 0.03)

    def test_monte_carlo_summary_is_reproducible_and_complete(self):
        rows = []
        for index in range(12):
            rows.extend([
                {'Date': f'2025-{index + 1:02d}-01', 'Home': 'A', 'Away': 'X',
                 'HG': 2, 'AG': index % 2},
                {'Date': f'2025-{index + 1:02d}-02', 'Home': 'Y', 'Away': 'B',
                 'HG': 1, 'AG': 1 + index % 2},
            ])
        history = pd.DataFrame(rows)
        args = (history, 'A', 'B', date(2026, 1, 1), 0.02, True, -1, 123)
        first = _monte_carlo_summary(*args)
        second = _monte_carlo_summary(*args)
        self.assertEqual(first, second)
        self.assertEqual(first['模拟次数'], 10_000)
        self.assertEqual(len(first['模拟Top3比分'].split(' / ')), 3)
        self.assertTrue(first['模拟让球'])
        self.assertTrue(first['模拟模型来源'].startswith('历史攻防双泊松蒙特卡洛'))
        self.assertIn('未使用赔率/正式模型/首发校正', first['模拟模型来源'])

    def test_monte_carlo_refuses_to_fake_independence_without_history(self):
        result = _monte_carlo_summary(
            None, 'A', 'B', date(2026, 1, 1), 0.0, False, -1, 123,
        )
        self.assertEqual(result['模拟次数'], 0)
        self.assertIn('历史攻防样本不足', result['模拟模型来源'])
        self.assertIn('未使用赔率/模型兜底', result['模拟模型来源'])

    def test_monte_carlo_rejects_market_fallback_rates(self):
        result = _monte_carlo_summary(
            None, 'A', 'B', date(2026, 1, 1), 0.0, False, -1, 123,
            fallback_goal_rates=(1.55, 1.05),
        )
        self.assertEqual(result['模拟次数'], 0)
        self.assertFalse(result['模拟胜负'])
        self.assertIn('未使用赔率/模型兜底', result['模拟模型来源'])

    def test_legacy_blank_simulation_is_backfilled_from_score_prior(self):
        from src.services.daily_sporttery import backfill_missing_simulations

        predictions = pd.DataFrame([{
            '赛事编号': '周六001', '比赛ID': 123, '比赛时间': '2026-08-15 18:00',
            '主队': 'A', '客队': 'B', '官方让球数': -1,
            '模拟胜负': '', '模拟模型来源': '历史攻防样本不足',
        }])
        result = backfill_missing_simulations(predictions).iloc[0]

        self.assertTrue(result['模拟胜负'])
        self.assertTrue(result['模拟让球'])
        self.assertTrue(result['模拟Top3比分'])
        self.assertIn('本地跨联赛真实比分先验', result['模拟模型来源'])

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

    @patch('src.services.daily_sporttery._load_cup_market_artifact', return_value=None)
    def test_confirmed_lineup_shift_is_bounded_and_applied(self, _artifact):
        row = _predict_supported_match(
            raw={
                'matchNumStr': '周六001', 'matchId': 1,
                'matchDate': '2026-08-22', 'matchTime': '19:30:00',
                'leagueAllName': '未训练联赛', 'sellStatus': '2',
            },
            league=None, home_cn='主队', away_cn='客队', home='主队', away='客队',
            odds={'H': 2.0, 'D': 3.2, 'A': 3.6}, handicap_odds=None,
            lineup_analysis={
                'status': '已确认', 'summary': '已确认首发',
                'probability_shift': 0.04,
            },
        )
        baseline = _market_baseline_probabilities({'H': 2.0, 'D': 3.2, 'A': 3.6})
        self.assertGreater(row['模型主胜概率'], baseline['result'][0])
        self.assertEqual(row['首发状态'], '已确认')
        self.assertIn('首发主队利好', row['最终结论'])

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
