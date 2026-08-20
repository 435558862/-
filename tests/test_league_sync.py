import unittest

import pandas as pd

from src.services.league_sync import (
    SYNC_LEAGUES, _korean_feature_dataset, _preserve_half_time_target,
)
from src.services.team_names import chinese_team_name, translate_fixture_columns


class LeagueSyncTests(unittest.TestCase):

    def test_default_sync_includes_added_prediction_leagues(self):
        self.assertEqual(
            set(SYNC_LEAGUES),
            {'英超', '西甲', '德甲', '意甲', '法甲', '瑞超', '葡超', '日职', '韩职'},
        )

    def test_korean_rows_are_converted_to_prediction_schema(self):
        rows = []
        for week in range(1, 11):
            rows.extend([
                {
                    'Result': 'HT:0-0, FT:1-0', 'Match': 'A vs B',
                    'Start Time': f'2026-01-{week:02d} 19:00:00',
                    'Ft1X2_01': 2.0, 'Ft1X2_02': 3.2, 'Ft1X2_03': 3.4,
                },
                {
                    'Result': 'HT:0-0, FT:0-1', 'Match': 'B vs A',
                    'Start Time': f'2026-02-{week:02d} 19:00:00',
                    'Ft1X2_01': 2.6, 'Ft1X2_02': 3.1, 'Ft1X2_03': 2.5,
                },
            ])
        dataset = _korean_feature_dataset(pd.DataFrame(rows))
        self.assertIn('Week', dataset.columns)
        self.assertIn('HAGD', dataset.columns)
        self.assertGreater(len(dataset.dropna()), 0)

    def test_verified_half_time_targets_survive_base_feed_sync(self):
        before = pd.DataFrame([{
            'Date': '2026-01-01', 'Home': 'Sirius', 'Away': 'Hammarby', 'HTR': 'D',
        }])
        updated = before.drop(columns='HTR')
        result = _preserve_half_time_target(before, updated)
        self.assertEqual(result.loc[0, 'HTR'], 'D')

    def test_new_league_team_names_are_chinese_for_display_only(self):
        source = pd.DataFrame([{'Home': 'Sirius', 'Away': 'Norrkoping'}])
        shown = translate_fixture_columns(source, '瑞超')
        self.assertEqual(shown.loc[0, 'Home'], '天狼星')
        self.assertEqual(shown.loc[0, 'Away'], '北雪平')
        self.assertEqual(source.loc[0, 'Home'], 'Sirius')
        self.assertEqual(chinese_team_name('日职', 'Sagan Tosu'), '鸟栖砂岩')


if __name__ == '__main__':
    unittest.main()
