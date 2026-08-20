import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.services import odds_tracking


def raw_match(match_id, h='1.58', d='3.55', a='4.65', update='20:02:46',
              line='-1', hh='2.93', hd='3.25', ha='2.08'):
    return {
        'matchId': match_id,
        'matchNumStr': '周三001',
        'leagueAllName': '欧洲超级杯',
        'homeTeamAllName': '巴黎圣日尔曼',
        'awayTeamAllName': '阿斯顿维拉',
        'had': {'h': h, 'd': d, 'a': a,
                'updateDate': '2026-08-12', 'updateTime': update},
        'hhad': {'goalLine': line, 'h': hh, 'd': hd, 'a': ha},
    }


class OddsTrackingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / 'odds_history.jsonl'

    def tearDown(self):
        self._tmp.cleanup()

    def test_records_compact_observations(self):
        appended = odds_tracking.record_odds_snapshots(
            [raw_match(2040831)], path=self.path, captured_at='t1',
        )
        self.assertEqual(appended, 1)
        rows = odds_tracking.read_odds_series(self.path)['2040831']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['had'], {'H': 1.58, 'D': 3.55, 'A': 4.65})
        self.assertEqual(rows[0]['hhad']['line'], -1.0)

    def test_skips_identical_latest_snapshot(self):
        odds_tracking.record_odds_snapshots(
            [raw_match(1)], path=self.path, captured_at='t1',
        )
        appended = odds_tracking.record_odds_snapshots(
            [raw_match(1)], path=self.path, captured_at='t2',
        )
        self.assertEqual(appended, 0)
        self.assertEqual(
            len(odds_tracking.read_odds_series(self.path)['1']), 1,
        )

    def test_appends_changed_odds_and_reports_drift(self):
        odds_tracking.record_odds_snapshots(
            [raw_match(1)], path=self.path, captured_at='t1',
        )
        appended = odds_tracking.record_odds_snapshots(
            [raw_match(1, h='1.50', d='3.60')], path=self.path, captured_at='t2',
        )
        self.assertEqual(appended, 1)
        summary = odds_tracking.drift_summary(1, path=self.path)
        self.assertEqual(summary['observations'], 2)
        self.assertEqual(summary['drift'], {'H': -0.08, 'D': 0.05, 'A': 0.0})

    def test_ignores_rows_without_odds(self):
        appended = odds_tracking.record_odds_snapshots(
            [{'matchId': 9, 'had': None}], path=self.path,
        )
        self.assertEqual(appended, 0)
        self.assertFalse(self.path.exists())

    def test_format_match_drift_labels(self):
        self.assertEqual(odds_tracking.format_match_drift(7, path=self.path), '')
        odds_tracking.record_odds_snapshots(
            [raw_match(7)], path=self.path, captured_at='t1',
        )
        self.assertEqual(
            odds_tracking.format_match_drift(7, path=self.path), '首次记录',
        )
        odds_tracking.record_odds_snapshots(
            [raw_match(7, h='1.50', d='3.70')], path=self.path, captured_at='t2',
        )
        self.assertEqual(
            odds_tracking.format_match_drift(7, path=self.path), '主↓0.08·平↑0.15',
        )

    def test_market_intent_labels(self):
        odds = {'H': 1.58, 'D': 3.55, 'A': 4.65}
        self.assertEqual(
            odds_tracking.market_intent_label(odds), '市场偏向主·暂无变动',
        )
        self.assertEqual(
            odds_tracking.market_intent_label(
                odds, {'H': -0.10, 'D': 0.05, 'A': 0.05},
            ),
            '市场偏向主·变动也挺热门',
        )
        self.assertEqual(
            odds_tracking.market_intent_label(
                odds, {'H': 0.10, 'D': -0.10, 'A': 0.0},
            ),
            '市场偏向主·变动向平，防冷',
        )
        balanced = {'H': 2.2, 'D': 3.3, 'A': 3.1}
        self.assertEqual(
            odds_tracking.market_intent_label(balanced, {'A': -0.15}),
            '市场三方格局·变动向客',
        )

    def test_snapshot_had_direction_uses_latest_odds(self):
        self.assertEqual(
            odds_tracking.snapshot_had_direction(7, path=self.path), '',
        )
        odds_tracking.record_odds_snapshots(
            [raw_match(7)], path=self.path, captured_at='t1',
        )
        self.assertEqual(
            odds_tracking.snapshot_had_direction(7, path=self.path),
            '主胜（56.0%）',
        )
        odds_tracking.record_odds_snapshots(
            [raw_match(7, h='3.10', d='3.20', a='2.05')],
            path=self.path, captured_at='t2',
        )
        self.assertEqual(
            odds_tracking.snapshot_had_direction(7, path=self.path),
            '客胜（43.4%）',
        )

    def test_odds_early_warning_flags_draw_and_direction_conflict(self):
        self.assertEqual(
            odds_tracking.odds_early_warning(7, '胜', path=self.path),
            '无快照',
        )
        odds_tracking.record_odds_snapshots(
            [raw_match(7)], path=self.path, captured_at='t1',
        )
        self.assertEqual(
            odds_tracking.odds_early_warning(7, '胜', path=self.path),
            '主胜 56%',
        )
        self.assertEqual(
            odds_tracking.odds_early_warning(7, '负', path=self.path),
            '主胜 56%｜盘口主胜，与模型负相反',
        )
        # Draw-heavy market: home 2.50 / draw 2.90 / away 2.70.
        odds_tracking.record_odds_snapshots(
            [raw_match(8, h='2.50', d='2.90', a='2.70')],
            path=self.path, captured_at='t1',
        )
        self.assertEqual(
            odds_tracking.odds_early_warning(8, '胜', path=self.path),
            '主胜 36%｜防平（盘口平31%）',
        )
        # Draw odds shortening across snapshots.
        odds_tracking.record_odds_snapshots(
            [raw_match(7, d='3.20')], path=self.path, captured_at='t2',
        )
        self.assertEqual(
            odds_tracking.odds_early_warning(7, '胜', path=self.path),
            '主胜 55%｜平赔走低0.35，防平',
        )


if __name__ == '__main__':
    unittest.main()
