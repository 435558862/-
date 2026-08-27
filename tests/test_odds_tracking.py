import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.services import odds_tracking


def raw_match(match_id, h='1.58', d='3.55', a='4.65', update='20:02:46',
              line='-1', hh='2.93', hd='3.25', ha='2.08'):
    return {
        'matchId': match_id,
        'matchDate': '2026-08-13',
        'matchTime': '20:00:00',
        'matchNumStr': '周三001',
        'leagueAllName': '欧洲超级杯',
        'homeTeamAllName': '巴黎圣日尔曼',
        'awayTeamAllName': '阿斯顿维拉',
        'had': {'h': h, 'd': d, 'a': a,
                'updateDate': '2026-08-12', 'updateTime': update},
        'hhad': {'goalLine': line, 'h': hh, 'd': hd, 'a': ha},
    }


class OddsTrackingTests(unittest.TestCase):

    def test_official_history_records_full_had_and_handicap_opening(self):
        history = {
            'leagueAllName': '测试联赛', 'homeTeamAllName': '主队',
            'awayTeamAllName': '客队',
            'hadList': [
                {'updateDate': '2026-08-27', 'updateTime': '09:21:43',
                 'h': '1.87', 'd': '3.20', 'a': '3.55'},
                {'updateDate': '2026-08-27', 'updateTime': '11:07:17',
                 'h': '1.83', 'd': '3.25', 'a': '3.65'},
            ],
            'hhadList': [
                {'updateDate': '2026-08-27', 'updateTime': '09:21:43',
                 'goalLine': '-1', 'h': '3.85', 'd': '3.43', 'a': '1.73'},
                {'updateDate': '2026-08-27', 'updateTime': '11:07:59',
                 'goalLine': '-1', 'h': '3.76', 'd': '3.35', 'a': '1.77'},
            ],
        }
        rows = odds_tracking.official_history_snapshots('2041078', history)
        self.assertEqual(rows[0]['market_update'], '2026-08-27 09:21:43')
        self.assertEqual(rows[0]['had'], {'H': 1.87, 'D': 3.2, 'A': 3.55})
        self.assertEqual(rows[0]['hhad'], {
            'line': -1.0, 'H': 3.85, 'D': 3.43, 'A': 1.73,
        })
        self.assertEqual(
            odds_tracking.record_official_history(
                '2041078', history, path=self.path,
            ),
            3,
        )
        stored = odds_tracking.read_odds_series(self.path)['2041078']
        self.assertEqual(stored[0]['hhad']['H'], 3.85)
        self.assertEqual(
            odds_tracking.record_official_history(
                '2041078', history, path=self.path,
            ),
            0,
        )

    def test_market_flow_gate_agrees_and_conflicts(self):
        series = {'1': [
            {'had': {'H': 2.00, 'D': 3.20, 'A': 3.60}},
            {'had': {'H': 1.75, 'D': 3.45, 'A': 4.20}},
        ]}
        self.assertEqual(
            odds_tracking.market_flow_gate('1', '胜', series=series)['state'],
            'agree',
        )
        self.assertEqual(
            odds_tracking.market_flow_gate('1', '负', series=series)['state'],
            'conflict',
        )

    def test_market_flow_gate_reports_time_normalized_speed(self):
        series = {'1': [
            {'captured_at': '2026-08-13T10:00:00+08:00',
             'had': {'H': 2.00, 'D': 3.20, 'A': 3.60}},
            {'captured_at': '2026-08-13T12:00:00+08:00',
             'had': {'H': 1.75, 'D': 3.45, 'A': 4.20}},
        ]}
        gate = odds_tracking.market_flow_gate('1', '胜', series=series)
        self.assertGreater(gate['speed_per_hour'], 0)
        self.assertEqual(gate['observations'], 2)

    def test_market_quality_exposes_return_and_secondary_market_changes(self):
        series = {'1': [
            {'had': {'H': 2.0, 'D': 3.2, 'A': 3.6},
             'hhad': {'line': -1.0},
             'ttg': {f's{i}': str(8.0 - i * 0.5) for i in range(8)}},
            {'had': {'H': 1.9, 'D': 3.3, 'A': 3.8},
             'hhad': {'line': -2.0},
             'ttg': {f's{i}': str(8.5 - i * 0.5) for i in range(8)}},
        ]}
        metrics = odds_tracking.market_quality_metrics('1', series=series)
        self.assertGreater(metrics['return_rate'], 0.8)
        self.assertEqual(metrics['hhad_line_change'], -1.0)
        self.assertIsNotNone(metrics['ttg_expected_change'])
        self.assertFalse(metrics['multi_company_available'])
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

    def test_snapshot_records_time_to_kickoff_without_future_information(self):
        odds_tracking.record_odds_snapshots(
            [raw_match(3)], path=self.path,
            captured_at='2026-08-12T12:00:00+08:00',
        )
        row = odds_tracking.read_odds_series(self.path)['3'][0]
        self.assertEqual(row['hours_to_kickoff'], 32.0)
        self.assertEqual(row['snapshot_window'], '早盘档')

    def test_same_odds_are_kept_once_when_entering_a_new_time_window(self):
        odds_tracking.record_odds_snapshots(
            [raw_match(4)], path=self.path,
            captured_at='2026-08-12T22:00:00+08:00',
        )
        odds_tracking.record_odds_snapshots(
            [raw_match(4)], path=self.path,
            captured_at='2026-08-13T14:00:00+08:00',
        )
        rows = odds_tracking.read_odds_series(self.path)['4']
        self.assertEqual([row['snapshot_window'] for row in rows], [
            '赛前24小时档', '赛前6小时档',
        ])

    def test_unchanged_file_reuses_parsed_series_and_append_invalidates_it(self):
        odds_tracking.record_odds_snapshots(
            [raw_match(1)], path=self.path, captured_at='t1',
        )
        first = odds_tracking.read_odds_series(self.path)
        self.assertIs(first, odds_tracking.read_odds_series(self.path))
        odds_tracking.record_odds_snapshots(
            [raw_match(1, h='1.49')], path=self.path, captured_at='t2',
        )
        second = odds_tracking.read_odds_series(self.path)
        self.assertIsNot(first, second)
        self.assertEqual(len(second['1']), 2)

    def test_bounded_chart_read_keeps_opening_and_latest_rows(self):
        for index in range(6):
            odds_tracking.record_odds_snapshots(
                [raw_match(1, h=str(1.80 - index / 100))],
                path=self.path, captured_at=f't{index}',
            )
        rows = odds_tracking.read_odds_series(
            self.path, max_rows_per_match=3, keep_opening=True,
        )['1']
        self.assertEqual([row['captured_at'] for row in rows], ['t0', 't4', 't5'])
        self.assertEqual(len(odds_tracking.read_odds_series(self.path)['1']), 6)

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

    def test_market_flow_reports_waiting_and_direction(self):
        odds_tracking.record_odds_snapshots(
            [raw_match(20)], path=self.path, captured_at='t1',
        )
        self.assertEqual(odds_tracking.format_market_flow(20, path=self.path), '待积累')
        odds_tracking.record_odds_snapshots(
            [raw_match(20, h='1.40', d='3.90', a='5.20')],
            path=self.path, captured_at='t2',
        )
        label = odds_tracking.format_market_flow(20, path=self.path)
        self.assertEqual(label, '购买方向：胜')

    def test_market_flow_exposes_small_direction_without_weakening_gate(self):
        series = {'1': [
            {'captured_at': '2026-01-01T00:00:00+00:00',
             'had': {'H': 2.00, 'D': 3.20, 'A': 3.60}},
            {'captured_at': '2026-01-01T01:00:00+00:00',
             'had': {'H': 1.99, 'D': 3.20, 'A': 3.60}},
        ]}
        label = odds_tracking.format_market_flow('1', series=series)
        self.assertEqual(label, '暂无明确购买方向')
        self.assertEqual(
            odds_tracking.market_flow_gate('1', '胜', series=series)['state'],
            'stable',
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
