from src.services.market_trends import (
    build_trend_rows, implied_probabilities, live_snapshot_from_match,
    summarize_trend,
)


def _snapshot(stamp, h, d, a, line=-1, over_bias=False):
    ttg = {f's{i}': 8.0 for i in range(8)}
    if over_bias:
        for i in range(3, 8):
            ttg[f's{i}'] = 4.0
    return {
        'captured_at': stamp,
        'had_update': stamp,
        'had': {'H': h, 'D': d, 'A': a},
        'hhad': {'line': line},
        'ttg': ttg,
    }


def test_implied_probabilities_remove_bookmaker_margin():
    result = implied_probabilities({'H': 2.0, 'D': 3.2, 'A': 3.8})
    assert result is not None
    assert abs(sum(result.values()) - 1.0) < 1e-9
    assert max(result, key=result.get) == 'H'


def test_time_filter_keeps_preceding_point_for_a_visible_direction():
    series = {'7': [
        _snapshot('2026-08-25T00:00:00+00:00', 2.0, 3.2, 3.8),
        _snapshot('2026-08-25T20:00:00+00:00', 1.9, 3.3, 4.0),
    ]}
    rows = build_trend_rows('7', series, hours=6)
    assert len(rows) == 2
    assert rows[0]['is_opening'] is True
    assert rows[0]['label'].startswith('初盘（首次记录）')


def test_time_filter_always_keeps_opening_when_multiple_late_points_exist():
    series = {'7': [
        _snapshot('2026-08-24T00:00:00+00:00', 2.1, 3.1, 3.7),
        _snapshot('2026-08-25T18:00:00+00:00', 2.0, 3.2, 3.8),
        _snapshot('2026-08-25T20:00:00+00:00', 1.9, 3.3, 4.0),
    ]}
    rows = build_trend_rows('7', series, hours=6)
    assert len(rows) == 3
    assert rows[0]['captured_at'] == '2026-08-24T00:00:00+00:00'
    assert rows[0]['is_opening'] is True
    assert all(row['is_opening'] is False for row in rows[1:])


def test_summary_reports_direction_handicap_total_and_stability():
    rows = build_trend_rows('7', {'7': [
        _snapshot('2026-08-25T10:00:00+00:00', 2.0, 3.2, 3.8, -1),
        _snapshot('2026-08-25T12:00:00+00:00', 1.7, 3.5, 4.4, -2, True),
    ]})
    summary = summarize_trend(rows)
    assert summary['direction'] == '胜'
    assert summary['flow'] == '胜'
    assert summary['stability'] == '稳定'
    assert summary['handicap'] == '让球 -1→-2'
    assert summary['total_goals'] == '大球升温'
    assert summary['observations'] == 2


def test_summary_marks_repeated_favorite_switches_as_unstable():
    rows = build_trend_rows('7', {'7': [
        _snapshot('2026-08-25T10:00:00+00:00', 1.8, 3.2, 4.2),
        _snapshot('2026-08-25T11:00:00+00:00', 4.2, 3.2, 1.8),
        _snapshot('2026-08-25T12:00:00+00:00', 1.8, 3.2, 4.2),
    ]})
    summary = summarize_trend(rows)
    assert summary['stability'] == '反复'
    assert '等待临场确认' in summary['conclusion']


def test_empty_trend_is_explicit():
    assert summarize_trend([])['conclusion'] == '暂无可用赔率快照'


def test_live_official_row_becomes_a_visible_chart_point():
    raw = {
        'matchId': 2041001,
        'had': {'h': '1.80', 'd': '3.20', 'a': '4.20'},
        'hhad': {'goalLine': '0'},
        'ttg': {f's{i}': str(8.0 - i * 0.5) for i in range(8)},
    }
    snapshot = live_snapshot_from_match(
        raw, '2026-08-25T15:00:00+00:00',
    )
    assert snapshot is not None
    assert snapshot['had'] == {'H': 1.8, 'D': 3.2, 'A': 4.2}
    assert snapshot['hhad']['line'] == 0.0
    rows = build_trend_rows('2041001', {'2041001': [snapshot]})
    assert len(rows) == 1
    assert abs(sum(rows[0][key] for key in ('H', 'D', 'A')) - 1.0) < 1e-9


def test_chart_row_exposes_full_handicap_prices():
    snapshot = _snapshot(
        '2026-08-25T15:00:00+00:00', 1.8, 3.2, 4.2, line=-1,
    )
    snapshot['hhad'].update({'H': 3.85, 'D': 3.43, 'A': 1.73})
    row = build_trend_rows('7', {'7': [snapshot]})[0]
    assert (row['had_H'], row['had_D'], row['had_A']) == (1.8, 3.2, 4.2)
    assert row['hhad_line'] == -1
    assert (row['hhad_H'], row['hhad_D'], row['hhad_A']) == (3.85, 3.43, 1.73)
