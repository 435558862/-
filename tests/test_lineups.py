from datetime import datetime

from src.services import lineups
from src.services.lineups import (
    _analyse, _match_fixture, _record, lineup_poll_interval_seconds, SHANGHAI,
)


def test_lineup_polling_accelerates_toward_kickoff():
    assert lineup_poll_interval_seconds(80) == 15 * 60
    assert lineup_poll_interval_seconds(40) == 5 * 60
    assert lineup_poll_interval_seconds(10) == 2 * 60


def test_fixture_match_requires_both_codes_and_close_kickoff():
    raw = {
        'matchDate': '2026-08-22', 'matchTime': '19:30:00',
        'homeTeamAbbEnName': 'HUL', 'awayTeamAbbEnName': 'MNU',
    }
    rows = [{
        'id': 1, 'date': '2026-08-22T19:30:00+08:00',
        'home_code': 'HUL', 'away_code': 'MNU',
    }]
    assert _match_fixture(raw, rows)['id'] == 1


def test_fixture_match_falls_back_to_team_names_when_api_codes_are_missing():
    raw = {
        'matchDate': '2026-08-22', 'matchTime': '18:00:00',
        'homeTeamAllName': '大阪樱花', 'homeTeamAbbName': '大阪樱花',
        'awayTeamAllName': '清水鼓动', 'awayTeamAbbName': '清水鼓动',
        'homeTeamAbbEnName': 'CEE', 'awayTeamAbbEnName': 'SHI',
    }
    rows = [{
        'id': 1556025, 'date': '2026-08-22T18:00:00+08:00',
        'home_code': None, 'away_code': None,
        'home_name': 'Cerezo Osaka', 'away_name': 'Shimizu S-pulse',
    }]
    assert _match_fixture(raw, rows)['id'] == 1556025


def test_free_plan_fetches_fixture_details_with_singular_id(monkeypatch):
    calls = []

    def request(endpoint, params):
        calls.append((endpoint, params))
        return [{'fixture': {'id': params['id']}}]

    monkeypatch.setattr(lineups, '_request', request)
    rows = lineups._details([10, 20, 10])
    assert [row['fixture']['id'] for row in rows] == [10, 20]
    assert calls == [
        ('fixtures', {'id': 10}),
        ('fixtures', {'id': 20}),
    ]


def _fixture():
    def lineup(team_id, name, formation, offset):
        return {
            'team': {'id': team_id, 'name': name}, 'formation': formation,
            'startXI': [{'player': {
                'id': offset + index, 'name': f'{name}{index}',
                'pos': 'G' if index == 0 else 'D' if index < 5 else 'M',
            }} for index in range(11)],
        }
    return {
        'fixture': {'id': 99},
        'teams': {'home': {'id': 1, 'name': 'Home'}, 'away': {'id': 2, 'name': 'Away'}},
        'lineups': [lineup(1, 'Home', '4-3-3', 100), lineup(2, 'Away', '4-4-2', 200)],
    }


def test_only_complete_confirmed_elevens_are_accepted():
    record = _record(_fixture(), 'sporttery-1')
    assert record['sporttery_id'] == 'sporttery-1'
    assert len(record['home']['starters']) == 11
    assert record['away']['formation'] == '4-4-2'


def test_first_confirmed_lineup_does_not_invent_probability_adjustment():
    analysis = _analyse(_record(_fixture(), '1'), [])
    assert analysis['status'] == '已确认'
    assert analysis['probability_shift'] == 0.0
    assert '已确认首发' in analysis['summary']
    assert analysis['home_penalty'] == 0
    assert analysis['away_goalkeeper_changed'] is False


def test_repoll_does_not_compare_lineup_with_itself():
    from copy import deepcopy
    current = _record(_fixture(), '1')
    current['captured_at'] = '2026-09-05T18:00:00+08:00'
    previous = deepcopy(current)
    previous['fixture_id'] = 98
    previous['captured_at'] = '2026-09-01T18:00:00+08:00'
    previous['home']['starters'][1]['id'] = 999
    first = _analyse(current, [previous])
    repeated = _analyse(current, [previous, deepcopy(current)])
    assert first['home_rotation'] == 1
    assert repeated == first


def test_later_lineup_is_not_used_as_historical_evidence():
    from copy import deepcopy
    current = _record(_fixture(), '1')
    current['captured_at'] = '2026-09-05T18:00:00+08:00'
    future = deepcopy(current)
    future['fixture_id'] = 100
    future['captured_at'] = '2026-09-06T18:00:00+08:00'
    assert _analyse(current, [future])['home_rotation'] is None
