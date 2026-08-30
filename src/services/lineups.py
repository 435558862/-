"""Confirmed pre-match lineups with conservative, history-backed analysis."""

import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from src.services.team_names import _chinese_to_english, _normalize_chinese_team

API_ROOT = 'https://v3.football.api-sports.io'
KEY_PATH = Path('storage/network/.api_football_key')
ROOT = Path('storage/jingcai/lineups')
HISTORY_PATH = ROOT / 'confirmed_history.jsonl'
SHANGHAI = ZoneInfo('Asia/Shanghai')


def lineup_poll_interval_seconds(minutes_to_kickoff):
    """Return a quota-conscious adaptive interval for confirmed lineups."""
    try:
        minutes = float(minutes_to_kickoff)
    except (TypeError, ValueError):
        return 15 * 60
    if minutes <= 15:
        return 2 * 60
    if minutes <= 45:
        return 5 * 60
    return 15 * 60


def lineup_api_configured():
    """Expose configuration state without ever revealing the secret key."""
    return bool(_key())

# Official Sporttery names which are not present in the trained-league alias
# catalog (notably cup/second-tier fixtures), mapped to API-Football names.
LINEUP_NAME_OVERRIDES = {
    '磐田喜悦': 'Jubilo Iwata',
    '德岛漩涡': 'Tokushima Vortis',
    '全北现代': 'Jeonbuk Motors',
    '蔚山现代': 'Ulsan Hyundai FC',
    '赫尔城': 'Hull City',
    '曼彻斯特联': 'Manchester United',
    '米尔沃尔': 'Millwall',
    '诺维奇': 'Norwich',
}


def _key():
    configured = os.environ.get('API_FOOTBALL_KEY', '').strip()
    if configured:
        return configured
    try:
        return KEY_PATH.read_text(encoding='utf-8').strip()
    except OSError:
        return ''


def _request(endpoint, params):
    if not _key():
        return []
    response = requests.get(
        f'{API_ROOT}/{endpoint}', params=params,
        headers={'x-apisports-key': _key()}, timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get('errors'):
        raise RuntimeError(f'API-Football错误：{payload["errors"]}')
    return list(payload.get('response') or [])


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def _catalog(day):
    path = ROOT / f'catalog-{day}.json'
    try:
        age = datetime.now(SHANGHAI) - datetime.fromtimestamp(path.stat().st_mtime, SHANGHAI)
        if age <= timedelta(hours=6):
            cached = json.loads(path.read_text(encoding='utf-8'))
            if not cached or 'home_name' in cached[0]:
                return cached
    except (OSError, ValueError, TypeError):
        pass
    rows = _request('fixtures', {'date': day, 'timezone': 'Asia/Shanghai'})
    compact = [{
        'id': row.get('fixture', {}).get('id'),
        'date': row.get('fixture', {}).get('date'),
        'home_code': row.get('teams', {}).get('home', {}).get('code'),
        'away_code': row.get('teams', {}).get('away', {}).get('code'),
        'home_name': row.get('teams', {}).get('home', {}).get('name'),
        'away_name': row.get('teams', {}).get('away', {}).get('name'),
    } for row in rows]
    _write(path, compact)
    return compact


def _kickoff(raw):
    try:
        return datetime.fromisoformat(
            f'{str(raw.get("matchDate"))[:10]}T{str(raw.get("matchTime"))[:8]}'
        ).replace(tzinfo=SHANGHAI)
    except (TypeError, ValueError):
        return None


def _match_fixture(raw, rows):
    kickoff = _kickoff(raw)
    home = str(raw.get('homeTeamAbbEnName') or '').upper().strip()
    away = str(raw.get('awayTeamAbbEnName') or '').upper().strip()
    aliases = _chinese_to_english()

    def model_name(side):
        candidates = (
            raw.get(f'{side}TeamAllName'), raw.get(f'{side}TeamAbbName'),
        )
        for candidate in candidates:
            override = LINEUP_NAME_OVERRIDES.get(str(candidate or '').strip())
            if override:
                return override
        normalized = {_normalize_chinese_team(value) for value in candidates if value}
        for mapping in aliases.values():
            for chinese, english in mapping.items():
                if _normalize_chinese_team(chinese) in normalized:
                    return str(english)
        return ''

    def normalize_english(value):
        value = unicodedata.normalize('NFKD', str(value or ''))
        value = value.encode('ascii', 'ignore').decode('ascii').lower()
        return re.sub(r'[^a-z0-9]+', '', value)

    expected_home = normalize_english(model_name('home'))
    expected_away = normalize_english(model_name('away'))
    candidates = []
    if kickoff is None:
        return None
    for row in rows:
        try:
            api_time = datetime.fromisoformat(str(row['date'])).astimezone(SHANGHAI)
        except (TypeError, ValueError):
            continue
        difference = abs((api_time - kickoff).total_seconds())
        if difference > 7200:
            continue
        code_match = (
            home and away
            and str(row.get('home_code') or '').upper() == home
            and str(row.get('away_code') or '').upper() == away
        )
        api_home = normalize_english(row.get('home_name'))
        api_away = normalize_english(row.get('away_name'))
        home_score = SequenceMatcher(None, expected_home, api_home).ratio() if expected_home else 0
        away_score = SequenceMatcher(None, expected_away, api_away).ratio() if expected_away else 0
        name_match = min(home_score, away_score) >= 0.62 and (home_score + away_score) / 2 >= 0.72
        if code_match or name_match:
            score = 2.0 if code_match else home_score + away_score
            candidates.append((-score, difference, row))
    return min(candidates, default=(0, 0, None), key=lambda item: (item[0], item[1]))[2]


def _details(ids):
    result = []
    # API-Football's free plan rejects the batch ``ids`` parameter.  Fetch only
    # the small set of fixtures inside the lineup window one by one using the
    # supported singular ``id`` parameter.
    for fixture_id in dict.fromkeys(ids):
        result.extend(_request('fixtures', {'id': int(fixture_id)}))
    return result


def _record(fixture, sporttery_id):
    lineups = fixture.get('lineups') or []
    teams = fixture.get('teams') or {}
    if len(lineups) != 2:
        return None
    by_id = {str(row.get('team', {}).get('id')): row for row in lineups}

    def side(name):
        team = teams.get(name) or {}
        row = by_id.get(str(team.get('id')))
        starters = (row or {}).get('startXI') or []
        if len(starters) != 11:
            return None
        return {
            'team_id': team.get('id'), 'team_name': team.get('name') or '',
            'formation': row.get('formation') or '',
            'starters': [{
                'id': item.get('player', {}).get('id'),
                'name': item.get('player', {}).get('name') or '',
                'position': item.get('player', {}).get('pos') or '',
            } for item in starters],
        }

    home, away = side('home'), side('away')
    if home is None or away is None:
        return None
    return {
        'fixture_id': fixture.get('fixture', {}).get('id'),
        'sporttery_id': str(sporttery_id),
        'captured_at': datetime.now(SHANGHAI).isoformat(timespec='seconds'),
        'home': home, 'away': away,
    }


def _history():
    try:
        rows = []
        with HISTORY_PATH.open('r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows
    except OSError:
        return []


def _append(records, previous):
    known = {str(row.get('fixture_id')) for row in previous}
    fresh = [row for row in records if str(row.get('fixture_id')) not in known]
    if not fresh:
        return
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open('a', encoding='utf-8') as handle:
        for row in fresh:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _team(side, history):
    previous = [match[key] for match in history for key in ('home', 'away')
                if (match.get(key) or {}).get('team_id') == side['team_id']]
    current = {row['id'] for row in side['starters'] if row.get('id')}
    result = {'rotation': None, 'missing_core': 0, 'goalkeeper_changed': False}
    if previous:
        latest = {row['id'] for row in previous[-1]['starters'] if row.get('id')}
        result['rotation'] = 11 - len(current & latest)
    if len(previous) >= 3:
        recent = previous[-8:]
        counts = Counter(row['id'] for match in recent for row in match['starters'] if row.get('id'))
        core = {player for player, count in counts.items()
                if count >= math.ceil(len(recent) * 0.60)}
        result['missing_core'] = len(core - current)
        keepers = [row['id'] for match in previous[-3:] for row in match['starters']
                   if row.get('position') == 'G' and row.get('id')]
        current_keepers = {row['id'] for row in side['starters']
                           if row.get('position') == 'G' and row.get('id')}
        if keepers:
            result['goalkeeper_changed'] = Counter(keepers).most_common(1)[0][0] not in current_keepers
    return result


def _analyse(record, history):
    home, away = _team(record['home'], history), _team(record['away'], history)
    def penalty(value):
        result = value['missing_core'] + (1.5 if value['goalkeeper_changed'] else 0)
        if value['rotation'] is not None:
            result += max(0, value['rotation'] - 3) * 0.4
        return result
    shift = max(-0.04, min(0.04, (penalty(away) - penalty(home)) * 0.008))
    if len(history) < 3:
        shift = 0.0
    parts = [f'已确认首发 {record["home"]["formation"] or "未知"}/{record["away"]["formation"] or "未知"}']
    if home['rotation'] is not None and away['rotation'] is not None:
        parts.append(f'轮换 主{home["rotation"]}/客{away["rotation"]}')
    if home['missing_core'] or away['missing_core']:
        parts.append(f'核心缺阵 主{home["missing_core"]}/客{away["missing_core"]}')
    warnings = []
    for label, value in (('主队', home), ('客队', away)):
        if value['goalkeeper_changed']:
            warnings.append(f'{label}门将变化')
        if value['missing_core']:
            warnings.append(f'{label}核心缺阵{value["missing_core"]}人')
        if value['rotation'] is not None and value['rotation'] >= 5:
            warnings.append(f'{label}异常轮换{value["rotation"]}人')
    return {
        'status': '已确认', 'summary': '｜'.join(parts),
        'home_formation': record['home']['formation'],
        'away_formation': record['away']['formation'],
        'home_starting': '、'.join(row['name'] for row in record['home']['starters']),
        'away_starting': '、'.join(row['name'] for row in record['away']['starters']),
        'home_rotation': home['rotation'], 'away_rotation': away['rotation'],
        'home_missing_core': home['missing_core'], 'away_missing_core': away['missing_core'],
        'home_goalkeeper_changed': home['goalkeeper_changed'],
        'away_goalkeeper_changed': away['goalkeeper_changed'],
        'home_penalty': penalty(home), 'away_penalty': penalty(away),
        'probability_shift': shift,
        'warning_level': '高' if any(
            value['goalkeeper_changed'] or value['missing_core'] >= 2
            or (value['rotation'] is not None and value['rotation'] >= 6)
            for value in (home, away)
        ) else ('中' if warnings else '无'),
        'warnings': warnings,
    }


def fetch_lineup_analysis(matches):
    """Fetch only confirmed lineups for matches starting within 90 minutes."""
    if not _key():
        return {}
    now, matched, catalogs, unmatched = datetime.now(SHANGHAI), [], {}, {}
    try:
        for raw in matches:
            kickoff = _kickoff(raw)
            until_kickoff = kickoff - now if kickoff is not None else None
            if until_kickoff is None or not (
                    -timedelta(hours=3) <= until_kickoff <= timedelta(minutes=90)):
                continue
            day = kickoff.date().isoformat()
            catalogs.setdefault(day, _catalog(day))
            fixture = _match_fixture(raw, catalogs[day])
            if fixture:
                matched.append((raw, fixture))
            else:
                unmatched[str(raw.get('matchId') or '')] = {
                    'status': '未匹配',
                    'summary': '阵容源未匹配到对应官方场次',
                }
        if not matched:
            return unmatched
        detail = _details([int(fixture['id']) for _, fixture in matched])
        by_id = {int(row.get('fixture', {}).get('id')): row for row in detail}
        records, pending = [], {}
        for raw, fixture in matched:
            match_id = str(raw.get('matchId') or '')
            record = _record(by_id.get(int(fixture['id']), {}), match_id)
            if record is None:
                pending[match_id] = {'status': '待公布', 'summary': '首发尚未公布'}
            else:
                records.append(record)
        history = _history()
        result = dict(unmatched)
        result.update({row['sporttery_id']: _analyse(row, history) for row in records})
        result.update(pending)
        _append(records, history)
        return result
    except Exception:
        logging.exception('首发阵容同步失败，本次不调整预测。')
        return {}
