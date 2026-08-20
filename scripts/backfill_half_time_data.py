#!/usr/bin/env python3
"""Backfill real half-time scores for leagues whose odds CSV omits them.

ESPN's public scoreboard contains goal events.  A half-time score is accepted
only when every scoring event reconciles exactly with the stored full-time
score, and the dated fixture can be matched unambiguously.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import unicodedata
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


LEAGUES = {
    '瑞超': 'swe.1',
    '日职': 'jpn.1',
}
NETWORK_DIR = Path('storage/network/half_time')
BACKUP_DIR = Path('storage/backups/before-half-time-backfill-20260810')


def _normalise_team(value: str) -> str:
    text = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]', '', text)


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalise_team(left), _normalise_team(right)).ratio()


def _download_season(slug: str, year: int) -> dict:
    url = (
        f'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard'
        f'?limit=1000&dates={year}'
    )
    result = subprocess.run(
        ['curl', '-L', '--fail', '--silent', '--show-error', url],
        check=True, capture_output=True, timeout=60,
    )
    return json.loads(result.stdout)


def _half_time_fixture(event: dict) -> dict | None:
    competition = (event.get('competitions') or [{}])[0]
    competitors = competition.get('competitors') or []
    if len(competitors) != 2:
        return None
    sides = {item.get('homeAway'): item for item in competitors}
    if 'home' not in sides or 'away' not in sides:
        return None
    try:
        full_home = int(sides['home']['score'])
        full_away = int(sides['away']['score'])
    except (KeyError, TypeError, ValueError):
        return None

    home_id = str(sides['home']['id'])
    away_id = str(sides['away']['id'])
    total_home = total_away = half_home = half_away = 0
    for detail in competition.get('details') or []:
        if not detail.get('scoringPlay') or detail.get('shootout'):
            continue
        value = int(detail.get('scoreValue') or 1)
        team_id = str((detail.get('team') or {}).get('id', ''))
        minute = float((detail.get('clock') or {}).get('value', 999999)) / 60.0
        if team_id == home_id:
            total_home += value
            if minute <= 45:
                half_home += value
        elif team_id == away_id:
            total_away += value
            if minute <= 45:
                half_away += value

    # Reject incomplete play-by-play instead of inventing a half-time result.
    if (total_home, total_away) != (full_home, full_away):
        return None
    return {
        'Date': pd.to_datetime(event['date'], utc=True).tz_convert('Asia/Shanghai').date().isoformat(),
        'Home': sides['home']['team']['displayName'],
        'Away': sides['away']['team']['displayName'],
        'HG': full_home,
        'AG': full_away,
        'HTHG': half_home,
        'HTAG': half_away,
    }


def _collect(league: str, slug: str, start_year: int, end_year: int) -> pd.DataFrame:
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    fixtures: list[dict] = []
    for year in range(start_year, end_year + 1):
        payload = _download_season(slug, year)
        cache = NETWORK_DIR / f'{league}-{year}.json'
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        fixtures.extend(filter(None, (_half_time_fixture(event) for event in payload.get('events', []))))
        print(f'{league} {year}: 接口比赛={len(payload.get("events", []))}，可校验半场={sum(1 for x in fixtures if x["Date"].startswith(str(year)))}')
    return pd.DataFrame(fixtures).drop_duplicates(['Date', 'Home', 'Away'])


def _match(dataset: pd.DataFrame, source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in source.to_dict('records'):
        buckets[(row['Date'], int(row['HG']), int(row['AG']))].append(row)

    output = dataset.copy()
    output['HTHG'] = pd.NA
    output['HTAG'] = pd.NA
    audit = []
    used: set[tuple] = set()
    for idx, row in output.iterrows():
        key = (str(row['Date'])[:10], int(row['HG']), int(row['AG']))
        ranked = []
        for candidate in buckets.get(key, []):
            identity = (candidate['Date'], candidate['Home'], candidate['Away'])
            if identity in used:
                continue
            score = (_similarity(row['Home'], candidate['Home']) + _similarity(row['Away'], candidate['Away'])) / 2
            ranked.append((score, identity, candidate))
        ranked.sort(reverse=True, key=lambda item: item[0])
        if not ranked or ranked[0][0] < 0.45:
            continue
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.08:
            continue
        score, identity, candidate = ranked[0]
        used.add(identity)
        output.at[idx, 'HTHG'] = int(candidate['HTHG'])
        output.at[idx, 'HTAG'] = int(candidate['HTAG'])
        audit.append({
            'Date': row['Date'], 'Home': row['Home'], 'Away': row['Away'],
            'SourceHome': candidate['Home'], 'SourceAway': candidate['Away'],
            'HG': row['HG'], 'AG': row['AG'], 'HTHG': candidate['HTHG'],
            'HTAG': candidate['HTAG'], 'MatchScore': score,
        })
    valid = output['HTHG'].notna() & output['HTAG'].notna()
    output.loc[valid, 'HTR'] = output.loc[valid].apply(
        lambda row: 'H' if row['HTHG'] > row['HTAG'] else ('A' if row['HTHG'] < row['HTAG'] else 'D'),
        axis=1,
    )
    return output, pd.DataFrame(audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--league', action='append', choices=LEAGUES, help='默认处理瑞超和日职')
    parser.add_argument('--apply', action='store_true', help='校验后写入正式数据集')
    args = parser.parse_args()
    selected = args.league or list(LEAGUES)
    for league in selected:
        path = Path('storage/leagues') / league / 'data/dataset.csv'
        dataset = pd.read_csv(path)
        source = _collect(league, LEAGUES[league], int(dataset['Season'].min()), datetime.now().year)
        merged, audit = _match(dataset, source)
        coverage = len(audit) / len(dataset) if len(dataset) else 0
        audit_path = NETWORK_DIR / f'{league}-半场比分匹配审计.csv'
        audit.to_csv(audit_path, index=False)
        print(f'{league}: 匹配={len(audit)}/{len(dataset)} ({coverage:.1%})，审计={audit_path}')
        if args.apply:
            if len(audit) < 100:
                raise RuntimeError(f'{league}真实半场匹配不足100场，拒绝写入。')
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, BACKUP_DIR / f'{league}-dataset.csv')
            # HTHG/HTAG are useful for auditing but must never become pre-match
            # model features. Persist only the categorical training target HTR.
            merged.drop(columns=['HTHG', 'HTAG']).to_csv(path, index=False)
            print(f'{league}: 已写入 {path}')


if __name__ == '__main__':
    main()
