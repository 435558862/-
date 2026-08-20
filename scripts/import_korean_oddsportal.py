#!/usr/bin/env python3
"""Import CC0 OddsPortal K League history into the local prediction schema."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.database.league import LeagueDatabase
from src.preprocessing.statistics import StatisticsEngine


TEAM_ALIASES = {
    'Bucheon FC 1995': 'Bucheon',
    'Gimcheon Sangmu': 'Sangmu',
    'Jeju Utd': 'Jeju',
    'Suwon FC': 'Suwon City',
    'Ulsan Hyundai': 'Ulsan',
}
SOURCE_ARCHIVE = Path('storage/network/k_league_oddsportal_2014_2021.csv')
BACKUP = Path('storage/backups/before-korean-history-import-20260810/韩职-dataset.csv')


def build_history(leagues_path: Path, matches_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    leagues = pd.read_csv(leagues_path)
    selected = leagues[
        leagues['country'].eq('south-korea')
        & leagues['name'].str.contains(r'k-league-1|k-league-classic', case=False, regex=True)
    ]
    matches = pd.read_csv(matches_path)
    source = matches[matches['liga_id'].isin(selected['id'])].copy()
    source['Date'] = pd.to_datetime(source['timestamp'], unit='s', utc=True).dt.tz_convert(
        'Asia/Seoul',
    ).dt.strftime('%Y-%m-%d')
    source.rename(columns={
        'home': 'Home', 'away': 'Away', 'score_h': 'HG', 'score_a': 'AG',
        'm_o1': '1', 'm_oX': 'X', 'm_o2': '2',
    }, inplace=True)
    for column in ('Home', 'Away'):
        source[column] = source[column].replace(TEAM_ALIASES)
    source = source.dropna(subset=['Date', 'Home', 'Away', 'HG', 'AG', '1', 'X', '2'])
    source = source.drop_duplicates(['Date', 'Home', 'Away'], keep='last').sort_values('Date')
    raw = source[['Date', 'Home', 'Away', 'HG', 'AG', '1', 'X', '2']].copy()
    raw[['HG', 'AG']] = raw[['HG', 'AG']].astype(int)
    raw['Season'] = pd.to_datetime(raw['Date']).dt.year
    raw['Result'] = np.select(
        [raw['HG'] > raw['AG'], raw['HG'] < raw['AG']], ['H', 'A'], default='D',
    )
    raw['HTR'] = np.nan
    raw['Week'] = raw.groupby('Season').cumcount() + 1
    stats = StatisticsEngine.get_basic_stat_columns()
    featured = StatisticsEngine(match_history_window=4, goal_diff_margin=2).compute_stats(raw, stats)
    return source.reset_index(drop=True), featured.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--leagues', required=True, type=Path)
    parser.add_argument('--matches', required=True, type=Path)
    args = parser.parse_args()

    source, historical = build_history(args.leagues, args.matches)
    league_db = LeagueDatabase()
    current = league_db.load_league('韩职')
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2('storage/leagues/韩职/data/dataset.csv', BACKUP)
    SOURCE_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(SOURCE_ARCHIVE, index=False)

    combined = pd.concat([historical, current], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(['Date', 'Home', 'Away'], keep='last')
    combined = combined.sort_values(['Date', 'Home'], ascending=[False, True]).reset_index(drop=True)
    league_db.save_league(combined, league_db.index['韩职'])
    core_required = combined.drop(columns=['HTR'], errors='ignore').columns
    print(
        f'韩职合并完成：历史源={len(source)}，总场次={len(combined)}，'
        f'核心可训练={len(combined.dropna(subset=core_required))}，'
        f'半全场可训练={len(combined.dropna())}'
    )


if __name__ == '__main__':
    main()
