import json
import logging
import os
import re
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.preprocessing.statistics import StatisticsEngine
from src.preprocessing.utils.normalization import NormalizerType
from src.preprocessing.utils.target import TargetType


BIG_FIVE = ('英超', '西甲', '德甲', '意甲', '法甲')
SYNC_LEAGUES = (*BIG_FIVE, '瑞超', '葡超', '日职', '韩职')
SYNC_STATE_PATH = Path('storage/network/sync_state.json')
KOREA_SOURCE_PATH = Path('storage/network/k_league_sgodds.csv')
KOREA_HISTORY_PATH = Path('storage/network/k_league_oddsportal_2014_2021.csv')
KOREA_DATA_PAGE = 'https://sgodds.com/football/data'
# K-League source coverage is sparse at the start of a season.  A four-match
# home/away window discarded almost every 2025-2026 row; validation showed that
# a one-match window retains the recent fixtures without degrading held-out
# result/score performance and improves totals performance.
KOREA_MATCH_HISTORY_WINDOW = 1


def _korean_feature_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert SG Odds rows to the same feature schema used by other leagues."""
    completed = raw['Result'].astype(str).str.extract(r'FT:(\d+)-(\d+)')
    half_time = raw['Result'].astype(str).str.extract(r'HT:(\d+)-(\d+)')
    teams = raw['Match'].astype(str).str.split(' vs ', n=1, expand=True)
    valid = completed.notna().all(axis=1) & teams.notna().all(axis=1)
    frame = pd.DataFrame({
        'Date': pd.to_datetime(raw.loc[valid, 'Start Time']).dt.strftime('%Y-%m-%d'),
        'Season': pd.to_datetime(raw.loc[valid, 'Start Time']).dt.year,
        'Home': teams.loc[valid, 0].str.strip(),
        'Away': teams.loc[valid, 1].str.strip(),
        'HG': completed.loc[valid, 0].astype(int),
        'AG': completed.loc[valid, 1].astype(int),
        '1': pd.to_numeric(raw.loc[valid, 'Ft1X2_01']),
        'X': pd.to_numeric(raw.loc[valid, 'Ft1X2_02']),
        '2': pd.to_numeric(raw.loc[valid, 'Ft1X2_03']),
    })
    frame['Result'] = np.select(
        [frame['HG'] > frame['AG'], frame['HG'] < frame['AG']], ['H', 'A'], default='D',
    )
    half_home = pd.to_numeric(half_time.loc[valid, 0], errors='coerce')
    half_away = pd.to_numeric(half_time.loc[valid, 1], errors='coerce')
    frame['HTR'] = np.select(
        [half_home > half_away, half_home < half_away], ['H', 'A'], default='D',
    )
    frame = frame.dropna().drop_duplicates(
        subset=['Date', 'Home', 'Away'], keep='last',
    ).sort_values(['Date', 'Home']).reset_index(drop=True)
    frame['Week'] = frame.groupby('Season').cumcount() + 1
    stats = StatisticsEngine.get_basic_stat_columns()
    return StatisticsEngine(
        match_history_window=KOREA_MATCH_HISTORY_WINDOW, goal_diff_margin=2,
    ).compute_stats(
        frame, stats,
    )


def _korean_history_feature_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the archived OddsPortal history without inventing half-time results."""
    frame = raw[['Date', 'Home', 'Away', 'HG', 'AG', '1', 'X', '2']].copy()
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    for column in ('HG', 'AG', '1', 'X', '2'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['Date', 'Home', 'Away', 'HG', 'AG', '1', 'X', '2'])
    frame[['HG', 'AG']] = frame[['HG', 'AG']].astype(int)
    frame['Season'] = pd.to_datetime(frame['Date']).dt.year
    frame['Result'] = np.select(
        [frame['HG'] > frame['AG'], frame['HG'] < frame['AG']], ['H', 'A'], default='D',
    )
    frame['HTR'] = np.nan
    frame = frame.drop_duplicates(['Date', 'Home', 'Away'], keep='last').sort_values('Date')
    frame['Week'] = frame.groupby('Season').cumcount() + 1
    stats = StatisticsEngine.get_basic_stat_columns()
    return StatisticsEngine(
        match_history_window=KOREA_MATCH_HISTORY_WINDOW, goal_diff_margin=2,
    ).compute_stats(
        frame.reset_index(drop=True), stats,
    )


def _download_korean_rows() -> pd.DataFrame:
    """Resolve the current K-League CSV link and download it."""
    request = Request(KOREA_DATA_PAGE, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(request, timeout=30) as response:
        page = response.read().decode('utf-8', errors='replace')
    match = re.search(r'https?://sgodds\.com/downloads/[^"\']*k-league\.csv', page)
    if match is None:
        match = re.search(r'/downloads/[^"\']*k-league\.csv', page)
    if match is None:
        raise RuntimeError('韩职数据页未找到 K League CSV 下载地址。')
    url = match.group(0)
    if url.startswith('/'):
        url = f'https://sgodds.com{url}'
    with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60) as response:
        return pd.read_csv(BytesIO(response.read()))


def _sync_korean_league(league_db: LeagueDatabase) -> pd.DataFrame:
    archived = pd.read_csv(KOREA_SOURCE_PATH) if KOREA_SOURCE_PATH.exists() else None
    try:
        latest = _download_korean_rows()
    except Exception as error:
        if archived is None or archived.empty:
            raise RuntimeError(f'韩职网络不可用且没有本地缓存：{error}') from error
        logging.warning('韩职在线数据获取失败，继续使用本地缓存：%s', error)
        latest = pd.DataFrame(columns=archived.columns)
    raw = (
        pd.concat([archived, latest], ignore_index=True).drop_duplicates(
            subset=['Start Time', 'Match'], keep='last',
        )
        if archived is not None else latest
    )
    KOREA_SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(KOREA_SOURCE_PATH, index=False)
    dataset = _korean_feature_dataset(raw)
    if KOREA_HISTORY_PATH.exists():
        history = _korean_history_feature_dataset(pd.read_csv(KOREA_HISTORY_PATH))
        dataset = pd.concat([history, dataset], ignore_index=True, sort=False)
        dataset = dataset.drop_duplicates(['Date', 'Home', 'Away'], keep='last')
        dataset = dataset.sort_values(
            ['Date', 'Home'], ascending=[False, True],
        ).reset_index(drop=True)
    league_db.save_league(dataset, league_db.index['韩职'])
    return dataset


def _model_parameters(config: dict) -> dict:
    """Convert a saved model config back to safe constructor parameters."""
    params = {key: value for key, value in config.items() if key not in {'cls', 'train', 'eval'}}
    normalizer = params.get('normalizer')
    if isinstance(normalizer, StandardScaler):
        params['normalizer'] = NormalizerType.STANDARD
    elif isinstance(normalizer, MinMaxScaler):
        params['normalizer'] = NormalizerType.MIN_MAX
    elif isinstance(normalizer, MaxAbsScaler):
        params['normalizer'] = NormalizerType.MAX_ABS
    elif isinstance(normalizer, TransformerMixin):
        params['normalizer'] = None
    return params


def retrain_saved_models(league_id: str, dataset: pd.DataFrame) -> int:
    """Refit every saved model with its selected parameters and all known matches."""
    model_db = ModelDatabase(league_id)
    trained = 0
    for model_id in model_db.get_model_ids():
        if '早期模型' in model_id:
            continue
        config = model_db.load_model_config(model_id)
        target_type = config.get('target_type')
        if target_type == TargetType.HALF_FULL:
            clean = dataset.dropna().reset_index(drop=True)
            # Protect a previously trained half-full model when an upstream
            # source adds full-time rows without enough verified HTR history.
            if len(clean) < 100:
                continue
        else:
            # Missing HTR must not discard otherwise complete matches when
            # retraining result, totals, or score models.
            clean = dataset.drop(columns=['HTR'], errors='ignore')
            clean = clean.dropna().reset_index(drop=True)
        history_tuning = config.get('train', {}).get('history_weight_tuning', {})
        history_years = history_tuning.get('history_years')
        fit_data = clean
        if history_years is not None:
            dates = pd.to_datetime(clean['Date'])
            cutoff = dates.max() - pd.DateOffset(years=int(history_years))
            fit_data = clean.loc[dates >= cutoff].reset_index(drop=True)
        model = config['cls'](**_model_parameters(config))
        model.fit(fit_data)
        new_config = model.get_default_model_config()
        new_config['train'] = dict(config.get('train', {}))
        new_config['train']['last_incremental_retrain'] = {
            'at': datetime.now(timezone.utc).isoformat(),
            'samples': len(fit_data),
            'history_years': history_years,
        }
        model_db.save_model(model, new_config)
        trained += 1
    return trained


def _preserve_half_time_target(before: pd.DataFrame, updated: pd.DataFrame) -> pd.DataFrame:
    """Keep verified HTR values when the base odds feed omits half-time data."""
    if 'HTR' not in before.columns:
        return updated
    known = before[['Date', 'Home', 'Away', 'HTR']].dropna(subset=['HTR']).drop_duplicates(
        ['Date', 'Home', 'Away'], keep='last',
    )
    result = updated.drop(columns=['HTR'], errors='ignore').merge(
        known, on=['Date', 'Home', 'Away'], how='left',
    )
    return result


def sync_five_leagues(
        league_ids: Iterable[str] = SYNC_LEAGUES,
        retrain_on_change: bool = True,
) -> Dict[str, dict]:
    """Synchronize all prediction leagues and refit models when matches changed."""
    league_db = LeagueDatabase()
    available = set(league_db.get_league_ids())
    results: Dict[str, dict] = {}
    for league_id in league_ids:
        try:
            if league_id not in available:
                results[league_id] = {'status': 'missing', 'added': 0, 'models_retrained': 0}
                continue
            before = league_db.load_league(league_id)
            before_keys = set(zip(before['Date'].astype(str), before['Home'], before['Away']))
            updated = (
                _sync_korean_league(league_db)
                if league_id == '韩职'
                else league_db.update_league(league_id)
            )
            if league_id in {'瑞超', '日职'}:
                updated = _preserve_half_time_target(before, updated)
                league_db.save_league(updated, league_db.index[league_id])
            after_keys = set(zip(updated['Date'].astype(str), updated['Home'], updated['Away']))
            added = len(after_keys - before_keys)
            retrained = retrain_saved_models(league_id, updated) if added and retrain_on_change else 0
            results[league_id] = {
                'status': 'updated' if added else 'current',
                'added': added,
                'total': len(updated),
                'latest_date': str(updated['Date'].max()),
                'models_retrained': retrained,
            }
        except Exception as error:
            results[league_id] = {
                'status': 'rejected',
                'added': 0,
                'models_retrained': 0,
                'error': str(error),
            }

    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SYNC_STATE_PATH.open('w', encoding='utf-8') as handle:
        json.dump({
            'last_sync': datetime.now(timezone.utc).isoformat(),
            'results': results,
        }, handle, ensure_ascii=False, indent=2)
    return results


def sync_is_due(interval_hours: int = 24) -> bool:
    if not SYNC_STATE_PATH.exists():
        return True
    age = datetime.now(timezone.utc).timestamp() - SYNC_STATE_PATH.stat().st_mtime
    return age >= interval_hours * 3600
