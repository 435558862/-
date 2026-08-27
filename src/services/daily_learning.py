"""Daily Sporttery review and guarded generic-model learning loop."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.network.fixtures.sporttery import SportteryResultClient
from src.services.draw_calibration import AUDIT_PATH as DRAW_AUDIT_PATH, train_draw_calibrator


REPORT_ROOT = Path('storage/jingcai/reports')
LEARNING_ROOT = Path('storage/jingcai/learning')
SETTLED_PATH = LEARNING_ROOT / 'settled_predictions.csv'
OFFICIAL_HISTORY_PATH = LEARNING_ROOT / 'official_market_history.csv'
STATUS_PATH = LEARNING_ROOT / 'status.json'
GENERIC_MODEL_PATH = Path('storage/models/market/daily_generic_1x2.joblib')
AUDIT_ROOT = Path('storage/models/market/audits/daily_learning')
CHAMPION_ROOT = Path('storage/models/market/champions/daily_learning')
SELECTION_PROFILE_PATH = LEARNING_ROOT / 'selection_profile.json'
REPORT_PATTERN = re.compile(r'^(\d{4}-\d{2}-\d{2})-竞彩预测\.csv$')
MIN_TRAIN_ROWS = 300
VALIDATION_ROWS = 60
TEST_ROWS = 60
HOLDOUT_FRACTION = 0.10
MAX_HOLDOUT_ROWS = 500
RETRAIN_STEP = 10
TRAINING_EVALUATION_VERSION = 4
MODEL_GUARD_MIN_SAMPLES = 30
MODEL_GUARD_WINDOW = 60
DEFAULT_OFFICIAL_LOOKBACK_DAYS = 7
FULL_OFFICIAL_LOOKBACK_DAYS = 365
NUMERIC_FEATURES = [
    'p_home', 'p_draw', 'p_away', 'overround', 'entropy',
    'favorite_gap', 'home_away_gap',
]
RICH_NUMERIC_FEATURES = NUMERIC_FEATURES + [
    'log_odds_home', 'log_odds_draw', 'log_odds_away',
    'draw_vs_sides', 'p_home_squared', 'p_draw_squared', 'p_away_squared',
]
FEATURE_PROFILES = {
    'market_league_v1': {
        'numeric': NUMERIC_FEATURES,
        'categorical': ['league'],
    },
    'market_shape_v2': {
        'numeric': RICH_NUMERIC_FEATURES,
        'categorical': ['league', 'month', 'favorite_side'],
    },
    'market_team_v2': {
        'numeric': RICH_NUMERIC_FEATURES,
        'categorical': ['league', 'month', 'favorite_side', 'home', 'away'],
    },
}
EVOLUTION_CANDIDATES = (
    ('market_league_v1', 0.03, None),
    ('market_league_v1', 0.10, None),
    ('market_league_v1', 0.30, None),
    ('market_league_v1', 1.00, None),
    ('market_league_v1', 0.10, 1200),
    ('market_league_v1', 0.30, 2400),
    ('market_shape_v2', 0.01, None),
    ('market_shape_v2', 0.03, None),
    ('market_shape_v2', 0.10, None),
    ('market_shape_v2', 0.10, 1200),
    ('market_team_v2', 0.01, None),
    ('market_team_v2', 0.03, None),
    ('market_team_v2', 0.10, 1200),
)
EVOLUTION_FOLDS = 3


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    temporary.replace(path)


def _normalize_match_id(value) -> str:
    text = str(value or '').strip()
    return text[:-2] if text.endswith('.0') and text[:-2].isdigit() else text


def _parse_score(value: str) -> Optional[tuple[int, int]]:
    match = re.fullmatch(r'\s*(\d+)\s*[:：-]\s*(\d+)\s*', str(value or ''))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _float(row: pd.Series, column: str) -> float:
    try:
        return float(row.get(column))
    except (TypeError, ValueError):
        return float('nan')


def _first_text(*values) -> str:
    for value in values:
        if pd.notna(value):
            text = str(value).strip()
            if text and text.casefold() != 'nan':
                return text
    return ''


def _prediction_model_category(prediction: pd.Series) -> str:
    result_category = _first_text(
        prediction.get('胜负模型类别'), prediction.get('result_model_category'),
    )
    if result_category:
        return result_category
    category = _first_text(prediction.get('模型类别'), prediction.get('model_category'))
    dedicated = _first_text(
        prediction.get('专用模型联赛'), prediction.get('dedicated_league'),
    )
    basis = _first_text(prediction.get('预测依据'), prediction.get('prediction_basis'))
    if dedicated:
        return f'{dedicated}专用模型'
    if category and category != '通用/市场模型':
        return category
    if '通用模型' in basis:
        return '通用模型'
    if '欧战' in basis and '校准' in basis:
        return '欧战校准模型'
    return '市场基线'


def _accuracy_by_model(settled: pd.DataFrame) -> dict:
    """Build rolling, model-specific audits instead of one blended hit rate."""
    if settled.empty:
        return {}
    frame = settled.copy()
    if 'model_category' not in frame.columns:
        frame['model_category'] = frame.apply(_prediction_model_category, axis=1)
    else:
        missing = frame['model_category'].fillna('').astype(str).eq('')
        if missing.any():
            frame.loc[missing, 'model_category'] = frame.loc[missing].apply(
                _prediction_model_category, axis=1,
            )
    for column in ('match_date', 'prediction_date'):
        if column not in frame.columns:
            frame[column] = ''
    result = {}
    for category, group in frame.sort_values(
            ['match_date', 'prediction_date'], kind='stable',
    ).groupby('model_category', dropna=False):
        recent = group.tail(MODEL_GUARD_WINDOW).copy()
        actual = pd.to_numeric(recent.get('actual_result'), errors='coerce')
        model_hit = pd.to_numeric(recent.get('result_hit'), errors='coerce')
        market_probability = recent[[
            'market_p_home', 'market_p_draw', 'market_p_away',
        ]].apply(pd.to_numeric, errors='coerce')
        valid = actual.notna() & model_hit.notna() & market_probability.notna().all(axis=1)
        recent = recent.loc[valid]
        actual = actual.loc[valid].astype(int)
        model_hit = model_hit.loc[valid]
        market_probability = market_probability.loc[valid]
        samples = len(recent)
        if samples == 0:
            continue
        market_hit = market_probability.to_numpy().argmax(axis=1) == actual.to_numpy()
        accuracy = float(model_hit.mean())
        market_accuracy = float(np.mean(market_hit))
        fallback = bool(
            samples >= MODEL_GUARD_MIN_SAMPLES
            and (accuracy < 0.45 or accuracy + 0.02 < market_accuracy)
        )
        result[str(category or '未标记模型')] = {
            'samples': samples,
            'window': MODEL_GUARD_WINDOW,
            'accuracy': accuracy,
            'market_accuracy': market_accuracy,
            'edge_vs_market': accuracy - market_accuracy,
            'action': 'fallback_market' if fallback else 'active',
            'status': '自动回退市场基线' if fallback else (
                '样本积累中' if samples < MODEL_GUARD_MIN_SAMPLES else '继续启用'
            ),
        }
    return result


def model_result_is_allowed(model_category: str) -> bool:
    """Use the last persisted live audit as a circuit breaker."""
    if not STATUS_PATH.exists():
        return True
    try:
        status = json.loads(STATUS_PATH.read_text(encoding='utf-8'))
        audit = status.get('accuracy_by_model', {}).get(model_category, {})
        return audit.get('action') != 'fallback_market'
    except (OSError, ValueError, TypeError):
        return True


def model_result_blend_weight(model_category: str) -> float:
    """Return a conservative live-audit weight for a dedicated model.

    New or tied models are blended with the market instead of replacing it;
    only a sufficiently sampled model with positive live edge earns more weight.
    """
    if not STATUS_PATH.exists():
        return 0.35
    try:
        status = json.loads(STATUS_PATH.read_text(encoding='utf-8'))
        audit = status.get('accuracy_by_model', {}).get(model_category, {})
        samples = int(audit.get('samples') or 0)
        edge = float(audit.get('edge_vs_market') or 0.0)
        if audit.get('action') == 'fallback_market':
            return 0.0
        # Before 30 settled live predictions the dedicated model is shadowed:
        # it contributes only 10%, so it can accumulate an audit without being
        # allowed to dominate a recommendation. Between 30 and 49 it earns a
        # cautious weight only with a positive edge; 50+ samples unlock the
        # normal dynamic blend.
        if samples < MODEL_GUARD_MIN_SAMPLES:
            return 0.10
        if samples < 50:
            return 0.25 if edge > 0.0 else 0.10
        return min(0.70, max(0.10, 0.35 + edge * 4.0))
    except (OSError, ValueError, TypeError):
        return 0.35


def _prediction_reports(today: date) -> pd.DataFrame:
    frames = []
    for path in REPORT_ROOT.glob('*-竞彩预测.csv'):
        match = REPORT_PATTERN.match(path.name)
        if not match:
            continue
        prediction_day = date.fromisoformat(match.group(1))
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, UnicodeError):
            continue
        if frame.empty or '比赛ID' not in frame.columns:
            continue
        frame['_prediction_date'] = prediction_day.isoformat()
        frame['_source_report'] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result['_match_id'] = result['比赛ID'].map(_normalize_match_id)
    result = result[result['_match_id'].ne('')].copy()
    # A match offered on multiple days contributes one observation: use the
    # freshest pre-match odds rather than overweighting that fixture.
    return result.sort_values('_prediction_date').drop_duplicates(
        '_match_id', keep='last',
    )


def _load_settled() -> pd.DataFrame:
    if not SETTLED_PATH.exists() or SETTLED_PATH.stat().st_size == 0:
        return pd.DataFrame()
    try:
        result = pd.read_csv(SETTLED_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if 'match_id' in result.columns:
        result['match_id'] = result['match_id'].map(_normalize_match_id)
    return result


def _load_official_history() -> pd.DataFrame:
    if not OFFICIAL_HISTORY_PATH.exists() or OFFICIAL_HISTORY_PATH.stat().st_size == 0:
        return pd.DataFrame()
    try:
        result = pd.read_csv(OFFICIAL_HISTORY_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if 'match_id' in result.columns:
        result['match_id'] = result['match_id'].map(_normalize_match_id)
    return result


def _result_rows_for_dates(
        client: SportteryResultClient,
        begin: date,
        end: date,
) -> list[dict]:
    rows = []
    cursor = begin
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=29))
        rows.extend(client.settled_matches(cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return rows


def _settled_record(prediction: pd.Series, result: dict) -> Optional[dict]:
    score = _parse_score(result.get('sectionsNo999'))
    if score is None or str(result.get('matchResultStatus')) != '2':
        return None
    home_goals, away_goals = score
    actual_result = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
    labels = ['胜', '平', '负']
    predicted_score = _first_text(prediction.get('首选比分'))
    predicted_score_second = _first_text(prediction.get('次选比分'))
    predicted_score_third = _first_text(prediction.get('第三比分'))
    predicted_score_upset = _first_text(
        prediction.get('比分爆冷'), prediction.get('爆冷比分'),
    )
    predicted_score_aggressive = _first_text(prediction.get('大小球进取比分'))
    score_candidates = [
        ('首', predicted_score), ('次1', predicted_score_second),
        ('次2', predicted_score_third), ('冷', predicted_score_upset),
        ('进', predicted_score_aggressive),
    ]
    actual_score = f'{home_goals}-{away_goals}'
    score_hit_source = next(
        (label for label, value in score_candidates if _parse_score(value) == score), '',
    )
    predicted_ou = _first_text(prediction.get('大小球首选'))
    actual_over = home_goals + away_goals > 2

    handicap_line = _float(prediction, '官方让球数')
    predicted_handicap = _first_text(prediction.get('让球首选'))
    predicted_handicap_second = _first_text(prediction.get('让球次选'))
    actual_handicap = ''
    if np.isfinite(handicap_line):
        adjusted = home_goals + handicap_line - away_goals
        actual_handicap = '胜' if adjusted > 1e-9 else '平' if abs(adjusted) <= 1e-9 else '负'

    half_score = _parse_score(result.get('sectionsNo1'))
    actual_half_full = ''
    if half_score is not None:
        half_result = 0 if half_score[0] > half_score[1] else 1 if half_score[0] == half_score[1] else 2
        actual_half_full = labels[half_result] + labels[actual_result]
    predicted_half_full = _first_text(prediction.get('半全场首选'))
    predicted_half_full_second = _first_text(prediction.get('半全场次选'))
    # Freeze the independent Monte Carlo output alongside the professional
    # model picks. Review must never reconstruct one side from the other.
    monte_carlo_fields = {
        'monte_carlo_count': prediction.get('模拟次数'),
        'monte_carlo_top3_score': _first_text(prediction.get('模拟Top3比分')),
        'monte_carlo_result': _first_text(prediction.get('模拟胜负')),
        'monte_carlo_handicap': _first_text(prediction.get('模拟让球')),
        'monte_carlo_total': _first_text(prediction.get('模拟总进球')),
        'monte_carlo_half_full': _first_text(prediction.get('模拟半全场')),
        'monte_carlo_confidence': _first_text(prediction.get('模拟可信度')),
        'monte_carlo_risk': _first_text(prediction.get('蒙特风险')),
        'monte_carlo_source': _first_text(prediction.get('模拟模型来源')),
    }
    return {
        'prediction_date': prediction['_prediction_date'],
        'match_id': prediction['_match_id'],
        'match_date': str(result.get('matchDate') or prediction.get('比赛时间') or ''),
        'match_time': _first_text(prediction.get('比赛时间')),
        'match_number': str(prediction.get('赛事编号') or result.get('matchNumStr') or ''),
        'league': str(prediction.get('联赛') or result.get('leagueName') or ''),
        'home': str(prediction.get('主队') or result.get('allHomeTeam') or ''),
        'away': str(prediction.get('客队') or result.get('allAwayTeam') or ''),
        'odds_home': _float(prediction, '官方胜奖金'),
        'odds_draw': _float(prediction, '官方平奖金'),
        'odds_away': _float(prediction, '官方负奖金'),
        'market_p_home': _float(prediction, '市场去水主胜概率'),
        'market_p_draw': _float(prediction, '市场去水平局概率'),
        'market_p_away': _float(prediction, '市场去水客胜概率'),
        'model_p_home': _float(prediction, '模型主胜概率'),
        'model_p_draw': _float(prediction, '模型平局概率'),
        'model_p_away': _float(prediction, '模型客胜概率'),
        'predicted_result': str(prediction.get('胜平负首选') or ''),
        'advice': str(prediction.get('建议状态') or ''),
        'prediction_basis': str(prediction.get('预测依据') or ''),
        'model_category': _prediction_model_category(prediction),
        'dedicated_league': str(prediction.get('专用模型联赛') or ''),
        'confidence': str(prediction.get('置信等级') or ''),
        'draw_probability_change': _float(prediction, '平局概率变化'),
        'hhad_line_change': _float(prediction, '让球线变化'),
        'ttg_expected_change': _float(prediction, '总进球预期变化'),
        'predicted_score': predicted_score,
        'predicted_score_second': predicted_score_second,
        'predicted_score_third': predicted_score_third,
        'predicted_score_upset': predicted_score_upset,
        'predicted_score_aggressive': predicted_score_aggressive,
        'predicted_over_under': predicted_ou,
        'handicap_line': handicap_line,
        'predicted_handicap': predicted_handicap,
        'predicted_handicap_second': predicted_handicap_second,
        'predicted_half_full': predicted_half_full,
        'predicted_half_full_second': predicted_half_full_second,
        **monte_carlo_fields,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'actual_score': actual_score,
        'actual_result': actual_result,
        'actual_result_label': labels[actual_result],
        'actual_handicap': actual_handicap,
        'actual_half_full': actual_half_full,
        'result_hit': int(str(prediction.get('胜平负首选') or '') == labels[actual_result]),
        'score_hit': int(_parse_score(predicted_score) == score),
        'score_hit_any': int(bool(score_hit_source)),
        'score_hit_source': score_hit_source,
        'handicap_hit': (
            int(predicted_handicap == actual_handicap)
            if predicted_handicap and actual_handicap else np.nan
        ),
        'handicap_second_hit': (
            int(predicted_handicap_second == actual_handicap)
            if predicted_handicap_second and actual_handicap else np.nan
        ),
        'half_full_hit': (
            int(predicted_half_full == actual_half_full)
            if predicted_half_full and actual_half_full else np.nan
        ),
        'half_full_second_hit': (
            int(predicted_half_full_second == actual_half_full)
            if predicted_half_full_second and actual_half_full else np.nan
        ),
        'over_under_hit': int(
            predicted_ou == ('大于2.5球' if actual_over else '小于2.5球')
        ),
        'official_half_score': str(result.get('sectionsNo1') or ''),
        'official_status': str(result.get('poolStatus') or ''),
        'settled_at': datetime.now().isoformat(timespec='seconds'),
        'source_report': prediction['_source_report'],
    }


def _official_training_record(result: dict) -> Optional[dict]:
    """Convert an official completed result into a market-training sample."""
    score = _parse_score(result.get('sectionsNo999'))
    if score is None or str(result.get('matchResultStatus')) != '2':
        return None
    try:
        odds = [float(result[key]) for key in ('h', 'd', 'a')]
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(odds).all() or not all(value > 1.0 for value in odds):
        return None
    home_goals, away_goals = score
    actual = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
    match_day = str(result.get('matchDate') or '')
    return {
        'prediction_date': match_day,
        'match_id': _normalize_match_id(result.get('matchId')),
        'match_date': match_day,
        'match_number': str(result.get('matchNumStr') or ''),
        'league': str(result.get('leagueName') or result.get('leagueNameAbbr') or ''),
        'home': str(result.get('allHomeTeam') or result.get('homeTeam') or ''),
        'away': str(result.get('allAwayTeam') or result.get('awayTeam') or ''),
        'odds_home': odds[0],
        'odds_draw': odds[1],
        'odds_away': odds[2],
        'home_goals': home_goals,
        'away_goals': away_goals,
        'actual_result': actual,
        'actual_result_label': ['胜', '平', '负'][actual],
        'source_type': 'official_result_market_odds',
        'imported_at': datetime.now().isoformat(timespec='seconds'),
    }


def _save_frame(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _backfill_official_history(
        today: date,
        client: SportteryResultClient,
        full_backfill: bool,
) -> tuple[pd.DataFrame, int]:
    history = _load_official_history()
    if not history.empty and 'match_date' in history.columns:
        latest = pd.to_datetime(history['match_date'], errors='coerce').max()
        begin = latest.date() if not pd.isna(latest) else today - timedelta(days=7)
    else:
        lookback = (
            FULL_OFFICIAL_LOOKBACK_DAYS if full_backfill
            else DEFAULT_OFFICIAL_LOOKBACK_DAYS
        )
        begin = today - timedelta(days=lookback)
    results = _result_rows_for_dates(client, begin, today)
    records = [
        record for record in (_official_training_record(row) for row in results)
        if record is not None
    ]
    before = set(history.get('match_id', pd.Series(dtype=str)).astype(str))
    if records:
        history = pd.concat([history, pd.DataFrame(records)], ignore_index=True)
        history = history.sort_values(['match_date', 'match_id']).drop_duplicates(
            'match_id', keep='last',
        )
        _save_frame(OFFICIAL_HISTORY_PATH, history)
    after = set(history.get('match_id', pd.Series(dtype=str)).astype(str))
    return history, len(after - before)


def _combined_training_data(
        settled: pd.DataFrame,
        official_history: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not official_history.empty:
        frames.append(official_history)
    if not settled.empty:
        review_rows = settled.copy()
        review_rows['source_type'] = 'saved_pre_match_prediction'
        frames.append(review_rows)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Saved pre-match rows are appended last and therefore take precedence over
    # final-official odds when both sources contain the same fixture.
    return combined.drop_duplicates('match_id', keep='last')


def _market_probabilities(data: pd.DataFrame) -> np.ndarray:
    odds = data[['odds_home', 'odds_draw', 'odds_away']].to_numpy(dtype=np.float64)
    inverse = 1.0 / odds
    return inverse / inverse.sum(axis=1, keepdims=True)


def _text_column(data: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=str)
    return data[column].fillna(default).astype(str)


def _features(
        data: pd.DataFrame,
        profile: str = 'market_league_v1',
) -> pd.DataFrame:
    if profile not in FEATURE_PROFILES:
        raise ValueError(f'未知特征方案：{profile}')
    probability = _market_probabilities(data)
    inverse = 1.0 / data[['odds_home', 'odds_draw', 'odds_away']].to_numpy(
        dtype=np.float64,
    )
    ordered = np.sort(probability, axis=1)
    match_date = pd.to_datetime(
        data.get('match_date', pd.Series('', index=data.index)), errors='coerce',
    )
    favorite_index = probability.argmax(axis=1)
    result = pd.DataFrame({
        'p_home': probability[:, 0],
        'p_draw': probability[:, 1],
        'p_away': probability[:, 2],
        'overround': inverse.sum(axis=1),
        'entropy': -np.sum(
            probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1,
        ),
        'favorite_gap': ordered[:, -1] - ordered[:, -2],
        'home_away_gap': probability[:, 0] - probability[:, 2],
        'log_odds_home': np.log(data['odds_home'].to_numpy(dtype=float)),
        'log_odds_draw': np.log(data['odds_draw'].to_numpy(dtype=float)),
        'log_odds_away': np.log(data['odds_away'].to_numpy(dtype=float)),
        'draw_vs_sides': probability[:, 1] - (
            probability[:, 0] + probability[:, 2]
        ) / 2.0,
        'p_home_squared': probability[:, 0] ** 2,
        'p_draw_squared': probability[:, 1] ** 2,
        'p_away_squared': probability[:, 2] ** 2,
        'league': _text_column(data, 'league', '未知联赛'),
        'month': match_date.dt.month.fillna(0).astype(int).astype(str),
        'favorite_side': pd.Series(
            np.asarray(['主', '平', '客'])[favorite_index], index=data.index,
        ),
        'home': _text_column(data, 'home', '未知主队'),
        'away': _text_column(data, 'away', '未知客队'),
    }, index=data.index)
    columns = (
        FEATURE_PROFILES[profile]['numeric']
        + FEATURE_PROFILES[profile]['categorical']
    )
    return result[columns]


def _build_model(c: float, profile: str = 'market_league_v1') -> Pipeline:
    specification = FEATURE_PROFILES[profile]
    transform = ColumnTransformer([
        ('numeric', StandardScaler(), specification['numeric']),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), specification['categorical']),
    ])
    return Pipeline([
        ('features', transform),
        ('classifier', LogisticRegression(C=c, max_iter=2000, random_state=42)),
    ])


def _ordered_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    result = np.zeros((len(features), 3), dtype=np.float64)
    for source, target_class in enumerate(model.named_steps['classifier'].classes_):
        result[:, int(target_class)] = raw[:, source]
    return result


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict:
    return {
        'samples': int(len(target)),
        'accuracy': float(accuracy_score(target, probability.argmax(axis=1))),
        'log_loss': float(log_loss(target, probability, labels=[0, 1, 2])),
    }


def _threshold_result(
        frame: pd.DataFrame,
        threshold: float,
) -> dict:
    probability = _market_probabilities(frame)
    target = frame['actual_result'].to_numpy(dtype=np.int32)
    eligible = probability.max(axis=1) >= threshold
    hit = probability.argmax(axis=1) == target
    return {
        'samples': int(eligible.sum()),
        'coverage': float(eligible.mean()),
        'accuracy': float(hit[eligible].mean()) if eligible.any() else 0.0,
    }


def _choose_selection_threshold(
        full: pd.DataFrame,
        recent: pd.DataFrame,
        minimum_threshold: float,
        target_accuracy: float,
) -> float:
    for threshold in np.arange(minimum_threshold, 0.801, 0.005):
        full_result = _threshold_result(full, float(threshold))
        recent_result = _threshold_result(recent, float(threshold))
        if (
            full_result['samples'] >= 200
            and recent_result['samples'] >= 80
            and full_result['accuracy'] >= target_accuracy
            and recent_result['accuracy'] >= target_accuracy
        ):
            return round(float(threshold), 3)
    return 0.80


def _learn_selection_profile(
        data: pd.DataFrame,
        today: date,
) -> Optional[dict]:
    required = {
        'odds_home', 'odds_draw', 'odds_away', 'actual_result', 'match_date',
    }
    if data.empty or not required.issubset(data.columns):
        return None
    valid = data.dropna(subset=list(required)).copy()
    odds = valid[['odds_home', 'odds_draw', 'odds_away']].to_numpy(dtype=float)
    valid = valid[
        np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    ]
    sort_columns = ['match_date'] + (
        ['prediction_date'] if 'prediction_date' in valid.columns else []
    )
    valid = valid.sort_values(sort_columns, kind='stable').reset_index(drop=True)
    if len(valid) < 500:
        return None
    # Thresholds are selected without seeing the newest chronological audit
    # block. Reported hit rates include that untouched block, preventing the
    # same matches from both choosing and validating a recommendation line.
    audit_rows = max(100, min(500, int(len(valid) * 0.20)))
    calibration = valid.iloc[:-audit_rows].copy()
    audit = valid.iloc[-audit_rows:].copy()
    recent = calibration.tail(min(1000, len(calibration))).copy()
    # Keep a small safety margin above 70% so ordinary sampling noise does not
    # turn a borderline tier into a main recommendation.
    high_threshold = _choose_selection_threshold(valid, recent, 0.625, 0.71)
    selected_threshold = _choose_selection_threshold(
        valid, recent, max(0.675, high_threshold + 0.025), 0.75,
    )
    observe_threshold = _choose_selection_threshold(valid, recent, 0.55, 0.64)
    if observe_threshold >= high_threshold:
        observe_threshold = 0.55

    rows = []
    for threshold, grade in (
        (selected_threshold, '精选主推'),
        (high_threshold, '高置信主推'),
        (observe_threshold, '观察'),
        (0.0, '跳过'),
    ):
        full_result = _threshold_result(calibration, threshold)
        recent_result = _threshold_result(recent, threshold)
        audit_result = _threshold_result(audit, threshold)
        rows.append({
            'threshold': threshold,
            'grade': grade,
            # Use the weaker of long-run and recent hit rates in the UI.
            'accuracy': min(
                full_result['accuracy'], recent_result['accuracy'],
                audit_result['accuracy'],
            ),
            'coverage': audit_result['coverage'],
            'samples': audit_result['samples'],
            'full_samples': full_result['samples'],
            'audit_samples': audit_result['samples'],
        })
    profile = {
        'learned_at': datetime.now().isoformat(timespec='seconds'),
        'as_of': today.isoformat(),
        'total_samples': len(valid),
        'recent_samples': len(audit),
        'calibration_samples': len(calibration),
        'audit_samples': len(audit),
        'period': (
            f'{str(audit["match_date"].min())[:10]}至'
            f'{str(audit["match_date"].max())[:10]}'
        ),
        'rows': rows,
    }
    _write_json(SELECTION_PROFILE_PATH, profile)
    return profile


def load_selection_profile() -> Optional[dict]:
    if not SELECTION_PROFILE_PATH.exists():
        return None
    try:
        profile = json.loads(SELECTION_PROFILE_PATH.read_text(encoding='utf-8'))
        rows = profile.get('rows')
        return profile if isinstance(rows, list) and rows else None
    except (OSError, ValueError, TypeError):
        logging.exception('自主学习筛选阈值加载失败，使用内置安全阈值。')
        return None


def _holdout_rows(samples: int, minimum: int) -> int:
    """Use a stable chronological holdout that grows with the learning history."""
    proportional = int(samples * HOLDOUT_FRACTION)
    return max(minimum, min(MAX_HOLDOUT_ROWS, proportional))


@lru_cache(maxsize=1)
def load_generic_artifact():
    if not GENERIC_MODEL_PATH.exists():
        return None
    try:
        artifact = joblib.load(GENERIC_MODEL_PATH)
        return artifact if artifact.get('deployable') else None
    except Exception:
        logging.exception('每日通用模型加载失败。')
        return None


def predict_generic_probabilities(
        league: str,
        odds: dict,
        home: str = '',
        away: str = '',
        match_date: str = '',
) -> Optional[np.ndarray]:
    if not model_result_is_allowed('通用模型'):
        return None
    artifact = load_generic_artifact()
    if artifact is None:
        return None
    row = pd.DataFrame([{
        'league': league or '未知联赛',
        'home': home or '未知主队',
        'away': away or '未知客队',
        'match_date': match_date,
        'odds_home': odds['H'], 'odds_draw': odds['D'], 'odds_away': odds['A'],
    }])
    try:
        profile = artifact.get('feature_profile', 'market_league_v1')
        learned = _ordered_probability(
            artifact['model'], _features(row, profile),
        )[0]
        market = _market_probabilities(row)[0]
        weight = float(artifact['model_weight'])
        result = weight * learned + (1.0 - weight) * market
        return result / result.sum()
    except Exception:
        logging.exception('每日通用模型预测失败，回退市场基线。')
        return None


def _recency_weights(rows: int, half_life: Optional[int]) -> Optional[np.ndarray]:
    if not half_life:
        return None
    age = np.arange(rows - 1, -1, -1, dtype=np.float64)
    return np.clip(np.exp2(-age / float(half_life)), 0.10, 1.0)


def _fit_candidate(
        frame: pd.DataFrame,
        profile: str,
        c: float,
        half_life: Optional[int],
) -> Pipeline:
    model = _build_model(c, profile)
    fit_arguments = {}
    weights = _recency_weights(len(frame), half_life)
    if weights is not None:
        fit_arguments['classifier__sample_weight'] = weights
    model.fit(
        _features(frame, profile),
        frame['actual_result'].to_numpy(dtype=np.int32),
        **fit_arguments,
    )
    return model


def _best_blend(
        target: np.ndarray,
        learned: np.ndarray,
        market: np.ndarray,
) -> tuple[float, dict]:
    selected_weight, selected_metrics, selected_rank = 0.0, None, None
    for weight in np.arange(0.0, 0.501, 0.025):
        probability = weight * learned + (1.0 - weight) * market
        metrics = _metrics(target, probability)
        rank = (metrics['accuracy'], -metrics['log_loss'], -float(weight))
        if selected_rank is None or rank > selected_rank:
            selected_weight = float(weight)
            selected_metrics = metrics
            selected_rank = rank
    return selected_weight, selected_metrics


def _evolution_folds(samples: int, validation_rows: int, test_rows: int) -> list[tuple[int, int]]:
    pretest_end = samples - test_rows
    minimum_fold_train = max(15, MIN_TRAIN_ROWS // 2)
    possible = max(1, (pretest_end - minimum_fold_train) // validation_rows)
    fold_count = min(EVOLUTION_FOLDS, possible)
    return [
        (
            pretest_end - (fold_count - index) * validation_rows,
            pretest_end - (fold_count - index - 1) * validation_rows,
        )
        for index in range(fold_count)
    ]


def _train_if_ready(data: pd.DataFrame, previous_status: dict) -> dict:
    required = {'odds_home', 'odds_draw', 'odds_away', 'actual_result'}
    if data.empty or not required.issubset(data.columns):
        return {
            'model_status': '积累样本中',
            'model_samples': 0,
            'minimum_training_samples': MIN_TRAIN_ROWS,
            'next_training_at': MIN_TRAIN_ROWS,
        }
    valid = data.dropna(subset=[
        'odds_home', 'odds_draw', 'odds_away', 'actual_result',
    ]).copy()
    odds = valid[['odds_home', 'odds_draw', 'odds_away']].to_numpy(dtype=float)
    valid = valid[
        np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    ].sort_values(['match_date', 'prediction_date']).reset_index(drop=True)
    samples = len(valid)
    if samples < MIN_TRAIN_ROWS:
        return {
            'model_status': '积累样本中',
            'model_samples': samples,
            'minimum_training_samples': MIN_TRAIN_ROWS,
            'next_training_at': MIN_TRAIN_ROWS,
        }
    last_attempt = int(previous_status.get('last_training_attempt_rows') or 0)
    evaluation_version = int(
        previous_status.get('training_evaluation_version') or 0,
    )
    if (
        evaluation_version >= TRAINING_EVALUATION_VERSION
        and last_attempt >= MIN_TRAIN_ROWS
        and samples < last_attempt + RETRAIN_STEP
    ):
        waiting = {
            'model_status': previous_status.get('model_status', '等待下一批样本'),
            'model_samples': samples,
            'minimum_training_samples': MIN_TRAIN_ROWS,
            'next_training_at': last_attempt + RETRAIN_STEP,
            'last_training_attempt_rows': last_attempt,
            'training_evaluation_version': TRAINING_EVALUATION_VERSION,
        }
        for key in (
            'last_training_passed', 'challenger_test_accuracy',
            'market_test_accuracy', 'evolution_attempts',
            'champion_generation', 'selected_candidate',
        ):
            if key in previous_status:
                waiting[key] = previous_status[key]
        return waiting

    validation_rows = _holdout_rows(samples, VALIDATION_ROWS)
    test_rows = _holdout_rows(samples, TEST_ROWS)
    folds = _evolution_folds(samples, validation_rows, test_rows)
    pretest_end = samples - test_rows
    test = valid.iloc[pretest_end:]
    y_test = test['actual_result'].to_numpy(dtype=np.int32)
    test_market = _market_probabilities(test)

    selected, selected_rank = None, None
    candidates = []
    validation_targets, validation_markets = [], []
    for validation_start, validation_end in folds:
        validation = valid.iloc[validation_start:validation_end]
        validation_targets.append(
            validation['actual_result'].to_numpy(dtype=np.int32),
        )
        validation_markets.append(_market_probabilities(validation))
    combined_target = np.concatenate(validation_targets)
    combined_market = np.vstack(validation_markets)
    validation_market_metrics = _metrics(combined_target, combined_market)

    for candidate_index, (profile, c, half_life) in enumerate(EVOLUTION_CANDIDATES):
        learned_parts = []
        try:
            for validation_start, validation_end in folds:
                fold_train = valid.iloc[:validation_start]
                fold_validation = valid.iloc[validation_start:validation_end]
                model = _fit_candidate(fold_train, profile, c, half_life)
                learned_parts.append(_ordered_probability(
                    model, _features(fold_validation, profile),
                ))
        except (ValueError, TypeError) as error:
            candidates.append({
                'feature_profile': profile,
                'C': c,
                'half_life_rows': half_life,
                'error': str(error),
            })
            continue
        combined_learned = np.vstack(learned_parts)
        weight, metrics = _best_blend(
            combined_target, combined_learned, combined_market,
        )
        fold_metrics = []
        cursor = 0
        for fold_index, target_part in enumerate(validation_targets):
            length = len(target_part)
            probability = (
                weight * combined_learned[cursor:cursor + length]
                + (1.0 - weight) * validation_markets[fold_index]
            )
            candidate_metrics = _metrics(target_part, probability)
            market_fold_metrics = _metrics(target_part, validation_markets[fold_index])
            fold_metrics.append({
                'fold': fold_index + 1,
                **candidate_metrics,
                'market_accuracy': market_fold_metrics['accuracy'],
                'accuracy_edge': (
                    candidate_metrics['accuracy'] - market_fold_metrics['accuracy']
                ),
            })
            cursor += length
        worst_fold_edge = min(row['accuracy_edge'] for row in fold_metrics)
        row = {
            'feature_profile': profile,
            'C': c,
            'half_life_rows': half_life,
            'model_weight': weight,
            **metrics,
            'worst_fold_edge': worst_fold_edge,
            'folds': fold_metrics,
        }
        candidates.append(row)
        rank = (
            row['accuracy'], -row['log_loss'], row['worst_fold_edge'],
            -weight, -candidate_index,
        )
        if selected_rank is None or rank > selected_rank:
            selected_rank, selected = rank, row

    if selected is None:
        raise RuntimeError('所有自主进化候选模型训练失败。')
    profile = str(selected['feature_profile'])
    c = float(selected['C'])
    half_life = selected.get('half_life_rows')
    challenger = _fit_candidate(valid.iloc[:pretest_end], profile, c, half_life)
    learned_test = _ordered_probability(challenger, _features(test, profile))
    weight = float(selected['model_weight'])
    challenger_probability = weight * learned_test + (1.0 - weight) * test_market
    challenger_metrics = _metrics(y_test, challenger_probability)
    market_metrics = _metrics(y_test, test_market)

    champion_metrics = None
    champion = load_generic_artifact()
    if champion is not None:
        champion_profile = champion.get('feature_profile', 'market_league_v1')
        champion_learned = _ordered_probability(
            champion['model'], _features(test, champion_profile),
        )
        champion_probability = (
            float(champion['model_weight']) * champion_learned
            + (1.0 - float(champion['model_weight'])) * test_market
        )
        champion_metrics = _metrics(y_test, champion_probability)

    cross_validation_gate = (
        selected['accuracy'] >= validation_market_metrics['accuracy']
        and selected['log_loss'] <= validation_market_metrics['log_loss'] + 0.005
        and selected['worst_fold_edge'] >= -0.02
    )
    market_gate = (
        weight > 0.0
        and cross_validation_gate
        and (
            challenger_metrics['log_loss'] <= market_metrics['log_loss'] - 0.002
            or (
                challenger_metrics['accuracy'] >= market_metrics['accuracy'] + 0.005
                and challenger_metrics['log_loss'] <= market_metrics['log_loss'] + 0.01
            )
        )
    )
    champion_gate = (
        champion_metrics is None
        or (
            challenger_metrics['accuracy'] >= champion_metrics['accuracy']
            and challenger_metrics['log_loss'] <= champion_metrics['log_loss'] + 0.005
        )
    )
    passed = bool(market_gate and champion_gate)
    attempts = int(previous_status.get('evolution_attempts') or 0) + 1
    generation = int(previous_status.get('champion_generation') or 0)
    audit = {
        'attempted_at': datetime.now().isoformat(timespec='seconds'),
        'training_mode': 'autonomous_champion_challenger',
        'evaluation_version': TRAINING_EVALUATION_VERSION,
        'evolution_attempt': attempts,
        'samples': samples,
        'split': {
            'initial_train': folds[0][0],
            'validation_folds': len(folds),
            'validation_each': validation_rows,
            'validation_total': len(combined_target),
            'pretest_train': pretest_end,
            'test': len(test),
        },
        'selected_on_validation': selected,
        'validation_market': validation_market_metrics,
        'test_market': market_metrics,
        'test_challenger': challenger_metrics,
        'test_champion': champion_metrics,
        'passed': passed,
        'candidates': candidates,
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(AUDIT_ROOT / f'audit-{samples}-{date.today().isoformat()}.json', audit)

    if passed:
        final_model = _fit_candidate(valid, profile, c, half_life)
        generation += 1
        artifact = {
            'model': final_model,
            'model_weight': weight,
            'feature_profile': profile,
            'C': c,
            'half_life_rows': half_life,
            'deployable': True,
            'trained_rows': samples,
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'feature_version': 2,
            'champion_generation': generation,
            'report': audit,
        }
        GENERIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if GENERIC_MODEL_PATH.exists():
            CHAMPION_ROOT.mkdir(parents=True, exist_ok=True)
            backup_name = (
                f'generation-{generation - 1}-'
                f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.joblib'
            )
            shutil.copy2(GENERIC_MODEL_PATH, CHAMPION_ROOT / backup_name)
        temporary = GENERIC_MODEL_PATH.with_suffix('.joblib.tmp')
        joblib.dump(artifact, temporary)
        temporary.replace(GENERIC_MODEL_PATH)
        load_generic_artifact.cache_clear()
        model_status = f'自主进化第{generation}代冠军已通过并启用'
    else:
        model_status = '自主进化候选未稳定胜出，保留当前冠军/市场基线'
    return {
        'model_status': model_status,
        'model_samples': samples,
        'minimum_training_samples': MIN_TRAIN_ROWS,
        'next_training_at': samples + RETRAIN_STEP,
        'last_training_attempt_rows': samples,
        'training_evaluation_version': TRAINING_EVALUATION_VERSION,
        'last_training_passed': passed,
        'challenger_test_accuracy': challenger_metrics['accuracy'],
        'market_test_accuracy': market_metrics['accuracy'],
        'evolution_attempts': attempts,
        'champion_generation': generation,
        'selected_candidate': {
            'feature_profile': profile,
            'C': c,
            'half_life_rows': half_life,
            'model_weight': weight,
        },
    }


def review_and_learn(
        today: Optional[date] = None,
        result_client: Optional[SportteryResultClient] = None,
        full_backfill: bool = False,
) -> dict:
    """Settle past predictions, update review stats, and challenge the champion."""
    today = today or date.today()
    LEARNING_ROOT.mkdir(parents=True, exist_ok=True)
    previous_status = {}
    if STATUS_PATH.exists():
        try:
            previous_status = json.loads(STATUS_PATH.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            pass
    predictions = _prediction_reports(today)
    settled = _load_settled()
    settled_ids = set(settled.get('match_id', pd.Series(dtype=str)).astype(str))
    pending = predictions[
        ~predictions.get('_match_id', pd.Series(dtype=str)).isin(settled_ids)
    ].copy() if not predictions.empty else pd.DataFrame()
    if not pending.empty:
        match_dates = pd.to_datetime(pending.get('比赛时间'), errors='coerce').dt.date
        pending['_actual_match_date'] = match_dates
        pending = pending[
            pending['_actual_match_date'].notna()
            & pending['_actual_match_date'].le(today)
        ].copy()

    new_records = []
    errors = []
    client = result_client or SportteryResultClient()
    if not pending.empty:
        try:
            results = _result_rows_for_dates(
                client,
                min(pending['_actual_match_date']),
                min(today, max(pending['_actual_match_date'])),
            )
            by_id = {_normalize_match_id(row.get('matchId')): row for row in results}
            for _, prediction in pending.iterrows():
                result = by_id.get(prediction['_match_id'])
                if result is not None:
                    record = _settled_record(prediction, result)
                    if record is not None:
                        new_records.append(record)
        except Exception as exception:
            logging.exception('每日赛果复盘失败。')
            errors.append(f'预测复盘：{exception}')

    if new_records:
        settled = pd.concat([settled, pd.DataFrame(new_records)], ignore_index=True)
        settled = settled.sort_values(['match_date', 'prediction_date']).drop_duplicates(
            'match_id', keep='last',
        )
        _save_frame(SETTLED_PATH, settled)

    official_history = _load_official_history()
    new_official = 0
    try:
        official_history, new_official = _backfill_official_history(
            today, client, full_backfill,
        )
    except Exception as exception:
        logging.exception('官方历史市场样本补同步失败。')
        errors.append(f'历史赔率：{exception}')
    training_data = _combined_training_data(settled, official_history)
    training = _train_if_ready(training_data, previous_status)
    draw_audit = previous_status.get('draw_calibration', {})
    if not draw_audit and DRAW_AUDIT_PATH.exists():
        try:
            draw_audit = json.loads(DRAW_AUDIT_PATH.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            draw_audit = {}
    if new_records or new_official or not DRAW_AUDIT_PATH.exists():
        try:
            draw_audit = train_draw_calibrator(training_data)
        except Exception:
            logging.exception('平局校准候选训练失败，继续保留旧校准。')
            draw_audit = {'status': '训练异常，保留旧校准', 'deployable': False}
    selection_profile = _learn_selection_profile(training_data, today)
    selected = settled[
        settled.get('advice', pd.Series(dtype=str)).fillna('').astype(str).str.contains('主推')
    ] if not settled.empty else settled
    summary = {
        'last_review': datetime.now().isoformat(timespec='seconds'),
        'newly_settled': len(new_records),
        'settled_samples': len(settled),
        'new_official_history': new_official,
        'official_history_samples': len(official_history),
        'total_training_samples': len(training_data),
        'pending_samples': max(0, len(pending) - len(new_records)),
        'result_accuracy': (
            float(settled['result_hit'].mean()) if not settled.empty else None
        ),
        'selected_accuracy': (
            float(selected['result_hit'].mean()) if not selected.empty else None
        ),
        'score_accuracy': (
            float(settled['score_hit'].mean()) if not settled.empty else None
        ),
        'over_under_accuracy': (
            float(settled['over_under_hit'].mean()) if not settled.empty else None
        ),
        'accuracy_by_model': _accuracy_by_model(settled),
        'selection_profile': selection_profile,
        'draw_calibration': draw_audit,
        'review_error': '；'.join(errors),
        **training,
    }
    _write_json(STATUS_PATH, summary)
    return summary
