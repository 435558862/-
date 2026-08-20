"""Daily Sporttery review and guarded generic-model learning loop."""

from __future__ import annotations

import json
import logging
import re
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


REPORT_ROOT = Path('storage/jingcai/reports')
LEARNING_ROOT = Path('storage/jingcai/learning')
SETTLED_PATH = LEARNING_ROOT / 'settled_predictions.csv'
OFFICIAL_HISTORY_PATH = LEARNING_ROOT / 'official_market_history.csv'
STATUS_PATH = LEARNING_ROOT / 'status.json'
GENERIC_MODEL_PATH = Path('storage/models/market/daily_generic_1x2.joblib')
AUDIT_ROOT = Path('storage/models/market/audits/daily_learning')
REPORT_PATTERN = re.compile(r'^(\d{4}-\d{2}-\d{2})-竞彩预测\.csv$')
MIN_TRAIN_ROWS = 300
VALIDATION_ROWS = 60
TEST_ROWS = 60
HOLDOUT_FRACTION = 0.10
MAX_HOLDOUT_ROWS = 500
RETRAIN_STEP = 25
DEFAULT_OFFICIAL_LOOKBACK_DAYS = 7
FULL_OFFICIAL_LOOKBACK_DAYS = 365
NUMERIC_FEATURES = [
    'p_home', 'p_draw', 'p_away', 'overround', 'entropy',
    'favorite_gap', 'home_away_gap',
]


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
    predicted_score = str(prediction.get('首选比分') or '')
    predicted_ou = str(prediction.get('大小球首选') or '')
    actual_over = home_goals + away_goals > 2
    return {
        'prediction_date': prediction['_prediction_date'],
        'match_id': prediction['_match_id'],
        'match_date': str(result.get('matchDate') or prediction.get('比赛时间') or ''),
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
        'predicted_score': predicted_score,
        'predicted_over_under': predicted_ou,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'actual_result': actual_result,
        'actual_result_label': labels[actual_result],
        'result_hit': int(str(prediction.get('胜平负首选') or '') == labels[actual_result]),
        'score_hit': int(predicted_score == f'{home_goals}-{away_goals}'),
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


def _features(data: pd.DataFrame) -> pd.DataFrame:
    probability = _market_probabilities(data)
    inverse = 1.0 / data[['odds_home', 'odds_draw', 'odds_away']].to_numpy(
        dtype=np.float64,
    )
    ordered = np.sort(probability, axis=1)
    return pd.DataFrame({
        'p_home': probability[:, 0],
        'p_draw': probability[:, 1],
        'p_away': probability[:, 2],
        'overround': inverse.sum(axis=1),
        'entropy': -np.sum(
            probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1,
        ),
        'favorite_gap': ordered[:, -1] - ordered[:, -2],
        'home_away_gap': probability[:, 0] - probability[:, 2],
        'league': data['league'].fillna('未知联赛').astype(str),
    })


def _build_model(c: float) -> Pipeline:
    transform = ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC_FEATURES),
        ('league', OneHotEncoder(handle_unknown='ignore'), ['league']),
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


def predict_generic_probabilities(league: str, odds: dict) -> Optional[np.ndarray]:
    artifact = load_generic_artifact()
    if artifact is None:
        return None
    row = pd.DataFrame([{
        'league': league or '未知联赛',
        'odds_home': odds['H'], 'odds_draw': odds['D'], 'odds_away': odds['A'],
    }])
    try:
        learned = _ordered_probability(artifact['model'], _features(row))[0]
        market = _market_probabilities(row)[0]
        weight = float(artifact['model_weight'])
        result = weight * learned + (1.0 - weight) * market
        return result / result.sum()
    except Exception:
        logging.exception('每日通用模型预测失败，回退市场基线。')
        return None


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
    if last_attempt >= MIN_TRAIN_ROWS and samples < last_attempt + RETRAIN_STEP:
        return {
            'model_status': previous_status.get('model_status', '等待下一批样本'),
            'model_samples': samples,
            'minimum_training_samples': MIN_TRAIN_ROWS,
            'next_training_at': last_attempt + RETRAIN_STEP,
            'last_training_attempt_rows': last_attempt,
        }

    validation_rows = _holdout_rows(samples, VALIDATION_ROWS)
    test_rows = _holdout_rows(samples, TEST_ROWS)
    train_end = samples - validation_rows - test_rows
    validation_end = samples - test_rows
    train, validation, test = (
        valid.iloc[:train_end], valid.iloc[train_end:validation_end], valid.iloc[validation_end:],
    )
    y_train = train['actual_result'].to_numpy(dtype=np.int32)
    y_validation = validation['actual_result'].to_numpy(dtype=np.int32)
    y_test = test['actual_result'].to_numpy(dtype=np.int32)
    x_train, x_validation, x_test = _features(train), _features(validation), _features(test)
    validation_market = _market_probabilities(validation)
    test_market = _market_probabilities(test)

    selected, selected_rank = None, None
    candidates = []
    for c in (0.03, 0.1, 0.3, 1.0):
        model = _build_model(c)
        model.fit(x_train, y_train)
        learned = _ordered_probability(model, x_validation)
        for weight in np.arange(0.0, 0.501, 0.05):
            probability = weight * learned + (1.0 - weight) * validation_market
            row = {'C': c, 'model_weight': float(weight), **_metrics(y_validation, probability)}
            candidates.append(row)
            rank = (row['accuracy'], -row['log_loss'], -weight)
            if selected_rank is None or rank > selected_rank:
                selected_rank, selected = rank, row

    challenger = _build_model(float(selected['C']))
    combined = pd.concat([train, validation], ignore_index=True)
    challenger.fit(_features(combined), combined['actual_result'].to_numpy(dtype=np.int32))
    learned_test = _ordered_probability(challenger, x_test)
    weight = float(selected['model_weight'])
    challenger_probability = weight * learned_test + (1.0 - weight) * test_market
    challenger_metrics = _metrics(y_test, challenger_probability)
    market_metrics = _metrics(y_test, test_market)

    champion_metrics = None
    champion = load_generic_artifact()
    if champion is not None:
        champion_learned = _ordered_probability(champion['model'], x_test)
        champion_probability = (
            float(champion['model_weight']) * champion_learned
            + (1.0 - float(champion['model_weight'])) * test_market
        )
        champion_metrics = _metrics(y_test, champion_probability)

    market_gate = (
        weight > 0.0
        and challenger_metrics['accuracy'] >= market_metrics['accuracy']
        and (
            challenger_metrics['log_loss'] <= market_metrics['log_loss'] - 0.002
            or (
                challenger_metrics['accuracy'] >= market_metrics['accuracy'] + 0.01
                and challenger_metrics['log_loss'] <= market_metrics['log_loss'] + 0.02
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
    audit = {
        'attempted_at': datetime.now().isoformat(timespec='seconds'),
        'samples': samples,
        'split': {
            'train': len(train), 'validation': len(validation), 'test': len(test),
        },
        'selected_on_validation': selected,
        'validation_market': _metrics(y_validation, validation_market),
        'test_market': market_metrics,
        'test_challenger': challenger_metrics,
        'test_champion': champion_metrics,
        'passed': passed,
        'candidates': candidates,
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(AUDIT_ROOT / f'audit-{samples}-{date.today().isoformat()}.json', audit)

    if passed:
        final_model = _build_model(float(selected['C']))
        final_model.fit(_features(valid), valid['actual_result'].to_numpy(dtype=np.int32))
        artifact = {
            'model': final_model,
            'model_weight': weight,
            'deployable': True,
            'trained_rows': samples,
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'feature_version': 1,
            'report': audit,
        }
        GENERIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = GENERIC_MODEL_PATH.with_suffix('.joblib.tmp')
        joblib.dump(artifact, temporary)
        temporary.replace(GENERIC_MODEL_PATH)
        load_generic_artifact.cache_clear()
        model_status = '新冠军模型已通过并启用'
    else:
        model_status = '候选未胜过基线，保留旧模型'
    return {
        'model_status': model_status,
        'model_samples': samples,
        'minimum_training_samples': MIN_TRAIN_ROWS,
        'next_training_at': samples + RETRAIN_STEP,
        'last_training_attempt_rows': samples,
        'last_training_passed': passed,
        'challenger_test_accuracy': challenger_metrics['accuracy'],
        'market_test_accuracy': market_metrics['accuracy'],
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
        'review_error': '；'.join(errors),
        **training,
    }
    _write_json(STATUS_PATH, summary)
    return summary
