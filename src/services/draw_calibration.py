"""Leakage-safe draw calibration with chronological champion/challenger gates."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ARTIFACT_PATH = Path('storage/models/market/draw_calibrator.joblib')
AUDIT_PATH = Path('storage/jingcai/learning/draw_calibration_audit.json')
NUMERIC = [
    'p_home', 'p_draw', 'p_away', 'side_gap', 'favorite_gap',
    'league_draw_prior', 'home_recent_draw_rate', 'away_recent_draw_rate',
    'draw_flow', 'hhad_line_change', 'ttg_expected_change',
]
OPTIONAL_SOURCE_COLUMNS = {
    'draw_flow': 'draw_probability_change',
    'hhad_line_change': 'hhad_line_change',
    'ttg_expected_change': 'ttg_expected_change',
}


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _market(data: pd.DataFrame) -> np.ndarray:
    odds = data[['odds_home', 'odds_draw', 'odds_away']].to_numpy(float)
    inverse = 1.0 / odds
    return inverse / inverse.sum(axis=1, keepdims=True)


def _prepare(data: pd.DataFrame, prior_strength: int) -> tuple[pd.DataFrame, dict]:
    frame = data.sort_values(['match_date', 'prediction_date'], kind='stable').reset_index(drop=True)
    market = _market(frame)
    global_games = global_draws = 0
    league_games, league_draws = defaultdict(int), defaultdict(int)
    team_games, team_draws = defaultdict(int), defaultdict(int)
    rows = []
    for index, row in frame.iterrows():
        league = str(row.get('league') or '未知联赛')
        home, away = str(row.get('home') or ''), str(row.get('away') or '')
        global_prior = (global_draws + 27.0) / (global_games + 100.0)
        league_prior = (
            league_draws[league] + global_prior * prior_strength
        ) / (league_games[league] + prior_strength)
        home_rate = (team_draws[home] + global_prior * 20) / (team_games[home] + 20)
        away_rate = (team_draws[away] + global_prior * 20) / (team_games[away] + 20)
        probability = market[index]
        ordered = np.sort(probability)
        values = {
            'p_home': probability[0], 'p_draw': probability[1], 'p_away': probability[2],
            'side_gap': abs(probability[0] - probability[2]),
            'favorite_gap': ordered[-1] - ordered[-2],
            'league_draw_prior': league_prior,
            'home_recent_draw_rate': home_rate, 'away_recent_draw_rate': away_rate,
            'league': league,
        }
        for feature, source in OPTIONAL_SOURCE_COLUMNS.items():
            try:
                value = float(row.get(source))
                values[feature] = value if np.isfinite(value) else 0.0
            except (TypeError, ValueError):
                values[feature] = 0.0
        rows.append(values)
        target = int(row['actual_result'])
        is_draw = int(target == 1)
        global_games += 1
        global_draws += is_draw
        league_games[league] += 1
        league_draws[league] += is_draw
        for team in (home, away):
            if team:
                team_games[team] += 1
                team_draws[team] += is_draw
    state = {
        'global_games': global_games, 'global_draws': global_draws,
        'league_games': dict(league_games), 'league_draws': dict(league_draws),
        'team_games': dict(team_games), 'team_draws': dict(team_draws),
        'prior_strength': prior_strength,
    }
    return pd.DataFrame(rows), state


def _model(c: float) -> Pipeline:
    return Pipeline([
        ('features', ColumnTransformer([
            ('numeric', StandardScaler(), NUMERIC),
            ('league', OneHotEncoder(handle_unknown='ignore'), ['league']),
        ])),
        ('classifier', LogisticRegression(C=c, max_iter=2000, random_state=42)),
    ])


def _apply(base: np.ndarray, draw_probability: np.ndarray, strength: float) -> np.ndarray:
    result = np.asarray(base, dtype=float).copy()
    target = np.clip(
        (1.0 - strength) * result[:, 1] + strength * draw_probability,
        0.10, 0.45,
    )
    side_sum = result[:, 0] + result[:, 2]
    result[:, 0] = result[:, 0] / side_sum * (1.0 - target)
    result[:, 2] = result[:, 2] / side_sum * (1.0 - target)
    result[:, 1] = target
    return result


def _old(base: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    result = base.copy()
    target = 0.55 * base[:, 1] + 0.45 * features['league_draw_prior'].to_numpy(float)
    gap = features['side_gap'].to_numpy(float)
    target = np.where(gap <= 0.10, target * 1.06, np.where(gap >= 0.30, target * 0.94, target))
    return _apply(result, target, 0.25)


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(axis=1)
    actual_draw = target == 1
    predicted_draw = prediction == 1
    one_hot = np.eye(3)[target]
    return {
        'samples': int(len(target)),
        'accuracy': float(np.mean(prediction == target)),
        'brier': float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        'log_loss': float(log_loss(target, probability, labels=[0, 1, 2])),
        'draw_precision': float(precision_score(actual_draw, predicted_draw, zero_division=0)),
        'draw_recall': float(recall_score(actual_draw, predicted_draw, zero_division=0)),
        'draw_predictions': int(predicted_draw.sum()),
    }


def train_draw_calibrator(data: pd.DataFrame) -> dict:
    required = {'match_date', 'prediction_date', 'league', 'home', 'away',
                'odds_home', 'odds_draw', 'odds_away', 'actual_result'}
    valid = data.dropna(subset=list(required)).copy() if required.issubset(data.columns) else pd.DataFrame()
    if len(valid) < 800:
        return {'status': '样本不足', 'samples': len(valid), 'deployable': False}
    valid = valid.sort_values(['match_date', 'prediction_date'], kind='stable').reset_index(drop=True)
    test_rows = min(500, max(160, int(len(valid) * 0.15)))
    test_start = len(valid) - test_rows
    validation_rows = min(400, max(160, int(test_start * 0.12)))
    folds = [
        (test_start - validation_rows * offset, test_start - validation_rows * (offset - 1))
        for offset in (3, 2, 1)
        if test_start - validation_rows * offset >= 500
    ]
    candidates, best = [], None
    for prior_strength in (30, 100, 300):
        features, _ = _prepare(valid, prior_strength)
        for c in (0.01, 0.03, 0.10, 0.30):
            learned_parts, targets, bases = [], [], []
            try:
                for start, end in folds:
                    estimator = _model(c)
                    y_train = (valid.iloc[:start]['actual_result'].to_numpy(int) == 1).astype(int)
                    estimator.fit(features.iloc[:start], y_train)
                    learned_parts.append(estimator.predict_proba(features.iloc[start:end])[:, 1])
                    targets.append(valid.iloc[start:end]['actual_result'].to_numpy(int))
                    bases.append(_market(valid.iloc[start:end]))
            except ValueError:
                continue
            learned, target, base = np.concatenate(learned_parts), np.concatenate(targets), np.vstack(bases)
            for strength in (0.15, 0.25, 0.35, 0.50, 0.70):
                metric = _metrics(target, _apply(base, learned, strength))
                row = {'prior_strength': prior_strength, 'C': c, 'strength': strength, **metric}
                candidates.append(row)
                rank = (metric['accuracy'], -metric['log_loss'], -metric['brier'])
                if best is None or rank > best[0]:
                    best = (rank, row)
    if best is None:
        return {'status': '候选训练失败', 'samples': len(valid), 'deployable': False}
    selected = best[1]
    features, state = _prepare(valid, selected['prior_strength'])
    # Learn league-specific calibration strength on the last pre-test block.
    # Small leagues inherit the global strength instead of fitting noisy knobs.
    league_strengths = {}
    strength_train_end = max(500, test_start - validation_rows)
    strength_model = _model(selected['C'])
    strength_model.fit(
        features.iloc[:strength_train_end],
        (valid.iloc[:strength_train_end]['actual_result'].to_numpy(int) == 1).astype(int),
    )
    strength_slice = valid.iloc[strength_train_end:test_start]
    strength_learned = strength_model.predict_proba(
        features.iloc[strength_train_end:test_start],
    )[:, 1]
    strength_base = _market(strength_slice)
    for league, indexes in strength_slice.groupby('league').groups.items():
        positions = strength_slice.index.get_indexer(indexes)
        if len(positions) < 50:
            continue
        target_part = strength_slice.loc[indexes, 'actual_result'].to_numpy(int)
        selected_strength, selected_rank = selected['strength'], None
        for strength in (0.15, 0.25, 0.35, 0.50, 0.70):
            metric = _metrics(
                target_part,
                _apply(strength_base[positions], strength_learned[positions], strength),
            )
            rank = (metric['accuracy'], -metric['log_loss'], -metric['brier'])
            if selected_rank is None or rank > selected_rank:
                selected_strength, selected_rank = strength, rank
        league_strengths[str(league)] = selected_strength
    pretest = valid.iloc[:test_start]
    estimator = _model(selected['C'])
    estimator.fit(features.iloc[:test_start], (pretest['actual_result'].to_numpy(int) == 1).astype(int))
    learned = estimator.predict_proba(features.iloc[test_start:])[:, 1]
    target = valid.iloc[test_start:]['actual_result'].to_numpy(int)
    base = _market(valid.iloc[test_start:])
    test_strength = np.asarray([
        league_strengths.get(str(league), selected['strength'])
        for league in valid.iloc[test_start:]['league']
    ], dtype=float)
    challenger = _metrics(target, _apply(base, learned, test_strength))
    incumbent = _metrics(target, _old(base, features.iloc[test_start:]))
    passed = bool(
        challenger['accuracy'] >= incumbent['accuracy']
        and challenger['log_loss'] <= incumbent['log_loss']
        and challenger['brier'] <= incumbent['brier']
        and challenger['draw_precision'] >= incumbent['draw_precision']
        and (
            challenger['accuracy'] > incumbent['accuracy']
            or challenger['log_loss'] < incumbent['log_loss'] - 0.001
        )
    )
    audit = {
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'samples': len(valid), 'test_samples': test_rows,
        'candidate': selected, 'league_strengths': league_strengths,
        'challenger': challenger, 'incumbent': incumbent,
        'deployable': passed,
        'status': '新版时间外测试胜出并启用' if passed else '新版未全面胜出，保留旧校准',
    }
    _write_json(AUDIT_PATH, audit)
    if passed:
        final_model = _model(selected['C'])
        final_model.fit(features, (valid['actual_result'].to_numpy(int) == 1).astype(int))
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': final_model, 'state': state,
            'strength': selected['strength'], 'league_strengths': league_strengths,
            'audit': audit,
        }, ARTIFACT_PATH)
        load_draw_artifact.cache_clear()
    return audit


@lru_cache(maxsize=1)
def load_draw_artifact():
    try:
        artifact = joblib.load(ARTIFACT_PATH)
        return artifact if artifact.get('audit', {}).get('deployable') else None
    except Exception:
        return None


def calibrate_draw(base_probability, league, home, away, market_probability,
                   draw_flow=0.0, hhad_line_change=0.0, ttg_expected_change=0.0):
    artifact = load_draw_artifact()
    if artifact is None:
        return None
    state = artifact['state']
    global_prior = (state['global_draws'] + 27.0) / (state['global_games'] + 100.0)
    league_games = int(state['league_games'].get(league, 0))
    league_draws = int(state['league_draws'].get(league, 0))
    prior_strength = int(state['prior_strength'])
    league_prior = (league_draws + global_prior * prior_strength) / (league_games + prior_strength)
    def team_rate(team):
        games = int(state['team_games'].get(team, 0))
        draws = int(state['team_draws'].get(team, 0))
        return (draws + global_prior * 20) / (games + 20)
    market = np.asarray(market_probability, dtype=float)
    ordered = np.sort(market)
    features = pd.DataFrame([{
        'p_home': market[0], 'p_draw': market[1], 'p_away': market[2],
        'side_gap': abs(market[0] - market[2]), 'favorite_gap': ordered[-1] - ordered[-2],
        'league_draw_prior': league_prior,
        'home_recent_draw_rate': team_rate(home), 'away_recent_draw_rate': team_rate(away),
        'draw_flow': float(draw_flow or 0),
        'hhad_line_change': float(hhad_line_change or 0),
        'ttg_expected_change': float(ttg_expected_change or 0), 'league': league,
    }])
    learned = artifact['model'].predict_proba(features)[:, 1]
    strength = artifact.get('league_strengths', {}).get(league, artifact['strength'])
    return _apply(np.asarray(base_probability, float).reshape(1, 3), learned,
                  float(strength))[0]
