"""Tune a leakage-resistant UCL score/total-goals correction model.

The official 1X2 market is first converted into neutral Poisson scoring rates.
Candidate models then learn only a conservative correction to those rates.  All
selection is chronological; the 2022 and 2024 seasons are a locked final audit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import poisson, skellam
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COMPETITION = 'europe/champions-league'
EURO_CUPS = {'europe/champions-league', 'europe/europa-league'}
FINAL_TEST_SEASONS = {2022, 2024}
ODDS_COLUMNS = ['Cote 1', 'Cote X', 'Cote 2']


def market_probability(frame: pd.DataFrame) -> np.ndarray:
    odds = frame[ODDS_COLUMNS].to_numpy(dtype=np.float64)
    inverse = 1.0 / odds
    return inverse / inverse.sum(axis=1, keepdims=True)


def clean_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = [*ODDS_COLUMNS, 'Score Domicile', 'Score Extérieur']
    data = data.dropna(subset=required).copy()
    odds = data[ODDS_COLUMNS].to_numpy(dtype=np.float64)
    scores = data[['Score Domicile', 'Score Extérieur']].to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
        & np.isfinite(scores).all(axis=1) & (scores >= 0).all(axis=1)
    )
    data = data.loc[valid].copy()
    data['season_start'] = data['Saison'].astype(str).str[:4].astype(int)
    date_and_stage = data['Date'].astype(str).str.split(' - ', n=1, expand=True)
    data['match_date'] = pd.to_datetime(
        date_and_stage[0].str.strip(), format='%d %b %Y', errors='coerce',
    )
    missing = data['match_date'].isna()
    data.loc[missing, 'match_date'] = pd.to_datetime(
        date_and_stage.loc[missing, 0].str.strip(), errors='coerce',
    )
    data['stage'] = date_and_stage[1].fillna('unknown').str.strip()
    return data.dropna(subset=['match_date']).reset_index(drop=True)


def fit_market_lambdas(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized nearest-Poisson fit to de-vigged H/D/A probabilities."""
    values = np.arange(0.20, 4.201, 0.05, dtype=np.float64)
    home_grid, away_grid = np.meshgrid(values, values, indexing='ij')
    home_flat, away_flat = home_grid.ravel(), away_grid.ravel()
    draw = skellam.pmf(0, home_flat, away_flat)
    away_win = skellam.cdf(-1, home_flat, away_flat)
    home_win = 1.0 - skellam.cdf(0, home_flat, away_flat)
    grid = np.column_stack([home_win, draw, away_win])
    home_result = np.empty(len(probabilities), dtype=np.float64)
    away_result = np.empty(len(probabilities), dtype=np.float64)
    for start in range(0, len(probabilities), 256):
        stop = min(len(probabilities), start + 256)
        distance = np.square(
            probabilities[start:stop, None, :] - grid[None, :, :],
        ).sum(axis=2)
        best = distance.argmin(axis=1)
        home_result[start:stop] = home_flat[best]
        away_result[start:stop] = away_flat[best]
    return home_result, away_result


def add_market_lambdas(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    home, away = fit_market_lambdas(market_probability(result))
    result['market_home_goals'] = home
    result['market_away_goals'] = away
    return result


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    odds = frame[ODDS_COLUMNS].to_numpy(dtype=np.float64)
    probability = market_probability(frame)
    lambdas = frame[['market_home_goals', 'market_away_goals']].to_numpy(
        dtype=np.float64,
    )
    stage = frame['stage'].str.casefold()
    month_angle = 2.0 * np.pi * frame['match_date'].dt.month.to_numpy() / 12.0
    return np.column_stack([
        probability,
        np.log(odds),
        lambdas,
        np.log(lambdas),
        lambdas[:, 0] - lambdas[:, 1],
        lambdas[:, 0] + lambdas[:, 1],
        probability[:, 0] - probability[:, 2],
        probability[:, 1] ** 2,
        np.sin(month_angle),
        np.cos(month_angle),
        stage.str.contains('qualif').to_numpy(dtype=np.float64),
        stage.str.contains('group').to_numpy(dtype=np.float64),
        stage.str.contains('play off').to_numpy(dtype=np.float64),
        stage.str.contains('league phase').to_numpy(dtype=np.float64),
    ])


@dataclass(frozen=True)
class Candidate:
    name: str
    scope: str
    kind: str
    params: dict

    def build(self):
        if self.kind == 'poisson':
            return make_pipeline(
                StandardScaler(),
                PoissonRegressor(alpha=self.params['alpha'], max_iter=2000),
            )
        return HistGradientBoostingRegressor(
            loss='poisson',
            learning_rate=0.035,
            max_iter=180,
            max_leaf_nodes=self.params['max_leaf_nodes'],
            min_samples_leaf=self.params['min_samples_leaf'],
            l2_regularization=6.0,
            random_state=42,
        )


def candidates() -> list[Candidate]:
    result = []
    for scope in ('ucl', 'euro_cups'):
        for alpha in (0.03, 0.1, 0.3, 1.0):
            result.append(Candidate(
                f'{scope}-poisson-A{alpha}', scope, 'poisson', {'alpha': alpha},
            ))
        for nodes, leaf in ((7, 60), (7, 120), (15, 120)):
            result.append(Candidate(
                f'{scope}-hist-L{nodes}-N{leaf}', scope, 'hist',
                {'max_leaf_nodes': nodes, 'min_samples_leaf': leaf},
            ))
    return result


def scope_mask(data: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == 'ucl':
        return data['Championat'].eq(TARGET_COMPETITION).to_numpy()
    return data['Championat'].isin(EURO_CUPS).to_numpy()


def prediction_metrics(frame: pd.DataFrame, home: np.ndarray, away: np.ndarray) -> dict:
    home = np.clip(np.asarray(home, dtype=np.float64), 0.08, 5.5)
    away = np.clip(np.asarray(away, dtype=np.float64), 0.08, 5.5)
    observed_home = frame['Score Domicile'].to_numpy(dtype=np.int32)
    observed_away = frame['Score Extérieur'].to_numpy(dtype=np.int32)
    score_nll = (
        home - observed_home * np.log(home) + gammaln(observed_home + 1)
        + away - observed_away * np.log(away) + gammaln(observed_away + 1)
    )
    predicted_home = np.floor(home).astype(np.int32)
    predicted_away = np.floor(away).astype(np.int32)
    over_probability = 1.0 - poisson.cdf(2, home + away)
    actual_over = observed_home + observed_away > 2
    predicted_over = over_probability >= 0.5
    result_probability = np.column_stack([
        1.0 - skellam.cdf(0, home, away),
        skellam.pmf(0, home, away),
        skellam.cdf(-1, home, away),
    ])
    actual_result = np.where(
        observed_home > observed_away, 0,
        np.where(observed_home == observed_away, 1, 2),
    )
    return {
        'samples': int(len(frame)),
        'score_nll': float(score_nll.mean()),
        'score_top1_accuracy': float(np.mean(
            (predicted_home == observed_home) & (predicted_away == observed_away),
        )),
        'over_under_accuracy': float(np.mean(predicted_over == actual_over)),
        'result_accuracy': float(np.mean(result_probability.argmax(axis=1) == actual_result)),
        'goal_mae': float(np.mean(
            np.abs(home - observed_home) + np.abs(away - observed_away),
        ) / 2.0),
    }


def fit_predict(
        candidate: Candidate,
        data: pd.DataFrame,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = train_mask & scope_mask(data, candidate.scope)
    features = feature_matrix(data)
    home_model, away_model = candidate.build(), candidate.build()
    home_model.fit(features[selected], data.loc[selected, 'Score Domicile'])
    away_model.fit(features[selected], data.loc[selected, 'Score Extérieur'])
    return (
        home_model.predict(features[test_mask]),
        away_model.predict(features[test_mask]),
    )


def weighted(rows: list[dict], metric: str) -> float:
    samples = sum(row['samples'] for row in rows)
    return sum(row[metric] * row['samples'] for row in rows) / samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    data = clean_data(args.data)
    data = data[data['Championat'].isin(EURO_CUPS)].reset_index(drop=True)
    data = add_market_lambdas(data)
    target = data['Championat'].eq(TARGET_COMPETITION).to_numpy()
    validation_seasons = [2017, 2018, 2019, 2020, 2021]
    final_test = target & data['season_start'].isin(FINAL_TEST_SEASONS).to_numpy()
    final_frame = data.loc[final_test]
    market_final = prediction_metrics(
        final_frame,
        final_frame['market_home_goals'].to_numpy(),
        final_frame['market_away_goals'].to_numpy(),
    )
    report = {
        'data_source': 'Kaggle rayenjlassi/more-than-20k-footballsoccer-match v7',
        'license': 'CC BY-NC-SA 4.0',
        'validation_seasons': validation_seasons,
        'final_test_seasons': sorted(FINAL_TEST_SEASONS),
        'market_final': market_final,
        'candidates': [],
    }
    print('UCL rows', int(target.sum()), 'final test', int(final_test.sum()))
    print('market final', market_final)

    best, best_rank = None, None
    for candidate in candidates():
        fold_rows = []
        for season in validation_seasons:
            fold_test = target & data['season_start'].eq(season).to_numpy()
            fold_train = data['season_start'].lt(season).to_numpy()
            if not fold_test.any():
                continue
            home, away = fit_predict(candidate, data, fold_train, fold_test)
            row = prediction_metrics(data.loc[fold_test], home, away)
            baseline = prediction_metrics(
                data.loc[fold_test],
                data.loc[fold_test, 'market_home_goals'].to_numpy(),
                data.loc[fold_test, 'market_away_goals'].to_numpy(),
            )
            row['season'] = season
            row['market_score_nll'] = baseline['score_nll']
            row['market_over_under_accuracy'] = baseline['over_under_accuracy']
            row['market_score_top1_accuracy'] = baseline['score_top1_accuracy']
            fold_rows.append(row)
        final_home, final_away = fit_predict(
            candidate, data,
            data['season_start'].lt(min(FINAL_TEST_SEASONS)).to_numpy(),
            final_test,
        )
        row = {
            'name': candidate.name,
            'scope': candidate.scope,
            'kind': candidate.kind,
            'params': candidate.params,
            'rolling_score_nll': weighted(fold_rows, 'score_nll'),
            'rolling_market_score_nll': weighted(fold_rows, 'market_score_nll'),
            'rolling_over_under_accuracy': weighted(fold_rows, 'over_under_accuracy'),
            'rolling_market_over_under_accuracy': weighted(
                fold_rows, 'market_over_under_accuracy',
            ),
            'rolling_score_top1_accuracy': weighted(fold_rows, 'score_top1_accuracy'),
            'rolling_market_score_top1_accuracy': weighted(
                fold_rows, 'market_score_top1_accuracy',
            ),
            'folds': fold_rows,
            'final_test': prediction_metrics(final_frame, final_home, final_away),
        }
        report['candidates'].append(row)
        rank = (-row['rolling_score_nll'], row['rolling_over_under_accuracy'])
        if best_rank is None or rank > best_rank:
            best_rank, best = rank, candidate
        print(
            candidate.name,
            f"nll={row['rolling_score_nll']:.4f}",
            f"market={row['rolling_market_score_nll']:.4f}",
            f"final={row['final_test']['score_nll']:.4f}",
        )

    cached_folds = []
    for season in validation_seasons:
        fold_test = target & data['season_start'].eq(season).to_numpy()
        if not fold_test.any():
            continue
        learned = fit_predict(
            best, data, data['season_start'].lt(season).to_numpy(), fold_test,
        )
        cached_folds.append((data.loc[fold_test], *learned))
    blend_rows = []
    for weight in np.arange(0.0, 1.001, 0.05):
        rows = []
        for frame, learned_home, learned_away in cached_folds:
            # Geometric blending preserves positive rates and behaves like a
            # conservative correction on the Poisson log scale.
            home = np.exp(
                weight * np.log(np.clip(learned_home, 0.08, None))
                + (1.0 - weight) * np.log(frame['market_home_goals']),
            )
            away = np.exp(
                weight * np.log(np.clip(learned_away, 0.08, None))
                + (1.0 - weight) * np.log(frame['market_away_goals']),
            )
            rows.append(prediction_metrics(frame, home, away))
        blend_rows.append({
            'model_weight': float(weight),
            'score_nll': weighted(rows, 'score_nll'),
            'over_under_accuracy': weighted(rows, 'over_under_accuracy'),
            'score_top1_accuracy': weighted(rows, 'score_top1_accuracy'),
        })
    selected_blend = min(
        blend_rows, key=lambda row: (row['score_nll'], -row['over_under_accuracy']),
    )
    train_all = data['season_start'].lt(min(FINAL_TEST_SEASONS)).to_numpy()
    learned_home, learned_away = fit_predict(best, data, train_all, final_test)
    weight = selected_blend['model_weight']
    final_home = np.exp(
        weight * np.log(np.clip(learned_home, 0.08, None))
        + (1.0 - weight) * np.log(final_frame['market_home_goals']),
    )
    final_away = np.exp(
        weight * np.log(np.clip(learned_away, 0.08, None))
        + (1.0 - weight) * np.log(final_frame['market_away_goals']),
    )
    blended_final = prediction_metrics(final_frame, final_home, final_away)
    non_losing_folds = sum(
        row['score_nll'] <= row['market_score_nll']
        for row in next(item for item in report['candidates'] if item['name'] == best.name)['folds']
    )
    passed = bool(
        weight > 0.0
        and selected_blend['score_nll']
        <= report['candidates'][0]['rolling_market_score_nll'] - 0.005
        and non_losing_folds >= 3
        and blended_final['score_nll'] <= market_final['score_nll'] - 0.002
        and blended_final['over_under_accuracy'] >= market_final['over_under_accuracy']
        and blended_final['score_top1_accuracy']
        >= market_final['score_top1_accuracy'] - 0.01
    )
    report.update({
        'selected': best.name,
        'blend_search': blend_rows,
        'selected_blend': selected_blend,
        'blended_final': blended_final,
        'deployment_gate': {
            'passed': passed,
            'non_losing_folds': int(non_losing_folds),
            'requires_rolling_nll_gain': 0.005,
            'requires_final_nll_gain': 0.002,
            'requires_final_ou_not_below_market': True,
        },
    })

    selected_scope = scope_mask(data, best.scope)
    features = feature_matrix(data)
    final_home_model, final_away_model = best.build(), best.build()
    final_home_model.fit(features[selected_scope], data.loc[selected_scope, 'Score Domicile'])
    final_away_model.fit(features[selected_scope], data.loc[selected_scope, 'Score Extérieur'])
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = {
        'home_model': final_home_model,
        'away_model': final_away_model,
        'model_weight': weight,
        'feature_version': 1,
        'competition': TARGET_COMPETITION,
        'trained_rows': int(selected_scope.sum()),
        'deployable': passed,
        'report': report,
    }
    joblib.dump(artifact, args.output / 'champions_league_goals.joblib')
    (args.output / 'goal_training_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('SELECTED', best.name, selected_blend)
    print('FINAL', blended_final, 'DEPLOYABLE', passed)


if __name__ == '__main__':
    main()
