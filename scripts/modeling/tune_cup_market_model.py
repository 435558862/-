"""Tune an odds-only cup fallback with chronological, competition-aware tests."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COMPETITION = 'europe/champions-league'
EURO_CUPS = {'europe/champions-league', 'europe/europa-league'}
RESULT_MAP = {1.0: 0, 0.0: 1, 2.0: 2}
FINAL_TEST_SEASONS = {2022, 2024}
HOME_TEAM = 'Équipe Domicile'
AWAY_TEAM = 'Équipe Extérieur'

# Only deterministic, manually verified aliases are allowed.  The historical
# Elo file covers 19 European countries, so an absent club is left missing
# rather than being assigned to the closest-looking (and possibly wrong) club.
ELO_ALIASES = {
    'ac milan': 'Milan',
    'as roma': 'Roma',
    'atl madrid': 'Ath Madrid',
    'aek athens': 'AEK',
    'b monchengladbach': 'MGladbach',
    'bayer leverkusen': 'Leverkusen',
    'club brugge kv': 'Club Brugge',
    'cska moscow': 'CSKA Moskva',
    'dep la coruna': 'La Coruna',
    'eintracht frankfurt': 'Ein Frankfurt',
    'fc copenhagen': 'FC Kobenhavn',
    'fc porto': 'Porto',
    'fcsb': 'Steaua',
    'hamburger': 'Hamburg',
    'lokomotiv moscow': 'Lok Moskva',
    'malmo ff': 'Malmoe',
    'manchester city': 'Man City',
    'manchester utd': 'Man United',
    'olympiacos piraeus': 'Olympiakos',
    'psg': 'Paris SG',
    'psv': 'PSV Eindhoven',
    'schalke': 'Schalke 04',
    'sk rapid': 'Rapid Wien',
    'spartak moscow': 'Spartak Moskva',
    'sporting cp': 'Sp Lisbon',
    'tampere utd': 'Tampere',
}


def normalize_team(value: str) -> str:
    value = unicodedata.normalize('NFKD', str(value))
    value = ''.join(char for char in value if not unicodedata.combining(char))
    value = value.casefold().replace('&', ' and ')
    value = re.sub(r'\([^)]{2,4}\)\s*$', '', value)
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    return ' '.join(value.split())


def parse_season(value: str) -> int:
    return int(str(value).split('-', 1)[0])


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    odds = frame[['Cote 1', 'Cote X', 'Cote 2']].to_numpy(dtype=np.float64)
    inverse = 1.0 / odds
    probability = inverse / inverse.sum(axis=1, keepdims=True)
    ordered = np.sort(probability, axis=1)
    entropy = -np.sum(
        probability * np.log(np.clip(probability, 1e-12, 1.0)),
        axis=1,
        keepdims=True,
    )
    month_angle = 2.0 * np.pi * frame['match_date'].dt.month.to_numpy() / 12.0
    stage = frame['stage'].str.casefold()
    stage_features = np.column_stack([
        stage.str.contains('qualif').to_numpy(dtype=np.float64),
        stage.str.contains('group').to_numpy(dtype=np.float64),
        stage.str.contains('play off').to_numpy(dtype=np.float64),
        stage.str.contains('league phase').to_numpy(dtype=np.float64),
    ])
    elo_features = np.column_stack([
        (frame['home_elo'].to_numpy(dtype=np.float64) - 1500.0) / 400.0,
        (frame['away_elo'].to_numpy(dtype=np.float64) - 1500.0) / 400.0,
        frame['elo_difference'].to_numpy(dtype=np.float64) / 400.0,
        frame['elo_missing'].to_numpy(dtype=np.float64),
    ])
    return np.hstack([
        probability,
        np.log(odds),
        inverse.sum(axis=1, keepdims=True),
        entropy,
        (ordered[:, -1] - ordered[:, -2]).reshape(-1, 1),
        probability ** 2,
        (probability[:, 0] - probability[:, 2]).reshape(-1, 1),
        np.sin(month_angle).reshape(-1, 1),
        np.cos(month_angle).reshape(-1, 1),
        stage_features,
        elo_features,
    ])


def market_probability(frame: pd.DataFrame) -> np.ndarray:
    odds = frame[['Cote 1', 'Cote X', 'Cote 2']].to_numpy(dtype=np.float64)
    inverse = 1.0 / odds
    return inverse / inverse.sum(axis=1, keepdims=True)


def clean_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    odds_columns = ['Cote 1', 'Cote X', 'Cote 2']
    data = data.dropna(subset=[*odds_columns, 'Gangnant']).copy()
    odds = data[odds_columns].to_numpy(dtype=np.float64)
    valid = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    data = data.loc[valid & data['Gangnant'].isin(RESULT_MAP)].copy()
    data['season_start'] = data['Saison'].map(parse_season)
    data['target'] = data['Gangnant'].map(RESULT_MAP).astype(int)
    date_and_stage = data['Date'].astype(str).str.split(' - ', n=1, expand=True)
    data['match_date'] = pd.to_datetime(
        date_and_stage[0].str.strip(), format='%d %b %Y', errors='coerce',
    )
    # The source mixes abbreviated and full English month names. Pandas accepts
    # both with %b; the generic parser is a safe fallback for any odd row.
    missing_date = data['match_date'].isna()
    data.loc[missing_date, 'match_date'] = pd.to_datetime(
        date_and_stage.loc[missing_date, 0].str.strip(), errors='coerce',
    )
    data['stage'] = date_and_stage[1].fillna('unknown').str.strip()
    data = data.dropna(subset=['match_date']).copy()
    data['home_elo'] = 1500.0
    data['away_elo'] = 1500.0
    data['elo_difference'] = 0.0
    data['elo_missing'] = 1.0
    return data.reset_index(drop=True)


def add_historical_elo(data: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict]:
    snapshots = pd.read_csv(path, usecols=['date', 'club', 'elo']).dropna().copy()
    snapshots['date'] = pd.to_datetime(snapshots['date'], errors='coerce')
    snapshots = snapshots.dropna(subset=['date'])
    exact_names: dict[str, str] = {}
    duplicates: set[str] = set()
    for club in snapshots['club'].unique():
        key = normalize_team(club)
        if key in exact_names and exact_names[key] != club:
            duplicates.add(key)
        else:
            exact_names[key] = club
    for key in duplicates:
        exact_names.pop(key, None)

    available = set(snapshots['club'])
    aliases = {
        key: club for key, club in ELO_ALIASES.items() if club in available
    }

    def resolve(team: str) -> str | None:
        key = normalize_team(team)
        return exact_names.get(key) or aliases.get(key)

    home_club = data[HOME_TEAM].map(resolve)
    away_club = data[AWAY_TEAM].map(resolve)
    by_club = {
        club: (
            group.sort_values('date')['date'].to_numpy(dtype='datetime64[ns]'),
            group.sort_values('date')['elo'].to_numpy(dtype=np.float64),
        )
        for club, group in snapshots.groupby('club', sort=False)
    }

    # A two-day lag prevents a same-day result from leaking into its own input.
    cutoff_dates = (data['match_date'] - pd.Timedelta(days=2)).to_numpy(
        dtype='datetime64[ns]',
    )

    def lookup(clubs: pd.Series) -> np.ndarray:
        result = np.full(len(data), np.nan, dtype=np.float64)
        for index, club in enumerate(clubs):
            if club is None or club not in by_club:
                continue
            dates, ratings = by_club[club]
            position = int(np.searchsorted(dates, cutoff_dates[index], side='right') - 1)
            if position >= 0:
                result[index] = ratings[position]
        return result

    home_rating = lookup(home_club)
    away_rating = lookup(away_club)
    both = np.isfinite(home_rating) & np.isfinite(away_rating)
    enriched = data.copy()
    enriched['home_elo'] = np.where(np.isfinite(home_rating), home_rating, 1500.0)
    enriched['away_elo'] = np.where(np.isfinite(away_rating), away_rating, 1500.0)
    enriched['elo_difference'] = np.where(both, home_rating - away_rating, 0.0)
    enriched['elo_missing'] = (~both).astype(np.float64)
    target = enriched['Championat'].eq(TARGET_COMPETITION).to_numpy()
    coverage = {
        'all_both_ratings': int(both.sum()),
        'all_rows': int(len(enriched)),
        'ucl_both_ratings': int((both & target).sum()),
        'ucl_rows': int(target.sum()),
        'ucl_coverage': float((both & target).sum() / max(1, target.sum())),
        'mapping': 'normalized exact names plus manually verified aliases; no fuzzy matching',
        'lag_days': 2,
    }
    return enriched, coverage


@dataclass(frozen=True)
class Candidate:
    name: str
    scope: str
    kind: str
    params: dict

    def build(self):
        if self.kind == 'logistic':
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=self.params['C'], max_iter=2000,
                    class_weight=self.params.get('class_weight'),
                    random_state=42,
                ),
            )
        if self.kind == 'hist':
            return HistGradientBoostingClassifier(
                learning_rate=self.params['learning_rate'],
                max_iter=self.params['max_iter'],
                max_leaf_nodes=self.params['max_leaf_nodes'],
                min_samples_leaf=self.params['min_samples_leaf'],
                l2_regularization=self.params['l2_regularization'],
                random_state=42,
            )
        raise ValueError(self.kind)


def candidates() -> list[Candidate]:
    result = []
    # The broad all-competition scope was slower and less stable in the first
    # audit. Keep the search deliberately small to reduce hyperparameter
    # overfitting on a modest cup sample.
    for scope in ('ucl', 'euro_cups'):
        for c in (0.01, 0.1, 1.0):
            result.append(Candidate(
                name=f'{scope}-logistic-C{c}', scope=scope,
                kind='logistic', params={'C': c},
            ))
        for leaf_nodes in (7, 15):
            for min_leaf in (60, 120):
                result.append(Candidate(
                    name=f'{scope}-hist-L{leaf_nodes}-N{min_leaf}', scope=scope,
                    kind='hist', params={
                        'learning_rate': 0.035, 'max_iter': 180,
                        'max_leaf_nodes': leaf_nodes,
                        'min_samples_leaf': min_leaf,
                        'l2_regularization': 4.0,
                    },
                ))
    return result


def scope_mask(data: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == 'ucl':
        return data['Championat'].eq(TARGET_COMPETITION).to_numpy()
    if scope == 'euro_cups':
        return data['Championat'].isin(EURO_CUPS).to_numpy()
    return np.ones(len(data), dtype=bool)


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    return {
        'samples': int(len(y)),
        'accuracy': float(accuracy_score(y, probability.argmax(axis=1))),
        'log_loss': float(log_loss(y, probability, labels=[0, 1, 2])),
    }


def fit_predict(
        candidate: Candidate,
        data: pd.DataFrame,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
) -> np.ndarray:
    scoped_train = train_mask & scope_mask(data, candidate.scope)
    model = candidate.build()
    model.fit(feature_matrix(data.loc[scoped_train]), data.loc[scoped_train, 'target'])
    return model.predict_proba(feature_matrix(data.loc[test_mask]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--elo-data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    data = clean_data(args.data)
    elo_coverage = None
    if args.elo_data:
        data, elo_coverage = add_historical_elo(data, args.elo_data)
    target = data['Championat'].eq(TARGET_COMPETITION).to_numpy()
    validation_seasons = [2017, 2018, 2019, 2020, 2021]
    final_test = target & data['season_start'].isin(FINAL_TEST_SEASONS).to_numpy()
    final_market = market_probability(data.loc[final_test])
    y_final = data.loc[final_test, 'target'].to_numpy(dtype=np.int32)
    report = {
        'data_source': 'Kaggle rayenjlassi/more-than-20k-footballsoccer-match v7',
        'license': 'CC BY-NC-SA 4.0',
        'target_competition': TARGET_COMPETITION,
        'validation_seasons': validation_seasons,
        'final_test_seasons': sorted(FINAL_TEST_SEASONS),
        'market_final': metrics(y_final, final_market),
        'elo_coverage': elo_coverage,
        'candidates': [],
    }

    print('UCL rows', int(target.sum()), 'final test', int(final_test.sum()))
    print('market final', report['market_final'])
    best_candidate = None
    best_rank = None
    for candidate in candidates():
        fold_rows = []
        for season in validation_seasons:
            fold_test = target & data['season_start'].eq(season).to_numpy()
            fold_train = data['season_start'].lt(season).to_numpy()
            if fold_test.sum() == 0:
                continue
            probability = fit_predict(candidate, data, fold_train, fold_test)
            y = data.loc[fold_test, 'target'].to_numpy(dtype=np.int32)
            fold_rows.append({
                'season': season,
                **metrics(y, probability),
                'market_accuracy': metrics(y, market_probability(data.loc[fold_test]))['accuracy'],
            })
        total_samples = sum(row['samples'] for row in fold_rows)
        weighted_accuracy = sum(
            row['accuracy'] * row['samples'] for row in fold_rows
        ) / total_samples
        market_accuracy = sum(
            row['market_accuracy'] * row['samples'] for row in fold_rows
        ) / total_samples
        weighted_loss = sum(
            row['log_loss'] * row['samples'] for row in fold_rows
        ) / total_samples
        final_probability = fit_predict(
            candidate, data,
            data['season_start'].lt(min(FINAL_TEST_SEASONS)).to_numpy(),
            final_test,
        )
        final_metrics = metrics(y_final, final_probability)
        row = {
            'name': candidate.name,
            'scope': candidate.scope,
            'kind': candidate.kind,
            'params': candidate.params,
            'rolling_accuracy': weighted_accuracy,
            'rolling_market_accuracy': market_accuracy,
            'rolling_log_loss': weighted_loss,
            'folds': fold_rows,
            'final_test': final_metrics,
        }
        report['candidates'].append(row)
        # Rank by rolling accuracy first. Final test is a locked audit and is
        # deliberately not used to select hyperparameters.
        rank = (weighted_accuracy, -weighted_loss)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_candidate = candidate
        print(
            candidate.name,
            f'rolling={weighted_accuracy:.4f} market={market_accuracy:.4f}',
            f'final={final_metrics["accuracy"]:.4f}',
        )

    report['selected'] = best_candidate.name
    train_all = data['season_start'].lt(min(FINAL_TEST_SEASONS)).to_numpy()
    probability = fit_predict(best_candidate, data, train_all, final_test)

    # Select a conservative market/model blend using rolling validation only.
    blend_rows = []
    cached_folds = []
    for season in validation_seasons:
        fold_test = target & data['season_start'].eq(season).to_numpy()
        fold_train = data['season_start'].lt(season).to_numpy()
        if not fold_test.any():
            continue
        cached_folds.append((
            data.loc[fold_test, 'target'].to_numpy(dtype=np.int32),
            fit_predict(best_candidate, data, fold_train, fold_test),
            market_probability(data.loc[fold_test]),
        ))
    for weight in np.arange(0.0, 1.001, 0.05):
        fold_scores = []
        for y, learned, market in cached_folds:
            blended = weight * learned + (1.0 - weight) * market
            fold_scores.append(metrics(y, blended))
        n = sum(row['samples'] for row in fold_scores)
        blend_rows.append({
            'model_weight': float(weight),
            'accuracy': sum(row['accuracy'] * row['samples'] for row in fold_scores) / n,
            'log_loss': sum(row['log_loss'] * row['samples'] for row in fold_scores) / n,
        })
    selected_blend = max(
        blend_rows, key=lambda row: (row['accuracy'], -row['log_loss']),
    )
    final_blended = (
        selected_blend['model_weight'] * probability
        + (1.0 - selected_blend['model_weight']) * final_market
    )
    report['blend_search'] = blend_rows
    report['selected_blend'] = selected_blend
    report['blended_final'] = metrics(y_final, final_blended)

    market_rolling = report['candidates'][0]['rolling_market_accuracy']
    winning_folds = sum(
        row['accuracy'] >= row['market_accuracy']
        for row in next(
            item for item in report['candidates']
            if item['name'] == best_candidate.name
        )['folds']
    )
    report['deployment_gate'] = {
        'passed': bool(
            selected_blend['model_weight'] > 0.0
            and selected_blend['accuracy'] >= market_rolling + 0.002
            and winning_folds >= 3
            and report['blended_final']['accuracy'] >= report['market_final']['accuracy']
            and report['blended_final']['log_loss'] <= report['market_final']['log_loss'] + 0.02
        ),
        'minimum_rolling_accuracy_gain': 0.002,
        'minimum_non_losing_folds': 3,
        'non_losing_folds': int(winning_folds),
        'requires_final_accuracy_not_below_market': True,
        'requires_final_log_loss_within': 0.02,
    }

    # Refit on every completed row for deployment after the untouched audit.
    final_model = best_candidate.build()
    final_scope = scope_mask(data, best_candidate.scope)
    final_model.fit(feature_matrix(data.loc[final_scope]), data.loc[final_scope, 'target'])
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': final_model,
        'model_weight': selected_blend['model_weight'],
        'feature_version': 1,
        'competition': TARGET_COMPETITION,
        'trained_rows': int(final_scope.sum()),
        'deployable': report['deployment_gate']['passed'],
        'report': report,
    }, args.output / 'champions_league_1x2.joblib')
    (args.output / 'training_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('SELECTED', best_candidate.name, selected_blend)
    print('FINAL', report['blended_final'])


if __name__ == '__main__':
    main()
