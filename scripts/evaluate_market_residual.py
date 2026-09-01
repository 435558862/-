"""Evaluate whether enhanced pre-match features improve the 1X2 market.

This is an offline challenger audit.  It never overwrites production models.
The newest 15% of matches remain sealed until all feature/model/blend choices
have been made on the preceding chronological validation block.
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from src.database.league import LeagueDatabase


LEAGUES = ('英超', '西甲', '德甲', '意甲', '法甲')
XG_ROOT = Path('storage/enhanced/xg')
LINEUP_PATH = Path('storage/enhanced/lineups/transfermarkt-lineup-features.csv')
REPORT = Path('storage/reports/市场残差模型评测-20260901.csv')
DETAIL_REPORT = Path('storage/reports/市场残差模型评测逐联赛明细-20260901.csv')
RESULT_MAP = {'H': 0, 'D': 1, 'A': 2}
FORM_FEATURES = (
    'HW', 'AW', 'HL', 'AL', 'HGF', 'AGF', 'HGA', 'AGA', 'HGD', 'AGD',
    'HW%', 'HL%', 'AW%', 'AL%', 'HSTF', 'ASTF', 'HCF', 'ACF', 'Week',
)
XG_FEATURES = (
    'HXGF5', 'HXGA5', 'HXGF10', 'HXGA10',
    'AXGF5', 'AXGA5', 'AXGF10', 'AXGA10', 'HXGD5', 'AXGD5',
)
LINEUP_FEATURES = (
    'HLineupContinuity5', 'ALineupContinuity5',
    'HLineupCore5', 'ALineupCore5',
)


def _name(value) -> str:
    text = unicodedata.normalize('NFKD', str(value or ''))
    text = ''.join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace('&', ' and ')
    text = re.sub(r'\b(fc|cf|afc|sc|club|football|calcio|deportivo)\b', ' ', text)
    return re.sub(r'[^a-z0-9]+', '', text)


def _fixture_score(left_home, left_away, right_home, right_away) -> float:
    return ratio(_name(left_home), _name(right_home)) + ratio(
        _name(left_away), _name(right_away)
    )


def _attach_by_fixture(base: pd.DataFrame, extra: pd.DataFrame,
                       feature_columns: tuple[str, ...], prefix: str):
    """Date-constrained fuzzy fixture join with one-to-one matching."""
    right_by_date = {
        day: group for day, group in extra.groupby('Date', sort=False)
    }
    values = pd.DataFrame(np.nan, index=base.index, columns=feature_columns)
    scores = pd.Series(np.nan, index=base.index, dtype=float)
    for day, left_group in base.groupby('Date', sort=False):
        right = right_by_date.get(day)
        if right is None or right.empty:
            continue
        available = set(right.index)
        ranked = []
        for left_index, left in left_group.iterrows():
            for right_index in available:
                candidate = right.loc[right_index]
                score = _fixture_score(
                    left['Home'], left['Away'], candidate['Home'], candidate['Away'],
                )
                ranked.append((score, left_index, right_index))
        used_left, used_right = set(), set()
        for score, left_index, right_index in sorted(ranked, reverse=True):
            if score < 145 or left_index in used_left or right_index in used_right:
                continue
            values.loc[left_index, list(feature_columns)] = right.loc[
                right_index, list(feature_columns)
            ].to_numpy()
            scores.loc[left_index] = score
            used_left.add(left_index)
            used_right.add(right_index)
    result = base.join(values.add_prefix(prefix))
    result[f'{prefix}match_score'] = scores
    return result


def _market_probabilities(frame: pd.DataFrame) -> np.ndarray:
    odds = frame[['1', 'X', '2']].to_numpy(float)
    inverse = 1.0 / odds
    return inverse / inverse.sum(axis=1, keepdims=True)


def _metrics(y, probabilities) -> dict:
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(3)[y]
    return {
        'accuracy': float(accuracy_score(y, prediction)),
        'log_loss': float(log_loss(y, probabilities, labels=[0, 1, 2])),
        'brier': float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def _blend(market, challenger, weight):
    result = (1.0 - weight) * market + weight * challenger
    return result / result.sum(axis=1, keepdims=True)


def evaluate_league(league: str, lineup: pd.DataFrame):
    base = LeagueDatabase().load_league(league).copy()
    base['Date'] = pd.to_datetime(base['Date']).dt.strftime('%Y-%m-%d')
    required = ['Date', 'Home', 'Away', 'Result', '1', 'X', '2']
    base = base.dropna(subset=required)
    valid_odds = np.isfinite(base[['1', 'X', '2']].to_numpy(float)).all(axis=1)
    base = base.loc[valid_odds & base[['1', 'X', '2']].gt(1.0).all(axis=1)].copy()
    xg = pd.read_csv(XG_ROOT / f'{league}-features.csv')
    xg['Date'] = pd.to_datetime(xg['Date']).dt.strftime('%Y-%m-%d')
    frame = _attach_by_fixture(base, xg, XG_FEATURES, 'xg_')
    frame = _attach_by_fixture(frame, lineup, LINEUP_FEATURES, 'lu_')
    frame = frame.loc[frame['xg_match_score'].notna()].copy()
    frame = frame.sort_values('Date').reset_index(drop=True)

    market = _market_probabilities(frame)
    frame['market_home'], frame['market_draw'], frame['market_away'] = market.T
    frame['market_entropy'] = -np.sum(market * np.log(market), axis=1)
    frame['market_gap'] = np.sort(market, axis=1)[:, -1] - np.sort(market, axis=1)[:, -2]
    frame['xg_attack_gap5'] = frame['xg_HXGF5'] - frame['xg_AXGF5']
    frame['xg_defence_gap5'] = frame['xg_AXGA5'] - frame['xg_HXGA5']
    frame['xg_attack_gap10'] = frame['xg_HXGF10'] - frame['xg_AXGF10']
    frame['xg_defence_gap10'] = frame['xg_AXGA10'] - frame['xg_HXGA10']
    frame['lineup_continuity_gap'] = (
        frame['lu_HLineupContinuity5'] - frame['lu_ALineupContinuity5']
    )
    frame['lineup_core_gap'] = frame['lu_HLineupCore5'] - frame['lu_ALineupCore5']
    feature_columns = [
        'market_home', 'market_draw', 'market_away', 'market_entropy', 'market_gap',
        *[f'xg_{column}' for column in XG_FEATURES],
        'xg_attack_gap5', 'xg_defence_gap5', 'xg_attack_gap10', 'xg_defence_gap10',
        *[column for column in FORM_FEATURES if column in frame],
        *[f'lu_{column}' for column in LINEUP_FEATURES],
        'lineup_continuity_gap', 'lineup_core_gap',
    ]
    x = frame[feature_columns].apply(pd.to_numeric, errors='coerce')
    # Historic lineup coverage is sparse. Missing values receive a neutral
    # train-only median plus explicit availability indicators.
    x['xg_available'] = frame['xg_match_score'].notna().astype(float)
    x['lineup_available'] = frame['lu_match_score'].notna().astype(float)
    y = frame['Result'].map(RESULT_MAP).astype(int).to_numpy()
    n = len(frame)
    train_end, validation_end = int(n * 0.70), int(n * 0.85)
    train, validation, test = slice(0, train_end), slice(train_end, validation_end), slice(validation_end, n)
    medians = x.iloc[train].median().fillna(0.0)
    x = x.fillna(medians).to_numpy(float)

    candidates = []
    for leaf_size in (30, 60, 100):
        for l2 in (5.0, 15.0, 30.0):
            model = HistGradientBoostingClassifier(
                learning_rate=0.035, max_iter=180, max_leaf_nodes=7,
                min_samples_leaf=leaf_size, l2_regularization=l2,
                random_state=0, early_stopping=False,
            ).fit(x[train], y[train])
            validation_model = model.predict_proba(x[validation])
            for weight in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
                probability = _blend(
                    market[validation], validation_model, weight,
                )
                value = _metrics(y[validation], probability)
                candidates.append((value['log_loss'], -value['accuracy'], leaf_size, l2, weight))
    _, _, leaf_size, l2, weight = min(candidates)
    final_model = HistGradientBoostingClassifier(
        learning_rate=0.035, max_iter=180, max_leaf_nodes=7,
        min_samples_leaf=leaf_size, l2_regularization=l2,
        random_state=0, early_stopping=False,
    ).fit(x[:validation_end], y[:validation_end])
    challenger = _blend(
        market[test], final_model.predict_proba(x[test]), weight,
    )
    market_metrics = _metrics(y[test], market[test])
    challenger_metrics = _metrics(y[test], challenger)
    details = frame.iloc[test][['Date', 'Home', 'Away', 'Result']].copy()
    details['联赛'] = league
    details['市场预测'] = np.array(['H', 'D', 'A'])[market[test].argmax(axis=1)]
    details['残差预测'] = np.array(['H', 'D', 'A'])[challenger.argmax(axis=1)]
    details['市场命中'] = details['市场预测'].eq(details['Result']).astype(int)
    details['残差命中'] = details['残差预测'].eq(details['Result']).astype(int)
    return {
        '联赛': league, '匹配样本': n,
        '起始日期': frame['Date'].min(), '结束日期': frame['Date'].max(),
        'xG匹配率': float(frame['xg_match_score'].notna().mean()),
        '阵容特征覆盖率': float(frame['lu_match_score'].notna().mean()),
        '最终测试样本': len(details), '叶样本': leaf_size, 'L2': l2,
        '残差权重': weight,
        '市场准确率': market_metrics['accuracy'],
        '残差准确率': challenger_metrics['accuracy'],
        '准确率变化': challenger_metrics['accuracy'] - market_metrics['accuracy'],
        '市场LogLoss': market_metrics['log_loss'],
        '残差LogLoss': challenger_metrics['log_loss'],
        'LogLoss变化': challenger_metrics['log_loss'] - market_metrics['log_loss'],
        '市场Brier': market_metrics['brier'],
        '残差Brier': challenger_metrics['brier'],
        'Brier变化': challenger_metrics['brier'] - market_metrics['brier'],
        '是否全面胜出': bool(
            challenger_metrics['accuracy'] > market_metrics['accuracy']
            and challenger_metrics['log_loss'] < market_metrics['log_loss']
            and challenger_metrics['brier'] < market_metrics['brier']
        ),
    }, details


def main():
    lineup = pd.read_csv(LINEUP_PATH)
    lineup['Date'] = pd.to_datetime(lineup['Date']).dt.strftime('%Y-%m-%d')
    rows, details = [], []
    for league in LEAGUES:
        row, detail = evaluate_league(league, lineup)
        rows.append(row)
        details.append(detail)
        print(pd.Series(row).to_string(), flush=True)
        print('', flush=True)
    report = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT, index=False, encoding='utf-8-sig')
    pd.concat(details, ignore_index=True).to_csv(
        DETAIL_REPORT, index=False, encoding='utf-8-sig',
    )
    print(report.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
