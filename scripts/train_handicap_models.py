#!/usr/bin/env python3
"""Train one dedicated, line-aware handicap probability model per league.

Historical league files do not contain China's official integer handicap.  Model
selection therefore uses a documented pre-match proxy: home favourites give one
goal (-1), home underdogs receive one goal (+1).  Live predictions always use
the actual Sporttery goal line supplied for that fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from scripts.optimize_structured_models import LEAGUES, REPORT_PATH, split_dataset
from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.football import GoalDistributionModel
from src.preprocessing.utils.target import TargetType
from src.services.daily_sporttery import _handicap_probabilities


def proxy_lines(df: pd.DataFrame) -> np.ndarray:
    """Approximate the offered integer line using only pre-match information."""
    home_odds = pd.to_numeric(df['1']).to_numpy(dtype=float)
    away_odds = pd.to_numeric(df['2']).to_numpy(dtype=float)
    return np.where(home_odds <= away_odds, -1.0, 1.0)


def handicap_targets(df: pd.DataFrame, lines: np.ndarray) -> np.ndarray:
    difference = df['HG'].to_numpy(dtype=float) + lines - df['AG'].to_numpy(dtype=float)
    return np.where(difference > 0, 0, np.where(difference == 0, 1, 2)).astype(np.int32)


def metrics(model: GoalDistributionModel, df: pd.DataFrame) -> dict:
    probabilities = model.predict_proba(df)
    classes = np.asarray(model.classifier.classes_, dtype=np.int32)
    lines = proxy_lines(df)
    handicap = np.vstack([
        _handicap_probabilities(row, classes, line)
        for row, line in zip(probabilities, lines)
    ])
    targets = handicap_targets(df, lines)
    ranking = np.argsort(handicap, axis=1)[:, ::-1]
    return {
        'accuracy': float(np.mean(ranking[:, 0] == targets)),
        'top2': float(np.mean(np.any(ranking[:, :2] == targets[:, None], axis=1))),
        'log_loss': float(log_loss(targets, handicap, labels=[0, 1, 2])),
    }


def train_one(league: str) -> dict:
    raw = LeagueDatabase().load_league(league).drop(columns=['HTR'], errors='ignore')
    dataset = raw.dropna().sort_values('Date', ascending=False).reset_index(drop=True)
    train, validation, train_validation, test = split_dataset(dataset)
    candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0):
        for rho in (-0.10, -0.05, 0.0):
            model = GoalDistributionModel(
                league, f'{league}让球胜负模型', TargetType.SCORE,
                algorithm='poisson_linear', alpha=alpha, rho=rho,
                recency_half_life_years=8.0,
            )
            model.fit(train)
            result = metrics(model, validation)
            candidates.append((result['accuracy'], -result['log_loss'], alpha, rho, result))
    _, _, alpha, rho, validation_metrics = max(candidates)
    final_model = GoalDistributionModel(
        league, f'{league}让球胜负模型', TargetType.SCORE,
        algorithm='poisson_linear', alpha=alpha, rho=rho,
        recency_half_life_years=8.0,
    )
    final_model.fit(train_validation)
    test_metrics = metrics(final_model, test)
    config = final_model.get_default_model_config()
    config['train'] = {
        'eval_samples_size': 15.0,
        'tuning': {
            'method': '赛前赔率±1代理盘口选型；实战使用官方真实让球数',
            'algorithm': f'专用进球分布 alpha={alpha:g}, rho={rho:.2f}',
            'validation_accuracy': validation_metrics['accuracy'],
            'validation_top2_accuracy': validation_metrics['top2'],
            'validation_log_loss': validation_metrics['log_loss'],
            'test_accuracy': test_metrics['accuracy'],
            'top2_accuracy': test_metrics['top2'],
            'test_log_loss': test_metrics['log_loss'],
            'handicap_history': 'proxy',
        },
    }
    ModelDatabase(league).save_model(final_model, config)
    print(
        f'{league}: 首选={test_metrics["accuracy"]:.1%} '
        f'Top2={test_metrics["top2"]:.1%} 测试={len(test)}', flush=True,
    )
    return {
        '联赛': league,
        '预测类型': '让球胜负',
        '模型': f'{league}让球胜负模型',
        '验证胜出算法': f'专用进球分布 alpha={alpha:g}, rho={rho:.2f}',
        '验证首选命中率': validation_metrics['accuracy'],
        '验证Top3': validation_metrics['top2'],
        '验证Top5': np.nan,
        '验证LogLoss': validation_metrics['log_loss'],
        '最终测试首选命中率': test_metrics['accuracy'],
        '最终测试Top3': test_metrics['top2'],
        '最终测试Top5': np.nan,
        '最终测试LogLoss': test_metrics['log_loss'],
        '训练样本': len(train),
        '验证样本': len(validation),
        '最终测试样本': len(test),
        '备注': '历史为赛前赔率±1代理盘口；当天预测使用官方真实让球数',
    }


def main() -> None:
    reports = pd.read_csv(REPORT_PATH)
    reports = reports[reports['预测类型'] != '让球胜负']
    new_rows = [train_one(league) for league in LEAGUES]
    pd.concat([reports, pd.DataFrame(new_rows)], ignore_index=True, sort=False).to_csv(
        REPORT_PATH, index=False,
    )


if __name__ == '__main__':
    main()
