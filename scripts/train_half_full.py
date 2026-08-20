"""Tune and train half-time/full-time models for every league with real HTR data."""

from pathlib import Path

import pandas as pd

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.randomforest import RandomForest
from src.models.evaluation import probability_metrics
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.target import TargetType


LEAGUES = ('英超', '西甲', '德甲', '意甲', '法甲', '葡超', '瑞超', '日职', '韩职')
PARAM_GRID = (
    {'max_depth': 7, 'min_samples_leaf': 12, 'min_samples_split': 24, 'max_features': 'sqrt'},
    {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 16, 'max_features': 'sqrt'},
    {'max_depth': 14, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt'},
)
REPORT = Path('storage/reports/半全场模型选优报告.csv')


def build_model(league: str, params: dict, estimators: int = 500):
    return RandomForest(
        league_id=league, model_id=f'{league}半全场模型',
        target_type=TargetType.HALF_FULL, calibrate_probabilities=False,
        n_estimators=estimators, criterion='gini', class_weight=True, **params,
    )


def train_one(league: str):
    raw = LeagueDatabase().load_league(league)
    if 'HTR' not in raw.columns or raw['HTR'].notna().sum() < 100:
        print(f'[{league}] 跳过：真实半场数据不足（{raw["HTR"].notna().sum() if "HTR" in raw else 0}）')
        return {'联赛': league, '状态': '真实半场数据不足'}
    dataset = raw.dropna().reset_index(drop=True)
    train_validation, test = train_test_split(dataset, 15.0)
    train, validation = train_test_split(train_validation, 15.0 / 85.0 * 100.0)
    candidates = []
    for number, params in enumerate(PARAM_GRID, 1):
        model = build_model(league, params)
        model.fit(train)
        metrics = probability_metrics(model, validation, TargetType.HALF_FULL)
        candidates.append((params, metrics))
        print(f'[{league}] 参数{number}: acc={metrics["accuracy"]:.3f} loss={metrics["log_loss"]:.3f}')
    params, validation_metrics = max(
        candidates, key=lambda item: (item[1]['accuracy'], -item[1]['log_loss']),
    )
    model = build_model(league, params, estimators=1000)
    model.fit(train_validation)
    test_metrics = probability_metrics(model, test, TargetType.HALF_FULL)
    config = model.get_default_model_config()
    config['train'] = {
        'method': '70%训练 / 15%时间验证选参 / 15%最近比赛独立测试',
        'selected_params': params, 'validation': validation_metrics, 'test': test_metrics,
    }
    ModelDatabase(league).save_model(model, config)
    print(f'[{league}] 半全场模型已保存，测试={test_metrics["accuracy"]:.3f}')
    return {
        '联赛': league, '状态': '已训练', '有效样本': len(dataset),
        '验证准确率': validation_metrics['accuracy'],
        '测试准确率': test_metrics['accuracy'], '测试样本': test_metrics['samples'],
        '测试LogLoss': test_metrics['log_loss'],
    }


if __name__ == '__main__':
    rows = [train_one(league) for league in LEAGUES]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
