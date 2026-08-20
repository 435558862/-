#!/usr/bin/env python3
"""Rebuild and tune the three core K-League models with time-aware validation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.football import (
    GoalDistributionModel,
    MarketBlendResultModel,
    WeightedLogisticModel,
)
from src.models.evaluation import probability_metrics, walk_forward_evaluate
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.target import TargetType
from src.services.league_sync import (
    KOREA_HISTORY_PATH,
    KOREA_SOURCE_PATH,
    _korean_feature_dataset,
    _korean_history_feature_dataset,
)


REPORT = Path('storage/reports/韩职核心模型滚动调优报告-20260810.csv')
BACKUP = Path('storage/backups/before-korean-window-retune-20260810')


def rebuild_dataset() -> pd.DataFrame:
    current = _korean_feature_dataset(pd.read_csv(KOREA_SOURCE_PATH))
    history = _korean_history_feature_dataset(pd.read_csv(KOREA_HISTORY_PATH))
    dataset = pd.concat([history, current], ignore_index=True, sort=False)
    dataset = dataset.drop_duplicates(['Date', 'Home', 'Away'], keep='last')
    return dataset.sort_values(['Date', 'Home'], ascending=[False, True]).reset_index(drop=True)


def candidates(target: TargetType):
    common = {'league_id': '韩职', 'model_id': '临时调优', 'target_type': target}
    if target == TargetType.RESULT:
        for c in (0.01, 0.03, 0.1, 0.3):
            for half_life in (2.0, 4.0, 8.0, None):
                name = f'时间逻辑回归 C={c:g}/半衰期={half_life}'
                yield name, lambda c=c, h=half_life: WeightedLogisticModel(
                    **common, c=c, recency_half_life_years=h,
                )
        for c in (0.01, 0.03, 0.1):
            for weight in (0.0, 0.15, 0.30):
                name = f'赔率融合 C={c:g}/模型权重={weight:.2f}'
                yield name, lambda c=c, w=weight: MarketBlendResultModel(
                    **common, c=c, model_weight=w, recency_half_life_years=8.0,
                )
        return
    for alpha in (0.01, 0.1, 1.0, 10.0):
        for half_life in (2.0, 4.0, 8.0, None):
            for shrinkage in (0.0, 0.10):
                name = (
                    f'泊松 alpha={alpha:g}/半衰期={half_life}/'
                    f'收缩={shrinkage:.2f}'
                )
                yield name, lambda a=alpha, h=half_life, s=shrinkage: GoalDistributionModel(
                    **common, algorithm='poisson_linear', alpha=a, rho='auto',
                    mean_shrinkage=s, recency_half_life_years=h,
                )


def tune(dataset: pd.DataFrame, target: TargetType, stem: str) -> list[dict]:
    train_validation, final_test = train_test_split(dataset, 15.0)
    evaluated = []
    for name, factory in candidates(target):
        folds = walk_forward_evaluate(
            factory, train_validation, target, folds=4, test_fraction=0.08,
        )
        evaluated.append((name, factory, folds))
        print(
            f'[{stem}] {name}: 滚动命中={folds.accuracy.mean():.3f} '
            f'LogLoss={folds.log_loss.mean():.3f}', flush=True,
        )

    # The product reports first-choice hit rate as its primary metric.  Average
    # four chronological folds first; use probability quality only as a tie-break.
    name, factory, folds = max(
        evaluated,
        key=lambda row: (row[2].accuracy.mean(), -row[2].log_loss.mean()),
    )
    model = factory()
    model._model_id = f'韩职{stem}模型'
    model.fit(train_validation)
    final = probability_metrics(model, final_test, target)
    config = model.get_default_model_config()
    config['train'] = {
        'method': '4折扩展窗口选型 / 最近15%独立测试',
        'selected_algorithm': name,
        'match_history_window': 1,
        'walk_forward_accuracy': float(folds.accuracy.mean()),
        'walk_forward_log_loss': float(folds.log_loss.mean()),
        'test': final,
    }
    ModelDatabase('韩职').save_model(model, config)
    print(
        f'[保存/{stem}] {name}: 测试命中={final["accuracy"]:.3f} '
        f'LogLoss={final["log_loss"]:.3f} n={final["samples"]}', flush=True,
    )
    rows = folds.assign(
        任务=stem, 算法=name, 数据量=len(dataset),
        最终测试命中率=final['accuracy'], 最终测试LogLoss=final['log_loss'],
        最终测试样本=final['samples'],
    )
    return rows.to_dict('records')


def main() -> None:
    dataset_path = Path('storage/leagues/韩职/data/dataset.csv')
    models_path = Path('storage/leagues/韩职/models')
    BACKUP.mkdir(parents=True, exist_ok=True)
    backup_data = BACKUP / 'dataset.csv'
    backup_models = BACKUP / 'models'
    if not backup_data.exists():
        shutil.copy2(dataset_path, backup_data)
    if models_path.exists() and not backup_models.exists():
        shutil.copytree(models_path, backup_models)

    rebuilt = rebuild_dataset()
    league_db = LeagueDatabase()
    old_league = league_db.index['韩职']
    tuned_league = old_league.clone(
        start_year=old_league.start_year,
        league_id='韩职',
        match_history_window=1,
        goal_diff_margin=old_league.goal_diff_margin,
        stats_columns=old_league.stats_columns,
        odd_1_range=old_league.odd_1_range,
        odd_x_range=old_league.odd_x_range,
        odd_2_range=old_league.odd_2_range,
    )
    league_db.save_league(rebuilt, tuned_league)
    core = rebuilt.drop(columns=['HTR'], errors='ignore').dropna().reset_index(drop=True)
    print(f'韩职重建：总计={len(rebuilt)}，核心有效={len(core)}', flush=True)

    report = []
    for target, stem in (
        (TargetType.RESULT, '胜平负'),
        (TargetType.OVER_UNDER, '大小球'),
        (TargetType.SCORE, '比分'),
    ):
        report.extend(tune(core, target, stem))
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report).to_csv(REPORT, index=False)


if __name__ == '__main__':
    main()
