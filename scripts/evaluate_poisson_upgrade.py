"""Compare incumbent and upgraded Poisson models with walk-forward backtests.

This script is deliberately read-only with respect to saved models. It writes a
CSV report, so candidate selection does not contaminate or replace production.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.database.league import LeagueDatabase
from src.models.classifiers.football import GoalDistributionModel
from src.models.evaluation import walk_forward_evaluate
from src.preprocessing.utils.target import TargetType


LEAGUES = ('英超', '西甲', '德甲', '意甲', '法甲')


def candidates(league: str, target: TargetType):
    model_id = f'{league}{"比分" if target == TargetType.SCORE else "大小球"}模型'
    common = dict(
        league_id=league,
        model_id=model_id,
        target_type=target,
        recency_half_life_years=8.0,
    )
    return {
        '泊松基准': lambda: GoalDistributionModel(
            **common, algorithm='poisson_linear', alpha=0.1, rho=-0.05,
        ),
        '自动Dixon-Coles': lambda: GoalDistributionModel(
            **common, algorithm='poisson_linear', alpha=0.1, rho='auto',
        ),
        '自动Dixon-Coles+均值收缩10%': lambda: GoalDistributionModel(
            **common, algorithm='poisson_linear', alpha=0.1, rho='auto',
            mean_shrinkage=0.10,
        ),
        '非线性泊松+自动Dixon-Coles': lambda: GoalDistributionModel(
            **common, algorithm='hist_poisson', rho='auto',
            algorithm_params={'max_leaf_nodes': 15, 'min_samples_leaf': 25},
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--test-fraction', type=float, default=0.05)
    parser.add_argument('--task', choices=('score', 'over-under', 'both'), default='both')
    parser.add_argument(
        '--output', default='storage/reports/泊松模型滚动回测升级报告.csv',
    )
    args = parser.parse_args()
    targets = {
        'score': (TargetType.SCORE,),
        'over-under': (TargetType.OVER_UNDER,),
        'both': (TargetType.SCORE, TargetType.OVER_UNDER),
    }[args.task]

    rows = []
    for target in targets:
        for league in LEAGUES:
            dataset = LeagueDatabase().load_league(league).dropna().reset_index(drop=True)
            for name, factory in candidates(league, target).items():
                print(f'[{league}/{target.value}] {name}', flush=True)
                result = walk_forward_evaluate(
                    factory, dataset, target, folds=args.folds,
                    test_fraction=args.test_fraction,
                )
                for row in result.to_dict(orient='records'):
                    row.update({'联赛': league, '任务': target.value, '候选模型': name})
                    rows.append(row)

    report = pd.DataFrame(rows)
    summary = report.groupby(['联赛', '任务', '候选模型'], as_index=False).agg(
        folds=('fold', 'count'),
        samples=('samples', 'sum'),
        accuracy=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        log_loss=('log_loss', 'mean'),
        brier=('brier', 'mean'),
        mean_confidence=('mean_confidence', 'mean'),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    fold_output = output.with_name(f'{output.stem}-逐折{output.suffix}')
    report.to_csv(fold_output, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f'汇总：{output}\n逐折：{fold_output}', flush=True)


if __name__ == '__main__':
    main()
