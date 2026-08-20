"""Create and train leagues that occur in the daily Sporttery feed."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.football import (
    GoalDistributionModel,
    MarketBlendResultModel,
    WeightedLogisticModel,
)
from src.models.evaluation import probability_metrics
from src.preprocessing.selection import train_test_split
from src.preprocessing.statistics import StatisticsEngine
from src.preprocessing.utils.target import TargetType
from src.network.leagues.league import League
from src.services.league_sync import KOREA_MATCH_HISTORY_WINDOW, _korean_feature_dataset


LEAGUES = {
    ('Sweden', 'Allsvenskan'): '瑞超',
    ('Portugal', 'Liga-1'): '葡超',
    ('Japan', 'J-1'): '日职',
}
KOREA_SOURCE = Path('storage/network/k_league_sgodds.csv')
MODEL_LEAGUES = ('英超', '西甲', '德甲', '意甲', '法甲', '瑞超', '葡超', '日职', '韩职')
REPORT = Path('storage/reports/全部联赛核心模型选优报告.csv')


def ensure_korean_league():
    """Import the locally archived SG Odds K-League results into league storage."""
    db = LeagueDatabase()
    if db.league_exists('韩职'):
        existing = db.load_league('韩职')
        if 'Week' in existing.columns and 'HTR' in existing.columns:
            print(f'[韩职] 数据已存在：{len(existing)} 场', flush=True)
            return
        print('[韩职] 旧数据缺少轮次字段，正在重建…', flush=True)
    if not KOREA_SOURCE.exists():
        raise FileNotFoundError(f'缺少韩职历史源：{KOREA_SOURCE}')
    raw = pd.read_csv(KOREA_SOURCE)
    frame = _korean_feature_dataset(raw)
    stats = StatisticsEngine.get_basic_stat_columns()
    league = League(
        country='South Korea', name='K-League-1', start_year=int(frame['Season'].min()),
        category='extra', url='https://sgodds.com/football/data',
        fixture='https://footystats.org/south-korea/k-league-1/fixtures',
        league_id='韩职', match_history_window=KOREA_MATCH_HISTORY_WINDOW,
        goal_diff_margin=2,
        stats_columns=stats,
    )
    db.save_league(frame, league)
    print(f'[韩职] 完成：原始{len(raw)}场，有效{len(frame.dropna())}场', flush=True)


def ensure_leagues():
    db = LeagueDatabase()
    available = {(item.country, item.name): item for item in db.leagues}
    basic = StatisticsEngine.get_basic_stat_columns()
    extended = StatisticsEngine.get_extended_stat_columns()
    for key, league_id in LEAGUES.items():
        if db.league_exists(league_id):
            print(f'[{league_id}] 数据已存在：{len(db.load_league(league_id))} 场', flush=True)
            continue
        source = available[key]
        stats = basic + extended if source.category == 'main' else basic
        league = source.clone(
            start_year=source.start_year,
            league_id=league_id,
            match_history_window=4,
            goal_diff_margin=2,
            stats_columns=stats,
        )
        print(f'[{league_id}] 下载并生成历史特征…', flush=True)
        dataset = db.create_league(league)
        if dataset is None or dataset.dropna().empty:
            raise RuntimeError(f'{league_id} 没有足够的有效历史数据。')
        print(f'[{league_id}] 完成：原始{len(dataset)}场，有效{len(dataset.dropna())}场', flush=True)
    ensure_korean_league()


def candidate_factories(league: str, model_id: str, target: TargetType):
    common = dict(league_id=league, model_id=model_id, target_type=target)
    if target == TargetType.RESULT:
        items = {}
        for c in (0.03, 0.1):
            for half_life in (2.0, 4.0, 8.0):
                items[f'时间逻辑回归 C={c}/半衰期={half_life}'] = (
                    lambda c=c, h=half_life: WeightedLogisticModel(
                        **common, c=c, recency_half_life_years=h,
                    )
                )
        for weight in (0.0, 0.15, 0.30):
            items[f'市场融合 权重={weight}'] = (
                lambda w=weight: MarketBlendResultModel(
                    **common, c=0.03, model_weight=w, recency_half_life_years=4.0,
                )
            )
        return items
    items = {}
    for alpha in (0.01, 0.1, 1.0):
        for shrinkage in (0.0, 0.10):
            items[f'泊松 alpha={alpha}/收缩={shrinkage}'] = (
                lambda a=alpha, s=shrinkage: GoalDistributionModel(
                    **common, algorithm='poisson_linear', alpha=a, rho='auto',
                    mean_shrinkage=s, recency_half_life_years=4.0,
                )
            )
    return items


def train_one(league: str, target: TargetType):
    stem = {
        TargetType.RESULT: '胜平负',
        TargetType.OVER_UNDER: '大小球',
        TargetType.SCORE: '比分',
    }[target]
    model_id = f'{league}{stem}模型'
    dataset = LeagueDatabase().load_league(league).dropna().reset_index(drop=True)
    train_validation, test = train_test_split(dataset, 15.0)
    train, validation = train_test_split(train_validation, 15.0 / 85.0 * 100.0)
    evaluated = []
    for name, factory in candidate_factories(league, model_id, target).items():
        model = factory()
        model.fit(train)
        metrics = probability_metrics(model, validation, target)
        evaluated.append((name, factory, metrics))
        print(f'  {name}: acc={metrics["accuracy"]:.3f} logloss={metrics["log_loss"]:.3f}', flush=True)

    # Accuracy first, log loss resolves close candidates without rounding away signal.
    name, factory, validation_metrics = max(
        evaluated, key=lambda row: (row[2]['accuracy'], -row[2]['log_loss']),
    )
    final_model = factory()
    final_model.fit(train_validation)
    test_metrics = probability_metrics(final_model, test, target)
    config = final_model.get_default_model_config()
    config['train'] = {
        'method': '70%历史训练 / 15%时间验证选型 / 15%最近比赛独立测试',
        'selected_algorithm': name,
        'validation': validation_metrics,
        'test': test_metrics,
    }
    ModelDatabase(league).save_model(final_model, config)
    print(f'[{league}/{stem}] 保存 {name}: 测试={test_metrics["accuracy"]:.3f}', flush=True)
    return {
        '联赛': league,
        '任务': stem,
        '模型': model_id,
        '算法': name,
        '验证命中率': validation_metrics['accuracy'],
        '测试命中率': test_metrics['accuracy'],
        '测试LogLoss': test_metrics['log_loss'],
        '测试Brier': test_metrics['brier'],
        '测试样本': test_metrics['samples'],
    }


def main():
    ensure_leagues()
    rows = []
    for league in MODEL_LEAGUES:
        for target in (TargetType.RESULT, TargetType.OVER_UNDER, TargetType.SCORE):
            rows.append(train_one(league, target))
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(REPORT, index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
