import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.football import (
    ConditionalHalfFullModel,
    GoalDistributionModel,
    MarketBlendResultModel,
    WeightedLogisticModel,
)
from src.models.classifiers.randomforest import RandomForest
from src.models.trainer import Trainer
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.normalization import NormalizerType
from src.preprocessing.utils.target import TargetType, construct_targets


warnings.filterwarnings('ignore')

LEAGUES = ['英超', '英冠', '西甲', '德甲', '意甲', '法甲', '葡超', '瑞超', '日职', '韩职']
TASKS = [
    ('胜平负', TargetType.RESULT),
    ('大小球', TargetType.OVER_UNDER),
    ('准确比分', TargetType.SCORE),
    ('半全场', TargetType.HALF_FULL),
    ('半场胜平负', TargetType.HALF_RESULT),
]
REPORT_PATH = 'storage/reports/全部模型深度调优报告.csv'


def backup_current_models():
    source_root = Path('storage/leagues')
    backup_root = Path('storage/backups/before-all-model-retune-20260810')
    backup_root.mkdir(parents=True, exist_ok=True)
    index_source = source_root / 'model_index.pkl'
    index_target = backup_root / 'model_index.pkl'
    if not index_target.exists():
        shutil.copy2(index_source, index_target)
    for league_id in LEAGUES:
        source = source_root / league_id / 'models'
        target = backup_root / league_id / 'models'
        if source.exists() and not target.exists():
            shutil.copytree(source, target)


def split_dataset(dataset):
    # Descending dates: newest 15% is the final test, preceding 15% validation.
    train_validation, test_df = train_test_split(dataset, 15.0)
    validation_ratio_of_remaining = 15.0 / 85.0 * 100.0
    train_df, validation_df = train_test_split(
        train_validation, validation_ratio_of_remaining,
    )
    return train_df, validation_df, train_validation, test_df


def clean_incumbent_params(config, final=False):
    params = {
        key: value for key, value in config.items()
        if key not in {'cls', 'train', 'eval'}
    }
    normalizer = params.get('normalizer')
    if isinstance(normalizer, StandardScaler):
        params['normalizer'] = NormalizerType.STANDARD
    elif isinstance(normalizer, MinMaxScaler):
        params['normalizer'] = NormalizerType.MIN_MAX
    elif isinstance(normalizer, MaxAbsScaler):
        params['normalizer'] = NormalizerType.MAX_ABS
    elif isinstance(normalizer, TransformerMixin):
        params['normalizer'] = None

    if config['cls'] is RandomForest and not final:
        params['n_estimators'] = min(int(params.get('n_estimators', 500)), 500)
    return params


def candidate(name, build):
    return {'name': name, 'build': build}


def incumbent_candidate(config):
    def build(final=False):
        params = clean_incumbent_params(config, final=final)
        return config['cls'](**params)
    return candidate('现有模型同参数重新训练', build)


def build_candidates(league_id, model_id, target_type, incumbent_config):
    # A newly added league/task has no incumbent model yet.
    items = [] if incumbent_config is None else [incumbent_candidate(incumbent_config)]

    if target_type == TargetType.RESULT:
        for c in (0.01, 0.03, 0.1, 0.3):
            for half_life in (2.0, 4.0, 8.0):
                items.append(candidate(
                    f'时间衰减逻辑回归 C={c:g}/半衰期{half_life:g}年',
                    lambda final=False, current_c=c, h=half_life: WeightedLogisticModel(
                        league_id, model_id, target_type,
                        c=current_c, recency_half_life_years=h,
                    ),
                ))
        for c in (0.01, 0.03, 0.1):
            for weight in (0.0, 0.15, 0.30, 0.45, 0.60):
                items.append(candidate(
                    f'赔率概率融合 C={c:g}/模型权重={weight:.2f}',
                    lambda final=False, current_c=c, w=weight: MarketBlendResultModel(
                        league_id, model_id,
                        model_weight=w,
                        c=current_c,
                        recency_half_life_years=8.0,
                    ),
                ))

    elif target_type == TargetType.OVER_UNDER:
        for c in (0.01, 0.03, 0.1, 0.3):
            for half_life in (2.0, 4.0, 8.0):
                items.append(candidate(
                    f'时间衰减逻辑回归 C={c:g}/半衰期{half_life:g}年',
                    lambda final=False, current_c=c, h=half_life: WeightedLogisticModel(
                        league_id, model_id, target_type,
                        c=current_c, recency_half_life_years=h,
                    ),
                ))
        for alpha in (0.01, 0.1, 1.0, 10.0):
            for half_life in (4.0, 8.0):
                items.append(candidate(
                    f'泊松总进球 alpha={alpha:g}/半衰期{half_life:g}年',
                    lambda final=False, a=alpha, h=half_life: GoalDistributionModel(
                        league_id, model_id, target_type,
                        algorithm='poisson_linear', alpha=a, rho=-0.05,
                        recency_half_life_years=h,
                    ),
                ))
        for leaf_nodes, leaf_size in ((7, 40), (15, 25), (31, 20)):
            items.append(candidate(
                f'非线性泊松 叶节点={leaf_nodes}/叶样本={leaf_size}',
                lambda final=False, nodes=leaf_nodes, size=leaf_size: GoalDistributionModel(
                    league_id, model_id, target_type,
                    algorithm='hist_poisson', rho=-0.05,
                    algorithm_params={'max_leaf_nodes': nodes, 'min_samples_leaf': size},
                    recency_half_life_years=8.0,
                ),
            ))
        for shrinkage in (0.0, 0.10, 0.20):
            items.append(candidate(
                f'自动Dixon-Coles泊松/均值收缩={shrinkage:.2f}',
                lambda final=False, s=shrinkage: GoalDistributionModel(
                    league_id, model_id, target_type,
                    algorithm='poisson_linear', alpha=0.1, rho='auto',
                    mean_shrinkage=s, recency_half_life_years=8.0,
                ),
            ))

    elif target_type == TargetType.SCORE:
        for alpha in (0.01, 0.1, 1.0, 10.0):
            for rho in (-0.15, -0.10, -0.05, 0.0, 0.05):
                items.append(candidate(
                    f'泊松/Dixon-Coles alpha={alpha:g}, rho={rho:.2f}',
                    lambda final=False, a=alpha, r=rho: GoalDistributionModel(
                        league_id, model_id, target_type,
                        algorithm='poisson_linear',
                        alpha=a,
                        rho=r,
                        recency_half_life_years=8.0,
                    ),
                ))
        for rho in (-0.15, -0.10, -0.05, 0.0, 0.05):
            for leaf_nodes in (7, 15, 31):
                items.append(candidate(
                    f'非线性泊松 rho={rho:.2f}/叶节点={leaf_nodes}',
                    lambda final=False, r=rho, nodes=leaf_nodes: GoalDistributionModel(
                        league_id, model_id, target_type,
                        algorithm='hist_poisson', rho=r,
                        algorithm_params={'max_leaf_nodes': nodes},
                        recency_half_life_years=8.0,
                ),
            ))
        for shrinkage in (0.0, 0.05, 0.10, 0.20):
            items.append(candidate(
                f'自动Dixon-Coles alpha=0.1/均值收缩={shrinkage:.2f}',
                lambda final=False, s=shrinkage: GoalDistributionModel(
                    league_id, model_id, target_type,
                    algorithm='poisson_linear', alpha=0.1, rho='auto',
                    mean_shrinkage=s, recency_half_life_years=8.0,
                ),
            ))

    elif target_type == TargetType.HALF_RESULT:
        for c in (0.03, 0.1):
            for half_life in (4.0, 8.0):
                items.append(candidate(
                    f'时间衰减半场逻辑回归 C={c:g}/半衰期{half_life:g}年',
                    lambda final=False, current_c=c, h=half_life: WeightedLogisticModel(
                        league_id, model_id, target_type,
                        c=current_c, recency_half_life_years=h,
                    ),
                ))

    elif target_type == TargetType.HALF_FULL:
        for c in (0.03, 0.1):
            for half_life in (4.0, 8.0):
                items.append(candidate(
                    f'半全场条件概率链 C={c:g}/半衰期{half_life:g}年',
                    lambda final=False, current_c=c, h=half_life: ConditionalHalfFullModel(
                        league_id, model_id,
                        half_c=current_c,
                        full_c=current_c,
                        recency_half_life_years=h,
                    ),
                ))
        items.append(candidate(
            '半全场条件概率链 C半场=0.03/C全场=0.1',
            lambda final=False: ConditionalHalfFullModel(
                league_id, model_id,
                half_c=0.03,
                full_c=0.1,
                recency_half_life_years=8.0,
            ),
        ))
    return items


def probability_metrics(model, df, target_type):
    y_true = construct_targets(df, target_type)
    probabilities = model.predict_proba(df)
    classes = np.asarray(model.classifier.classes_, dtype=np.int32)
    y_pred = classes[probabilities.argmax(axis=1)]
    accuracy = float(np.mean(y_pred == y_true))

    top_values = {}
    for k in (3, 5):
        width = min(k, probabilities.shape[1])
        indices = np.argpartition(probabilities, -width, axis=1)[:, -width:]
        top_values[k] = float(np.mean(
            np.any(classes[indices] == y_true[:, None], axis=1),
        ))

    class_index = {int(label): index for index, label in enumerate(classes)}
    true_probability = np.array([
        probabilities[row, class_index[label]]
        if int(label) in class_index else 1e-12
        for row, label in enumerate(y_true)
    ])
    log_loss = float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean())
    return {
        'accuracy': accuracy,
        'top3': top_values[3],
        'top5': top_values[5],
        'log_loss': log_loss,
        'y_true': y_true,
        'y_pred': y_pred,
        'probabilities': probabilities,
    }


def ranking(metrics, target_type):
    if target_type in {TargetType.SCORE, TargetType.HALF_FULL}:
        return (
            round(metrics['accuracy'], 3),
            metrics['top3'],
            metrics['top5'],
            -metrics['log_loss'],
        )
    return (metrics['accuracy'], -metrics['log_loss'])


def choose_selective_threshold(metrics):
    probabilities = metrics['probabilities']
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    rows = []
    for threshold in np.arange(0.35, 0.76, 0.01):
        mask = probabilities.max(axis=1) >= threshold
        count = int(mask.sum())
        coverage = float(mask.mean())
        accuracy = float(np.mean(y_pred[mask] == y_true[mask])) if count else 0.0
        rows.append((threshold, accuracy, coverage, count))
    eligible = [
        row for row in rows
        if row[1] >= 0.55 and row[2] >= 0.05 and row[3] >= 50
    ]
    if eligible:
        return max(eligible, key=lambda row: (row[2], row[1]))
    stable = [row for row in rows if row[3] >= 30]
    return max(stable or rows, key=lambda row: (row[1], row[2]))


def majority_and_significance(train_df, test_metrics, target_type):
    train_y = construct_targets(train_df, target_type)
    values, counts = np.unique(train_y, return_counts=True)
    majority = values[counts.argmax()]
    baseline_correct = test_metrics['y_true'] == majority
    model_correct = test_metrics['y_pred'] == test_metrics['y_true']
    better = int(np.sum(model_correct & ~baseline_correct))
    worse = int(np.sum(~model_correct & baseline_correct))
    p_value = (
        float(binomtest(min(better, worse), n=better + worse, p=0.5).pvalue)
        if better + worse else 1.0
    )
    return float(baseline_correct.mean()), p_value


def tune_one(league_id, task_name, target_type):
    model_stem = '比分' if target_type == TargetType.SCORE else task_name
    model_id = f'{league_id}{model_stem}模型'
    model_db = ModelDatabase(league_id)
    incumbent_config = model_db.load_model_config(model_id)
    raw_dataset = LeagueDatabase().load_league(league_id)
    if target_type in {TargetType.HALF_FULL, TargetType.HALF_RESULT} and (
            'HTR' not in raw_dataset.columns
            or raw_dataset['HTR'].notna().sum() < 100
    ):
        return {
            '联赛': league_id, '预测类型': task_name,
            '模型': model_id, '状态': '真实半场数据不足',
        }
    if target_type in {TargetType.HALF_FULL, TargetType.HALF_RESULT}:
        dataset = raw_dataset.dropna().reset_index(drop=True)
    else:
        dataset = raw_dataset.drop(columns=['HTR'], errors='ignore')
        dataset = dataset.dropna().reset_index(drop=True)
    train_df, validation_df, train_validation, test_df = split_dataset(dataset)
    print(
        f'\n[{league_id}/{task_name}] 训练={len(train_df)} '
        f'验证={len(validation_df)} 最终测试={len(test_df)}',
        flush=True,
    )

    evaluated = []
    for item in build_candidates(
            league_id, model_id, target_type, incumbent_config,
    ):
        model = item['build'](False)
        model.fit(train_df)
        metrics = probability_metrics(model, validation_df, target_type)
        evaluated.append({'item': item, 'metrics': metrics})
        print(
            f'  {item["name"]}: 首选={metrics["accuracy"]:.3f} '
            f'Top3={metrics["top3"]:.3f} Top5={metrics["top5"]:.3f} '
            f'LogLoss={metrics["log_loss"]:.3f}',
            flush=True,
        )

    best = max(evaluated, key=lambda row: ranking(row['metrics'], target_type))
    final_model = best['item']['build'](True)
    final_model, fit_metrics = Trainer().train(
        final_model, train_validation, test_df,
    )
    final = probability_metrics(final_model, test_df, target_type)
    baseline, p_value = majority_and_significance(
        train_validation, final, target_type,
    )

    tuning = {
        'method': '70%历史训练 / 15%时间验证选择模型 / 15%最近比赛最终测试',
        'algorithm': best['item']['name'],
        'compared_algorithms': [row['item']['name'] for row in evaluated],
        'validation_accuracy': best['metrics']['accuracy'],
        'validation_top3_accuracy': best['metrics']['top3'],
        'validation_top5_accuracy': best['metrics']['top5'],
        'validation_log_loss': best['metrics']['log_loss'],
        'test_accuracy': final['accuracy'],
        'top3_accuracy': final['top3'],
        'top5_accuracy': final['top5'],
        'test_log_loss': final['log_loss'],
        'majority_baseline': baseline,
        'mcnemar_p_value_vs_baseline': p_value,
        'test_samples': len(test_df),
    }

    if target_type == TargetType.HALF_RESULT:
        threshold, val_acc, val_coverage, _ = choose_selective_threshold(
            best['metrics'],
        )
        mask = final['probabilities'].max(axis=1) >= threshold
        selective_accuracy = (
            float(np.mean(final['y_pred'][mask] == final['y_true'][mask]))
            if mask.any() else 0.0
        )
        selective_samples = int(mask.sum())
        validation_passed = val_acc >= 0.55
        test_passed = selective_accuracy >= 0.55 and selective_samples >= 30
        tuning.update({
            'selective_threshold': float(threshold),
            'validation_selective_accuracy': val_acc,
            'validation_coverage': val_coverage,
            'selective_accuracy': selective_accuracy,
            'coverage': float(mask.mean()),
            'selective_samples': selective_samples,
            'selective_validation_passed': validation_passed,
            'selective_test_passed': test_passed,
            'selective_validated': bool(validation_passed and test_passed),
        })

    config = final_model.get_default_model_config()
    config['train'] = {
        'eval_samples_size': 15.0,
        'results': {'fit': fit_metrics},
        'tuning': tuning,
    }
    model_db.save_model(final_model, config)
    print(
        f'  [保存] {best["item"]["name"]} | '
        f'测试首选={final["accuracy"]:.3f} Top3={final["top3"]:.3f} '
        f'Top5={final["top5"]:.3f} 基线={baseline:.3f} p={p_value:.4g}',
        flush=True,
    )

    return {
        '联赛': league_id,
        '预测类型': task_name,
        '模型': model_id,
        '验证胜出算法': best['item']['name'],
        '验证首选命中率': best['metrics']['accuracy'],
        '验证Top3': best['metrics']['top3'],
        '验证Top5': best['metrics']['top5'],
        '验证LogLoss': best['metrics']['log_loss'],
        '最终测试首选命中率': final['accuracy'],
        '最终测试Top3': final['top3'],
        '最终测试Top5': final['top5'],
        '最终测试LogLoss': final['log_loss'],
        '简单基线': baseline,
        '相对基线优势': final['accuracy'] - baseline,
        '相对基线P值': p_value,
        '高置信度门槛': tuning.get('selective_threshold'),
        '高置信度命中率': tuning.get('selective_accuracy'),
        '高置信度覆盖率': tuning.get('coverage'),
        '高置信度样本数': tuning.get('selective_samples'),
        '训练样本': len(train_df),
        '验证样本': len(validation_df),
        '最终测试样本': len(test_df),
    }


def save_report(reports):
    os.makedirs('storage/reports', exist_ok=True)
    pd.DataFrame(reports).to_csv(REPORT_PATH, index=False)


if __name__ == '__main__':
    backup_current_models()
    if os.path.exists(REPORT_PATH):
        reports = pd.read_csv(REPORT_PATH).to_dict(orient='records')
    else:
        reports = []
    completed = {
        (row['联赛'], row['预测类型'])
        for row in reports
    }
    for task_name, target_type in TASKS:
        for league_id in LEAGUES:
            if (league_id, task_name) in completed:
                print(f'[断点续训] {league_id}/{task_name} 已完成，跳过。', flush=True)
                continue
            reports.append(tune_one(league_id, task_name, target_type))
            save_report(reports)
    print('\n' + pd.DataFrame(reports).to_string(index=False), flush=True)
    print(f'\n最终报告：{REPORT_PATH}', flush=True)
