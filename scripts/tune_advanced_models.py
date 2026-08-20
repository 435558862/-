import os
from typing import Dict

import numpy as np
import pandas as pd

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.logistic import LogisticRegressor
from src.models.classifiers.optimized import OptimizedEnsemble
from src.models.classifiers.randomforest import RandomForest
from src.models.trainer import Trainer
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.normalization import NormalizerType
from src.preprocessing.utils.target import TargetType, construct_targets


LEAGUES = ['英超', '西甲', '德甲', '意甲', '法甲']


def make_random_forest(
        league_id, model_id, target_type, saved_config=None, n_estimators=500,
):
    saved_config = saved_config or {}
    return RandomForest(
        league_id=league_id,
        model_id=model_id,
        target_type=target_type,
        calibrate_probabilities=False,
        n_estimators=n_estimators,
        criterion=saved_config.get('criterion', 'gini'),
        min_samples_leaf=saved_config.get('min_samples_leaf', 10),
        min_samples_split=saved_config.get('min_samples_split', 20),
        max_features=saved_config.get('max_features', 'sqrt'),
        max_depth=saved_config.get('max_depth', 8),
        class_weight=saved_config.get('class_weight', False),
    )


def make_candidate(spec: Dict, league_id, model_id, target_type, saved_config=None):
    algorithm = spec['algorithm']
    if algorithm == 'random_forest':
        return make_random_forest(
            league_id,
            model_id,
            target_type,
            saved_config=saved_config,
            n_estimators=spec.get('n_estimators', 500),
        )
    if algorithm == 'logistic':
        return LogisticRegressor(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=False,
            normalizer=NormalizerType.STANDARD,
            penalty='l2',
            fixed_c=spec['c'],
        )
    if algorithm == 'extra_trees':
        return OptimizedEnsemble(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=False,
            algorithm='extra_trees',
            algorithm_params={
                'n_estimators': spec.get('n_estimators', 600),
                'max_depth': spec.get('max_depth', 14),
                'min_samples_leaf': spec.get('min_samples_leaf', 7),
                'min_samples_split': spec.get('min_samples_split', 14),
                'max_features': spec.get('max_features', 'sqrt'),
                'class_weight': spec.get('class_weight'),
            },
        )
    if algorithm == 'hist_gradient_boosting':
        return OptimizedEnsemble(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=False,
            algorithm='hist_gradient_boosting',
            algorithm_params={
                'max_iter': 140,
                'learning_rate': 0.05,
                'max_leaf_nodes': 15,
                'min_samples_leaf': 25,
                'l2_regularization': 5.0,
                'early_stopping': False,
            },
        )
    raise ValueError(algorithm)


def prediction_metrics(model, df, target_type):
    y_true = construct_targets(df, target_type)
    y_prob = model.predict_proba(df)
    classes = np.asarray(model.classifier.classes_)
    y_pred = classes[y_prob.argmax(axis=1)]
    accuracy = float(np.mean(y_pred == y_true))

    top_values = {}
    for k in (3, 5):
        width = min(k, y_prob.shape[1])
        top_indices = np.argpartition(y_prob, -width, axis=1)[:, -width:]
        top_classes = classes[top_indices]
        top_values[k] = float(np.mean(np.any(top_classes == y_true[:, None], axis=1)))
    return accuracy, top_values[3], top_values[5]


def majority_baseline(train_df, test_df, target_type):
    train_y = construct_targets(train_df, target_type)
    test_y = construct_targets(test_df, target_type)
    classes, counts = np.unique(train_y, return_counts=True)
    return float(np.mean(test_y == classes[counts.argmax()]))


def candidate_specs(target_type):
    specs = [
        {'name': '随机森林（现有参数）', 'algorithm': 'random_forest'},
        {'name': '标准化逻辑回归 C=0.03', 'algorithm': 'logistic', 'c': 0.03},
        {'name': '标准化逻辑回归 C=0.1', 'algorithm': 'logistic', 'c': 0.1},
        {
            'name': '极端随机树',
            'algorithm': 'extra_trees',
            'n_estimators': 600,
            'max_depth': 14,
            'min_samples_leaf': 7,
            'min_samples_split': 14,
            'max_features': 'sqrt',
            'class_weight': None,
        },
    ]
    if target_type != TargetType.SCORE:
        specs.append({'name': '直方图梯度提升', 'algorithm': 'hist_gradient_boosting'})
    return specs


def tune_one(league_id, task_name, target_type):
    model_id = f'{league_id}{task_name}模型'
    model_db = ModelDatabase(league_id)
    saved_config = (
        model_db.load_model_config(model_id)
        if model_db.model_exists(model_id) else None
    )

    dataset = LeagueDatabase().load_league(league_id).dropna().reset_index(drop=True)
    if target_type == TargetType.HALF_FULL and 'HTR' not in dataset.columns:
        print(f'[{league_id} / {task_name}] 没有半场标签，跳过。', flush=True)
        return None

    train_validation, test_df = train_test_split(dataset, 20.0)
    train_df, validation_df = train_test_split(train_validation, 25.0)
    print(
        f'\n[{league_id} / {task_name}] '
        f'训练={len(train_df)} 验证={len(validation_df)} 测试={len(test_df)}',
        flush=True,
    )

    candidates = []
    for spec in candidate_specs(target_type):
        model = make_candidate(
            spec, league_id, model_id, target_type, saved_config=saved_config,
        )
        model.fit(train_df)
        accuracy, top3, top5 = prediction_metrics(model, validation_df, target_type)
        candidates.append({
            'spec': spec,
            'validation_accuracy': accuracy,
            'validation_top3': top3,
            'validation_top5': top5,
        })
        print(
            f'  {spec["name"]}: 首选={accuracy:.3f} '
            f'Top3={top3:.3f} Top5={top5:.3f}',
            flush=True,
        )

    best = max(
        candidates,
        key=lambda row: (
            row['validation_accuracy'],
            row['validation_top3'],
            row['validation_top5'],
        ),
    )
    final_spec = dict(best['spec'])
    if final_spec['algorithm'] in {'random_forest', 'extra_trees'}:
        final_spec['n_estimators'] = 1000
    final_model = make_candidate(
        final_spec, league_id, model_id, target_type, saved_config=saved_config,
    )
    final_model, metrics = Trainer().train(final_model, train_validation, test_df)
    test_accuracy, test_top3, test_top5 = prediction_metrics(
        final_model, test_df, target_type,
    )
    baseline = majority_baseline(train_validation, test_df, target_type)

    incumbent_metrics = None
    if saved_config is not None:
        incumbent, _ = model_db.load_model(model_id)
        incumbent_metrics = prediction_metrics(incumbent, test_df, target_type)

    replace = (
        incumbent_metrics is None
        or (test_accuracy, test_top3, test_top5) > incumbent_metrics
    )
    if replace:
        config = final_model.get_default_model_config()
        config['train'] = {
            'eval_samples_size': 20.0,
            'results': {'fit': metrics},
            'tuning': {
                'method': '60%训练 / 20%验证比较算法 / 20%最近比赛独立测试',
                'algorithm': best['spec']['name'],
                'validation_accuracy': best['validation_accuracy'],
                'validation_top3_accuracy': best['validation_top3'],
                'validation_top5_accuracy': best['validation_top5'],
                'test_accuracy': test_accuracy,
                'top3_accuracy': test_top3,
                'top5_accuracy': test_top5,
                'majority_baseline': baseline,
                'compared_algorithms': [row['spec']['name'] for row in candidates],
            },
        }
        model_db.save_model(final_model, config)
        kept_algorithm = best['spec']['name']
        kept_metrics = (test_accuracy, test_top3, test_top5)
        action = '已替换'
    else:
        kept_algorithm = saved_config.get('train', {}).get('tuning', {}).get(
            'algorithm', '现有随机森林',
        )
        kept_metrics = incumbent_metrics
        action = '保留现有模型'

    print(
        f'  [最终] {action}：{kept_algorithm}；'
        f'首选={kept_metrics[0]:.3f} Top3={kept_metrics[1]:.3f} '
        f'Top5={kept_metrics[2]:.3f} 基线={baseline:.3f}',
        flush=True,
    )
    return {
        '联赛': league_id,
        '任务': task_name,
        '模型': model_id,
        '处理': action,
        '保留算法': kept_algorithm,
        '首选命中率': kept_metrics[0],
        'Top3命中率': kept_metrics[1],
        'Top5命中率': kept_metrics[2],
        '简单基线': baseline,
        '验证集胜出算法': best['spec']['name'],
    }


if __name__ == '__main__':
    reports = []
    for league_id in LEAGUES:
        report = tune_one(league_id, '比分', TargetType.SCORE)
        if report:
            reports.append(report)
    for league_id in LEAGUES:
        report = tune_one(league_id, '半全场', TargetType.HALF_FULL)
        if report:
            reports.append(report)

    report_df = pd.DataFrame(reports)
    os.makedirs('storage/reports', exist_ok=True)
    report_df.to_csv('storage/reports/比分半全场算法对比报告.csv', index=False)
    print('\n' + report_df.to_string(index=False), flush=True)
