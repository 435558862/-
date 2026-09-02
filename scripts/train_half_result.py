import os

import numpy as np
import pandas as pd

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.logistic import LogisticRegressor
from src.models.trainer import Trainer
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.normalization import NormalizerType
from src.preprocessing.utils.target import TargetType, construct_targets


LEAGUES = ['英超', '英冠', '西甲', '德甲', '意甲', '法甲', '葡超', '瑞超', '日职', '韩职']
CS = [0.01, 0.03, 0.1, 0.3, 1.0]
THRESHOLDS = np.arange(0.40, 0.66, 0.01)


def make_model(league_id, c):
    return LogisticRegressor(
        league_id=league_id,
        model_id=f'{league_id}半场胜平负模型',
        target_type=TargetType.HALF_RESULT,
        calibrate_probabilities=False,
        normalizer=NormalizerType.STANDARD,
        penalty='l2',
        fixed_c=c,
    )


def accuracy(model, df):
    y_true = construct_targets(df, TargetType.HALF_RESULT)
    y_prob = model.predict_proba(df)
    classes = np.asarray(model.classifier.classes_)
    y_pred = classes[y_prob.argmax(axis=1)]
    return float(np.mean(y_pred == y_true)), y_true, y_pred, y_prob


def choose_threshold(y_true, y_pred, y_prob):
    candidates = []
    for threshold in THRESHOLDS:
        selected = y_prob.max(axis=1) >= threshold
        coverage = float(selected.mean())
        selected_accuracy = (
            float(np.mean(y_pred[selected] == y_true[selected]))
            if selected.any() else 0.0
        )
        candidates.append((threshold, selected_accuracy, coverage, int(selected.sum())))

    # Keep a safety margin on validation instead of merely touching 50%.
    # Selective accuracy commonly regresses a few points on the newest matches.
    eligible = [
        row for row in candidates
        if row[1] >= 0.55 and row[2] >= 0.05 and row[3] >= 60
    ]
    if eligible:
        return max(eligible, key=lambda row: (row[2], row[1]))
    return max(candidates, key=lambda row: (row[1], row[2]))


def train_one(league_id):
    dataset = LeagueDatabase().load_league(league_id).dropna().reset_index(drop=True)
    train_validation, test_df = train_test_split(dataset, 20.0)
    train_df, validation_df = train_test_split(train_validation, 25.0)

    best = None
    print(f'\n[{league_id} / 半场胜平负] 训练={len(train_df)} 验证={len(validation_df)} 测试={len(test_df)}', flush=True)
    for c in CS:
        model = make_model(league_id, c)
        model.fit(train_df)
        validation_accuracy, y_true, y_pred, y_prob = accuracy(model, validation_df)
        threshold = choose_threshold(y_true, y_pred, y_prob)
        print(
            f'  C={c}: 全部={validation_accuracy:.3f} '
            f'高置信度={threshold[1]:.3f} 覆盖={threshold[2]:.3f} 门槛={threshold[0]:.2f}',
            flush=True,
        )
        ranking = (validation_accuracy, threshold[1], threshold[2])
        if best is None or ranking > best['ranking']:
            best = {
                'c': c,
                'validation_accuracy': validation_accuracy,
                'threshold': threshold[0],
                'validation_selective_accuracy': threshold[1],
                'validation_coverage': threshold[2],
                'ranking': ranking,
            }

    final_model = make_model(league_id, best['c'])
    final_model, metrics = Trainer().train(final_model, train_validation, test_df)
    test_accuracy, y_true, y_pred, y_prob = accuracy(final_model, test_df)
    selected = y_prob.max(axis=1) >= best['threshold']
    selective_accuracy = (
        float(np.mean(y_pred[selected] == y_true[selected]))
        if selected.any() else 0.0
    )
    coverage = float(selected.mean())

    train_targets = construct_targets(train_validation, TargetType.HALF_RESULT)
    test_targets = construct_targets(test_df, TargetType.HALF_RESULT)
    values, counts = np.unique(train_targets, return_counts=True)
    majority = values[counts.argmax()]
    baseline = float(np.mean(test_targets == majority))

    config = final_model.get_default_model_config()
    config['train'] = {
        'eval_samples_size': 20.0,
        'results': {'fit': metrics},
        'tuning': {
            'method': '60%训练 / 20%验证选正则与置信度门槛 / 20%最近比赛独立测试',
            'algorithm': '标准化多项逻辑回归',
            'selected_c': best['c'],
            'validation_accuracy': best['validation_accuracy'],
            'test_accuracy': test_accuracy,
            'majority_baseline': baseline,
            'selective_threshold': float(best['threshold']),
            'selective_accuracy': selective_accuracy,
            'coverage': coverage,
            'validation_selective_accuracy': best['validation_selective_accuracy'],
            'validation_coverage': best['validation_coverage'],
        },
    }
    ModelDatabase(league_id).save_model(final_model, config)
    print(
        f'  [最终] 全部={test_accuracy:.3f} 基线={baseline:.3f} '
        f'高置信度={selective_accuracy:.3f} 覆盖={coverage:.3f} '
        f'门槛={best["threshold"]:.2f}',
        flush=True,
    )
    return {
        '联赛': league_id,
        '模型': final_model.model_id,
        '全部场次准确率': test_accuracy,
        '简单基线': baseline,
        '高置信度门槛': best['threshold'],
        '高置信度命中率': selective_accuracy,
        '覆盖率': coverage,
        'C': best['c'],
    }


if __name__ == '__main__':
    reports = [train_one(league_id) for league_id in LEAGUES]
    report_df = pd.DataFrame(reports)
    os.makedirs('storage/reports', exist_ok=True)
    report_df.to_csv('storage/reports/半场胜平负调教报告.csv', index=False)
    print('\n' + report_df.to_string(index=False), flush=True)
