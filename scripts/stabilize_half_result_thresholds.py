import numpy as np
import pandas as pd

from scripts.optimize_structured_models import (
    REPORT_PATH,
    clean_incumbent_params,
    probability_metrics,
    split_dataset,
)
from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.preprocessing.utils.target import TargetType


LEAGUES = ['英超', '西甲', '德甲', '意甲', '法甲']


def choose_threshold(metrics):
    rows = []
    confidence = metrics['probabilities'].max(axis=1)
    for threshold in np.arange(0.35, 0.76, 0.01):
        mask = confidence >= threshold
        count = int(mask.sum())
        accuracy = (
            float(np.mean(metrics['y_pred'][mask] == metrics['y_true'][mask]))
            if count else 0.0
        )
        rows.append((threshold, accuracy, float(mask.mean()), count))

    # A 60% validation target gives enough safety margin for temporal drift.
    eligible = [row for row in rows if row[1] >= 0.60 and row[3] >= 40]
    if eligible:
        return max(eligible, key=lambda row: (row[2], row[1])), True
    stable = [row for row in rows if row[3] >= 40]
    return max(stable, key=lambda row: (row[1], row[2])), False


if __name__ == '__main__':
    report = pd.read_csv(REPORT_PATH)
    for league_id in LEAGUES:
        model_id = f'{league_id}半场胜平负模型'
        model_db = ModelDatabase(league_id)
        config = model_db.load_model_config(model_id)
        dataset = LeagueDatabase().load_league(league_id).dropna().reset_index(drop=True)
        train_df, validation_df, _, test_df = split_dataset(dataset)

        candidate = config['cls'](**clean_incumbent_params(config, final=True))
        candidate.fit(train_df)
        validation = probability_metrics(
            candidate, validation_df, TargetType.HALF_RESULT,
        )
        selected, validated = choose_threshold(validation)
        threshold, validation_accuracy, validation_coverage, _ = selected

        final_model, _ = model_db.load_model(model_id)
        final = probability_metrics(
            final_model, test_df, TargetType.HALF_RESULT,
        )
        mask = final['probabilities'].max(axis=1) >= threshold
        test_accuracy = (
            float(np.mean(final['y_pred'][mask] == final['y_true'][mask]))
            if mask.any() else 0.0
        )
        coverage = float(mask.mean())
        samples = int(mask.sum())

        tuning = config['train']['tuning']
        tuning.update({
            'selective_threshold': float(threshold),
            'selective_validation_target': 0.60,
            'selective_validated': validated,
            'validation_selective_accuracy': validation_accuracy,
            'validation_coverage': validation_coverage,
            'selective_accuracy': test_accuracy,
            'coverage': coverage,
            'selective_samples': samples,
        })
        model_db.update_model_config(config)

        report_mask = (
            (report['联赛'] == league_id)
            & (report['预测类型'] == '半场胜平负')
        )
        report.loc[report_mask, '高置信度门槛'] = threshold
        report.loc[report_mask, '高置信度命中率'] = test_accuracy
        report.loc[report_mask, '高置信度覆盖率'] = coverage
        report.loc[report_mask, '高置信度样本数'] = samples
        report.loc[report_mask, '高置信度验证达标'] = validated

        print(
            f'{league_id}: 门槛={threshold:.2f} '
            f'验证={validation_accuracy:.1%}/{validation_coverage:.1%} '
            f'测试={test_accuracy:.1%}/{coverage:.1%} ({samples}场) '
            f'验证达标={"是" if validated else "否"}',
            flush=True,
        )

    report.to_csv(REPORT_PATH, index=False)
