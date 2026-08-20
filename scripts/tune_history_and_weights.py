"""Tune history length and recency weights for every Big-Five saved model."""

import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.normalization import NormalizerType
from src.preprocessing.utils.target import TargetType, construct_targets


LEAGUES = ('英超', '西甲', '德甲', '意甲', '法甲')
WINDOWS = (2, 3, 5, 8, None)
HALF_LIVES = (2.0, 4.0, 6.0, 8.0)
BACKUP = Path('storage/backups/before-history-weight-tuning-20260806')
REPORT = Path('storage/reports/历史窗口与时间权重调优-20260806.csv')


def backup_models():
    BACKUP.mkdir(parents=True, exist_ok=True)
    source = Path('storage/leagues/model_index.pkl')
    shutil.copy2(source, BACKUP / 'model_index.pkl')
    for league in LEAGUES:
        src = Path('storage/leagues') / league / 'models'
        dst = BACKUP / league / 'models'
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def params_from_config(config, validation=True):
    params = {key: value for key, value in config.items() if key not in {'cls', 'train', 'eval'}}
    normalizer = params.get('normalizer')
    if isinstance(normalizer, StandardScaler):
        params['normalizer'] = NormalizerType.STANDARD
    elif isinstance(normalizer, MinMaxScaler):
        params['normalizer'] = NormalizerType.MIN_MAX
    elif isinstance(normalizer, MaxAbsScaler):
        params['normalizer'] = NormalizerType.MAX_ABS
    elif isinstance(normalizer, TransformerMixin):
        params['normalizer'] = None
    if validation and 'n_estimators' in params:
        params['n_estimators'] = min(400, int(params['n_estimators']))
    return params


def history_slice(df, years):
    if years is None:
        return df.reset_index(drop=True)
    dates = pd.to_datetime(df['Date'])
    cutoff = dates.max() - pd.DateOffset(years=years)
    return df.loc[dates >= cutoff].reset_index(drop=True)


def metrics(model, df, target_type):
    y = construct_targets(df, target_type)
    probability = model.predict_proba(df)
    classes = np.asarray(model.classifier.classes_)
    prediction = classes[probability.argmax(axis=1)]
    accuracy = float(np.mean(prediction == y))
    top = {}
    for k in (3, 5):
        width = min(k, probability.shape[1])
        indices = np.argpartition(probability, -width, axis=1)[:, -width:]
        top[k] = float(np.mean(np.any(classes[indices] == y[:, None], axis=1)))
    class_index = {int(label): index for index, label in enumerate(classes)}
    true_probability = np.array([
        probability[row, class_index[int(label)]] if int(label) in class_index else 1e-12
        for row, label in enumerate(y)
    ])
    return accuracy, top[3], top[5], float(-np.log(np.clip(true_probability, 1e-12, 1)).mean())


def rank(value, target_type):
    accuracy, top3, top5, logloss = value
    if target_type in {TargetType.SCORE, TargetType.HALF_FULL}:
        return round(accuracy, 3), top3, top5, -logloss
    return accuracy, -logloss


def tune_model(league, model_id, dataset):
    db = ModelDatabase(league)
    config = db.load_model_config(model_id)
    target = config['target_type']
    train_validation, test = train_test_split(dataset, 15.0)
    train, validation = train_test_split(train_validation, 15.0 / 85.0 * 100.0)
    base_params = params_from_config(config, validation=True)
    supports_weight = 'recency_half_life_years' in base_params
    candidates = []
    for years in WINDOWS:
        weight_values = HALF_LIVES if supports_weight else (None,)
        for half_life in weight_values:
            params = dict(base_params)
            if supports_weight:
                params['recency_half_life_years'] = half_life
            fit_df = history_slice(train, years)
            if len(fit_df) < 300:
                continue
            model = config['cls'](**params)
            model.fit(fit_df)
            value = metrics(model, validation, target)
            candidates.append((years, half_life, value, len(fit_df)))
            label = '全部' if years is None else f'{years}年'
            weight = f'/半衰期{half_life:g}年' if supports_weight else ''
            print(f'  {label}{weight}: n={len(fit_df)} acc={value[0]:.3f} top3={value[1]:.3f} logloss={value[3]:.3f}', flush=True)

    best = max(candidates, key=lambda row: rank(row[2], target))
    years, half_life, validation_value, validation_samples = best
    final_params = params_from_config(config, validation=False)
    if supports_weight:
        final_params['recency_half_life_years'] = half_life
    final_train = history_slice(train_validation, years)
    final_model = config['cls'](**final_params)
    final_model.fit(final_train)
    test_value = metrics(final_model, test, target)
    new_config = final_model.get_default_model_config()
    old_train = dict(config.get('train', {}))
    old_train['history_weight_tuning'] = {
        'method': '70%历史训练 / 15%时间验证选择历史窗口与权重 / 15%最近比赛最终测试',
        'history_years': years,
        'recency_half_life_years': half_life,
        'validation_accuracy': validation_value[0],
        'validation_top3': validation_value[1],
        'validation_top5': validation_value[2],
        'validation_logloss': validation_value[3],
        'test_accuracy': test_value[0],
        'test_top3': test_value[1],
        'test_top5': test_value[2],
        'test_logloss': test_value[3],
    }
    new_config['train'] = old_train
    db.save_model(final_model, new_config)
    return {
        '联赛': league, '模型': model_id, '预测类型': target.value,
        '最佳历史窗口年': '全部' if years is None else years,
        '最佳半衰期年': half_life, '最终训练样本': len(final_train),
        '验证准确率': validation_value[0], '最终测试准确率': test_value[0],
        '测试Top3': test_value[1], '测试Top5': test_value[2], '测试LogLoss': test_value[3],
    }


if __name__ == '__main__':
    backup_models()
    reports = []
    for league in LEAGUES:
        dataset = LeagueDatabase().load_league(league).dropna().reset_index(drop=True)
        db = ModelDatabase(league)
        for model_id in sorted(m for m in db.get_model_ids() if '早期模型' not in m):
            print(f'\n[{league}/{model_id}]', flush=True)
            reports.append(tune_model(league, model_id, dataset))
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(reports).to_csv(REPORT, index=False)
    print(pd.DataFrame(reports).to_string(index=False))
