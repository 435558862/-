"""Time-aware probability evaluation utilities for football models."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from src.preprocessing.utils.target import TargetType, construct_targets


@dataclass(frozen=True)
class WalkForwardFold:
    train: pd.DataFrame
    test: pd.DataFrame
    fold: int


def expanding_window_folds(
        df: pd.DataFrame,
        folds: int = 5,
        test_fraction: float = 0.05,
        min_train_fraction: float = 0.60,
) -> Iterable[WalkForwardFold]:
    """Yield chronological expanding-window folds from newest-first data."""
    if not df['Date'].is_monotonic_decreasing:
        raise ValueError('Expected dates sorted newest first.')
    if folds < 1 or not 0 < test_fraction < 1 or not 0 < min_train_fraction < 1:
        raise ValueError('Invalid walk-forward settings.')

    chronological = df.iloc[::-1].reset_index(drop=True)
    n_rows = len(chronological)
    test_size = max(1, int(np.floor(n_rows * test_fraction)))
    first_test = max(int(np.floor(n_rows * min_train_fraction)), n_rows - folds * test_size)
    for fold in range(folds):
        start = first_test + fold * test_size
        stop = min(start + test_size, n_rows)
        if start >= n_rows:
            break
        yield WalkForwardFold(
            train=chronological.iloc[:start].iloc[::-1].reset_index(drop=True),
            test=chronological.iloc[start:stop].iloc[::-1].reset_index(drop=True),
            fold=fold + 1,
        )


def probability_metrics(model, df: pd.DataFrame, target_type: TargetType) -> Dict[str, float]:
    y_true = construct_targets(df, target_type)
    probabilities = np.asarray(model.predict_proba(df), dtype=np.float64)
    classes = np.asarray(model.classifier.classes_, dtype=np.int32)
    prediction = classes[probabilities.argmax(axis=1)]
    class_to_column = {int(value): i for i, value in enumerate(classes)}
    columns = np.array([class_to_column.get(int(value), -1) for value in y_true])
    present = columns >= 0
    true_probability = np.full(len(y_true), 1e-12, dtype=np.float64)
    true_probability[present] = probabilities[np.arange(len(y_true))[present], columns[present]]
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true))[present], columns[present]] = 1.0
    return {
        'samples': int(len(y_true)),
        'accuracy': float(np.mean(prediction == y_true)),
        'log_loss': float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean()),
        'brier': float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        'mean_confidence': float(probabilities.max(axis=1).mean()),
    }


def walk_forward_evaluate(
        model_factory: Callable[[], object],
        df: pd.DataFrame,
        target_type: TargetType,
        folds: int = 5,
        test_fraction: float = 0.05,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for split in expanding_window_folds(df, folds=folds, test_fraction=test_fraction):
        model = model_factory()
        model.fit(split.train)
        row = probability_metrics(model, split.test, target_type)
        row.update({
            'fold': split.fold,
            'train_samples': len(split.train),
            'test_start': split.test['Date'].min(),
            'test_end': split.test['Date'].max(),
        })
        rows.append(row)
    return pd.DataFrame(rows)
