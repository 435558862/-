import numpy as np
import pandas as pd
from enum import Enum
from sklearn.preprocessing import OneHotEncoder


class TargetType(Enum):
    """ The supported target types. """

    RESULT = 'result'
    OVER_UNDER = 'over-under'
    HALF_FULL = 'half-full'
    SCORE = 'score'
    HALF_RESULT = 'half-result'


SCORE_CAP = 6


def score_to_class(home_goals: int, away_goals: int) -> int:
    home = min(int(home_goals), SCORE_CAP)
    away = min(int(away_goals), SCORE_CAP)
    return home * (SCORE_CAP + 1) + away


def class_to_score(target_class: int) -> str:
    home, away = divmod(int(target_class), SCORE_CAP + 1)
    home_text = f'{SCORE_CAP}+' if home == SCORE_CAP else str(home)
    away_text = f'{SCORE_CAP}+' if away == SCORE_CAP else str(away)
    return f'{home_text}-{away_text}'


def construct_targets(df: pd.DataFrame, target_type: TargetType) -> np.ndarray:
    """ Constructs the dataset targets based on the selected classification task """

    if target_type == TargetType.RESULT:
        y = df['Result'].map({'H': 0, 'D': 1, 'A': 2}).fillna(0).to_numpy(dtype=np.int32)
    elif target_type == TargetType.OVER_UNDER:
        y = (df['HG'] + df['AG']).ge(2.5).to_numpy(dtype=np.int32)
    elif target_type == TargetType.HALF_FULL:
        labels = ['HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA']
        mapper = {label: index for index, label in enumerate(labels)}
        # Unknown targets are present in future fixtures; a placeholder is harmless
        # because prediction paths ignore y and only consume the constructed inputs.
        y = (df['HTR'].astype(str) + df['Result'].astype(str)).map(mapper).fillna(0).to_numpy(dtype=np.int32)
    elif target_type == TargetType.SCORE:
        y = np.array(
            [score_to_class(home, away) for home, away in zip(df['HG'], df['AG'])],
            dtype=np.int32,
        )
    elif target_type == TargetType.HALF_RESULT:
        y = df['HTR'].map({'H': 0, 'D': 1, 'A': 2}).fillna(0).to_numpy(dtype=np.int32)
    else:
        raise TypeError(f'Undefiend target type: "{target_type.name}"')

    return y


def one_hot_encode(y: np.ndarray, target_type: TargetType) -> np.ndarray:
    """ One-Hot encodes the provided targets. To ensure consistency,
        the target categories are fixed and depend on the target type.
    """

    if target_type == TargetType.RESULT:
        y_encoded = OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False).fit_transform(y.reshape(-1, 1))
    elif target_type == TargetType.HALF_FULL:
        y_encoded = OneHotEncoder(categories=[list(range(9))], sparse_output=False).fit_transform(y.reshape(-1, 1))
    elif target_type == TargetType.SCORE:
        y_encoded = OneHotEncoder(
            categories=[list(range((SCORE_CAP + 1) ** 2))],
            sparse_output=False,
        ).fit_transform(y.reshape(-1, 1))
    elif target_type == TargetType.HALF_RESULT:
        y_encoded = OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False).fit_transform(y.reshape(-1, 1))
    elif target_type == TargetType.OVER_UNDER:
        raise TypeError('OVER_UNDER targets do not support one-hot encoding, as it is binary classification task.')
    else:
        raise TypeError(f'Not supported target type: "{type(target_type)}"')

    return y_encoded
