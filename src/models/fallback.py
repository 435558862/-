"""Portable result models used when a competition-specific model is unavailable."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.model import ClassificationModel
from src.models.classifiers.football import ConditionalHalfFullEstimator, GoalDistributionEstimator
from src.preprocessing.utils.target import TargetType


ODDS_COLUMNS = ['1', 'X', '2']
COMMON_FORM_COLUMNS = [
    'HW', 'AW', 'HL', 'AL', 'HGF', 'AGF', 'HAGF', 'HGA', 'AGA', 'HAGA',
    'HGD', 'AGD', 'HAGD', 'HWGD', 'AWGD', 'HAWGD', 'HLGD', 'ALGD',
    'HALGD', 'HW%', 'HL%', 'AW%', 'AL%',
]
GENERIC_FEATURE_COLUMNS = ODDS_COLUMNS + COMMON_FORM_COLUMNS
XG_FEATURE_COLUMNS = [
    'HXGF5', 'HXGA5', 'HXGD5', 'AXGF5', 'AXGA5', 'AXGD5',
    'HXGF10', 'HXGA10', 'AXGF10', 'AXGA10',
]
LINEUP_FEATURE_COLUMNS = [
    'HLineupContinuity5', 'ALineupContinuity5',
    'HLineupCore5', 'ALineupCore5',
]
XG_ENHANCED_GENERIC_FEATURE_COLUMNS = GENERIC_FEATURE_COLUMNS + XG_FEATURE_COLUMNS
ENHANCED_GENERIC_FEATURE_COLUMNS = (
    XG_ENHANCED_GENERIC_FEATURE_COLUMNS + LINEUP_FEATURE_COLUMNS
)


def _market_probabilities(frame: pd.DataFrame) -> np.ndarray:
    odds = frame[ODDS_COLUMNS].to_numpy(dtype=np.float64)
    inverse = 1.0 / np.clip(odds, 1.01, None)
    return inverse / inverse.sum(axis=1, keepdims=True)


class ColumnResultEstimator(BaseEstimator):
    """Logistic result estimator with stable, explicitly named inputs."""

    def __init__(self, c: float = 0.1):
        self.c = float(c)
        self.scaler_ = StandardScaler()
        self.model_ = LogisticRegression(
            C=self.c, max_iter=5000, solver='lbfgs', random_state=0,
        )
        self.classes_ = np.arange(3, dtype=np.int32)

    def fit(self, x, y, sample_weight=None):
        values = self.scaler_.fit_transform(np.asarray(x, dtype=np.float64))
        self.model_.fit(values, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, x):
        values = self.scaler_.transform(np.asarray(x, dtype=np.float64))
        raw = self.model_.predict_proba(values)
        aligned = np.zeros((len(values), 3), dtype=np.float64)
        aligned[:, self.model_.classes_.astype(int)] = raw
        return aligned

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)


class PortableResultModel(ClassificationModel):
    """A cross-league model that is unaffected by dataframe column order/schema."""

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType = TargetType.RESULT,
            feature_columns: Optional[List[str]] = None,
            c: float = 0.1,
            market_weight: float = 0.5,
            recency_half_life_years: Optional[float] = 4.0,
            calibrate_probabilities: bool = False,
            normalizer=None,
            sampler=None,
            **kwargs,
    ):
        self._feature_columns = list(feature_columns or ODDS_COLUMNS)
        self._c = float(c)
        self._market_weight = float(market_weight)
        self._recency_half_life_years = recency_half_life_years
        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=TargetType.RESULT,
            calibrate_probabilities=False,
            normalizer=None,
            sampler=None,
            **kwargs,
        )

    @property
    def feature_columns(self) -> List[str]:
        return list(self._feature_columns)

    def _inputs(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self._feature_columns if column not in frame]
        if missing:
            raise ValueError(f'通用模型缺少输入列：{", ".join(missing)}')
        values = frame[self._feature_columns].apply(pd.to_numeric, errors='coerce')
        if values.isna().any().any():
            raise ValueError('通用模型输入包含空值或非数字。')
        return values.to_numpy(dtype=np.float64)

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        return ColumnResultEstimator(c=self._c)

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        required = self._feature_columns + ['Result']
        clean = train_df.dropna(subset=required).reset_index(drop=True)
        x = self._inputs(clean)
        y = clean['Result'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32)
        weights = None
        if self._recency_half_life_years is not None and 'Date' in clean:
            dates = pd.to_datetime(clean['Date'])
            age = (dates.max() - dates).dt.days.to_numpy(dtype=np.float64) / 365.25
            weights = np.power(0.5, age / self._recency_half_life_years)
            weights /= weights.mean()
        self._classifier = self.build_classifier(x.shape[1], 3)
        self._classifier.fit(x, y, sample_weight=weights)
        return pd.DataFrame()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        learned = self._classifier.predict_proba(self._inputs(df))
        market = _market_probabilities(df)
        probability = (
            (1.0 - self._market_weight) * learned
            + self._market_weight * market
        )
        return probability / probability.sum(axis=1, keepdims=True)

    def predict(self, df: pd.DataFrame, return_targets: bool = False):
        predicted = self.predict_proba(df).argmax(axis=1)
        target = None
        if return_targets and 'Result' in df:
            target = df['Result'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32)
        return predicted, target

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'c':
            return [0.01, 0.03, 0.1, 0.3]
        if param == 'market_weight':
            return [0.0, 0.25, 0.5, 0.75]
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'feature_columns': self._feature_columns,
            'c': self._c,
            'market_weight': self._market_weight,
            'recency_half_life_years': self._recency_half_life_years,
        })
        return model_config

class PortableGoalDistributionModel(ClassificationModel):
    """Cross-league score model with stable named inputs."""

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType = TargetType.SCORE,
            feature_columns: Optional[List[str]] = None,
            algorithm: str = 'poisson_linear',
            algorithm_params: Optional[Dict[str, Any]] = None,
            alpha: float = 1.0,
            rho: Union[float, str] = 'auto',
            mean_shrinkage: float = 0.1,
            score_calibration_weights: Optional[List[float]] = None,
            recency_half_life_years: Optional[float] = 4.0,
            calibrate_probabilities: bool = False,
            normalizer=None,
            sampler=None,
            **kwargs,
    ):
        self._feature_columns = list(feature_columns or GENERIC_FEATURE_COLUMNS)
        self._algorithm = algorithm
        self._algorithm_params = dict(algorithm_params or {})
        self._alpha = float(alpha)
        self._rho = rho
        self._mean_shrinkage = float(mean_shrinkage)
        self._score_calibration_weights = (
            None if score_calibration_weights is None
            else np.asarray(score_calibration_weights, dtype=np.float64)
        )
        self._recency_half_life_years = recency_half_life_years
        super().__init__(
            league_id=league_id, model_id=model_id, target_type=TargetType.SCORE,
            calibrate_probabilities=False, normalizer=None, sampler=None, **kwargs,
        )

    def _inputs(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self._feature_columns if column not in frame]
        if missing:
            raise ValueError(f'通用进球模型缺少输入列：{", ".join(missing)}')
        values = frame[self._feature_columns].apply(pd.to_numeric, errors='coerce')
        if values.isna().any().any():
            raise ValueError('通用进球模型输入包含空值或非数字。')
        return values.to_numpy(dtype=np.float64)

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        return GoalDistributionEstimator(
            target_type=TargetType.SCORE, algorithm=self._algorithm,
            alpha=self._alpha, rho=self._rho, mean_shrinkage=self._mean_shrinkage,
            algorithm_params=self._algorithm_params,
        )

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        required = self._feature_columns + ['HG', 'AG']
        clean = train_df.dropna(subset=required).reset_index(drop=True)
        weights = None
        if self._recency_half_life_years is not None and 'Date' in clean:
            dates = pd.to_datetime(clean['Date'])
            age = (dates.max() - dates).dt.days.to_numpy(dtype=np.float64) / 365.25
            weights = np.power(0.5, age / self._recency_half_life_years)
            weights /= weights.mean()
        self._classifier = self.build_classifier(len(self._feature_columns), 49)
        self._classifier.fit_goals(
            self._inputs(clean), clean['HG'].to_numpy(dtype=np.float64),
            clean['AG'].to_numpy(dtype=np.float64), sample_weight=weights,
        )
        return pd.DataFrame()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        probability = self._classifier.predict_proba(self._inputs(df))
        if self._score_calibration_weights is not None:
            if probability.shape[1] != len(self._score_calibration_weights):
                raise ValueError('比分校准权重与模型类别数量不一致。')
            probability = probability * self._score_calibration_weights
            probability = probability / probability.sum(axis=1, keepdims=True)
        return probability

    def predict(self, df: pd.DataFrame, return_targets: bool = False):
        predicted = self.predict_proba(df).argmax(axis=1)
        return predicted, None

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'alpha':
            return [0.01, 0.1, 1.0]
        if param == 'mean_shrinkage':
            return [0.0, 0.1, 0.2]
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'feature_columns': self._feature_columns,
            'algorithm': self._algorithm,
            'algorithm_params': self._algorithm_params,
            'alpha': self._alpha,
            'rho': self._rho,
            'mean_shrinkage': self._mean_shrinkage,
            'score_calibration_weights': (
                None if self._score_calibration_weights is None
                else self._score_calibration_weights.tolist()
            ),
            'recency_half_life_years': self._recency_half_life_years,
        })
        return model_config


class PortableHalfFullModel(ClassificationModel):
    """Cross-league half/full model with stable named inputs."""

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType = TargetType.HALF_FULL,
            feature_columns: Optional[List[str]] = None,
            half_c: float = 0.1,
            full_c: float = 0.1,
            recency_half_life_years: Optional[float] = 8.0,
            calibrate_probabilities: bool = False,
            normalizer=None,
            sampler=None,
            **kwargs,
    ):
        self._feature_columns = list(feature_columns or GENERIC_FEATURE_COLUMNS)
        self._half_c = float(half_c)
        self._full_c = float(full_c)
        self._recency_half_life_years = recency_half_life_years
        super().__init__(
            league_id=league_id, model_id=model_id, target_type=TargetType.HALF_FULL,
            calibrate_probabilities=False, normalizer=None, sampler=None, **kwargs,
        )

    def _inputs(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self._feature_columns if column not in frame]
        if missing:
            raise ValueError(f'通用半全场模型缺少输入列：{", ".join(missing)}')
        values = frame[self._feature_columns].apply(pd.to_numeric, errors='coerce')
        if values.isna().any().any():
            raise ValueError('通用半全场模型输入包含空值或非数字。')
        return values.to_numpy(dtype=np.float64)

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        return ConditionalHalfFullEstimator(self._half_c, self._full_c)

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        required = self._feature_columns + ['HTR', 'Result']
        clean = train_df.dropna(subset=required).reset_index(drop=True)
        weights = None
        if self._recency_half_life_years is not None and 'Date' in clean:
            dates = pd.to_datetime(clean['Date'])
            age = (dates.max() - dates).dt.days.to_numpy(dtype=np.float64) / 365.25
            weights = np.power(0.5, age / self._recency_half_life_years)
            weights /= weights.mean()
        self._classifier = self.build_classifier(len(self._feature_columns), 9)
        self._classifier.fit_outcomes(
            self._inputs(clean),
            clean['HTR'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32),
            clean['Result'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32),
            sample_weight=weights,
        )
        return pd.DataFrame()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._classifier.predict_proba(self._inputs(df))

    def predict(self, df: pd.DataFrame, return_targets: bool = False):
        predicted = self.predict_proba(df).argmax(axis=1)
        return predicted, None

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param in {'half_c', 'full_c'}:
            return [0.01, 0.03, 0.1, 0.3]
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'feature_columns': self._feature_columns,
            'half_c': self._half_c,
            'full_c': self._full_c,
            'recency_half_life_years': self._recency_half_life_years,
        })
        return model_config
