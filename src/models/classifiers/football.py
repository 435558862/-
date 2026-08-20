from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from src.models.model import ClassificationModel
from src.preprocessing.utils.target import TargetType


def _recency_weights(df: pd.DataFrame, half_life_years: Optional[float]) -> Optional[np.ndarray]:
    if half_life_years is None:
        return None
    dates = pd.to_datetime(df['Date'])
    age_years = (dates.max() - dates).dt.days.to_numpy(dtype=np.float64) / 365.25
    weights = np.power(0.5, age_years / float(half_life_years))
    return weights / weights.mean()


class GoalDistributionEstimator(BaseEstimator):
    """Predicts home/away goal means and converts them to coherent probabilities."""

    def __init__(
            self,
            target_type: TargetType,
            algorithm: str = 'poisson_linear',
            alpha: float = 1.0,
            rho: Union[float, str] = -0.05,
            mean_shrinkage: float = 0.0,
            algorithm_params: Optional[Dict[str, Any]] = None,
    ):
        self.target_type = target_type
        self.algorithm = algorithm
        self.alpha = alpha
        self.rho = rho
        self.mean_shrinkage = mean_shrinkage
        self.algorithm_params = dict(algorithm_params or {})
        self.scaler_ = None
        self.home_model_ = None
        self.away_model_ = None
        self.home_prior_ = None
        self.away_prior_ = None
        self.rho_ = None
        self.classes_ = (
            np.arange(49, dtype=np.int32)
            if target_type == TargetType.SCORE
            else np.arange(2, dtype=np.int32)
        )

    def _build_regressor(self):
        if self.algorithm == 'poisson_linear':
            return PoissonRegressor(
                alpha=self.alpha,
                max_iter=3000,
                tol=1e-8,
            )
        if self.algorithm == 'hist_poisson':
            params = {
                'loss': 'poisson',
                'learning_rate': 0.04,
                'max_iter': 220,
                'max_leaf_nodes': 15,
                'min_samples_leaf': 25,
                'l2_regularization': 5.0,
                'early_stopping': False,
                'random_state': 0,
            }
            params.update(self.algorithm_params)
            return HistGradientBoostingRegressor(**params)
        raise ValueError(f'Unknown goal algorithm: {self.algorithm}')

    def fit_goals(
            self,
            x: np.ndarray,
            home_goals: np.ndarray,
            away_goals: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ):
        train_x = np.asarray(x, dtype=np.float64)
        if not 0.0 <= float(self.mean_shrinkage) <= 1.0:
            raise ValueError('mean_shrinkage must be between 0 and 1.')
        self.home_prior_ = float(np.average(home_goals, weights=sample_weight))
        self.away_prior_ = float(np.average(away_goals, weights=sample_weight))
        if self.algorithm == 'poisson_linear':
            self.scaler_ = StandardScaler().fit(train_x)
            train_x = self.scaler_.transform(train_x)

        self.home_model_ = self._build_regressor()
        self.away_model_ = self._build_regressor()
        self.home_model_.fit(train_x, home_goals, sample_weight=sample_weight)
        self.away_model_.fit(train_x, away_goals, sample_weight=sample_weight)
        if self.rho == 'auto':
            self.rho_ = self._fit_rho(x, home_goals, away_goals, sample_weight)
        else:
            self.rho_ = float(self.rho)
        return self

    def _means(self, x: np.ndarray):
        predict_x = np.asarray(x, dtype=np.float64)
        if self.scaler_ is not None:
            predict_x = self.scaler_.transform(predict_x)
        home = self.home_model_.predict(predict_x)
        away = self.away_model_.predict(predict_x)
        # getattr keeps classifiers pickled before this option was introduced usable.
        shrinkage = float(getattr(self, 'mean_shrinkage', 0.0))
        if shrinkage:
            home = (1.0 - shrinkage) * home + shrinkage * self.home_prior_
            away = (1.0 - shrinkage) * away + shrinkage * self.away_prior_
        home = np.clip(home, 0.05, 6.5)
        away = np.clip(away, 0.05, 6.5)
        return home, away

    def _fit_rho(
            self,
            x: np.ndarray,
            home_goals: np.ndarray,
            away_goals: np.ndarray,
            sample_weight: Optional[np.ndarray],
    ) -> float:
        """Estimate Dixon-Coles rho by weighted training likelihood.

        Only the four low-score cells depend on rho. A conservative bounded grid
        keeps the correction positive and makes old serialized models compatible.
        """
        home_mean, away_mean = self._means(x)
        home_goals = np.asarray(home_goals, dtype=np.int32)
        away_goals = np.asarray(away_goals, dtype=np.int32)
        weights = (
            np.ones(len(home_goals), dtype=np.float64)
            if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        )
        best_rho, best_score = 0.0, -np.inf
        base00 = np.exp(-home_mean - away_mean)
        base01 = base00 * away_mean
        base10 = base00 * home_mean
        base11 = base00 * home_mean * away_mean
        for rho in np.linspace(-0.20, 0.15, 71):
            correction = np.ones(len(home_goals), dtype=np.float64)
            masks_and_values = (
                ((home_goals == 0) & (away_goals == 0), 1.0 - home_mean * away_mean * rho),
                ((home_goals == 0) & (away_goals == 1), 1.0 + home_mean * rho),
                ((home_goals == 1) & (away_goals == 0), 1.0 + away_mean * rho),
                ((home_goals == 1) & (away_goals == 1), 1.0 - rho),
            )
            for mask, value in masks_and_values:
                correction[mask] = value if np.ndim(value) == 0 else value[mask]
            if np.any(correction <= 0.0):
                continue
            normalization = (
                1.0
                + base00 * ((1.0 - home_mean * away_mean * rho) - 1.0)
                + base01 * ((1.0 + home_mean * rho) - 1.0)
                + base10 * ((1.0 + away_mean * rho) - 1.0)
                + base11 * ((1.0 - rho) - 1.0)
            )
            score = float(np.sum(weights * (
                np.log(correction) - np.log(np.clip(normalization, 1e-12, None))
            )))
            if score > best_score:
                best_rho, best_score = float(rho), score
        return best_rho

    @staticmethod
    def _capped_poisson(mean: np.ndarray) -> np.ndarray:
        goals = np.arange(6, dtype=np.float64)
        probabilities = np.exp(
            -mean[:, None]
            + goals[None, :] * np.log(mean[:, None])
            - gammaln(goals[None, :] + 1.0)
        )
        tail = np.maximum(0.0, 1.0 - probabilities.sum(axis=1, keepdims=True))
        return np.hstack([probabilities, tail])

    def _score_probabilities(self, x: np.ndarray) -> np.ndarray:
        home_mean, away_mean = self._means(x)
        home_prob = self._capped_poisson(home_mean)
        away_prob = self._capped_poisson(away_mean)
        joint = home_prob[:, :, None] * away_prob[:, None, :]

        # Dixon-Coles low-score correction. Negative rho increases correlated draws.
        fitted_rho = getattr(self, 'rho_', None)
        rho = float(fitted_rho if fitted_rho is not None else self.rho)
        joint[:, 0, 0] *= np.maximum(0.05, 1.0 - home_mean * away_mean * rho)
        joint[:, 0, 1] *= np.maximum(0.05, 1.0 + home_mean * rho)
        joint[:, 1, 0] *= np.maximum(0.05, 1.0 + away_mean * rho)
        joint[:, 1, 1] *= np.maximum(0.05, 1.0 - rho)
        joint /= joint.sum(axis=(1, 2), keepdims=True)
        return joint.reshape(-1, 49)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self._score_probabilities(x)
        if self.target_type == TargetType.SCORE:
            return scores

        grid = np.add.outer(np.arange(7), np.arange(7)).reshape(-1)
        under = scores[:, grid <= 2].sum(axis=1)
        return np.column_stack([under, 1.0 - under])

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)
        return self.classes_[probabilities.argmax(axis=1)]


class GoalDistributionModel(ClassificationModel):
    """Football-specific model for exact score or over/under 2.5."""

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            calibrate_probabilities: bool = False,
            normalizer=None,
            sampler=None,
            algorithm: str = 'poisson_linear',
            alpha: float = 1.0,
            rho: Union[float, str] = -0.05,
            mean_shrinkage: float = 0.0,
            algorithm_params: Optional[Dict[str, Any]] = None,
            recency_half_life_years: Optional[float] = None,
            **kwargs,
    ):
        if target_type not in {TargetType.SCORE, TargetType.OVER_UNDER}:
            raise ValueError('GoalDistributionModel supports score and over/under only.')
        self._algorithm = algorithm
        self._alpha = alpha
        self._rho = rho
        self._mean_shrinkage = mean_shrinkage
        self._algorithm_params = dict(algorithm_params or {})
        self._recency_half_life_years = recency_half_life_years
        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=False,
            normalizer=None,
            sampler=None,
            **kwargs,
        )

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        return GoalDistributionEstimator(
            target_type=self._target_type,
            algorithm=self._algorithm,
            alpha=self._alpha,
            rho=self._rho,
            mean_shrinkage=self._mean_shrinkage,
            algorithm_params=self._algorithm_params,
        )

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        clean = train_df.dropna().reset_index(drop=True)
        x, _, _ = self._dataset_preprocessor.preprocess_dataset(
            clean, target_type=self._target_type,
        )
        self._classifier = self.build_classifier(x.shape[1], 49)
        self._classifier.fit_goals(
            x,
            clean['HG'].to_numpy(dtype=np.float64),
            clean['AG'].to_numpy(dtype=np.float64),
            sample_weight=_recency_weights(clean, self._recency_half_life_years),
        )
        if eval_df is not None:
            self._num_eval_samples = len(eval_df)
        return self._evaluate_classifier(train_df=clean, eval_df=eval_df)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'algorithm':
            return ['poisson_linear', 'hist_poisson']
        if param == 'alpha':
            return [0.01, 0.1, 1.0, 10.0]
        if param == 'rho':
            return ['auto', -0.15, -0.10, -0.05, 0.0, 0.05]
        if param == 'mean_shrinkage':
            return [0.0, 0.05, 0.10, 0.20]
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'algorithm': self._algorithm,
            'alpha': self._alpha,
            'rho': self._rho,
            'mean_shrinkage': self._mean_shrinkage,
            'algorithm_params': self._algorithm_params,
            'recency_half_life_years': self._recency_half_life_years,
        })
        return model_config


class ConditionalHalfFullEstimator(BaseEstimator):
    """P(half result) × P(full result | half result, match inputs)."""

    def __init__(self, half_c: float = 0.1, full_c: float = 0.1):
        self.half_c = half_c
        self.full_c = full_c
        self.scaler_ = StandardScaler()
        self.half_model_ = LogisticRegression(
            C=half_c, max_iter=5000, solver='lbfgs', random_state=0,
        )
        self.full_model_ = LogisticRegression(
            C=full_c, max_iter=5000, solver='lbfgs', random_state=0,
        )
        self.classes_ = np.arange(9, dtype=np.int32)

    @staticmethod
    def _align(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
        aligned = np.zeros((len(probabilities), 3), dtype=np.float64)
        aligned[:, classes.astype(int)] = probabilities
        return aligned

    def fit_outcomes(
            self,
            x: np.ndarray,
            half_result: np.ndarray,
            full_result: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
    ):
        scaled = self.scaler_.fit_transform(np.asarray(x, dtype=np.float64))
        self.half_model_.fit(scaled, half_result, sample_weight=sample_weight)
        half_one_hot = np.eye(3, dtype=np.float64)[half_result]
        self.full_model_.fit(
            np.hstack([scaled, half_one_hot]),
            full_result,
            sample_weight=sample_weight,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scaled = self.scaler_.transform(np.asarray(x, dtype=np.float64))
        half_prob = self._align(
            self.half_model_.predict_proba(scaled),
            self.half_model_.classes_,
        )
        joint = np.zeros((len(scaled), 3, 3), dtype=np.float64)
        for half_class in range(3):
            assumed_half = np.zeros((len(scaled), 3), dtype=np.float64)
            assumed_half[:, half_class] = 1.0
            full_prob = self._align(
                self.full_model_.predict_proba(np.hstack([scaled, assumed_half])),
                self.full_model_.classes_,
            )
            joint[:, half_class, :] = half_prob[:, half_class, None] * full_prob
        joint /= joint.sum(axis=(1, 2), keepdims=True)
        return joint.reshape(-1, 9)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


class ConditionalHalfFullModel(ClassificationModel):
    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType = TargetType.HALF_FULL,
            calibrate_probabilities: bool = False,
            normalizer=None,
            sampler=None,
            half_c: float = 0.1,
            full_c: float = 0.1,
            recency_half_life_years: Optional[float] = None,
            **kwargs,
    ):
        self._half_c = half_c
        self._full_c = full_c
        self._recency_half_life_years = recency_half_life_years
        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=TargetType.HALF_FULL,
            calibrate_probabilities=False,
            normalizer=None,
            sampler=None,
            **kwargs,
        )

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        return ConditionalHalfFullEstimator(self._half_c, self._full_c)

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        clean = train_df.dropna().reset_index(drop=True)
        x, _, _ = self._dataset_preprocessor.preprocess_dataset(
            clean, target_type=TargetType.HALF_FULL,
        )
        half = clean['HTR'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32)
        full = clean['Result'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32)
        self._classifier = self.build_classifier(x.shape[1], 9)
        self._classifier.fit_outcomes(
            x,
            half,
            full,
            sample_weight=_recency_weights(clean, self._recency_half_life_years),
        )
        if eval_df is not None:
            self._num_eval_samples = len(eval_df)
        return self._evaluate_classifier(train_df=clean, eval_df=eval_df)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param in {'half_c', 'full_c'}:
            return [0.01, 0.03, 0.1, 0.3, 1.0]
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'half_c': self._half_c,
            'full_c': self._full_c,
            'recency_half_life_years': self._recency_half_life_years,
        })
        return model_config


class MarketBlendEstimator(BaseEstimator):
    def __init__(
            self,
            odds_indices: List[int],
            model_weight: float = 0.5,
            c: float = 0.1,
    ):
        self.odds_indices = list(odds_indices)
        self.model_weight = float(model_weight)
        self.c = float(c)
        self.scaler_ = StandardScaler()
        self.model_ = LogisticRegression(
            C=c, max_iter=5000, solver='lbfgs', random_state=0,
        )
        self.classes_ = np.arange(3, dtype=np.int32)

    def fit(self, x, y, sample_weight=None):
        scaled = self.scaler_.fit_transform(np.asarray(x, dtype=np.float64))
        self.model_.fit(scaled, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, x):
        raw_x = np.asarray(x, dtype=np.float64)
        learned_raw = self.model_.predict_proba(self.scaler_.transform(raw_x))
        learned = np.zeros((len(raw_x), 3), dtype=np.float64)
        learned[:, self.model_.classes_.astype(int)] = learned_raw
        odds = np.clip(raw_x[:, self.odds_indices], 1.01, None)
        market = 1.0 / odds
        market /= market.sum(axis=1, keepdims=True)
        blended = self.model_weight * learned + (1.0 - self.model_weight) * market
        return blended / blended.sum(axis=1, keepdims=True)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


class MarketBlendResultModel(ClassificationModel):
    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType = TargetType.RESULT,
            calibrate_probabilities: bool = False,
            normalizer=None,
            sampler=None,
            model_weight: float = 0.5,
            c: float = 0.1,
            recency_half_life_years: Optional[float] = None,
            **kwargs,
    ):
        self._model_weight = model_weight
        self._c = c
        self._recency_half_life_years = recency_half_life_years
        self._odds_indices = None
        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=TargetType.RESULT,
            calibrate_probabilities=False,
            normalizer=None,
            sampler=None,
            **kwargs,
        )

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        return MarketBlendEstimator(self._odds_indices, self._model_weight, self._c)

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        clean = train_df.dropna().reset_index(drop=True)
        feature_columns = clean.drop(
            columns=self._dataset_preprocessor.non_trainable_columns,
            errors='ignore',
        ).columns.tolist()
        self._odds_indices = [feature_columns.index(column) for column in ['1', 'X', '2']]
        x, y, _ = self._dataset_preprocessor.preprocess_dataset(
            clean, target_type=TargetType.RESULT,
        )
        self._classifier = self.build_classifier(x.shape[1], 3)
        self._classifier.fit(
            x,
            y,
            sample_weight=_recency_weights(clean, self._recency_half_life_years),
        )
        if eval_df is not None:
            self._num_eval_samples = len(eval_df)
        return self._evaluate_classifier(train_df=clean, eval_df=eval_df)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'model_weight':
            return [0.0, 0.25, 0.5, 0.75, 1.0]
        if param == 'c':
            return [0.01, 0.03, 0.1, 0.3]
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'model_weight': self._model_weight,
            'c': self._c,
            'recency_half_life_years': self._recency_half_life_years,
        })
        return model_config


class WeightedLogisticEstimator(BaseEstimator):
    def __init__(self, c: float = 0.1):
        self.c = c
        self.scaler_ = StandardScaler()
        self.model_ = LogisticRegression(
            C=c,
            max_iter=5000,
            solver='lbfgs',
            random_state=0,
        )
        self.classes_ = None

    def fit(self, x, y, sample_weight=None):
        scaled = self.scaler_.fit_transform(np.asarray(x, dtype=np.float64))
        self.model_.fit(scaled, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x):
        return self.model_.predict_proba(
            self.scaler_.transform(np.asarray(x, dtype=np.float64)),
        )

    def predict(self, x):
        return self.model_.predict(
            self.scaler_.transform(np.asarray(x, dtype=np.float64)),
        )


class WeightedLogisticModel(ClassificationModel):
    """Time-decayed multinomial logistic model for ordinary classification tasks."""

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            calibrate_probabilities: bool = False,
            normalizer=None,
            sampler=None,
            c: float = 0.1,
            recency_half_life_years: Optional[float] = 8.0,
            **kwargs,
    ):
        self._c = c
        self._recency_half_life_years = recency_half_life_years
        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=False,
            normalizer=None,
            sampler=None,
            **kwargs,
        )

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        return WeightedLogisticEstimator(c=self._c)

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        clean = train_df.dropna().reset_index(drop=True)
        x, y, _ = self._dataset_preprocessor.preprocess_dataset(
            clean, target_type=self._target_type,
        )
        self._classifier = self.build_classifier(x.shape[1], len(np.unique(y)))
        self._classifier.fit(
            x,
            y,
            sample_weight=_recency_weights(clean, self._recency_half_life_years),
        )
        if eval_df is not None:
            self._num_eval_samples = len(eval_df)
        return self._evaluate_classifier(train_df=clean, eval_df=eval_df)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'c':
            return [0.01, 0.03, 0.1, 0.3, 1.0]
        if param == 'recency_half_life_years':
            return [4.0, 6.0, 8.0, None]
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'c': self._c,
            'recency_half_life_years': self._recency_half_life_years,
        })
        return model_config
