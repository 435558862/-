from typing import Any, Dict, List, Optional, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from src.models.model import ClassificationModel
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType


class OptimizedEnsemble(ClassificationModel):
    """Small wrapper for the additional ensemble algorithms used by offline tuning."""

    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            calibrate_probabilities: bool,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            algorithm: str = 'extra_trees',
            algorithm_params: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        self._algorithm = algorithm
        self._algorithm_params = dict(algorithm_params or {})
        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=calibrate_probabilities,
            normalizer=normalizer,
            sampler=sampler,
            **kwargs,
        )

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        params = dict(self._algorithm_params)
        if self._algorithm == 'extra_trees':
            params.setdefault('n_estimators', 600)
            params.setdefault('random_state', 0)
            params.setdefault('n_jobs', -1)
            return ExtraTreesClassifier(**params)
        if self._algorithm == 'hist_gradient_boosting':
            params.setdefault('random_state', 0)
            return HistGradientBoostingClassifier(**params)
        raise ValueError(f'Undefined optimized ensemble algorithm: "{self._algorithm}".')

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == 'algorithm':
            return ['extra_trees', 'hist_gradient_boosting']
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            'algorithm': self._algorithm,
            'algorithm_params': self._algorithm_params,
        })
        return model_config
