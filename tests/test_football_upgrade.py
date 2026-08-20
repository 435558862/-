import unittest

import numpy as np
import pandas as pd

from src.models.classifiers.football import GoalDistributionEstimator
from src.models.evaluation import expanding_window_folds
from src.preprocessing.utils.target import TargetType


class FootballUpgradeTests(unittest.TestCase):

    def test_auto_rho_produces_normalized_score_probabilities(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(240, 4))
        home = rng.poisson(1.45, size=240)
        away = rng.poisson(1.10, size=240)
        estimator = GoalDistributionEstimator(
            TargetType.SCORE, rho='auto', mean_shrinkage=0.10,
        ).fit_goals(x, home, away)

        probabilities = estimator.predict_proba(x[:12])
        self.assertEqual(probabilities.shape, (12, 49))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertTrue(-0.20 <= estimator.rho_ <= 0.15)

    def test_walk_forward_never_trains_on_future_rows(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({'Date': dates[::-1]})
        folds = list(expanding_window_folds(df, folds=3, test_fraction=0.10))

        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertLess(
                pd.to_datetime(fold.train['Date']).max(),
                pd.to_datetime(fold.test['Date']).min(),
            )

    def test_pre_upgrade_estimator_remains_compatible(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(100, 3))
        estimator = GoalDistributionEstimator(TargetType.SCORE).fit_goals(
            x, rng.poisson(1.4, 100), rng.poisson(1.1, 100),
        )
        del estimator.mean_shrinkage
        del estimator.rho_
        probabilities = estimator.predict_proba(x[:4])
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))


if __name__ == '__main__':
    unittest.main()
