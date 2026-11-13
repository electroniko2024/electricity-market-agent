"""Native TreeSHAP output must reconstruct the model prediction."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ellectric.pipeline.market.features import FEATURE_COLUMNS
from ellectric.pipeline.market.forecasting import explain_tree_prediction
from ellectric.pipeline.market.runtime import import_lightgbm


class TreeExplanationTest(unittest.TestCase):
    def test_lightgbm_and_xgboost_contributions_reconstruct_predictions(self) -> None:
        rng = np.random.default_rng(17)
        features = pd.DataFrame(
            rng.normal(size=(32, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
        )
        features["slot_id"] = rng.integers(0, 96, size=len(features))
        features.loc[0, "forecast_bid_space_target_mw"] = np.nan
        target = (
            3.0 * features["slot_id"]
            + 20.0 * (features["price_lag_1d"] > 0).astype("float64")
        )
        row = features.iloc[[0]]

        lightgbm = import_lightgbm().LGBMRegressor(
            n_estimators=12,
            num_leaves=7,
            min_child_samples=1,
            n_jobs=1,
            random_state=3,
            verbosity=-1,
        )
        lightgbm.fit(features, target)

        from xgboost import XGBRegressor

        xgboost = XGBRegressor(
            n_estimators=12,
            max_depth=3,
            min_child_weight=1,
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            random_state=3,
            verbosity=0,
        )
        xgboost.fit(features, target, verbose=False)

        for name, model in (("LightGBM", lightgbm), ("XGBoost", xgboost)):
            with self.subTest(model=name):
                explanation = explain_tree_prediction(name, model, row)
                reconstructed = explanation["base_value"] + sum(
                    item["shap_value"] for item in explanation["feature_contributions"]
                )
                self.assertAlmostEqual(reconstructed, explanation["prediction"], delta=1e-4)
                self.assertEqual(len(explanation["feature_contributions"]), len(FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
