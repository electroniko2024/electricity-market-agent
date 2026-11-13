"""Common-target scoring checks for the fixed-test forecast reports."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ellectric.pipeline.market.forecasting import (
    rolling_origins,
    score_forecasts,
    snapshot_periods,
)


class ForecastMetricsTest(unittest.TestCase):
    def test_weekly_origins_cover_the_frozen_test_period_without_drift(self) -> None:
        origins = rolling_origins()

        self.assertEqual(len(origins), 16)
        self.assertEqual(origins[0], pd.Timestamp("2025-10-01"))
        self.assertEqual(origins[-1], pd.Timestamp("2026-01-14"))
        self.assertTrue(
            all((right - left).days == 7 for left, right in zip(origins, origins[1:]))
        )

    def test_tree_snapshot_periods_end_before_each_origin(self) -> None:
        origin, train_start, train_end, validation_start, validation_end = snapshot_periods(
            pd.Timestamp("2025-10-08", tz="Asia/Shanghai")
        )

        self.assertEqual(origin, pd.Timestamp("2025-10-08", tz="Asia/Shanghai"))
        self.assertEqual(train_start, pd.Timestamp("2024-01-08"))
        self.assertEqual(train_end, pd.Timestamp("2025-07-07"))
        self.assertEqual(validation_start, pd.Timestamp("2025-07-08"))
        self.assertEqual(validation_end, pd.Timestamp("2025-10-07"))

    def test_models_and_baselines_share_scored_points_and_smape_rule(self) -> None:
        samples = pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2025-10-01")] * 5,
                "time_label": ["00:15", "00:30", "00:45", "01:00", "01:15"],
                "slot_id": list(range(5)),
                "target_observed": [1, 1, 1, 0, 1],
                "y_raw": [0.0, -1.0, 1000.0, np.nan, 10.0],
                "baseline_yesterday": [0.0, 0.0, np.nan, 10.0, 10.0],
                "baseline_last_week": [0.0, -2.0, 1000.0, 10.0, 10.0],
            }
        )
        predictions = pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2025-10-01")] * 5,
                "time_label": samples["time_label"],
                "slot_id": list(range(5)),
                "LightGBM": [0.0, -2.0, 1000.0, 5.0, 10.0],
                "XGBoost": [0.0, 0.0, 1000.0, 5.0, 10.0],
                "PatchTST": [0.0, -1.0, 1000.0, 5.0, np.nan],
            }
        )

        report = score_forecasts(
            samples,
            predictions,
            start=pd.Timestamp("2025-10-01"),
            end=pd.Timestamp("2025-10-01"),
            spike_threshold=50.0,
        )

        self.assertEqual(report["eligible_observed_targets"], 4)
        self.assertEqual(report["common_scored_targets"], 2)
        self.assertEqual(report["models_and_baselines"]["LightGBM"]["mae"], 0.5)
        self.assertEqual(report["models_and_baselines"]["LightGBM"]["smape_zero_denominator_count"], 1)
        self.assertAlmostEqual(report["models_and_baselines"]["LightGBM"]["smape"], 1 / 3)
        self.assertEqual(report["models_and_baselines"]["LightGBM"]["negative_f1"], 1.0)
        self.assertEqual(report["models_and_baselines"]["baseline_yesterday"]["negative_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
