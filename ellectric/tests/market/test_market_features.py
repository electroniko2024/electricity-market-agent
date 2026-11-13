"""Leakage and missing-value checks for D+1 market samples."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ellectric.pipeline.market.features import (
    FEATURE_COLUMNS,
    build_forecast_samples,
    build_patchtst_frame,
)


def _market_frame(days: int = 9) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-01-01")
    for day_offset, date in enumerate(pd.date_range(start, periods=days)):
        for slot in range(96):
            label_minutes = (slot + 1) * 15
            label_at_local = date.tz_localize("Asia/Shanghai") + pd.Timedelta(
                minutes=label_minutes
            )
            rows.append(
                {
                    "trade_date": date.date(),
                    "time_label": "24:00" if slot == 95 else f"{(slot + 1) // 4:02d}:{((slot + 1) % 4) * 15:02d}",
                    "slot_id": slot,
                    "label_at_local": label_at_local,
                    "is_holiday": 0,
                    "is_weekend": int(date.dayofweek >= 5),
                    "rt_price": float(day_offset * 100 + slot)
                    if (day_offset, slot) not in {(6, 20), (7, 21)}
                    else np.nan,
                    "load_actual_mw": 1000 + day_offset,
                    "tie_line_actual_mw": 100 + day_offset,
                    "wind_actual_mw": 200 + day_offset,
                    "solar_actual_clipped_mw": 300 + day_offset,
                    "actual_bid_space_mw": 400 + day_offset,
                    "load_forecast_mw": 1100 + day_offset,
                    "tie_line_forecast_mw": 110 + day_offset,
                    "wind_forecast_mw": 210 + day_offset,
                    "solar_forecast_clipped_mw": 310 + day_offset,
                    "nuclear_forecast_mw": 10 + day_offset,
                    "captive_forecast_mw": 20 + day_offset,
                    "local_gen_forecast_mw": 30 + day_offset,
                    "forecast_bid_space_mw": np.nan if (day_offset, slot) == (7, 22) else 500 + day_offset,
                }
            )
    return pd.DataFrame(rows)


class MarketFeatureTest(unittest.TestCase):
    def test_lags_rollups_and_baselines_stop_before_target_day(self) -> None:
        market = _market_frame()
        samples = build_forecast_samples(market)
        target = samples.loc[
            (samples["trade_date"] == pd.Timestamp("2025-01-08"))
            & (samples["slot_id"] == 10)
        ].iloc[0]

        self.assertEqual(target["price_lag_1d"], 6 * 100 + 10)
        self.assertEqual(target["price_lag_7d"], 10)
        self.assertEqual(target["baseline_yesterday"], target["price_lag_1d"])
        self.assertEqual(target["baseline_last_week"], target["price_lag_7d"])
        self.assertAlmostEqual(target["price_7d_mean"], market.loc[: 6 * 96 + 95, "rt_price"].mean())
        self.assertEqual(target["load_actual_lag_1d_mw"], 1006)
        self.assertEqual(target["load_forecast_target_mw"], 1107)
        self.assertFalse(target["availability_verified"])
        self.assertEqual(target["availability_basis"], "assumed")

    def test_target_outcomes_do_not_change_same_day_features(self) -> None:
        market = _market_frame()
        baseline = build_forecast_samples(market)
        target_date = pd.Timestamp("2025-01-08")
        modified_market = market.copy()
        modified = pd.to_datetime(modified_market["trade_date"]) == target_date
        modified_market.loc[modified, "rt_price"] = 100_000.0
        changed = build_forecast_samples(modified_market)

        row_mask = (baseline["trade_date"] == target_date) & (baseline["slot_id"] == 10)
        original_row = baseline.loc[row_mask].iloc[0]
        changed_row = changed.loc[row_mask].iloc[0]
        self.assertNotEqual(original_row["y_raw"], changed_row["y_raw"])
        for feature in FEATURE_COLUMNS:
            self.assertEqual(original_row[feature], changed_row[feature], feature)

    def test_raw_missing_targets_and_missing_forecasts_are_not_imputed(self) -> None:
        samples = build_forecast_samples(_market_frame())
        missing_target = samples.loc[
            (samples["trade_date"] == pd.Timestamp("2025-01-08"))
            & (samples["slot_id"] == 21)
        ].iloc[0]
        missing_forecast = samples.loc[
            (samples["trade_date"] == pd.Timestamp("2025-01-08"))
            & (samples["slot_id"] == 22)
        ].iloc[0]

        self.assertTrue(pd.isna(missing_target["y_raw"]))
        self.assertEqual(missing_target["target_observed"], 0)
        self.assertTrue(pd.isna(missing_forecast["forecast_bid_space_target_mw"]))

    def test_patchtst_input_fill_preserves_timestamps_and_target_mask(self) -> None:
        market = _market_frame()
        patch_frame, metadata = build_patchtst_frame(market)
        missing_row = 6 * 96 + 20

        self.assertEqual(len(patch_frame), len(market))
        self.assertEqual(patch_frame.loc[missing_row, "available_mask"], 0.0)
        self.assertEqual(
            patch_frame.loc[missing_row, "y"],
            patch_frame.loc[missing_row - 1, "y"],
        )
        self.assertEqual(patch_frame.loc[missing_row, "ds"], market.loc[missing_row, "label_at_local"])
        self.assertEqual(metadata["input_forward_filled_points"], 2)
        self.assertEqual(metadata["observed_target_points"], len(market) - 2)


if __name__ == "__main__":
    unittest.main()
