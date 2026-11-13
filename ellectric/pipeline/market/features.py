"""Leakage-aware D+1 feature construction for the Shandong market dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ellectric.pipeline.market.data import MARKET_TIMEZONE

MARKET_KEY = ("trade_date", "slot_id")
REQUIRED_COLUMNS = {
    "trade_date",
    "time_label",
    "slot_id",
    "is_holiday",
    "is_weekend",
    "rt_price",
    "load_actual_mw",
    "wind_actual_mw",
    "solar_actual_clipped_mw",
    "actual_bid_space_mw",
    "load_forecast_mw",
    "tie_line_forecast_mw",
    "wind_forecast_mw",
    "solar_forecast_clipped_mw",
    "nuclear_forecast_mw",
    "captive_forecast_mw",
    "local_gen_forecast_mw",
    "forecast_bid_space_mw",
}

HISTORICAL_ACTUAL_COLUMNS = {
    "load_actual_mw": "load_actual_lag_1d_mw",
    "tie_line_actual_mw": "tie_line_actual_lag_1d_mw",
    "wind_actual_mw": "wind_actual_lag_1d_mw",
    "solar_actual_clipped_mw": "solar_actual_lag_1d_mw",
    "actual_bid_space_mw": "actual_bid_space_lag_1d_mw",
}

TARGET_FORECAST_COLUMNS = {
    "load_forecast_mw": "load_forecast_target_mw",
    "tie_line_forecast_mw": "tie_line_forecast_target_mw",
    "wind_forecast_mw": "wind_forecast_target_mw",
    "solar_forecast_clipped_mw": "solar_forecast_target_mw",
    "nuclear_forecast_mw": "nuclear_forecast_target_mw",
    "captive_forecast_mw": "captive_forecast_target_mw",
    "local_gen_forecast_mw": "local_gen_forecast_target_mw",
    "forecast_bid_space_mw": "forecast_bid_space_target_mw",
}

FEATURE_COLUMNS = (
    "slot_id",
    "slot_sin",
    "slot_cos",
    "weekday",
    "weekday_sin",
    "weekday_cos",
    "month",
    "month_sin",
    "month_cos",
    "is_holiday",
    "is_weekend",
    "market_rule_2025_active",
    "price_lag_1d",
    "price_lag_7d",
    "price_day_mean_lag_1d",
    "price_day_negative_share_lag_1d",
    "price_7d_mean",
    "price_7d_std",
    *HISTORICAL_ACTUAL_COLUMNS.values(),
    *TARGET_FORECAST_COLUMNS.values(),
)

SAMPLE_METADATA_COLUMNS = (
    "trade_date",
    "time_label",
    "slot_id",
    "forecast_origin",
    "target_observed",
    "y_raw",
    "baseline_yesterday",
    "baseline_last_week",
    "availability_verified",
    "availability_basis",
    "assumption_ids",
)


def _indexed_market_data(market: pd.DataFrame) -> tuple[pd.DataFrame, pd.MultiIndex]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(market.columns))
    if missing_columns:
        raise ValueError(f"市场数据缺少特征构造字段: {missing_columns}")

    frame = market.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["slot_id"] = pd.to_numeric(frame["slot_id"], errors="raise").astype("int16")
    if not frame["slot_id"].between(0, 95).all():
        raise ValueError("slot_id 必须位于 0..95")
    if frame.duplicated(list(MARKET_KEY)).any():
        raise ValueError("市场特征输入包含重复 trade_date + slot_id")

    expected_slots = set(range(96))
    incomplete_days = {
        day.strftime("%Y-%m-%d"): sorted(expected_slots - set(group["slot_id"].astype(int)))
        for day, group in frame.groupby("trade_date", sort=False)
        if set(group["slot_id"].astype(int)) != expected_slots
    }
    if incomplete_days:
        sample = dict(list(incomplete_days.items())[:5])
        raise ValueError(f"每个交易日必须保留完整 96 个槽位；不完整日期样例: {sample}")

    frame = frame.sort_values(list(MARKET_KEY), kind="stable").reset_index(drop=True)
    index = pd.MultiIndex.from_frame(frame[list(MARKET_KEY)])
    return frame, index


def _lookup_by_day_and_slot(
    source: pd.Series,
    index: pd.MultiIndex,
    dates: pd.Series,
    slots: pd.Series,
    day_offset: int,
) -> pd.Series:
    lookup = pd.MultiIndex.from_arrays(
        [dates - pd.Timedelta(days=day_offset), slots], names=list(MARKET_KEY)
    )
    values = source.reindex(lookup).to_numpy()
    return pd.Series(values, index=index)


def _daily_price_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("trade_date", sort=True)["rt_price"]
    daily = grouped.agg(
        price_day_mean="mean",
        price_observed_count="count",
        price_day_negative_count=lambda values: int((values < 0).sum()),
    )
    daily["price_day_negative_share"] = (
        daily["price_day_negative_count"] / daily["price_observed_count"].replace(0, np.nan)
    )

    valid = frame.assign(
        price_sum=frame["rt_price"].fillna(0.0),
        price_square_sum=frame["rt_price"].pow(2).fillna(0.0),
        price_count=frame["rt_price"].notna().astype("int32"),
    ).groupby("trade_date", sort=True)[["price_sum", "price_square_sum", "price_count"]].sum()
    sums = valid["price_sum"].rolling(7, min_periods=7).sum()
    squares = valid["price_square_sum"].rolling(7, min_periods=7).sum()
    counts = valid["price_count"].rolling(7, min_periods=7).sum().replace(0, np.nan)
    means = sums / counts
    sample_variance = (squares - sums.pow(2) / counts) / (counts - 1)
    daily["price_7d_mean"] = means
    daily["price_7d_std"] = np.sqrt(sample_variance.clip(lower=0))
    return daily


def build_forecast_samples(market: pd.DataFrame) -> pd.DataFrame:
    """Build one D+1 sample per target slot without consuming target-day outcomes.

    Source forecasts for the target day are included under the accepted midnight
    snapshot assumption; their publication time and revision history are unknown.
    """
    frame, market_index = _indexed_market_data(market)
    dates = frame["trade_date"]
    slots = frame["slot_id"]
    samples = frame.loc[:, ["trade_date", "time_label", "slot_id"]].copy()

    samples["forecast_origin"] = dates.dt.tz_localize(MARKET_TIMEZONE)
    samples["target_observed"] = frame["rt_price"].notna().astype("uint8")
    samples["y_raw"] = frame["rt_price"].astype("float64")
    samples["baseline_yesterday"] = _lookup_by_day_and_slot(
        frame.set_index(market_index)["rt_price"], market_index, dates, slots, 1
    ).to_numpy()
    samples["baseline_last_week"] = _lookup_by_day_and_slot(
        frame.set_index(market_index)["rt_price"], market_index, dates, slots, 7
    ).to_numpy()

    samples["slot_sin"] = np.sin(2 * np.pi * slots / 96)
    samples["slot_cos"] = np.cos(2 * np.pi * slots / 96)
    samples["weekday"] = dates.dt.dayofweek.astype("int8")
    samples["weekday_sin"] = np.sin(2 * np.pi * samples["weekday"] / 7)
    samples["weekday_cos"] = np.cos(2 * np.pi * samples["weekday"] / 7)
    samples["month"] = dates.dt.month.astype("int8")
    samples["month_sin"] = np.sin(2 * np.pi * (samples["month"] - 1) / 12)
    samples["month_cos"] = np.cos(2 * np.pi * (samples["month"] - 1) / 12)
    samples["is_holiday"] = pd.to_numeric(frame["is_holiday"], errors="coerce")
    samples["is_weekend"] = pd.to_numeric(frame["is_weekend"], errors="coerce")
    samples["market_rule_2025_active"] = (dates >= pd.Timestamp("2026-01-01")).astype("uint8")

    indexed = frame.set_index(market_index)
    for source_column, feature_column in HISTORICAL_ACTUAL_COLUMNS.items():
        samples[feature_column] = _lookup_by_day_and_slot(
            indexed[source_column], market_index, dates, slots, 1
        ).to_numpy()

    for source_column, feature_column in TARGET_FORECAST_COLUMNS.items():
        samples[feature_column] = pd.to_numeric(frame[source_column], errors="coerce")

    indexed_price = indexed["rt_price"]
    for offset, feature_column in ((1, "price_lag_1d"), (7, "price_lag_7d")):
        samples[feature_column] = _lookup_by_day_and_slot(
            indexed_price, market_index, dates, slots, offset
        ).to_numpy()

    daily = _daily_price_statistics(frame)
    history_dates = dates - pd.Timedelta(days=1)
    samples["price_day_mean_lag_1d"] = daily["price_day_mean"].reindex(history_dates).to_numpy()
    samples["price_day_negative_share_lag_1d"] = daily["price_day_negative_share"].reindex(
        history_dates
    ).to_numpy()
    samples["price_7d_mean"] = daily["price_7d_mean"].reindex(history_dates).to_numpy()
    samples["price_7d_std"] = daily["price_7d_std"].reindex(history_dates).to_numpy()

    samples["availability_verified"] = False
    samples["availability_basis"] = "assumed"
    samples["assumption_ids"] = [[
        "midnight_market_data_available",
        "target_day_forecast_snapshot_available_and_unrevised",
    ] for _ in range(len(samples))]

    output_columns = list(dict.fromkeys([*SAMPLE_METADATA_COLUMNS, *FEATURE_COLUMNS]))
    return samples.loc[:, output_columns].copy()


def build_patchtst_frame(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Create the dense univariate series plus a loss mask for PatchTST.

    The model input is causally forward-filled. ``available_mask`` retains the
    original observation status so filled labels are excluded from loss/validation.
    Leading rows before the first real target are discarded because no causal fill
    exists for them; all later timestamps, including missing targets, are retained.
    """
    frame, _ = _indexed_market_data(market)
    raw = pd.to_numeric(frame["rt_price"], errors="coerce").astype("float64")
    observed = raw.notna()
    if not observed.any():
        raise ValueError("PatchTST 目标序列没有任何有效实时价格")

    first_observed = int(np.flatnonzero(observed.to_numpy())[0])
    raw = raw.iloc[first_observed:].reset_index(drop=True)
    observed = observed.iloc[first_observed:].reset_index(drop=True)
    dense_values = raw.ffill()
    if dense_values.isna().any():
        raise AssertionError("首个有效标签之后仍有未填充实时价格")

    if "label_at_local" not in frame.columns:
        raise ValueError("PatchTST 序列需要 P1 生成的 label_at_local 时钟标签")
    timestamps = pd.to_datetime(
        frame["label_at_local"].iloc[first_observed:], errors="raise"
    ).reset_index(drop=True)
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(MARKET_TIMEZONE)
    else:
        timestamps = timestamps.dt.tz_convert(MARKET_TIMEZONE)
    patch_frame = pd.DataFrame(
        {
            "unique_id": "rt_price",
            "ds": timestamps,
            "y": dense_values.astype("float32"),
            "available_mask": observed.astype("float32"),
        }
    )
    fill_count = int((~observed).sum())
    return patch_frame, {
        "input_forward_filled_points": fill_count,
        "observed_target_points": int(observed.sum()),
        "leading_unobserved_points_discarded": first_observed,
    }
