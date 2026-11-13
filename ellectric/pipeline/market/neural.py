"""NeuralForecast PatchTST training and fixed-origin daily inference."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ellectric.pipeline.market.data import MARKET_TIMEZONE
from ellectric.pipeline.market.features import build_patchtst_frame
from ellectric.pipeline.market.forecasting import (
    ASSUMPTION_IDS,
    FORECAST_ORIGIN,
    TEST_END,
    TEST_START,
)

INPUT_SIZE = 7 * 96
HORIZON = 96
VALIDATION_DAYS = 92
VALIDATION_SIZE = VALIDATION_DAYS * 96


def train_patchtst_snapshot(
    market: pd.DataFrame,
    *,
    max_steps: int = 500,
    val_check_steps: int = 250,
    batch_size: int = 32,
    windows_batch_size: int = 128,
    random_seed: int = 42,
    forecast_origin: pd.Timestamp = FORECAST_ORIGIN,
    output_path: str | Path | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Fit a PatchTST snapshot using the 92 days immediately before its origin."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import PatchTST

    patch_frame, fill_metadata = build_patchtst_frame(market)
    origin = pd.Timestamp(forecast_origin)
    if origin.tz is None:
        origin = origin.tz_localize(MARKET_TIMEZONE)
    else:
        origin = origin.tz_convert(MARKET_TIMEZONE)
    pre_test = patch_frame.loc[patch_frame["ds"] <= origin].reset_index(drop=True)
    if len(pre_test) <= INPUT_SIZE + HORIZON + VALIDATION_SIZE:
        raise ValueError("PatchTST 数据不足以覆盖暖启动、训练和验证窗口")
    if pre_test["ds"].iloc[-1] != origin:
        raise ValueError(f"PatchTST 输入没有截止于发起时刻 {origin.isoformat()}")

    model = PatchTST(
        h=HORIZON,
        input_size=INPUT_SIZE,
        encoder_layers=3,
        n_heads=4,
        hidden_size=64,
        linear_hidden_size=128,
        patch_len=16,
        stride=8,
        revin=True,
        revin_affine=False,
        revin_subtract_last=True,
        scaler_type="identity",
        loss=MAE(),
        valid_loss=MAE(),
        max_steps=max_steps,
        early_stop_patience_steps=-1,
        val_check_steps=val_check_steps,
        batch_size=batch_size,
        valid_batch_size=batch_size,
        windows_batch_size=windows_batch_size,
        random_seed=random_seed,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
    )
    forecast = NeuralForecast(models=[model], freq="15min")
    forecast.fit(df=pre_test, val_size=VALIDATION_SIZE, verbose=False)

    train_end_label = origin - pd.Timedelta(minutes=15 * VALIDATION_SIZE)
    train_end_trade_date = (train_end_label - pd.Timedelta(days=1)).date()
    validation_start = train_end_label.date()
    validation_end = (origin - pd.Timedelta(days=1)).date()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model": "PatchTST",
        "neuralforecast_version": __import__("neuralforecast").__version__,
        "torch_version": __import__("torch").__version__,
        "target": "rt_price",
        "input_size": INPUT_SIZE,
        "horizon": HORIZON,
        "frequency": "15min",
        "train_data_end_trade_date": train_end_trade_date.isoformat(),
        "train_data_end_label_at_local": train_end_label.isoformat(),
        "validation_period": [validation_start.isoformat(), validation_end.isoformat()],
        "validation_size_points": VALIDATION_SIZE,
        "trained_as_of": origin.isoformat(),
        "training_steps_completed": max_steps,
        "training_controls": {
            "max_steps": max_steps,
            "val_check_steps": val_check_steps,
            "batch_size": batch_size,
            "windows_batch_size": windows_batch_size,
            "random_seed": random_seed,
            "device": "cpu",
        },
        **fill_metadata,
        "availability_verified": False,
        "availability_basis": "assumed",
        "assumption_ids": list(ASSUMPTION_IDS),
        "target_time_semantics": {
            "key": ["trade_date", "slot_id"],
            "interval_boundary": "unknown",
            "ds_source": "label_at_local as a regular sequence index only",
        },
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        forecast.save(path=str(output), save_dataset=False, overwrite=False)
    return forecast, patch_frame, metadata


def predict_patchtst_targets(
    forecast: Any,
    patch_frame: pd.DataFrame,
    target_dates: Iterable[date | str | pd.Timestamp],
    *,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Forecast each target date from its own midnight cutoff and return 96 points.

    `window` defaults to the test period; threshold calibration passes the validation period,
    which is out of sample for the fixed snapshot (trained through 2025-06-30).
    """
    window_start, window_end = window or (TEST_START, TEST_END)
    predictions: list[pd.DataFrame] = []
    for raw_date in target_dates:
        target_date = pd.Timestamp(raw_date)
        if target_date.tz is not None:
            target_date = target_date.tz_convert(MARKET_TIMEZONE).tz_localize(None)
        target_date = target_date.normalize()
        if target_date < window_start or target_date > window_end:
            raise ValueError(
                f"PatchTST 预测窗口为 {window_start.date()} 至 {window_end.date()}，收到 {target_date.date()}"
            )
        origin = target_date.tz_localize(MARKET_TIMEZONE)
        history = patch_frame.loc[patch_frame["ds"] <= origin].reset_index(drop=True)
        if len(history) < INPUT_SIZE:
            raise ValueError(f"{target_date.date()} 起报前历史不足 {INPUT_SIZE} 个点")
        if history.empty or history["ds"].iloc[-1] != origin:
            raise ValueError(
                f"{target_date.date()} 午夜起报点缺失；不可用未来观测补齐历史窗口"
            )

        result = forecast.predict(df=history, verbose=False)
        result = result.sort_values("ds", kind="stable").reset_index(drop=True)
        if len(result) != HORIZON:
            raise ValueError(
                f"PatchTST 对 {target_date.date()} 输出 {len(result)} 点，预期 {HORIZON}"
            )
        label_times = pd.to_datetime(result["ds"], errors="raise")
        if label_times.dt.tz is None:
            label_times = label_times.dt.tz_localize(MARKET_TIMEZONE)
        else:
            label_times = label_times.dt.tz_convert(MARKET_TIMEZONE)
        minutes_from_origin = (label_times - origin).dt.total_seconds() / 60
        slot_ids = (minutes_from_origin / 15 - 1).astype("int16")
        if not np.array_equal(slot_ids.to_numpy(), np.arange(HORIZON)):
            raise ValueError(f"PatchTST 对 {target_date.date()} 的槽位标签不连续")
        model_columns = [column for column in result.columns if column not in {"unique_id", "ds"}]
        if len(model_columns) != 1:
            raise ValueError(f"PatchTST 预期单变量预测列，实际为: {model_columns}")
        values = pd.to_numeric(result[model_columns[0]], errors="coerce")
        if values.isna().any():
            raise ValueError(f"PatchTST 对 {target_date.date()} 返回空预测")
        predictions.append(
            pd.DataFrame(
                {
                    "trade_date": target_date,
                    "time_label": [
                        "00:15" if slot == 0 else "24:00" if slot == 95 else
                        f"{(slot + 1) // 4:02d}:{((slot + 1) % 4) * 15:02d}"
                        for slot in range(HORIZON)
                    ],
                    "slot_id": slot_ids,
                    "PatchTST": values.to_numpy(dtype="float64"),
                }
            )
        )

    if not predictions:
        return pd.DataFrame(columns=["trade_date", "time_label", "slot_id", "PatchTST"])
    return pd.concat(predictions, ignore_index=True)
