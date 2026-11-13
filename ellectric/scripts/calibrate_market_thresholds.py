"""Calibrate model-divergence and forecast-residual thresholds on the validation period.

Out-of-sample by construction: tree models are refit on the training period only (with the
frozen snapshot's iteration counts) and PatchTST is the fixed snapshot, which was trained
through 2025-06-30. The test period (2025-10-01..2026-01-14) is never read here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ellectric.pipeline.market.features import build_forecast_samples, build_patchtst_frame
from ellectric.pipeline.market.forecasting import (
    FEATURE_COLUMNS,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    _feature_matrix,
    _period_rows,
    predict_tree_models,
)
from ellectric.pipeline.market.neural import predict_patchtst_targets

MODELS = ["LightGBM", "XGBoost", "PatchTST"]


def _train_only_trees(samples: pd.DataFrame, metadata: dict, n_jobs: int) -> dict:
    from ellectric.pipeline.market.runtime import import_lightgbm
    from xgboost import XGBRegressor

    lgb = import_lightgbm()
    train_end = VALIDATION_START - pd.Timedelta(days=1)
    rows = _period_rows(samples, TRAIN_START, train_end)
    x, y = _feature_matrix(rows), rows["y_raw"].astype("float32")
    versions = metadata["versions"]
    lgb_params = dict(versions["lightgbm_parameters"], n_jobs=n_jobs)
    xgb_params = dict(versions["xgboost_parameters"], n_jobs=n_jobs)
    iterations = metadata["best_iterations"]
    lgb_model = lgb.LGBMRegressor(n_estimators=iterations["LightGBM"], **lgb_params).fit(x, y)
    xgb_model = XGBRegressor(n_estimators=iterations["XGBoost"], **xgb_params)
    xgb_model.fit(x, y, verbose=False)
    return {"LightGBM": lgb_model, "XGBoost": xgb_model}, len(rows)


def main() -> int:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "market_processed"
    snapshot = data_dir / "model_run_2025-10-01"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=snapshot / "calibration_thresholds.json")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    from neuralforecast import NeuralForecast

    market = pd.read_parquet(data_dir / "sd_rt_15min.parquet")
    samples = build_forecast_samples(market)
    metadata = json.loads((snapshot / "models" / "tree_models.json").read_text(encoding="utf-8"))
    if list(metadata["feature_names"]) != list(FEATURE_COLUMNS):
        raise ValueError("feature columns differ from the frozen snapshot")
    trees, training_rows = _train_only_trees(samples, metadata, args.n_jobs)

    valid_rows = _period_rows(samples, VALIDATION_START, VALIDATION_END)
    tree_pred = predict_tree_models(trees, valid_rows)
    tree_pred["actual"] = valid_rows["y_raw"].astype("float64").to_numpy()

    patch = NeuralForecast.load(path=str(snapshot / "models" / "patchtst_2025-10-01"))
    patch_frame, _ = build_patchtst_frame(market)
    days = pd.date_range(VALIDATION_START, VALIDATION_END, freq="D")
    patch_pred = predict_patchtst_targets(
        patch, patch_frame, days, window=(VALIDATION_START, VALIDATION_END)
    )
    frame = tree_pred.merge(
        patch_pred.drop(columns="time_label"),
        on=["trade_date", "slot_id"],
        how="inner",
        validate="one_to_one",
    ).dropna(subset=MODELS + ["actual"])

    values = frame[MODELS].to_numpy()
    spread_std = values.std(axis=1)
    ensemble = values.mean(axis=1)
    residual = frame["actual"].to_numpy() - ensemble
    abs_residual = np.abs(residual)
    divergence_p90, divergence_p95 = np.quantile(spread_std, [0.90, 0.95])
    residual_p95, residual_p99 = np.quantile(abs_residual, [0.95, 0.99])
    high = spread_std >= divergence_p95
    payload = {
        "schema_version": 1,
        "calibration_period": [VALIDATION_START.date().isoformat(), VALIDATION_END.date().isoformat()],
        "scored_points": int(len(frame)),
        "method": {
            "trees": "LightGBM/XGBoost refit on training period only with frozen best_iterations",
            "patchtst": "fixed snapshot trained through 2025-06-30 (out of sample on validation)",
            "training_rows": int(training_rows),
            "ensemble": "unweighted mean of LightGBM, XGBoost, PatchTST",
        },
        "divergence": {
            "statistic": "population standard deviation of the three model forecasts per slot",
            "p90_yuan_per_mwh": float(divergence_p90),
            "p95_yuan_per_mwh": float(divergence_p95),
            "high_threshold_yuan_per_mwh": float(divergence_p95),
            "validation_ensemble_mae_high_divergence": float(abs_residual[high].mean()),
            "validation_ensemble_mae_other": float(abs_residual[~high].mean()),
        },
        "forecast_residual": {
            "statistic": "actual minus ensemble mean",
            "abs_p95_yuan_per_mwh": float(residual_p95),
            "abs_p99_yuan_per_mwh": float(residual_p99),
            "threshold_yuan_per_mwh": float(residual_p99),
            "validation_ensemble_mae": float(abs_residual.mean()),
        },
        "note": "thresholds flag disagreement and large errors; they are not probability intervals",
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
