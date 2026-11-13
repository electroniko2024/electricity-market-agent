"""Fixed-origin tree forecasting, common scoring, and local model artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ellectric.pipeline.market.data import MARKET_TIMEZONE
from ellectric.pipeline.market.features import FEATURE_COLUMNS
from ellectric.pipeline.market.runtime import import_lightgbm

TRAIN_START = pd.Timestamp("2024-01-08")
TRAIN_END = pd.Timestamp("2025-06-30")
VALIDATION_START = pd.Timestamp("2025-07-01")
VALIDATION_END = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-01-14")
FORECAST_ORIGIN = pd.Timestamp("2025-10-01", tz=MARKET_TIMEZONE)
VALIDATION_DAYS = int((VALIDATION_END - VALIDATION_START).days + 1)
SPIKE_THRESHOLD_CANDIDATE = 1000.0
ASSUMPTION_IDS = (
    "midnight_market_data_available",
    "target_day_forecast_snapshot_available_and_unrevised",
)


def _feature_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"预测样本缺少特征列: {missing}")
    return rows.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").astype("float32")


def _period_rows(
    samples: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    dates = pd.to_datetime(samples["trade_date"], errors="raise").dt.normalize()
    observed = pd.to_numeric(samples["target_observed"], errors="raise").eq(1)
    complete_history = pd.to_numeric(samples["price_7d_mean"], errors="coerce").notna()
    valid_label = pd.to_numeric(samples["y_raw"], errors="coerce").notna()
    mask = dates.between(start, end) & observed & valid_label & complete_history
    return samples.loc[mask].copy()


def snapshot_periods(
    forecast_origin: pd.Timestamp = FORECAST_ORIGIN,
    *,
    validation_days: int = VALIDATION_DAYS,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return an origin and the train/validation dates available at that origin."""
    if validation_days < 1:
        raise ValueError("validation_days 必须至少为 1")
    origin = pd.Timestamp(forecast_origin)
    if origin.tz is None:
        origin = origin.tz_localize(MARKET_TIMEZONE)
    else:
        origin = origin.tz_convert(MARKET_TIMEZONE)
    origin_date = origin.tz_localize(None).normalize()
    validation_start = origin_date - pd.Timedelta(days=validation_days)
    validation_end = origin_date - pd.Timedelta(days=1)
    train_end = validation_start - pd.Timedelta(days=1)
    return origin, TRAIN_START, train_end, validation_start, validation_end


def rolling_origins(
    start: pd.Timestamp = TEST_START,
    end: pd.Timestamp = TEST_END,
    *,
    interval_days: int = 7,
) -> list[pd.Timestamp]:
    """Return fixed retraining origins that cover the requested test interval."""
    if interval_days < 1:
        raise ValueError("interval_days 必须至少为 1")
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    if first > last:
        raise ValueError("回测开始日期不能晚于结束日期")
    return list(pd.date_range(first, last, freq=f"{interval_days}D"))


def _fit_model_candidates(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    max_estimators: int,
    early_stopping_rounds: int,
    n_jobs: int,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    lgb = import_lightgbm()
    import xgboost as xgb
    from xgboost import XGBRegressor
    LGBMRegressor = lgb.LGBMRegressor

    lgb_params = {
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }
    lgb_candidate = LGBMRegressor(n_estimators=max_estimators, **lgb_params)
    lgb_candidate.fit(
        train_x,
        train_y,
        eval_metric="l1",
        eval_X=valid_x,
        eval_y=valid_y,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    lgb_best_iteration = max(1, int(lgb_candidate.best_iteration_ or max_estimators))

    xgb_params = {
        "learning_rate": 0.04,
        "max_depth": 8,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "device": "cpu",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    xgb_candidate = XGBRegressor(
        n_estimators=max_estimators,
        early_stopping_rounds=early_stopping_rounds,
        **xgb_params,
    )
    xgb_candidate.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=False,
    )
    xgb_best_iteration = max(1, int(xgb_candidate.best_iteration + 1))

    combined_x = pd.concat([train_x, valid_x], ignore_index=True)
    combined_y = pd.concat([train_y, valid_y], ignore_index=True)
    lgb_final = LGBMRegressor(n_estimators=lgb_best_iteration, **lgb_params)
    lgb_final.fit(combined_x, combined_y)
    xgb_final = XGBRegressor(
        n_estimators=xgb_best_iteration,
        **xgb_params,
    )
    xgb_final.fit(combined_x, combined_y, verbose=False)

    return (
        {"LightGBM": lgb_final, "XGBoost": xgb_final},
        {"LightGBM": lgb_best_iteration, "XGBoost": xgb_best_iteration},
        {
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "lightgbm_parameters": lgb_params,
            "xgboost_parameters": xgb_params,
        },
    )


def train_tree_snapshot(
    samples: pd.DataFrame,
    *,
    forecast_origin: pd.Timestamp = FORECAST_ORIGIN,
    validation_days: int = VALIDATION_DAYS,
    max_estimators: int = 600,
    early_stopping_rounds: int = 50,
    n_jobs: int = 4,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select iterations on recent history, then refit through origin minus one day."""
    origin, train_start, train_end, validation_start, validation_end = snapshot_periods(
        forecast_origin, validation_days=validation_days
    )
    train_rows = _period_rows(samples, train_start, train_end)
    valid_rows = _period_rows(samples, validation_start, validation_end)
    if train_rows.empty or valid_rows.empty:
        raise ValueError(
            "固定切分缺少可用监督样本: "
            f"train={len(train_rows)}, validation={len(valid_rows)}"
        )

    train_x = _feature_matrix(train_rows)
    train_y = train_rows["y_raw"].astype("float32")
    valid_x = _feature_matrix(valid_rows)
    valid_y = valid_rows["y_raw"].astype("float32")
    models, best_iterations, versions = _fit_model_candidates(
        train_x,
        train_y,
        valid_x,
        valid_y,
        max_estimators=max_estimators,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    final_rows = pd.concat([train_rows, valid_rows], ignore_index=True)
    train_price = pd.to_numeric(train_rows["y_raw"], errors="raise")
    metadata = {
        "schema_version": 1,
        "model_family": "tree_regression",
        "feature_names": list(FEATURE_COLUMNS),
        "train_period": [train_start.date().isoformat(), train_end.date().isoformat()],
        "validation_period": [
            validation_start.date().isoformat(),
            validation_end.date().isoformat(),
        ],
        "train_data_end": validation_end.date().isoformat(),
        "trained_as_of": origin.isoformat(),
        "training_rows": int(len(train_rows)),
        "validation_rows": int(len(valid_rows)),
        "refit_rows": int(len(final_rows)),
        "best_iterations": best_iterations,
        "versions": versions,
        "random_state": random_state,
        "training_controls": {
            "max_estimators": max_estimators,
            "early_stopping_rounds": early_stopping_rounds,
            "n_jobs": n_jobs,
        },
        "availability_verified": False,
        "availability_basis": "assumed",
        "assumption_ids": list(ASSUMPTION_IDS),
        "target_time_semantics": {
            "key": ["trade_date", "slot_id"],
            "interval_boundary": "unknown",
            "time_label_preserved": True,
        },
        "spike_threshold_candidate_yuan_per_mwh": SPIKE_THRESHOLD_CANDIDATE,
        "training_price_p99_yuan_per_mwh": float(train_price.quantile(0.99)),
        "spike_threshold_yuan_per_mwh": float(train_price.quantile(0.99)),
        "spike_threshold_method": "99th percentile of observed training target prices",
    }
    return models, metadata


def train_fixed_tree_models(
    samples: pd.DataFrame,
    *,
    max_estimators: int = 600,
    early_stopping_rounds: int = 50,
    n_jobs: int = 4,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the frozen 2025-10-01 fixed-origin tree snapshot."""
    return train_tree_snapshot(
        samples,
        forecast_origin=FORECAST_ORIGIN,
        validation_days=VALIDATION_DAYS,
        max_estimators=max_estimators,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def predict_tree_models(
    models: dict[str, Any], samples: pd.DataFrame
) -> pd.DataFrame:
    features = _feature_matrix(samples)
    predictions = samples.loc[:, ["trade_date", "time_label", "slot_id"]].copy()
    for name, model in models.items():
        predictions[name] = np.asarray(model.predict(features), dtype="float64")
    return predictions


def _binary_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_negative = actual < 0
    predicted_negative = predicted < 0
    true_positive = int(np.logical_and(actual_negative, predicted_negative).sum())
    false_positive = int(np.logical_and(~actual_negative, predicted_negative).sum())
    false_negative = int(np.logical_and(actual_negative, ~predicted_negative).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "negative_precision": precision,
        "negative_recall": recall,
        "negative_f1": f1,
    }


def _regression_scores(
    actual: np.ndarray, predicted: np.ndarray, spike_threshold: float
) -> dict[str, float | int]:
    error = predicted - actual
    absolute_error = np.abs(error)
    smape_denominator = np.abs(actual) + np.abs(predicted)
    zero_denominator = smape_denominator == 0
    smape_values = np.divide(
        2 * absolute_error,
        smape_denominator,
        out=np.zeros_like(absolute_error),
        where=~zero_denominator,
    )
    spike_mask = actual >= spike_threshold
    return {
        "count": int(actual.size),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "smape": float(smape_values.mean()),
        "smape_zero_denominator_count": int(zero_denominator.sum()),
        "spike_threshold_yuan_per_mwh": float(spike_threshold),
        "spike_target_count": int(spike_mask.sum()),
        "spike_mae": float(absolute_error[spike_mask].mean()) if spike_mask.any() else 0.0,
        **_binary_scores(actual, predicted),
    }


def score_forecasts(
    samples: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    start: pd.Timestamp = TEST_START,
    end: pd.Timestamp = TEST_END,
    spike_threshold: float,
) -> dict[str, Any]:
    """Score every model and baseline on the exact same observed target points."""
    prediction_columns = [
        column
        for column in predictions.columns
        if column not in {"trade_date", "time_label", "slot_id"}
    ]
    required_predictions = {"LightGBM", "XGBoost", "PatchTST"}
    if not required_predictions.issubset(prediction_columns):
        raise ValueError(f"预测结果缺少模型列: {sorted(required_predictions - set(prediction_columns))}")

    keys = ["trade_date", "slot_id"]
    left = samples.loc[
        :, [*keys, "target_observed", "y_raw", "baseline_yesterday", "baseline_last_week"]
    ].copy()
    right = predictions.loc[:, [*keys, *prediction_columns]].copy()
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("预测评分键 trade_date + slot_id 必须唯一")
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    dates = pd.to_datetime(merged["trade_date"], errors="raise").dt.normalize()
    in_period = dates.between(start.normalize(), end.normalize())
    observed = pd.to_numeric(merged["target_observed"], errors="coerce").eq(1)
    required = ["y_raw", "baseline_yesterday", "baseline_last_week", *required_predictions]
    finite = np.isfinite(merged[required].to_numpy(dtype="float64")).all(axis=1)
    common = merged.loc[in_period & observed & finite].copy()
    if common.empty:
        raise ValueError("指定回测窗口没有模型与两条基线共同可评分的目标点")

    actual = common["y_raw"].to_numpy(dtype="float64")
    scores = {
        name: _regression_scores(actual, common[name].to_numpy(dtype="float64"), spike_threshold)
        for name in ["LightGBM", "XGBoost", "PatchTST", "baseline_yesterday", "baseline_last_week"]
    }
    eligible = int((in_period & observed).sum())
    return {
        "period": [start.date().isoformat(), end.date().isoformat()],
        "eligible_observed_targets": eligible,
        "common_scored_targets": int(len(common)),
        "common_coverage": float(len(common) / eligible) if eligible else 0.0,
        "models_and_baselines": scores,
        "smape_zero_denominator_rule": "count exact 0/0 pairs as zero and report their count",
    }


def explain_tree_prediction(
    model_name: str,
    model: Any,
    row: pd.DataFrame,
) -> dict[str, Any]:
    """Return native TreeSHAP contributions without adding a SHAP dependency."""
    features = _feature_matrix(row)
    if model_name == "LightGBM":
        contributions = np.asarray(model.booster_.predict(features, pred_contrib=True))[0]
    elif model_name == "XGBoost":
        import xgboost as xgb

        matrix = xgb.DMatrix(features, feature_names=list(FEATURE_COLUMNS))
        contributions = np.asarray(
            model.get_booster().predict(matrix, pred_contribs=True)
        )[0]
    else:
        raise ValueError(f"不支持树模型解释: {model_name}")

    feature_values = features.iloc[0]
    effects = [
        {
            "feature": feature,
            "value": None if pd.isna(feature_values[feature]) else float(feature_values[feature]),
            "shap_value": float(contributions[index]),
        }
        for index, feature in enumerate(FEATURE_COLUMNS)
    ]
    effects.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return {
        "model": model_name,
        "prediction": float(np.asarray(model.predict(features))[0]),
        "base_value": float(contributions[-1]),
        "feature_contributions": effects,
    }


def save_tree_model_bundle(
    models: dict[str, Any], metadata: dict[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    """Write local, versioned model and JSON metadata artifacts atomically."""
    import joblib

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "tree_models.joblib"
    metadata_path = target / "tree_models.json"

    with tempfile.NamedTemporaryFile(dir=target, suffix=".joblib.tmp", delete=False) as handle:
        model_tmp = Path(handle.name)
    try:
        joblib.dump({"models": models, "metadata": metadata}, model_tmp)
        os.replace(model_tmp, model_path)
    finally:
        model_tmp.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target, suffix=".json.tmp", delete=False
    ) as handle:
        metadata_tmp = Path(handle.name)
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(metadata_tmp, metadata_path)
    finally:
        metadata_tmp.unlink(missing_ok=True)
    return model_path, metadata_path
