"""Retrain tree and PatchTST snapshots every seven days over the frozen test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from ellectric.pipeline.market.features import build_forecast_samples, build_patchtst_frame
from ellectric.pipeline.market.forecasting import (
    TEST_END,
    TEST_START,
    predict_tree_models,
    rolling_origins,
    save_tree_model_bundle,
    score_forecasts,
    train_tree_snapshot,
)
from ellectric.pipeline.market.neural import (
    HORIZON,
    INPUT_SIZE,
    predict_patchtst_targets,
    train_patchtst_snapshot,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_tree_snapshot(
    tree_dir: Path,
    *,
    origin: pd.Timestamp,
    data_provenance: dict,
    n_estimators: int,
    early_stopping_rounds: int,
    n_jobs: int,
    seed: int,
) -> tuple[dict | None, dict | None]:
    model_path = tree_dir / "tree_models.joblib"
    metadata_path = tree_dir / "tree_models.json"
    existing = [path.exists() for path in (model_path, metadata_path)]
    if not any(existing):
        if tree_dir.exists() and any(tree_dir.iterdir()):
            raise RuntimeError(f"续跑树模型目录包含不完整产物: {tree_dir}")
        return None, None
    if not all(existing):
        raise RuntimeError(f"续跑树模型文件不完整: {tree_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_controls = {
        "max_estimators": n_estimators,
        "early_stopping_rounds": early_stopping_rounds,
        "n_jobs": n_jobs,
    }
    if metadata.get("trained_as_of") != origin.isoformat():
        raise RuntimeError(f"树模型起报时间与目录不符: {model_path}")
    if metadata.get("data_provenance") != data_provenance:
        raise RuntimeError(f"树模型数据指纹与当前输入不符: {model_path}")
    if metadata.get("training_controls") != expected_controls:
        raise RuntimeError(f"树模型训练参数与本次续跑不符: {model_path}")
    if metadata.get("random_state") != seed:
        raise RuntimeError(f"树模型随机种子与本次续跑不符: {model_path}")

    import joblib

    bundle = joblib.load(model_path)
    if bundle.get("metadata") != metadata:
        raise RuntimeError(f"树模型内外元数据不一致: {model_path}")
    models = bundle.get("models")
    if not isinstance(models, dict) or set(models) != {"LightGBM", "XGBoost"}:
        raise RuntimeError(f"树模型快照缺少预期模型: {model_path}")
    if not all(callable(getattr(model, "predict", None)) for model in models.values()):
        raise RuntimeError(f"树模型快照不可推理: {model_path}")
    return models, metadata


def _load_patchtst_snapshot(
    patch_dir: Path,
    *,
    max_steps: int,
    val_check_steps: int,
    seed: int,
) -> tuple[object | None, dict | None]:
    config_path = patch_dir / "configuration.json"
    weights_path = patch_dir / "PatchTST_0.safetensors"
    existing = [path.exists() for path in (config_path, weights_path)]
    if not any(existing):
        if patch_dir.exists() and any(patch_dir.iterdir()):
            raise RuntimeError(f"续跑 PatchTST 目录包含不完整产物: {patch_dir}")
        return None, None
    if not all(existing):
        raise RuntimeError(f"续跑 PatchTST 文件不完整: {patch_dir}")

    from neuralforecast import NeuralForecast

    forecast = NeuralForecast.load(path=str(patch_dir), verbose=False)
    if getattr(forecast, "freq", None) != "15min" or len(forecast.models) != 1:
        raise RuntimeError(f"PatchTST 频率或模型数量与预期不符: {patch_dir}")
    model = forecast.models[0]
    params = getattr(model, "hparams", {})
    expected_params = {
        "h": HORIZON,
        "input_size": INPUT_SIZE,
        "max_steps": max_steps,
        "val_check_steps": val_check_steps,
        "batch_size": 32,
        "valid_batch_size": 32,
        "windows_batch_size": 128,
        "random_seed": seed,
        "encoder_layers": 3,
        "n_heads": 4,
        "hidden_size": 64,
        "linear_hidden_size": 128,
        "patch_len": 16,
        "stride": 8,
        "revin": True,
        "revin_affine": False,
        "revin_subtract_last": True,
        "scaler_type": "identity",
    }
    mismatches = {
        name: {"expected": expected, "actual": params.get(name)}
        for name, expected in expected_params.items()
        if params.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"PatchTST 快照配置不匹配 {patch_dir}: {mismatches}")

    return forecast, {
        "artifact_state": "reused",
        "model": "PatchTST",
        "frequency": forecast.freq,
        "trained_as_of": None,
        "training_controls": {key: params[key] for key in expected_params},
        "provenance_note": "legacy snapshot has no dedicated PatchTST metadata file; colocated tree metadata and origin path are validated separately",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "ellectric" / "data" / "market_processed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-parquet", type=Path, default=data_dir / "sd_rt_15min.parquet")
    parser.add_argument("--dataset-manifest", type=Path, default=data_dir / "dataset_manifest.json")
    parser.add_argument(
        "--output-dir", type=Path, default=data_dir / "model_run_rolling_2025-10-01"
    )
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--patch-max-steps", type=int, default=500)
    parser.add_argument("--patch-val-check-steps", type=int, default=250)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete, matching snapshots and train missing model artifacts",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume 与 --overwrite 不能同时使用")

    for path in (args.market_parquet, args.dataset_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {path}")

    predictions_path = args.output_dir / "rolling_test_predictions.parquet"
    metrics_path = args.output_dir / "rolling_test_metrics.json"
    model_root = args.output_dir / "models"
    existing = [path for path in (predictions_path, metrics_path) if path.exists()]
    if args.resume and existing:
        raise FileExistsError(
            "续跑拒绝覆盖已有最终预测或指标文件: " + ", ".join(str(path) for path in existing)
        )
    if model_root.exists() and any(model_root.iterdir()):
        existing.append(model_root)
    if existing and not args.overwrite and not args.resume:
        raise FileExistsError(
            "目标产物已存在；为保护已有结果，默认不覆盖。使用 --overwrite 明确重建: "
            + ", ".join(str(path) for path in existing)
        )

    market = pd.read_parquet(args.market_parquet)
    samples = build_forecast_samples(market)
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    data_provenance = {
        "market_parquet": str(args.market_parquet.resolve()),
        "raw_power_csv_sha256": manifest["inputs"]["power_csv"]["sha256"],
        "raw_weather_parquet_sha256": manifest["inputs"]["weather_parquet"]["sha256"],
        "transform_version": manifest["transform_version"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    snapshot_reports: list[dict] = []
    score_threshold: float | None = None
    started_at = time.perf_counter()
    origins = rolling_origins(TEST_START, TEST_END, interval_days=7)

    for origin_number, origin_date in enumerate(origins, start=1):
        origin = origin_date.tz_localize("Asia/Shanghai")
        target_end = min(origin_date + pd.Timedelta(days=6), TEST_END)
        target_mask = pd.to_datetime(samples["trade_date"]).between(origin_date, target_end)
        target_samples = samples.loc[target_mask].copy()
        target_dates = sorted(pd.to_datetime(target_samples["trade_date"]).drop_duplicates())
        if not target_dates:
            raise RuntimeError(f"{origin_date.date()} 快照没有对应测试样本")

        snapshot_dir = model_root / origin_date.date().isoformat()
        tree_dir = snapshot_dir / "trees"
        patch_dir = snapshot_dir / "patchtst"
        tree_models = tree_metadata = None
        patch_forecast = patch_metadata = None
        tree_reused = False
        patch_reused = False
        if args.resume:
            tree_models, tree_metadata = _load_tree_snapshot(
                tree_dir,
                origin=origin,
                data_provenance=data_provenance,
                n_estimators=args.n_estimators,
                early_stopping_rounds=args.early_stopping_rounds,
                n_jobs=args.n_jobs,
                seed=args.seed,
            )
            patch_forecast, patch_metadata = _load_patchtst_snapshot(
                patch_dir,
                max_steps=args.patch_max_steps,
                val_check_steps=args.patch_val_check_steps,
                seed=args.seed,
            )
            tree_reused = tree_models is not None
            patch_reused = patch_forecast is not None

        if tree_models is None:
            tree_models, tree_metadata = train_tree_snapshot(
                samples,
                forecast_origin=origin,
                max_estimators=args.n_estimators,
                early_stopping_rounds=args.early_stopping_rounds,
                n_jobs=args.n_jobs,
                random_state=args.seed,
            )
            tree_metadata["data_provenance"] = data_provenance
        if score_threshold is None:
            score_threshold = float(tree_metadata["spike_threshold_yuan_per_mwh"])

        if not tree_reused:
            save_tree_model_bundle(tree_models, tree_metadata, tree_dir)

        if patch_forecast is None:
            patch_forecast, patch_frame, patch_metadata = train_patchtst_snapshot(
                market,
                max_steps=args.patch_max_steps,
                val_check_steps=args.patch_val_check_steps,
                random_seed=args.seed,
                forecast_origin=origin,
            )
            patch_metadata["data_provenance"] = data_provenance
            patch_forecast.save(
                path=str(patch_dir),
                save_dataset=False,
                overwrite=False,
            )
        else:
            patch_frame, _ = build_patchtst_frame(market)
            patch_metadata["trained_as_of"] = origin.isoformat()
            patch_metadata["data_provenance"] = data_provenance
            patch_metadata["provenance_binding"] = "validated from colocated tree metadata and snapshot directory origin; legacy patch-specific metadata was absent"

        tree_predictions = predict_tree_models(tree_models, target_samples)
        patch_predictions = predict_patchtst_targets(
            patch_forecast, patch_frame, target_dates
        )
        window_predictions = tree_predictions.merge(
            patch_predictions.drop(columns="time_label"),
            on=["trade_date", "slot_id"],
            how="inner",
            validate="one_to_one",
        )
        expected_rows = len(target_dates) * 96
        if len(window_predictions) != expected_rows:
            raise RuntimeError(
                f"{origin_date.date()} 预测覆盖不完整: "
                f"expected={expected_rows}, actual={len(window_predictions)}"
            )
        all_predictions.append(window_predictions)
        snapshot_reports.append(
            {
                "forecast_origin": origin.isoformat(),
                "target_period": [origin_date.date().isoformat(), target_end.date().isoformat()],
                "target_days": len(target_dates),
                "prediction_rows": len(window_predictions),
                "reused_tree_snapshot": tree_reused,
                "reused_patchtst_snapshot": patch_reused,
                "tree_model_sha256": _sha256(tree_dir / "tree_models.joblib"),
                "patchtst_weights_sha256": _sha256(patch_dir / "PatchTST_0.safetensors"),
                "tree_metadata": tree_metadata,
                "patchtst_metadata": patch_metadata,
            }
        )
        print(
            json.dumps(
                {
                    "snapshot": origin_number,
                    "snapshot_count": len(origins),
                    "origin": origin.isoformat(),
                    "target_end": target_end.date().isoformat(),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    if predictions.duplicated(["trade_date", "slot_id"]).any():
        raise RuntimeError("周滚动预测包含重复 trade_date + slot_id")
    expected_total = len(pd.date_range(TEST_START, TEST_END, freq="D")) * 96
    if len(predictions) != expected_total:
        raise RuntimeError(
            f"周滚动预测覆盖不完整: expected={expected_total}, actual={len(predictions)}"
        )

    evaluation = score_forecasts(
        samples,
        predictions,
        start=TEST_START,
        end=TEST_END,
        spike_threshold=float(score_threshold),
    )
    report = {
        "schema_version": 1,
        "backtest_mode": "weekly_rolling_refit",
        "period": [TEST_START.date().isoformat(), TEST_END.date().isoformat()],
        "retrain_interval_days": 7,
        "score_threshold_yuan_per_mwh": score_threshold,
        "score_threshold_basis": "first test-origin snapshot training target p99, frozen across test",
        "snapshot_count": len(snapshot_reports),
        "snapshots": snapshot_reports,
        "evaluation": evaluation,
        "elapsed_seconds": time.perf_counter() - started_at,
        "data_provenance": data_provenance,
        "test_predictions_path": str(predictions_path.resolve()),
    }
    predictions.to_parquet(predictions_path, index=False)
    _write_json(metrics_path, report)
    print(json.dumps({"metrics_path": str(metrics_path.resolve()), "evaluation": evaluation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
