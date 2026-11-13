"""Train the fixed-origin market models and score the frozen test period."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from ellectric.pipeline.market.features import build_forecast_samples
from ellectric.pipeline.market.forecasting import (
    TEST_END,
    TEST_START,
    predict_tree_models,
    save_tree_model_bundle,
    score_forecasts,
    train_fixed_tree_models,
)
from ellectric.pipeline.market.neural import (
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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "ellectric" / "data" / "market_processed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-parquet", type=Path, default=data_dir / "sd_rt_15min.parquet")
    parser.add_argument("--dataset-manifest", type=Path, default=data_dir / "dataset_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=data_dir / "model_run_2025-10-01")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--patch-max-steps", type=int, default=500)
    parser.add_argument("--patch-val-check-steps", type=int, default=250)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.market_parquet, args.dataset_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {path}")

    model_dir = args.output_dir / "models"
    tree_model_path = model_dir / "tree_models.joblib"
    tree_metadata_path = model_dir / "tree_models.json"
    patch_model_path = model_dir / "patchtst_2025-10-01"
    patch_metadata_path = model_dir / "patchtst_2025-10-01.json"
    predictions_path = args.output_dir / "fixed_test_predictions.parquet"
    metrics_path = args.output_dir / "fixed_test_metrics.json"
    outputs = [
        tree_model_path,
        tree_metadata_path,
        patch_model_path,
        patch_metadata_path,
        predictions_path,
        metrics_path,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
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

    tree_models, tree_metadata = train_fixed_tree_models(
        samples,
        max_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    tree_metadata["data_provenance"] = data_provenance
    save_tree_model_bundle(tree_models, tree_metadata, model_dir)

    patch_forecast, patch_frame, patch_metadata = train_patchtst_snapshot(
        market,
        max_steps=args.patch_max_steps,
        val_check_steps=args.patch_val_check_steps,
        random_seed=args.seed,
    )
    patch_metadata["data_provenance"] = data_provenance
    patch_forecast.save(
        path=str(patch_model_path),
        save_dataset=False,
        overwrite=args.overwrite,
    )
    _write_json(patch_metadata_path, patch_metadata)

    test_mask = samples["trade_date"].between(TEST_START, TEST_END)
    test_samples = samples.loc[test_mask].copy()
    tree_predictions = predict_tree_models(tree_models, test_samples)
    target_dates = sorted(test_samples["trade_date"].drop_duplicates().tolist())
    patch_predictions = predict_patchtst_targets(patch_forecast, patch_frame, target_dates)
    predictions = tree_predictions.merge(
        patch_predictions.drop(columns="time_label"),
        on=["trade_date", "slot_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(predictions) != len(test_samples):
        raise RuntimeError(
            f"固定测试预测覆盖不完整: samples={len(test_samples)}, predictions={len(predictions)}"
        )

    threshold = float(tree_metadata["spike_threshold_yuan_per_mwh"])
    metrics = score_forecasts(
        samples,
        predictions,
        start=TEST_START,
        end=TEST_END,
        spike_threshold=threshold,
    )
    report = {
        "schema_version": 1,
        "evaluation": metrics,
        "tree_snapshot": tree_metadata,
        "patchtst_snapshot": patch_metadata,
        "test_predictions_path": str(predictions_path.resolve()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)
    _write_json(metrics_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
