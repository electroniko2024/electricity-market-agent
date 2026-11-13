"""Export native TreeSHAP examples for a saved fixed-origin market model."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from ellectric.pipeline.market.features import build_forecast_samples
from ellectric.pipeline.market.forecasting import explain_tree_prediction


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
    run_dir = data_dir / "model_run_2025-10-01"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-parquet", type=Path, default=data_dir / "sd_rt_15min.parquet")
    parser.add_argument("--model-bundle", type=Path, default=run_dir / "models" / "tree_models.joblib")
    parser.add_argument("--date", default="2025-10-01")
    parser.add_argument("--slots", type=int, nargs="+", default=[0, 48, 95])
    parser.add_argument("--output", type=Path, default=run_dir / "tree_shap_examples.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.market_parquet, args.model_bundle):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {path}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在，为保护已有文件拒绝覆盖: {args.output}")
    if any(slot < 0 or slot > 95 for slot in args.slots):
        parser.error("slot 必须位于 0..95")

    market = pd.read_parquet(args.market_parquet)
    samples = build_forecast_samples(market)
    day = pd.Timestamp(args.date).normalize()
    bundle = joblib.load(args.model_bundle)
    models = bundle["models"]
    examples = []
    for slot in sorted(set(args.slots)):
        rows = samples.loc[
            (pd.to_datetime(samples["trade_date"]).dt.normalize() == day)
            & (samples["slot_id"] == slot)
        ]
        if len(rows) != 1:
            raise ValueError(f"{day.date()} slot_id={slot} 应有一条样本，实际 {len(rows)} 条")
        row = rows.iloc[[0]]
        actual = None if pd.isna(row.iloc[0]["y_raw"]) else float(row.iloc[0]["y_raw"])
        for model_name in ("LightGBM", "XGBoost"):
            explanation = explain_tree_prediction(model_name, models[model_name], row)
            reconstructed = explanation["base_value"] + sum(
                item["shap_value"] for item in explanation["feature_contributions"]
            )
            if not math.isclose(reconstructed, explanation["prediction"], rel_tol=1e-5, abs_tol=1e-4):
                raise AssertionError(
                    f"{model_name} TreeSHAP 分解不守恒: "
                    f"sum={reconstructed}, prediction={explanation['prediction']}"
                )
            examples.append(
                {
                    "trade_date": day.date().isoformat(),
                    "time_label": str(row.iloc[0]["time_label"]),
                    "slot_id": slot,
                    "actual_rt_price": actual,
                    **explanation,
                    "reconstruction_error": reconstructed - explanation["prediction"],
                }
            )

    report = {
        "schema_version": 1,
        "snapshot": str(args.model_bundle.parent),
        "method": "native TreeSHAP contributions from LightGBM and XGBoost",
        "target_time_semantics": {
            "key": ["trade_date", "slot_id"],
            "time_label_preserved": True,
            "interval_boundary": "unknown",
        },
        "availability_verified": False,
        "availability_basis": "assumed",
        "assumption_ids": bundle["metadata"]["assumption_ids"],
        "examples": examples,
    }
    _write_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "example_count": len(examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
