"""准备山东市场分析数据并输出 Parquet、DuckDB 与质量清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ellectric.pipeline.market.data import build_market_dataset


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "ellectric" / "data" / "shandong"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-csv", type=Path, default=data_dir / "山东_2024-2026_15min.csv")
    parser.add_argument("--weather-parquet", type=Path, default=data_dir / "weather_2024-2026.parquet")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "ellectric" / "data" / "market_processed",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = build_market_dataset(
        power_csv=args.power_csv,
        weather_parquet=args.weather_parquet,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
