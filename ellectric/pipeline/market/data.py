"""
山东市场分析数据准备 — Shandong Market Data Preparation
=========================================================

读取原始山东 15 分钟电力 CSV 和历史气象 Parquet，生成独立于旧展示链路
的分析数据、覆盖报告、来源清单及 DuckDB 数据库。

时间策略：
  trade_date + slot_id 是市场样本键
  time_label 保留原始值
  label_at_local / label_at_utc 仅表示源时钟标签
  区间起止及气象缓存时间基础均保持 unknown

原始文件保持只读；日前价格不进入读取列集合。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow

logger = logging.getLogger(__name__)

MARKET_TIMEZONE = "Asia/Shanghai"
TRANSFORM_VERSION = "market-data-v1"

BASE_SOURCE_COLUMNS = {
    "日期": "trade_date",
    "时刻": "time_label",
    "是否节假日": "is_holiday",
    "是否周末休息日": "is_weekend",
}

ACTUAL_SOURCE_COLUMNS = {
    "直调负荷(实际)": "load_actual_mw",
    "联络线受电负荷(实际)": "tie_line_actual_mw",
    "风电总加(实际)": "wind_actual_mw",
    "光伏总加(实际)": "solar_actual_mw",
    "抽蓄(实际)": "pumped_storage_actual_mw",
    "地方电厂发电总加(实际)": "local_gen_actual_mw",
    "非市场化核电总加(实际)": "nuclear_actual_mw",
    "自备机组总加(实际)": "captive_actual_mw",
}

FORECAST_SOURCE_COLUMNS = {
    "直调负荷(预测)": "load_forecast_mw",
    "联络线受电负荷(预测)": "tie_line_forecast_mw",
    "风电总加(预测)": "wind_forecast_mw",
    "光伏总加(预测)": "solar_forecast_mw",
    "非市场化核电总加(预测)": "nuclear_forecast_mw",
    "自备机组总加(预测)": "captive_forecast_mw",
    "地方电厂发电总加(预测)": "local_gen_forecast_mw",
}

WEATHER_JINAN_COLUMNS = (
    "temp_jinan",
    "ghi_jinan",
    "wind_speed_jinan",
    "precip_jinan",
    "humidity_jinan",
    "cloud_jinan",
)

_SOURCE_COLUMNS = tuple(
    BASE_SOURCE_COLUMNS
    | ACTUAL_SOURCE_COLUMNS
    | FORECAST_SOURCE_COLUMNS
    | {"实时价格": "rt_price"}
)
_NUMERIC_SOURCE_MAP = {
    **ACTUAL_SOURCE_COLUMNS,
    **FORECAST_SOURCE_COLUMNS,
    "实时价格": "rt_price",
}
_OUTPUT_FILES = (
    "sd_rt_15min.parquet",
    "weather_actual_hourly.parquet",
    "quality_report.json",
    "dataset_manifest.json",
    "sd_market.duckdb",
)


def _slot_id(time_label: str) -> int:
    """把源时钟标签映射到 0-based 日内槽位，不赋予区间边界含义。"""
    match = re.fullmatch(r"(\d{2}):(\d{2})", str(time_label).strip())
    if match is None:
        raise ValueError(f"无法解析电力时刻标签: {time_label!r}")

    hour, minute = (int(part) for part in match.groups())
    if minute not in (0, 15, 30, 45) or (hour == 24 and minute != 0):
        raise ValueError(f"时刻标签不在 15 分钟槽位上: {time_label!r}")
    total_minutes = hour * 60 + minute
    if not 15 <= total_minutes <= 24 * 60:
        raise ValueError(f"时刻标签超出 00:15..24:00 范围: {time_label!r}")
    return total_minutes // 15 - 1


def _to_numeric(series: pd.Series, source_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        rows = invalid[invalid].index[:5].tolist()
        raise ValueError(
            f"字段 {source_name!r} 含 {int(invalid.sum())} 个非数值内容，样本行: {rows}"
        )
    return numeric.astype("float64")


def _read_market_csv(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    missing = sorted(set(_SOURCE_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"电力 CSV 缺少实施所需字段: {missing}")

    # usecols 明确排除日前价格，避免将其带入新分析链路。
    source = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=list(_SOURCE_COLUMNS),
        dtype="string",
    )
    return source


def _normalise_market(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(set(_SOURCE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"电力数据缺少实施所需字段: {missing}")

    frame = pd.DataFrame(index=source.index)
    dates = pd.to_datetime(
        source["日期"].astype("string").str.strip(),
        format="%Y-%m-%d",
        errors="raise",
    )
    frame["trade_date"] = dates.dt.date
    frame["time_label"] = source["时刻"].astype("string").str.strip()
    frame["slot_id"] = frame["time_label"].map(_slot_id).astype("int16")

    is_2400 = frame["time_label"].eq("24:00")
    clock_label = frame["time_label"].mask(is_2400, "00:00")
    local_naive = pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + clock_label,
        format="%Y-%m-%d %H:%M",
        errors="raise",
    )
    local_naive.loc[is_2400] += pd.Timedelta(days=1)
    frame["label_at_local"] = local_naive.dt.tz_localize(MARKET_TIMEZONE)
    frame["label_at_utc"] = frame["label_at_local"].dt.tz_convert("UTC")

    for source_name, output_name in BASE_SOURCE_COLUMNS.items():
        if source_name in ("日期", "时刻"):
            continue
        raw = source[source_name].astype("string").str.strip()
        values = raw.map({"是": 1, "否": 0})
        invalid = raw.notna() & values.isna()
        if invalid.any():
            raise ValueError(f"字段 {source_name!r} 含未知标记: {raw[invalid].unique().tolist()}")
        frame[output_name] = values.astype("Int8")

    for source_name, output_name in _NUMERIC_SOURCE_MAP.items():
        values = source[source_name].astype("string").str.strip()
        frame[output_name] = _to_numeric(values, source_name)

    if frame.duplicated(["trade_date", "slot_id"]).any():
        duplicates = frame.loc[
            frame.duplicated(["trade_date", "slot_id"], keep=False),
            ["trade_date", "slot_id", "time_label"],
        ].head(10)
        raise ValueError(f"交易日与槽位键重复:\n{duplicates.to_string(index=False)}")

    frame["solar_actual_clipped_mw"] = frame["solar_actual_mw"].clip(lower=0)
    frame["solar_forecast_clipped_mw"] = frame["solar_forecast_mw"].clip(lower=0)
    frame["actual_bid_space_mw"] = (
        frame["load_actual_mw"]
        - frame["tie_line_actual_mw"]
        - frame["wind_actual_mw"]
        - frame["solar_actual_clipped_mw"]
        - frame["nuclear_actual_mw"]
        - frame["captive_actual_mw"]
        - frame["local_gen_actual_mw"]
    )
    frame["forecast_bid_space_mw"] = (
        frame["load_forecast_mw"]
        - frame["tie_line_forecast_mw"]
        - frame["wind_forecast_mw"]
        - frame["solar_forecast_clipped_mw"]
        - frame["nuclear_forecast_mw"]
        - frame["captive_forecast_mw"]
        - frame["local_gen_forecast_mw"]
    )

    frame = frame.sort_values(["trade_date", "slot_id"], kind="stable").reset_index(drop=True)
    counts = frame.groupby("trade_date", observed=True)["slot_id"].agg(["size", "nunique"])
    expected_slots = set(range(96))
    missing_slots = {
        day.isoformat(): sorted(expected_slots - set(group["slot_id"].astype(int)))
        for day, group in frame.groupby("trade_date", observed=True)
        if set(group["slot_id"].astype(int)) != expected_slots
    }

    quality = {
        "row_count": int(len(frame)),
        "trade_date_count": int(frame["trade_date"].nunique()),
        "trade_date_start": frame["trade_date"].min().isoformat(),
        "trade_date_end": frame["trade_date"].max().isoformat(),
        "daily_slot_anomalies": {
            day.isoformat(): {"rows": int(row.size), "unique_slots": int(row.nunique)}
            for day, row in counts.iterrows()
            if row["size"] != 96 or row["nunique"] != 96
        },
        "missing_slots_by_trade_date": missing_slots,
        "numeric_missing_counts": {
            name: int(frame[name].isna().sum()) for name in _NUMERIC_SOURCE_MAP.values()
        },
        "negative_rt_price_count": int((frame["rt_price"] < 0).sum()),
        "negative_solar_actual_count": int((frame["solar_actual_mw"] < 0).sum()),
        "solar_clip_applied_count": int((frame["solar_actual_mw"] < 0).sum()),
        "actual_bid_space_missing_count": int(frame["actual_bid_space_mw"].isna().sum()),
        "forecast_bid_space_missing_count": int(frame["forecast_bid_space_mw"].isna().sum()),
        "price_missing_segments": _missing_segments(frame, "rt_price"),
        "time_semantics": {
            "key": ["trade_date", "slot_id"],
            "interval_boundary": "unknown",
            "time_label_preserved": True,
            "label_timezone": MARKET_TIMEZONE,
            "label_at_fields_are_interval_boundaries": False,
        },
    }
    return frame, quality


def _missing_segments(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    missing = frame.loc[frame[column].isna(), ["trade_date", "slot_id", "time_label"]]
    segments: list[dict[str, Any]] = []
    for day, group in missing.groupby("trade_date", observed=True):
        points = group.sort_values("slot_id").to_dict("records")
        start = previous = points[0]
        for point in points[1:]:
            if int(point["slot_id"]) != int(previous["slot_id"]) + 1:
                segments.append(_segment_record(day, start, previous))
                start = point
            previous = point
        segments.append(_segment_record(day, start, previous))
    return segments


def _segment_record(day: Any, start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    return {
        "trade_date": day.isoformat(),
        "start_slot_id": int(start["slot_id"]),
        "end_slot_id": int(end["slot_id"]),
        "start_time_label": str(start["time_label"]),
        "end_time_label": str(end["time_label"]),
        "missing_points": int(end["slot_id"]) - int(start["slot_id"]) + 1,
    }


def _normalise_jinan_weather(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(path)
    if isinstance(raw.index, pd.DatetimeIndex):
        timestamps = raw.index
    elif "timestamp" in raw.columns:
        timestamps = pd.DatetimeIndex(raw.pop("timestamp"))
    else:
        raise ValueError("气象 Parquet 缺少时间索引或 timestamp 字段")

    missing = sorted(set(WEATHER_JINAN_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"气象 Parquet 缺少济南字段: {missing}")

    frame = raw.loc[:, WEATHER_JINAN_COLUMNS].copy().reset_index(drop=True)
    frame.insert(0, "timestamp_raw", timestamps)
    quality = {
        "row_count": int(len(frame)),
        "timestamp_dtype_as_stored": str(timestamps.dtype),
        "timestamp_start_as_stored": str(timestamps.min()),
        "timestamp_end_as_stored": str(timestamps.max()),
        "timestamp_timezone_basis": "unknown",
        "units_basis": "unknown",
        "observation_provenance": "unknown",
        "null_counts": {name: int(frame[name].isna().sum()) for name in WEATHER_JINAN_COLUMNS},
        "duplicate_timestamp_count": int(timestamps.duplicated().sum()),
        "resampled_or_shifted": False,
        "city": "jinan",
    }
    return frame, quality


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quality_rows(report: dict[str, Any]) -> pd.DataFrame:
    market = report["market"]
    weather = report["weather"]
    rows = [
        {
            "check_id": "market.daily_slots",
            "status": "pass" if not market["daily_slot_anomalies"] else "warning",
            "actual": str(len(market["daily_slot_anomalies"])),
            "expected": "0 anomalous trade dates",
            "notes": "每个交易日应有 96 个不同槽位。",
        },
        {
            "check_id": "market.price_missing",
            "status": "pass" if not market["numeric_missing_counts"]["rt_price"] else "warning",
            "actual": str(market["numeric_missing_counts"]["rt_price"]),
            "expected": "0 missing values",
            "notes": "缺失价格保留为空值，不插值。",
        },
        {
            "check_id": "market.interval_boundary",
            "status": "unknown",
            "actual": "unknown",
            "expected": "verified source convention",
            "notes": "只使用 trade_date + slot_id；不生成区间起止时刻。",
        },
        {
            "check_id": "weather.timezone_units_provenance",
            "status": "unknown",
            "actual": "unknown",
            "expected": "verified source metadata",
            "notes": "保留缓存原值，不时移、不转换单位、不对齐。",
        },
    ]
    if market["numeric_missing_counts"]["nuclear_forecast_mw"]:
        rows.append(
            {
                "check_id": "market.forecast_bid_space_coverage",
                "status": "warning",
                "actual": str(market["forecast_bid_space_missing_count"]),
                "expected": "complete forecast components",
                "notes": "核电等任一预测分项缺失时，完整预测竞价空间保持为空。",
            }
        )
    if weather["duplicate_timestamp_count"]:
        rows.append(
            {
                "check_id": "weather.duplicate_timestamp",
                "status": "warning",
                "actual": str(weather["duplicate_timestamp_count"]),
                "expected": "0 duplicates",
                "notes": "气象源时间戳未去重。",
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_market_dataset(
    power_csv: str | Path | None = None,
    weather_parquet: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """生成分析专用的 Parquet、DuckDB、质量报告与输入清单。"""
    repo_root = Path(__file__).resolve().parents[3]
    source_dir = repo_root / "ellectric" / "data" / "shandong"
    power_path = Path(power_csv) if power_csv else source_dir / "山东_2024-2026_15min.csv"
    weather_path = Path(weather_parquet) if weather_parquet else source_dir / "weather_2024-2026.parquet"
    target_dir = Path(output_dir) if output_dir else repo_root / "ellectric" / "data" / "market_processed"

    for path in (power_path, weather_path):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {path}")

    existing = [target_dir / name for name in _OUTPUT_FILES if (target_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "目标产物已存在；为保护现有数据，默认不覆盖。使用 --overwrite 明确重建: "
            + ", ".join(str(path) for path in existing)
        )

    market_source = _read_market_csv(power_path)
    market, market_quality = _normalise_market(market_source)
    weather, weather_quality = _normalise_jinan_weather(weather_path)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market": market_quality,
        "weather": weather_quality,
    }
    manifest = {
        "schema_version": 1,
        "transform_version": TRANSFORM_VERSION,
        "generated_at_utc": report["generated_at_utc"],
        "inputs": {
            "power_csv": {
                "path": str(power_path.resolve()),
                "sha256": _sha256(power_path),
                "size_bytes": power_path.stat().st_size,
                "rows": int(len(market)),
                "official_source_verified": False,
            },
            "weather_parquet": {
                "path": str(weather_path.resolve()),
                "sha256": _sha256(weather_path),
                "size_bytes": weather_path.stat().st_size,
                "rows": int(len(weather)),
                "time_basis": "unknown",
                "units_basis": "unknown",
                "observation_provenance": "unknown",
            },
        },
        "market_time": market_quality["time_semantics"],
        "scope": {
            "price_type": "real_time_only",
            "day_ahead_price_processed": False,
            "weather_city": "jinan_only",
            "weather_used_as_forecast_feature": False,
        },
        "outputs": list(_OUTPUT_FILES),
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "duckdb": duckdb.__version__,
        },
    }

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target_dir.name}-", dir=target_dir.parent) as temp:
        stage = Path(temp)
        market.to_parquet(stage / _OUTPUT_FILES[0], index=False, engine="pyarrow")
        weather.to_parquet(stage / _OUTPUT_FILES[1], index=False, engine="pyarrow")
        _write_json(stage / _OUTPUT_FILES[2], report)
        _write_json(stage / _OUTPUT_FILES[3], manifest)

        db_path = stage / _OUTPUT_FILES[4]
        quality_table = _quality_rows(report)
        with duckdb.connect(str(db_path)) as connection:
            connection.register("_market_frame", market)
            connection.register("_weather_frame", weather)
            connection.register("_quality_frame", quality_table)
            connection.execute("CREATE TABLE market_15min AS SELECT * FROM _market_frame")
            connection.execute(
                "CREATE TABLE weather_actual_hourly AS SELECT * FROM _weather_frame"
            )
            connection.execute("CREATE TABLE data_quality AS SELECT * FROM _quality_frame")

        target_dir.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            os.replace(stage / name, target_dir / name)

    logger.info("山东市场数据产物已生成: %s", target_dir)
    return {
        "output_dir": str(target_dir.resolve()),
        "market_rows": int(len(market)),
        "weather_rows": int(len(weather)),
        "price_missing": market_quality["numeric_missing_counts"]["rt_price"],
        "forecast_bid_space_missing": market_quality["forecast_bid_space_missing_count"],
        "interval_boundary": "unknown",
        "weather_time_basis": "unknown",
    }
