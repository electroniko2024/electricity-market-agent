"""Deterministic market tools for the Shandong analysis workflow."""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import math
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


ELLECTRIC_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ELLECTRIC_ROOT / "data" / "market_processed"
DATABASE_PATH = DATA_DIR / "sd_market.duckdb"
FIXED_SNAPSHOT_DIR = DATA_DIR / "model_run_2025-10-01"
ROLLING_SNAPSHOT_DIR = DATA_DIR / "model_run_rolling_2025-10-01"
POLICY_DOCUMENTS_PATH = DATA_DIR / "policy_documents.json"
POLICY_METADATA_PATH = DATA_DIR / "policy_metadata.json"
# The E3 ablation turns the effective-interval filter off (publish-date constraint stays on).
POLICY_EFFECTIVE_FILTER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "POLICY_EFFECTIVE_FILTER", default=True
)
TEST_START = date(2025, 10, 1)
TEST_END = date(2026, 1, 14)
MARKET_TIMEZONE = "Asia/Shanghai"

MARKET_NOTE = (
    "market rows are keyed by trade_date + slot_id; interval boundary semantics are unknown"
)
WEATHER_NOTE = (
    "weather timestamp basis, units, and observation availability are unknown; "
    "stored timestamps are not shifted or aligned"
)
AVAILABILITY_ASSUMPTION_IDS = [
    "midnight_market_data_available",
    "target_day_forecast_snapshot_available_and_unrevised",
]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def latest_available_trade_date(as_of: datetime) -> date:
    """Trade date D is treated as complete at D 24:00 market time (the documented availability assumption)."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of 必须包含时区偏移量")
    local = pd.Timestamp(as_of).tz_convert(MARKET_TIMEZONE)
    return (local.normalize() - pd.Timedelta(days=1)).date()


DETAIL_SPAN_ERROR = (
    "单次明细查询最多覆盖 32 个交易日。更长区间的汇总（均值、极值及其所在日期时段、负价/尖峰时段数，"
    "可按日或按月分组）用 query_stats，单次最多 366 天；需要明细时把区间拆成不超过 32 天的几段分别查询"
)


_MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def _slot_time_label(slot_id: int) -> str:
    """Clock label of a 15-minute slot, named by its end time (slot 0 -> 00:15, slot 95 -> 24:00)."""
    minutes = (slot_id + 1) * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class DateRangeInput(ToolInput):
    # Detail tools return one row per slot or event, so their span is capped; aggregates override it.
    max_span_days: ClassVar[int] = 31
    span_error: ClassVar[str] = DETAIL_SPAN_ERROR

    start_date: date = Field(description="YYYY-MM-DD；写 YYYY-MM 表示从该月 1 日开始")
    end_date: date = Field(description="YYYY-MM-DD；写 YYYY-MM 表示到该月最后一个已有数据的交易日")
    as_of: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def expand_month_bounds(cls, data: Any) -> Any:
        """Accept a month ("2025-02") as a range bound: month groups are returned as YYYY-MM, so a step can
        feed a previous step's month straight into a date range."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        start, end = data.get("start_date"), data.get("end_date")
        if isinstance(start, str) and _MONTH_PATTERN.match(start.strip()):
            data["start_date"] = date.fromisoformat(f"{start.strip()}-01")
        if isinstance(end, str) and _MONTH_PATTERN.match(end.strip()):
            first = date.fromisoformat(f"{end.strip()}-01")
            last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            as_of = data.get("as_of")
            try:
                as_of_value = datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of
                if isinstance(as_of_value, datetime):
                    # A month that is still running ends at the last trade date available at as_of.
                    last = max(first, min(last, latest_available_trade_date(as_of_value)))
            except ValueError:
                pass  # an invalid as_of is reported by field validation
            data["end_date"] = last
        return data

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_date < self.start_date:
            raise ValueError("end_date 不能早于 start_date")
        if (self.end_date - self.start_date).days > self.max_span_days:
            raise ValueError(self.span_error)
        if self.as_of is not None:
            latest = latest_available_trade_date(self.as_of)
            if self.end_date > latest:
                raise ValueError(
                    f"DATA_AFTER_AS_OF: end_date {self.end_date.isoformat()} 晚于 as_of 时可得的"
                    f"最后交易日 {latest.isoformat()}"
                )
        return self


_TIME_LABEL_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})$")


class PriceQueryInput(DateRangeInput):
    slot_id: int | None = Field(default=None, ge=0, le=95)
    time_label: str | None = Field(
        default=None,
        description="按时刻取某一个时段：该时段的结束时刻 HH:MM（15 分钟一档，00:15 到 24:00）。"
        "知道时刻时用它，不要自己换算 slot_id",
    )
    limit: int = Field(default=500, ge=1, le=5000)

    @model_validator(mode="after")
    def resolve_time_label(self):
        """A clock time selects its slot here, so no caller has to convert times to slot ids."""
        if self.time_label is None:
            return self
        match = _TIME_LABEL_PATTERN.match(self.time_label.strip())
        minutes = int(match.group(1)) * 60 + int(match.group(2)) if match else -1
        if minutes % 15 or not 15 <= minutes <= 24 * 60:
            raise ValueError(
                f"time_label {self.time_label!r} 不是时段结束时刻：应为 HH:MM，15 分钟一档，从 00:15 到 24:00"
            )
        slot_id = minutes // 15 - 1
        if self.slot_id is not None and self.slot_id != slot_id:
            raise ValueError(
                f"time_label {self.time_label} 对应 slot_id {slot_id}，与传入的 slot_id {self.slot_id} 不一致；只传其中一个"
            )
        self.slot_id = slot_id
        return self


class WeatherQueryInput(DateRangeInput):
    limit: int = Field(default=500, ge=1, le=5000)


# metric -> (table, column, unit). Market columns are keyed by trade_date + slot_id; weather is hourly.
STATS_METRICS: dict[str, tuple[str, str, str]] = {
    "rt_price": ("market_15min", "rt_price", "yuan_per_mwh"),
    "load_actual_mw": ("market_15min", "load_actual_mw", "MW"),
    "wind_actual_mw": ("market_15min", "wind_actual_mw", "MW"),
    "solar_actual_mw": ("market_15min", "solar_actual_mw", "MW"),
    "tie_line_actual_mw": ("market_15min", "tie_line_actual_mw", "MW"),
    "actual_bid_space_mw": ("market_15min", "actual_bid_space_mw", "MW"),
    "temp_jinan": ("weather_actual_hourly", "temp_jinan", "degC"),
}


class StatsInput(DateRangeInput):
    max_span_days: ClassVar[int] = 365
    span_error: ClassVar[str] = "query_stats 单次最多覆盖 366 天，请把区间拆成几段"

    metric: Literal[
        "rt_price",
        "load_actual_mw",
        "wind_actual_mw",
        "solar_actual_mw",
        "tie_line_actual_mw",
        "actual_bid_space_mw",
        "temp_jinan",
    ] = "rt_price"
    group_by: Literal["day", "month", "slot"] = "day"

    @model_validator(mode="after")
    def validate_grouping(self):
        if STATS_METRICS[self.metric][0] != "market_15min" and self.group_by == "slot":
            raise ValueError(f"{self.metric} 是逐小时气象数据，没有 slot，只能按 day 或 month 分组")
        return self


class ForecastPriceInput(ToolInput):
    target_date: date
    as_of: datetime
    mode: Literal["fixed", "rolling"] = "fixed"

    @model_validator(mode="after")
    def validate_origin(self):
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of 必须包含时区偏移量")
        return self


class AnomalyInput(DateRangeInput):
    include_single_negative_points: bool = True
    methods: list[Literal[
        "negative_price",
        "floor_price",
        "spike",
        "jump",
        "forecast_residual",
        "data_quality",
    ]] = Field(
        default_factory=lambda: [
            "negative_price",
            "floor_price",
            "spike",
            "jump",
            "forecast_residual",
            "data_quality",
        ],
        min_length=1,
    )


class DriverInput(DateRangeInput):
    # Omit event_ids to explain representative price events detected in the range;
    # the planner cannot know detect_anomaly's ids before that step has run.
    event_ids: list[str] | None = Field(default=None, min_length=1, max_length=20)
    max_events: int = Field(default=3, ge=1, le=5)


class PolicySearchInput(ToolInput):
    query: str = Field(min_length=2, max_length=1000)
    as_of: datetime | None = None
    policy_at: date

    @model_validator(mode="after")
    def validate_policy_at(self):
        if self.as_of is not None:
            if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
                raise ValueError("as_of 必须包含时区偏移量")
            as_of_date = pd.Timestamp(self.as_of).tz_convert(MARKET_TIMEZONE).date()
            if self.policy_at > as_of_date:
                raise ValueError(
                    f"POLICY_AT_AFTER_AS_OF: policy_at {self.policy_at.isoformat()} 晚于 as_of 日期 "
                    f"{as_of_date.isoformat()}"
                )
        return self


def _database() -> duckdb.DuckDBPyConnection:
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(f"标准化市场数据库不存在: {DATABASE_PATH}")
    return duckdb.connect(str(DATABASE_PATH), read_only=True)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def _records(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    names = [item[0] for item in cursor.description]
    return [
        {name: _json_value(value) for name, value in zip(names, row, strict=True)}
        for row in cursor.fetchall()
    ]


def _tool_result(
    *,
    tool_name: str,
    status: str,
    data: Any,
    source: str,
    limitations: list[str] | None = None,
    assumptions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tool": tool_name,
        "status": status,
        "source": source,
        "data": data,
        "limitations": limitations or [],
        "assumptions": assumptions or [],
        "availability_verified": False,
    }


def query_price(request: PriceQueryInput) -> dict[str, Any]:
    """Return source-preserving 15-minute real-time price rows."""
    connection = _database()
    try:
        rows = _records(
            connection.execute(
                """
                SELECT trade_date, time_label, slot_id, rt_price,
                       rt_price IS NOT NULL AS target_observed
                FROM market_15min
                WHERE trade_date BETWEEN ? AND ?
                  AND (? IS NULL OR slot_id = ?)
                ORDER BY trade_date, slot_id
                LIMIT ?
                """,
                [request.start_date, request.end_date, request.slot_id, request.slot_id, request.limit],
            )
        )
        count, missing = connection.execute(
            """
            SELECT count(*), count(*) FILTER (WHERE rt_price IS NULL)
            FROM market_15min
            WHERE trade_date BETWEEN ? AND ? AND (? IS NULL OR slot_id = ?)
            """,
            [request.start_date, request.end_date, request.slot_id, request.slot_id],
        ).fetchone()
    finally:
        connection.close()
    return _tool_result(
        tool_name="query_price",
        status="ok",
        source="sd_market.duckdb:market_15min",
        data={
            "period": [request.start_date.isoformat(), request.end_date.isoformat()],
            "slot_id": request.slot_id,
            "total_rows": count,
            "missing_price_rows": missing,
            "returned_rows": len(rows),
            "truncated": len(rows) < count,
            "rows": rows,
        },
        limitations=[MARKET_NOTE, "missing realtime prices are not imputed"],
    )


def _stats_sources(metric: str, group_by: str) -> tuple[str, str, str, list[str], str]:
    """(table, value column, group expression, extreme-row key columns, date expression) for a metric."""
    table, column, _unit = STATS_METRICS[metric]
    if table == "market_15min":
        groups = {"day": "trade_date", "month": "strftime(trade_date, '%Y-%m')", "slot": "slot_id"}
        return table, column, groups[group_by], ["trade_date", "slot_id", "time_label"], "trade_date"
    day = "CAST(timestamp_raw AS DATE)"
    groups = {"day": day, "month": f"strftime({day}, '%Y-%m')"}
    return table, column, groups[group_by], ["CAST(timestamp_raw AS VARCHAR) AS timestamp_raw"], day


def _extreme_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    table: str,
    column: str,
    group_expression: str,
    key_columns: list[str],
    date_expression: str,
    start: date,
    end: date,
) -> dict[Any, dict[str, dict[str, Any]]]:
    """The source row holding each group's min and max (ties -> earliest row), keyed by group value."""
    order_keys = ", ".join(name.rsplit(" AS ", 1)[0] for name in key_columns)
    keys = ", ".join(key_columns)
    cursor = connection.execute(
        f"""
        SELECT * FROM (
            SELECT {group_expression} AS group_value, {keys}, {column},
                   row_number() OVER (PARTITION BY {group_expression} ORDER BY {column} DESC, {order_keys}) AS rank_max,
                   row_number() OVER (PARTITION BY {group_expression} ORDER BY {column} ASC, {order_keys}) AS rank_min
            FROM {table}
            WHERE {date_expression} BETWEEN ? AND ? AND {column} IS NOT NULL
        ) WHERE rank_max = 1 OR rank_min = 1
        """,
        [start, end],
    )
    out: dict[Any, dict[str, dict[str, Any]]] = {}
    for record in _records(cursor):
        group = record.pop("group_value")
        rank_max, rank_min = record.pop("rank_max"), record.pop("rank_min")
        slot = out.setdefault(group, {})
        if rank_max == 1:
            slot["max_at"] = dict(record)
        if rank_min == 1:
            slot["min_at"] = dict(record)
    return out


# Group statistics worth ranking ("which day had the highest mean / the most negative slots").
_RANKED_GROUP_FIELDS = ("mean_price", "max_price", "min_price", "negative_price_slots", "spike_price_slots",
                        "mean_value", "max_value", "min_value")
_GROUP_IDENTITY_FIELDS = ("group_value", "time_label", "first_date", "last_date")


def _group_extremes(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """For each ranked statistic: the group with the largest and the smallest value, and how many groups tie."""
    out: dict[str, Any] = {}
    for name in _RANKED_GROUP_FIELDS:
        valued = [group for group in groups if isinstance(group.get(name), (int, float)) and group[name] == group[name]]
        if not valued:
            continue
        entry: dict[str, Any] = {}
        for side, pick in (("max", max), ("min", min)):
            best = pick(group[name] for group in valued)
            winners = [group for group in valued if group[name] == best]
            # Earliest group on ties, like max_at / min_at; the tie count says whether the answer is unique.
            entry[side] = {**{key: winners[0][key] for key in _GROUP_IDENTITY_FIELDS if key in winners[0]},
                           "value": best, "tied_groups": len(winners)}
        out[name] = entry
    return out


def query_stats(request: StatsInput) -> dict[str, Any]:
    """Statistics of one metric over up to 366 days, overall and grouped by day, month or slot.

    metric: rt_price (real-time price, default), load_actual_mw, wind_actual_mw, solar_actual_mw,
    tie_line_actual_mw, actual_bid_space_mw, or temp_jinan (hourly Jinan temperature; day/month only).
    Each total and group carries max_at / min_at: the source row (trade_date, slot_id, time_label, or
    timestamp_raw for weather) where the extreme occurs, so "which day / slot was highest" needs no row scan.
    rt_price also counts negative and spike (>= spike threshold) slots per group.
    group_extremes ranks the groups: for each group statistic (mean / max / min, negative and spike slot
    counts) the group with the largest and the smallest value and how many groups tie, so "which day or month
    had the most" is read directly. Month groups carry first_date / last_date, the days of that month inside
    the requested range, for use as the range of a follow-up query.
    """
    table, column, group_expression, key_columns, date_expression = _stats_sources(
        request.metric, request.group_by
    )
    is_price = request.metric == "rt_price"
    spike_threshold = _spike_threshold() if is_price else None
    if is_price:
        aggregates = f"""
                   count(*) AS total_slots,
                   count(rt_price) AS observed_prices,
                   count(*) - count(rt_price) AS missing_prices,
                   count(*) FILTER (WHERE rt_price < 0) AS negative_price_slots,
                   count(*) FILTER (WHERE rt_price >= {float(spike_threshold)!r}) AS spike_price_slots,
                   avg(rt_price) AS mean_price,
                   median(rt_price) AS median_price,
                   stddev_samp(rt_price) AS price_stddev,
                   min(rt_price) AS min_price,
                   max(rt_price) AS max_price,
                   quantile_cont(rt_price, 0.05) AS p05_price,
                   quantile_cont(rt_price, 0.95) AS p95_price"""
    else:
        aggregates = f"""
                   count({column}) AS observed_values,
                   count(*) - count({column}) AS missing_values,
                   avg({column}) AS mean_value,
                   median({column}) AS median_value,
                   stddev_samp({column}) AS value_stddev,
                   min({column}) AS min_value,
                   max({column}) AS max_value,
                   quantile_cont({column}, 0.05) AS p05_value,
                   quantile_cont({column}, 0.95) AS p95_value"""
    bounds = [request.start_date, request.end_date]
    extreme_args = {
        "table": table,
        "column": column,
        "key_columns": key_columns,
        "date_expression": date_expression,
        "start": request.start_date,
        "end": request.end_date,
    }
    connection = _database()
    try:
        totals = _records(
            connection.execute(
                f"SELECT {aggregates} FROM {table} WHERE {date_expression} BETWEEN ? AND ?", bounds
            )
        )[0]
        grouped_rows = _records(
            connection.execute(
                f"""
                SELECT {group_expression} AS group_value, {aggregates}
                FROM {table}
                WHERE {date_expression} BETWEEN ? AND ?
                GROUP BY {group_expression}
                ORDER BY group_value
                """,
                bounds,
            )
        )
        total_extremes = _extreme_rows(connection, group_expression="1", **extreme_args).get(1, {})
        group_extremes = _extreme_rows(connection, group_expression=group_expression, **extreme_args)
    finally:
        connection.close()

    limitations = [
        MARKET_NOTE if table == "market_15min" else WEATHER_NOTE,
        "statistics exclude missing values",
        "max_at / min_at give the earliest source row when several rows share the extreme value",
    ]
    if is_price:
        # The price output keeps its established field names (yuan_per_mwh suffix on totals).
        renamed = {
            "mean_price": "mean_price_yuan_per_mwh",
            "median_price": "median_price_yuan_per_mwh",
            "price_stddev": "price_stddev_yuan_per_mwh",
            "min_price": "min_price_yuan_per_mwh",
            "max_price": "max_price_yuan_per_mwh",
            "p05_price": "p05_price_yuan_per_mwh",
            "p95_price": "p95_price_yuan_per_mwh",
        }
        totals = {renamed.get(name, name): value for name, value in totals.items()}
        totals["negative_price_share"] = (
            totals["negative_price_slots"] / totals["observed_prices"] if totals["observed_prices"] else None
        )
        totals["spike_threshold_yuan_per_mwh"] = spike_threshold
        limitations.append("floor-price counts are null because period-specific legal limits are unverified")
    else:
        totals["unit"] = STATS_METRICS[request.metric][2]
    totals.update(total_extremes)

    groups = []
    for group in grouped_rows:
        if is_price:
            observed = group["observed_prices"]
            group["negative_price_share"] = group["negative_price_slots"] / observed if observed else None
            group["verified_floor_price_count"] = None
            group["floor_price_count_status"] = "unavailable_unverified_policy_limits"
        group.update(group_extremes.get(group["group_value"], {}))
        if request.group_by == "slot":
            # A slot group is identified by its clock time too, so a row read from a compacted view shows it.
            group["time_label"] = _slot_time_label(group["group_value"])
        elif request.group_by == "month":
            # The part of the month inside the requested range (first and last month may be partial).
            first = date.fromisoformat(f"{group['group_value']}-01")
            last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            group["first_date"] = max(first, request.start_date).isoformat()
            group["last_date"] = min(last, request.end_date).isoformat()
        groups.append(group)

    data: dict[str, Any] = {
        "period": [request.start_date.isoformat(), request.end_date.isoformat()],
        "metric": request.metric,
        "group_by": request.group_by,
        **totals,
        "group_extremes": _group_extremes(groups),
        "groups": groups,
    }
    if is_price:
        data["verified_floor_price_count"] = None
        data["floor_price_count_status"] = "unavailable_unverified_policy_limits"
    return _tool_result(
        tool_name="query_stats",
        status="ok",
        source=f"sd_market.duckdb:{table}",
        data=data,
        limitations=limitations,
    )


def query_load(request: PriceQueryInput) -> dict[str, Any]:
    """Return actual and source-published load and supply forecast rows."""
    connection = _database()
    try:
        rows = _records(
            connection.execute(
                """
                SELECT trade_date, time_label, slot_id,
                       load_actual_mw, load_forecast_mw,
                       tie_line_actual_mw, tie_line_forecast_mw,
                       wind_actual_mw, wind_forecast_mw,
                       solar_actual_mw, solar_forecast_mw,
                       nuclear_actual_mw, nuclear_forecast_mw,
                       captive_actual_mw, captive_forecast_mw,
                       local_gen_actual_mw, local_gen_forecast_mw,
                       actual_bid_space_mw, forecast_bid_space_mw
                FROM market_15min
                WHERE trade_date BETWEEN ? AND ?
                  AND (? IS NULL OR slot_id = ?)
                ORDER BY trade_date, slot_id
                LIMIT ?
                """,
                [request.start_date, request.end_date, request.slot_id, request.slot_id, request.limit],
            )
        )
        total = connection.execute(
            """
            SELECT count(*) FROM market_15min
            WHERE trade_date BETWEEN ? AND ? AND (? IS NULL OR slot_id = ?)
            """,
            [request.start_date, request.end_date, request.slot_id, request.slot_id],
        ).fetchone()[0]
    finally:
        connection.close()
    return _tool_result(
        tool_name="query_load",
        status="ok",
        source="sd_market.duckdb:market_15min",
        data={
            "period": [request.start_date.isoformat(), request.end_date.isoformat()],
            "returned_rows": len(rows),
            "total_rows": total,
            "truncated": len(rows) < total,
            "rows": rows,
        },
        limitations=[
            MARKET_NOTE,
            "historical source forecasts have no verified issue or revision time",
            "bid-space component semantics are inherited from the source processing assumptions",
        ],
    )


def query_weather(request: WeatherQueryInput) -> dict[str, Any]:
    """Return Jinan archived weather while preserving its unresolved time basis."""
    connection = _database()
    try:
        rows = _records(
            connection.execute(
                """
                SELECT CAST(timestamp_raw AS VARCHAR) AS timestamp_raw,
                       temp_jinan, ghi_jinan,
                       wind_speed_jinan, precip_jinan, humidity_jinan, cloud_jinan
                FROM weather_actual_hourly
                WHERE CAST(timestamp_raw AS DATE) BETWEEN ? AND ?
                ORDER BY timestamp_raw
                LIMIT ?
                """,
                [request.start_date, request.end_date, request.limit],
            )
        )
        total = connection.execute(
            """
            SELECT count(*) FROM weather_actual_hourly
            WHERE CAST(timestamp_raw AS DATE) BETWEEN ? AND ?
            """,
            [request.start_date, request.end_date],
        ).fetchone()[0]
    finally:
        connection.close()
    return _tool_result(
        tool_name="query_weather",
        status="ok",
        source="sd_market.duckdb:weather_actual_hourly",
        data={
            "period_by_stored_timestamp_date": [
                request.start_date.isoformat(),
                request.end_date.isoformat(),
            ],
            "returned_rows": len(rows),
            "total_rows": total,
            "truncated": len(rows) < total,
            "rows": rows,
        },
        limitations=[WEATHER_NOTE, "Jinan only; weather is not a forecast input"],
    )


def _load_snapshot(mode: Literal["fixed", "rolling"], origin: pd.Timestamp):
    import joblib

    if mode == "fixed":
        directory = FIXED_SNAPSHOT_DIR
        tree_bundle_path = directory / "models" / "tree_models.joblib"
        patch_path = directory / "models" / "patchtst_2025-10-01"
        metadata_path = directory / "models" / "tree_models.json"
    else:
        run_metadata_path = ROLLING_SNAPSHOT_DIR / "rolling_test_metrics.json"
        if not run_metadata_path.is_file():
            raise ValueError(
                "ROLLING_SNAPSHOT_NOT_AVAILABLE: weekly rolling backtest has not completed"
            )
        report = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        available = [
            pd.Timestamp(item["forecast_origin"])
            for item in report["snapshots"]
            if pd.Timestamp(item["forecast_origin"]) <= origin
        ]
        if not available:
            raise ValueError("ROLLING_SNAPSHOT_NOT_AVAILABLE: no snapshot precedes this origin")
        selected = max(available)
        directory = ROLLING_SNAPSHOT_DIR / "models" / selected.date().isoformat()
        tree_bundle_path = directory / "trees" / "tree_models.joblib"
        patch_path = directory / "patchtst"
        metadata_path = directory / "trees" / "tree_models.json"

    if not tree_bundle_path.is_file() or not patch_path.is_dir() or not metadata_path.is_file():
        raise ValueError(f"MODEL_SNAPSHOT_NOT_AVAILABLE: required artifacts are missing in {directory}")
    bundle = joblib.load(tree_bundle_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trained_as_of = pd.Timestamp(metadata["trained_as_of"])
    if trained_as_of > origin:
        raise ValueError("MODEL_SNAPSHOT_AFTER_FORECAST_ORIGIN")
    return directory, bundle["models"], metadata, patch_path


_NEURAL_LOCK = threading.Lock()


@functools.lru_cache(maxsize=4)
def _load_patchtst(path: str, _mtime_ns: int):
    from neuralforecast import NeuralForecast

    return NeuralForecast.load(path=path)


def _model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-model extremes stated explicitly, so prompt compaction can never hide them."""
    summary: dict[str, Any] = {}
    for model in ("LightGBM", "XGBoost", "PatchTST", "ensemble_mean"):
        low = min(rows, key=lambda row: row[model])
        high = max(rows, key=lambda row: row[model])
        summary[model] = {
            "min_yuan_per_mwh": low[model],
            "min_time_label": low["time_label"],
            "max_yuan_per_mwh": high[model],
            "max_time_label": high["time_label"],
            "mean_yuan_per_mwh": round(sum(row[model] for row in rows) / len(rows), 3),
            "negative_slot_count": sum(1 for row in rows if row[model] < 0),
        }
    summary["any_model_negative"] = any(
        summary[model]["negative_slot_count"] > 0 for model in ("LightGBM", "XGBoost", "PatchTST")
    )
    return summary


def forecast_price(request: ForecastPriceInput) -> dict[str, Any]:
    """Return a 96-slot D+1 forecast from a compliant fixed or rolling snapshot."""
    if not TEST_START <= request.target_date <= TEST_END:
        raise ValueError(
            f"TARGET_OUTSIDE_SUPPORTED_PERIOD: {TEST_START.isoformat()}..{TEST_END.isoformat()}"
        )
    origin = pd.Timestamp(request.target_date).tz_localize(MARKET_TIMEZONE)
    as_of = pd.Timestamp(request.as_of)
    if as_of.tzinfo is None:
        raise ValueError("as_of 必须包含时区偏移量")
    as_of = as_of.tz_convert(MARKET_TIMEZONE)
    if origin > as_of:
        raise ValueError("FORECAST_ORIGIN_AFTER_AS_OF")

    from ellectric.pipeline.market.features import build_forecast_samples, build_patchtst_frame
    from ellectric.pipeline.market.forecasting import predict_tree_models
    from ellectric.pipeline.market.neural import predict_patchtst_targets

    market_path = DATA_DIR / "sd_rt_15min.parquet"
    if not market_path.is_file():
        raise FileNotFoundError(f"标准化市场 Parquet 不存在: {market_path}")
    snapshot_dir, tree_models, metadata, patch_path = _load_snapshot(request.mode, origin)
    market = pd.read_parquet(market_path)
    samples = build_forecast_samples(market)
    target_samples = samples.loc[
        pd.to_datetime(samples["trade_date"]).dt.normalize() == pd.Timestamp(request.target_date)
    ].copy()
    if len(target_samples) != 96:
        raise ValueError(f"TARGET_DAY_COVERAGE_ERROR: expected=96, actual={len(target_samples)}")
    tree_predictions = predict_tree_models(tree_models, target_samples)

    patch_frame, _ = build_patchtst_frame(market)
    # Torch inference already uses all cores; one model call at a time avoids thrashing.
    with _NEURAL_LOCK:
        patch_forecast = _load_patchtst(str(patch_path), patch_path.stat().st_mtime_ns)
        patch_predictions = predict_patchtst_targets(
            patch_forecast, patch_frame, [request.target_date]
        )
    predictions = tree_predictions.merge(
        patch_predictions.drop(columns="time_label"),
        on=["trade_date", "slot_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(predictions) != 96 or predictions.isna().any().any():
        raise ValueError("FORECAST_OUTPUT_INVALID: expected 96 complete model predictions")
    calibration = _calibration()
    high_threshold = (
        calibration["divergence"]["high_threshold_yuan_per_mwh"] if calibration else None
    )
    rows = []
    for row in predictions.itertuples(index=False):
        values = [float(row.LightGBM), float(row.XGBoost), float(row.PatchTST)]
        spread = float(pd.Series(values).std(ddof=0))
        rows.append(
            {
                "trade_date": row.trade_date.date().isoformat(),
                "time_label": row.time_label,
                "slot_id": int(row.slot_id),
                "LightGBM": round(values[0], 3),
                "XGBoost": round(values[1], 3),
                "PatchTST": round(values[2], 3),
                "ensemble_mean": round(sum(values) / 3, 3),
                "model_std": round(spread, 3),
                "model_range": round(max(values) - min(values), 3),
                "high_divergence": None if high_threshold is None else spread >= high_threshold,
            }
        )
    divergence = _divergence_summary(rows, calibration)
    return _tool_result(
        tool_name="forecast_price",
        status="ok",
        source=f"{request.mode}_snapshot:{snapshot_dir}",
        data={
            "target_date": request.target_date.isoformat(),
            "forecast_origin": origin.isoformat(),
            "as_of": as_of.isoformat(),
            "trained_as_of": metadata["trained_as_of"],
            "snapshot_mode": request.mode,
            "replayed_historical_snapshot": origin < pd.Timestamp.now(tz=MARKET_TIMEZONE),
            "prediction_count": len(rows),
            "model_summary": _model_summary(rows),
            "divergence": divergence,
            "predictions": rows,
        },
        limitations=[
            MARKET_NOTE,
            "historical backtest assumes midnight data availability and unrevised target-day forecasts",
            "LightGBM/XGBoost use target-day supply forecasts; PatchTST uses only historical price",
            "model disagreement is not a calibrated probability interval",
        ],
        assumptions=metadata["assumption_ids"],
    )


PRICE_LIMITS_PATH = DATA_DIR / "price_limits.json"


def _price_floors() -> list[dict[str, Any]] | None:
    """User-confirmed real-time clearing floors per period; None until a confirmed file exists."""
    if not PRICE_LIMITS_PATH.is_file():
        return None
    payload = _cached_json(str(PRICE_LIMITS_PATH), PRICE_LIMITS_PATH.stat().st_mtime_ns)
    if payload.get("verification_status") != "user_confirmed":
        return None
    return payload.get("periods", [])


def _floor_events(prices: pd.DataFrame) -> tuple[list[dict[str, Any]], str]:
    periods = _price_floors()
    if periods is None:
        return [], "FLOOR_PRICE_NOT_DETECTED: no period-specific legal price floor is verified"
    tolerance = 0.01
    floors = []
    for day in prices["trade_date"].astype(str):
        match = [p for p in periods if p["from"] <= day and (p.get("to") is None or day < p["to"])]
        floors.append(float(match[0]["rt_clearing_floor_yuan_per_mwh"]) if match else math.nan)
    frame = prices.assign(floor=floors)
    uncovered = sorted(set(frame.loc[frame["floor"].isna(), "trade_date"].astype(str)))
    price = pd.to_numeric(frame["rt_price"], errors="coerce")
    at_floor = frame.loc[(price - frame["floor"]).abs() <= tolerance].copy()
    events = []
    if not at_floor.empty:
        at_floor["run"] = at_floor["absolute_slot"].diff().ne(1).cumsum()
        for _, group in at_floor.groupby("run", sort=True):
            first, last = group.iloc[0], group.iloc[-1]
            events.append({
                "event_type": "floor_price",
                "start": {"trade_date": first["trade_date"], "time_label": first["time_label"], "slot_id": int(first["slot_id"])},
                "end": {"trade_date": last["trade_date"], "time_label": last["time_label"], "slot_id": int(last["slot_id"])},
                "points": len(group),
                "floor_yuan_per_mwh": float(first["floor"]),
                "threshold_status": "user_confirmed_clearing_floor",
            })
    note = "floor_price events use user-confirmed clearing floors (tolerance 0.01 yuan/MWh)"
    if uncovered:
        note += f"; no confirmed floor for {uncovered[0]}..{uncovered[-1]} ({len(uncovered)} days)"
    return events, note


CALIBRATION_PATH = FIXED_SNAPSHOT_DIR / "calibration_thresholds.json"
FIXED_TEST_PREDICTIONS_PATH = FIXED_SNAPSHOT_DIR / "fixed_test_predictions.parquet"


def _calibration() -> dict[str, Any] | None:
    """Validation-period thresholds from scripts/calibrate_market_thresholds.py, if present."""
    if not CALIBRATION_PATH.is_file():
        return None
    return _cached_json(str(CALIBRATION_PATH), CALIBRATION_PATH.stat().st_mtime_ns)


@functools.lru_cache(maxsize=4)
def _cached_json(path: str, _mtime_ns: int) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _divergence_summary(rows: list[dict[str, Any]], calibration: dict[str, Any] | None) -> dict[str, Any]:
    stds = [row["model_std"] for row in rows]
    widest = max(rows, key=lambda row: row["model_std"])
    summary: dict[str, Any] = {
        "statistic": "per-slot population std of LightGBM, XGBoost, PatchTST",
        "mean_model_std": round(sum(stds) / len(stds), 3),
        "max_model_std": widest["model_std"],
        "max_model_std_slot": {"slot_id": widest["slot_id"], "time_label": widest["time_label"]},
        "interpretation": "disagreement between models, not a probability interval",
    }
    if calibration is None:
        summary["status"] = "uncalibrated"
        return summary
    high = [row for row in rows if row["high_divergence"]]
    calibrated = calibration["divergence"]
    summary.update(
        {
            "status": "calibrated",
            "high_threshold_yuan_per_mwh": round(calibrated["high_threshold_yuan_per_mwh"], 3),
            "threshold_basis": f"validation p95, {calibration['calibration_period'][0]}..{calibration['calibration_period'][1]}",
            "high_divergence_slot_count": len(high),
            "high_divergence_slots": [
                {"slot_id": row["slot_id"], "time_label": row["time_label"], "model_std": row["model_std"]}
                for row in high
            ],
            "validation_ensemble_mae_high_divergence": round(
                calibrated["validation_ensemble_mae_high_divergence"], 3
            ),
            "validation_ensemble_mae_other": round(calibrated["validation_ensemble_mae_other"], 3),
        }
    )
    return summary


def _residual_events(prices: pd.DataFrame, calibration: dict[str, Any]) -> list[dict[str, Any]]:
    """Runs of slots where |actual - saved fixed-snapshot ensemble| reaches the validation p99."""
    forecasts = _cached_parquet(
        str(FIXED_TEST_PREDICTIONS_PATH), FIXED_TEST_PREDICTIONS_PATH.stat().st_mtime_ns
    )
    threshold = float(calibration["forecast_residual"]["threshold_yuan_per_mwh"])
    frame = prices.assign(trade_date_key=pd.to_datetime(prices["trade_date"])).merge(
        forecasts.assign(trade_date_key=pd.to_datetime(forecasts["trade_date"]))[
            ["trade_date_key", "slot_id", "ensemble_mean"]
        ],
        on=["trade_date_key", "slot_id"],
        how="inner",
    )
    frame["residual"] = pd.to_numeric(frame["rt_price"], errors="coerce") - frame["ensemble_mean"]
    flagged = frame.loc[frame["residual"].abs() >= threshold].copy()
    if flagged.empty:
        return []
    flagged["sign"] = np.sign(flagged["residual"])
    flagged["run"] = (
        flagged["absolute_slot"].diff().ne(1) | flagged["sign"].diff().ne(0)
    ).cumsum()
    events = []
    for _, group in flagged.groupby("run", sort=True):
        first, last = group.iloc[0], group.iloc[-1]
        peak = group.loc[group["residual"].abs().idxmax()]
        events.append(
            {
                "event_type": "forecast_residual_high" if first["sign"] > 0 else "forecast_residual_low",
                "start": {"trade_date": first["trade_date"], "time_label": first["time_label"], "slot_id": int(first["slot_id"])},
                "end": {"trade_date": last["trade_date"], "time_label": last["time_label"], "slot_id": int(last["slot_id"])},
                "points": len(group),
                "peak_residual_yuan_per_mwh": round(float(peak["residual"]), 3),
                "peak_actual_yuan_per_mwh": round(float(peak["rt_price"]), 3),
                "peak_forecast_yuan_per_mwh": round(float(peak["ensemble_mean"]), 3),
                "threshold_yuan_per_mwh": round(threshold, 3),
                "threshold_status": "validation_p99_abs_residual_of_fixed_snapshot_ensemble",
            }
        )
    return events


@functools.lru_cache(maxsize=2)
def _cached_parquet(path: str, _mtime_ns: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["ensemble_mean"] = frame[["LightGBM", "XGBoost", "PatchTST"]].mean(axis=1)
    return frame


def detect_anomaly(request: AnomalyInput) -> dict[str, Any]:
    """Identify observed price events and separate market data-quality events."""
    connection = _database()
    try:
        rows = _records(
            connection.execute(
                """
                SELECT trade_date, time_label, slot_id, rt_price,
                       load_actual_mw, load_forecast_mw,
                       tie_line_actual_mw, tie_line_forecast_mw,
                       wind_actual_mw, wind_forecast_mw,
                       solar_actual_clipped_mw, solar_forecast_clipped_mw,
                       nuclear_actual_mw, nuclear_forecast_mw,
                       captive_actual_mw, captive_forecast_mw,
                       local_gen_actual_mw, local_gen_forecast_mw,
                       actual_bid_space_mw, forecast_bid_space_mw
                FROM market_15min
                WHERE trade_date BETWEEN ? AND ?
                ORDER BY trade_date, slot_id
                """,
                [request.start_date, request.end_date],
            )
        )
    finally:
        connection.close()
    if not rows:
        empty_notes = [MARKET_NOTE, "no market rows in the requested range"]
        if "floor_price" in request.methods:
            empty_notes.append("FLOOR_PRICE_NOT_DETECTED: no period-specific legal price floor is verified")
        if "forecast_residual" in request.methods:
            empty_notes.append("FORECAST_RESIDUAL_NOT_DETECTED: no market rows to compare with saved forecasts")
        return _tool_result(
            tool_name="detect_anomaly",
            status="ok",
            source="sd_market.duckdb:market_15min",
            data={"events": [], "event_count": 0},
            limitations=empty_notes,
        )

    prices = pd.DataFrame(rows)
    prices["date"] = pd.to_datetime(prices["trade_date"])
    prices["absolute_slot"] = (prices["date"] - pd.Timestamp("1970-01-01")).dt.days * 96 + prices[
        "slot_id"
    ].astype("int64")
    valid_prices = pd.to_numeric(prices["rt_price"], errors="coerce")
    threshold = _spike_threshold()
    methods = set(request.methods)
    events: list[dict[str, Any]] = []

    if "negative_price" in methods:
        negative = prices.loc[valid_prices < 0].copy()
        if not negative.empty:
            negative["run"] = negative["absolute_slot"].diff().ne(1).cumsum()
            for _, group in negative.groupby("run", sort=True):
                if len(group) >= 2:
                    events.append(_negative_event(group, "negative_price_run"))
                elif request.include_single_negative_points:
                    events.append(_negative_event(group, "isolated_negative_price"))

    if "spike" in methods:
        spikes = prices.loc[valid_prices >= threshold]
        for row in spikes.itertuples(index=False):
            point = {
                "trade_date": row.trade_date,
                "time_label": row.time_label,
                "slot_id": int(row.slot_id),
            }
            events.append(
                {
                    "event_type": "high_price_candidate",
                    "start": point,
                    "end": point,
                    "points": 1,
                    "price_yuan_per_mwh": row.rt_price,
                    "threshold_yuan_per_mwh": threshold,
                    "threshold_status": "training_p99_candidate_not_legal_price_limit",
                }
            )

    if "jump" in methods:
        prior = prices["rt_price"].shift(1)
        adjacent = prices["absolute_slot"].diff().eq(1)
        jumps = prices.loc[
            adjacent
            & prices["rt_price"].notna()
            & prior.notna()
            & ((prices["rt_price"] - prior).abs() >= 300)
        ]
        for index, row in jumps.iterrows():
            point = {
                "trade_date": row["trade_date"],
                "time_label": row["time_label"],
                "slot_id": int(row["slot_id"]),
            }
            events.append(
                {
                    "event_type": "price_jump_candidate",
                    "start": point,
                    "end": point,
                    "previous_price_yuan_per_mwh": float(prior.loc[index]),
                    "current_price_yuan_per_mwh": float(row["rt_price"]),
                    "change_yuan_per_mwh": float(row["rt_price"] - prior.loc[index]),
                    "threshold_yuan_per_mwh": 300.0,
                    "threshold_status": "candidate_requires_validation",
                }
            )

    residual_note = None
    if "forecast_residual" in methods:
        calibration = _calibration()
        in_test = request.start_date <= TEST_END and request.end_date >= TEST_START
        if calibration is None or not FIXED_TEST_PREDICTIONS_PATH.is_file():
            residual_note = "FORECAST_RESIDUAL_NOT_DETECTED: calibration or saved forecasts are missing"
        elif not in_test:
            residual_note = (
                "FORECAST_RESIDUAL_NOT_DETECTED: saved compliant forecasts exist only for "
                f"{TEST_START.isoformat()}..{TEST_END.isoformat()}"
            )
        else:
            events.extend(_residual_events(prices, calibration))
            residual_note = (
                "forecast residuals use the saved fixed-snapshot ensemble (issued at each D 24:00); "
                "a residual flags a surprise relative to the model, not a market violation"
            )

    if "data_quality" in methods:
        required_columns = [
            "rt_price",
            "load_actual_mw",
            "load_forecast_mw",
            "tie_line_actual_mw",
            "tie_line_forecast_mw",
            "wind_actual_mw",
            "wind_forecast_mw",
            "solar_actual_clipped_mw",
            "solar_forecast_clipped_mw",
            "nuclear_actual_mw",
            "nuclear_forecast_mw",
            "captive_actual_mw",
            "captive_forecast_mw",
            "local_gen_actual_mw",
            "local_gen_forecast_mw",
            "actual_bid_space_mw",
            "forecast_bid_space_mw",
        ]
        quality_rows: list[dict[str, Any]] = []
        for row in prices.loc[prices[required_columns].isna().any(axis=1)].to_dict("records"):
            row["missing_fields"] = [column for column in required_columns if pd.isna(row[column])]
            quality_rows.append(row)

        grouped_quality: list[list[dict[str, Any]]] = []
        for row in quality_rows:
            if (
                not grouped_quality
                or int(row["absolute_slot"]) != int(grouped_quality[-1][-1]["absolute_slot"]) + 1
                or row["missing_fields"] != grouped_quality[-1][-1]["missing_fields"]
            ):
                grouped_quality.append([row])
            else:
                grouped_quality[-1].append(row)
        for group in grouped_quality:
            first, last = group[0], group[-1]
            events.append(
                {
                    "event_type": "data_quality_missing_values",
                    "start": {
                        "trade_date": first["trade_date"],
                        "time_label": first["time_label"],
                        "slot_id": int(first["slot_id"]),
                    },
                    "end": {
                        "trade_date": last["trade_date"],
                        "time_label": last["time_label"],
                        "slot_id": int(last["slot_id"]),
                    },
                    "points": len(group),
                    "missing_fields": first["missing_fields"],
                }
            )

        present = {
            (str(row.trade_date), int(row.slot_id)) for row in prices.itertuples(index=False)
        }
        for day_offset in range((request.end_date - request.start_date).days + 1):
            day = request.start_date + timedelta(days=day_offset)
            missing_slots = [slot for slot in range(96) if (day.isoformat(), slot) not in present]
            runs: list[list[int]] = []
            for slot in missing_slots:
                if not runs or slot != runs[-1][-1] + 1:
                    runs.append([slot])
                else:
                    runs[-1].append(slot)
            for run in runs:
                first_slot, last_slot = run[0], run[-1]
                events.append(
                    {
                        "event_type": "data_quality_missing_slots",
                        "start": {
                            "trade_date": day.isoformat(),
                            "time_label": _slot_time_label(first_slot),
                            "slot_id": first_slot,
                        },
                        "end": {
                            "trade_date": day.isoformat(),
                            "time_label": _slot_time_label(last_slot),
                            "slot_id": last_slot,
                        },
                        "points": len(run),
                        "missing_fields": ["market_15min_row"],
                    }
                )

    for event in events:
        identity = {
            "event_type": event["event_type"],
            "start": event["start"],
            "end": event["end"],
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        event["event_id"] = "evt_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    events.sort(
        key=lambda event: (
            event["start"]["trade_date"],
            event["start"]["slot_id"],
            event["event_type"],
        )
    )
    limitations = [
        MARKET_NOTE,
        "candidate thresholds are not validated legal price bounds",
        "negative-price and jump events describe observed patterns, not causation or violations",
    ]
    if "floor_price" in methods:
        floor_events, floor_note = _floor_events(prices)
        events.extend(floor_events)
        for event in floor_events:
            identity = json.dumps({"event_type": event["event_type"], "start": event["start"], "end": event["end"]},
                                  sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            event["event_id"] = "evt_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        events.sort(key=lambda event: (event["start"]["trade_date"], event["start"]["slot_id"], event["event_type"]))
        limitations.append(floor_note)
    if residual_note:
        limitations.append(residual_note)
    return _tool_result(
        tool_name="detect_anomaly",
        status="ok",
        source="sd_market.duckdb:market_15min",
        data={
            "period": [request.start_date.isoformat(), request.end_date.isoformat()],
            "event_count": len(events),
            "events": events,
        },
        limitations=limitations,
    )


def _negative_event(group: pd.DataFrame, event_type: str) -> dict[str, Any]:
    first, last = group.iloc[0], group.iloc[-1]
    return {
        "event_type": event_type,
        "start": {
            "trade_date": first["trade_date"],
            "time_label": first["time_label"],
            "slot_id": int(first["slot_id"]),
        },
        "end": {
            "trade_date": last["trade_date"],
            "time_label": last["time_label"],
            "slot_id": int(last["slot_id"]),
        },
        "points": len(group),
        "min_price_yuan_per_mwh": float(group["rt_price"].min()),
        "max_price_yuan_per_mwh": float(group["rt_price"].max()),
    }


def _spike_threshold() -> float:
    metadata_path = FIXED_SNAPSHOT_DIR / "models" / "tree_models.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return float(metadata["spike_threshold_yuan_per_mwh"])
    return 1000.0


@functools.lru_cache(maxsize=2)
def _driver_model_context(
    tree_bundle_path: str, _bundle_mtime_ns: int, market_path: str, _market_mtime_ns: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load feature samples and the fixed tree bundle once per file version, not once per event."""
    import joblib

    from ellectric.pipeline.market.features import build_forecast_samples

    samples = build_forecast_samples(pd.read_parquet(market_path))
    return samples, joblib.load(tree_bundle_path)


def _direction(value: float, positive: str, negative: str) -> str:
    return positive if value > 0 else negative if value < 0 else "equal"


def _explain_driver_point(trade_date: date, slot_id: int) -> dict[str, Any]:
    """Explain one representative event slot without using a future model snapshot."""
    connection = _database()
    try:
        columns = [
            "time_label",
            "is_holiday",
            "is_weekend",
            "rt_price",
            "actual_bid_space_mw",
            "forecast_bid_space_mw",
            "load_actual_mw",
            "load_forecast_mw",
            "tie_line_actual_mw",
            "tie_line_forecast_mw",
            "wind_actual_mw",
            "wind_forecast_mw",
            "solar_actual_clipped_mw",
            "solar_forecast_clipped_mw",
            "nuclear_actual_mw",
            "nuclear_forecast_mw",
            "captive_actual_mw",
            "captive_forecast_mw",
            "local_gen_actual_mw",
            "local_gen_forecast_mw",
        ]
        row = connection.execute(
            "SELECT " + ", ".join(columns) + " FROM market_15min WHERE trade_date = ? AND slot_id = ?",
            [trade_date, slot_id],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("MARKET_SLOT_NOT_FOUND")
    values = dict(zip(columns, row, strict=True))
    pairs = [
        ("load", "load_actual_mw", "load_forecast_mw", 1),
        ("tie_line", "tie_line_actual_mw", "tie_line_forecast_mw", -1),
        ("wind", "wind_actual_mw", "wind_forecast_mw", -1),
        ("solar", "solar_actual_clipped_mw", "solar_forecast_clipped_mw", -1),
        ("nuclear", "nuclear_actual_mw", "nuclear_forecast_mw", -1),
        ("captive", "captive_actual_mw", "captive_forecast_mw", -1),
        ("local_generation", "local_gen_actual_mw", "local_gen_forecast_mw", -1),
    ]
    contributions = []
    missing_terms = []
    for name, actual_name, forecast_name, sign in pairs:
        actual = values[actual_name]
        forecast = values[forecast_name]
        if actual is None or forecast is None:
            missing_terms.append(name)
            contribution = None
            delta = None
        else:
            delta = round(float(actual) - float(forecast), 3)
            contribution = round(sign * delta, 3)
        contributions.append(
            {
                "component": name,
                "actual_mw": _json_value(actual),
                "forecast_mw": _json_value(forecast),
                "actual_minus_forecast_mw": delta,
                # Explicit labels so a report never has to infer direction from a signed contribution.
                "actual_vs_forecast": None if delta is None else _direction(delta, "above", "below"),
                "actual_minus_forecast_bid_space_contribution_mw": contribution,
                "bid_space_effect": None
                if contribution is None
                else _direction(contribution, "increase", "decrease"),
            }
        )
    total_contribution = (
        None
        if missing_terms
        else round(sum(item["actual_minus_forecast_bid_space_contribution_mw"] for item in contributions), 3)
    )

    model_explanations: list[dict[str, Any]] = []
    tree_bundle_path = FIXED_SNAPSHOT_DIR / "models" / "tree_models.joblib"
    market_path = DATA_DIR / "sd_rt_15min.parquet"
    if tree_bundle_path.is_file() and market_path.is_file():
        from ellectric.pipeline.market.forecasting import explain_tree_prediction

        samples, bundle = _driver_model_context(
            str(tree_bundle_path),
            tree_bundle_path.stat().st_mtime_ns,
            str(market_path),
            market_path.stat().st_mtime_ns,
        )
        target = samples.loc[
            (pd.to_datetime(samples["trade_date"]).dt.date == trade_date)
            & (samples["slot_id"] == slot_id)
        ]
        metadata_path = FIXED_SNAPSHOT_DIR / "models" / "tree_models.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        point_as_of = pd.Timestamp(trade_date).tz_localize(MARKET_TIMEZONE)
        trained_as_of = pd.Timestamp(metadata["trained_as_of"]) if metadata.get("trained_as_of") else None
        if len(target) == 1 and trained_as_of is not None and trained_as_of <= point_as_of:
            for name in ("LightGBM", "XGBoost"):
                explanation = explain_tree_prediction(name, bundle["models"][name], target.iloc[[0]])
                explanation["trained_as_of"] = trained_as_of.isoformat()
                model_explanations.append(explanation)
    return {
        "trade_date": trade_date.isoformat(),
        "slot_id": slot_id,
        "time_label": values["time_label"],
        "rt_price_yuan_per_mwh": _json_value(values["rt_price"]),
        "actual_bid_space_mw": _json_value(values["actual_bid_space_mw"]),
        "forecast_bid_space_mw": _json_value(values["forecast_bid_space_mw"]),
        "supply_error_complete": not missing_terms,
        "missing_supply_terms": missing_terms,
        "supply_error_contributions": contributions,
        "reconstructed_actual_minus_forecast_bid_space_mw": total_contribution,
        "tree_model_explanations": model_explanations,
        "model_explanation_status": "available" if model_explanations else "unavailable_as_of_or_snapshot_missing",
        "calendar": {
            "is_holiday": _json_value(values["is_holiday"]),
            "is_weekend": _json_value(values["is_weekend"]),
        },
    }


def _representative_event_ids(events: list[dict[str, Any]], limit: int) -> list[str]:
    """Pick price events across event types in detection order; data-quality events are skipped."""
    by_type: dict[str, list[str]] = {}
    for event in events:
        if not event["event_type"].startswith("data_quality"):
            by_type.setdefault(event["event_type"], []).append(event["event_id"])
    selected: list[str] = []
    while len(selected) < limit and any(by_type.values()):
        for queue in by_type.values():
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
    return selected


def explain_drivers(request: DriverInput) -> dict[str, Any]:
    """Return evidence bundles for selected anomaly events in a date range."""
    anomaly_result = detect_anomaly(
        AnomalyInput(
            start_date=request.start_date,
            end_date=request.end_date,
            methods=["negative_price", "spike", "jump", "data_quality"],
        )
    )
    event_by_id = {event["event_id"]: event for event in anomaly_result["data"]["events"]}
    if request.event_ids is None:
        event_ids = _representative_event_ids(anomaly_result["data"]["events"], request.max_events)
        selection = "auto_round_robin_by_event_type"
    else:
        event_ids = list(request.event_ids)
        selection = "requested"
        unknown_ids = [event_id for event_id in event_ids if event_id not in event_by_id]
        if unknown_ids:
            raise ValueError(f"EVENT_IDS_NOT_FOUND_IN_RANGE: {unknown_ids}")

    # One shared candidate search per call (cached across calls): the reranker costs ~20 s on CPU.
    policy = search_policy(
        PolicySearchInput(query="山东电力市场实时价格", policy_at=request.start_date, as_of=request.as_of)
    )
    policy_reference = {
        "status": policy["status"],
        "policy_at": request.start_date.isoformat(),
        "candidate_count": len(policy["data"].get("evidence", [])),
        "shared_result_pointer": "/data/policy_evidence",
    }
    explanations = []
    for event_id in event_ids:
        event = event_by_id[event_id]
        start_date = date.fromisoformat(event["start"]["trade_date"])
        end_date = date.fromisoformat(event["end"]["trade_date"])
        slot_id = int(event["start"]["slot_id"])
        representative = None
        if event["event_type"] != "data_quality_missing_slots":
            representative = _explain_driver_point(start_date, slot_id)
        weather = query_weather(
            WeatherQueryInput(start_date=start_date, end_date=end_date, limit=500)
        )
        explanations.append(
            {
                "event_id": event_id,
                "event": event,
                "representative_market_point": representative,
                "weather_context": {
                    "period_by_stored_timestamp_date": weather["data"]["period_by_stored_timestamp_date"],
                    "rows": weather["data"]["rows"],
                    "truncated": weather["data"]["truncated"],
                    "time_alignment": "not_performed",
                },
                "policy_evidence": policy_reference,
            }
        )
    price_event_types = list(
        dict.fromkeys(
            event["event_type"]
            for event in anomaly_result["data"]["events"]
            if not event["event_type"].startswith("data_quality")
        )
    )
    explained_types = list(dict.fromkeys(item["event"]["event_type"] for item in explanations))
    event_type_coverage = {
        "detected_price_event_types": price_event_types,
        "explained_event_types": explained_types,
        "uncovered_price_event_types": [
            event_type for event_type in price_event_types if event_type not in explained_types
        ],
    }
    sampling_note = []
    if len(explanations) < len(event_by_id):
        sampling_note.append(
            "only representative events are explained (one per event type in rotation); "
            "other events of the same type are not attributed one by one"
        )
    return _tool_result(
        tool_name="explain_drivers",
        status="ok",
        source="sd_market.duckdb:market_15min; weather_actual_hourly; local policy inventory",
        data={
            "period": [request.start_date.isoformat(), request.end_date.isoformat()],
            "event_count": len(explanations),
            "detected_event_count": len(event_by_id),
            "event_selection": selection,
            "event_type_coverage": event_type_coverage,
            "events": explanations,
            "policy_evidence": policy,
        },
        limitations=[
            MARKET_NOTE,
            "supply-space contribution signs follow the frozen accounting formula; source business semantics remain unverified",
            "TreeSHAP explains model predictions, not the actual market outcome or the forecast residual",
            WEATHER_NOTE,
            "weather records are returned by stored timestamp date only; no market-slot alignment is asserted",
            "policy context remains unavailable until policy coverage and version metadata are verified",
            *sampling_note,
        ],
        assumptions=AVAILABILITY_ASSUMPTION_IDS,
    )


def search_policy(request: PolicySearchInput) -> dict[str, Any]:
    """Fail closed until document versions and effective periods are verified."""
    from ellectric.policy.retrieval import search_local_policy

    index_dir = DATA_DIR / "policy_index"
    if (index_dir / "policy_index_manifest.json").is_file():
        as_of_date = (
            pd.Timestamp(request.as_of).tz_convert(MARKET_TIMEZONE).date().isoformat()
            if request.as_of is not None
            else None
        )
        result = search_local_policy(
            query=request.query,
            policy_at=request.policy_at.isoformat(),
            index_dir=index_dir,
            as_of_date=as_of_date,
            effective_filter=POLICY_EFFECTIVE_FILTER.get(),
            metadata_path=POLICY_METADATA_PATH,
        )
        return _tool_result(
            tool_name="search_policy",
            status=result["status"],
            source="local policy attachments; BM25 + bge-m3 dense recall, RRF, bge-reranker-v2-m3",
            data=result,
            limitations=result.get("limitations", []),
        )

    if not POLICY_DOCUMENTS_PATH.is_file():
        return _tool_result(
            tool_name="search_policy",
            status="unavailable",
            source="local policy inventory",
            data={"query": request.query, "evidence": []},
            limitations=["POLICY_INVENTORY_NOT_AVAILABLE"],
        )
    inventory = json.loads(POLICY_DOCUMENTS_PATH.read_text(encoding="utf-8"))
    pending = any(document.get("metadata_pending") for document in inventory.get("documents", []))
    if not inventory.get("policy_coverage_verified") or pending:
        return _tool_result(
            tool_name="search_policy",
            status="unavailable",
            source="local policy inventory",
            data={"query": request.query, "policy_at": request.policy_at.isoformat(), "evidence": []},
            limitations=[
                "POLICY_COVERAGE_UNVERIFIED",
                "document status, publication/effective dates, and replacement relationships are unresolved",
            ],
        )
    return _tool_result(
        tool_name="search_policy",
        status="unavailable",
        source="local policy inventory",
        data={"query": request.query, "policy_at": request.policy_at.isoformat(), "evidence": []},
        limitations=["POLICY_RETRIEVAL_INDEX_NOT_BUILT"],
    )


TOOL_MODELS: dict[str, type[ToolInput]] = {
    "query_price": PriceQueryInput,
    "query_stats": StatsInput,
    "query_load": PriceQueryInput,
    "query_weather": WeatherQueryInput,
    "forecast_price": ForecastPriceInput,
    "detect_anomaly": AnomalyInput,
    "explain_drivers": DriverInput,
    "search_policy": PolicySearchInput,
}

TOOL_FUNCTIONS = {
    "query_price": query_price,
    "query_stats": query_stats,
    "query_load": query_load,
    "query_weather": query_weather,
    "forecast_price": forecast_price,
    "detect_anomaly": detect_anomaly,
    "explain_drivers": explain_drivers,
    "search_policy": search_policy,
}


# What the dataset is, which the schema alone cannot say (the database holds one province and one market).
DATASET_SCOPE = {
    "region": "山东省全省（一个价区）。没有其他省份的数据，也没有分地市、分节点的数据",
    "price": "只有实时现货出清价 rt_price（元/MWh）。没有日前价格、中长期合同价、零售或分时电价",
    "slot": "每个交易日 96 个 15 分钟时段，每个时段用结束时刻标记（time_label，00:15 到 24:00）："
            "07:00 指 06:45-07:00 这个时段。按时刻取数时把 time_label 传给 query_price / query_load",
    "weather": "只有济南一个站点的逐小时实况，没有其他城市",
    "forecast": "forecast_price 只做 D+1：目标日必须是 as_of 时最后一个可得交易日的下一天，更远的日期不能预测",
    # Listing documents here would show titles published after as_of; retrieval applies the date filters.
    "policy": "本地山东电力市场政策文件，由 search_policy 按 policy_at 检索；有没有适用条款以检索结果为准",
}
_CATALOG_KEY_COLUMNS = {"trade_date", "slot_id", "time_label", "label_at_local", "label_at_utc", "timestamp_raw"}


@functools.lru_cache(maxsize=4)
def _catalog_coverage(_path: str, _mtime_ns: int) -> dict[str, Any]:
    with _database() as connection:
        tables = {}
        for table, column in (("market_15min", "trade_date"), ("weather_actual_hourly", "timestamp_raw")):
            columns = [row[0] for row in connection.execute(f'DESCRIBE "{table}"').fetchall()]
            first, last = connection.execute(
                f'SELECT CAST(MIN("{column}") AS DATE), CAST(MAX("{column}") AS DATE) FROM "{table}"'
            ).fetchone()
            tables[table] = {
                "columns": [name for name in columns if name not in _CATALOG_KEY_COLUMNS],
                "first_date": first,
                "last_date": last,
            }
    return tables


def data_catalog(as_of: datetime) -> dict[str, Any]:
    """What the tools can answer at ``as_of``: dataset scope plus table columns and date coverage read from
    the database, with the coverage end clipped to the last trade date available at ``as_of``."""
    latest = latest_available_trade_date(as_of)
    coverage = _catalog_coverage(str(DATABASE_PATH), DATABASE_PATH.stat().st_mtime_ns)
    tables = {
        table: {
            "columns": info["columns"],
            "first_date": info["first_date"].isoformat(),
            "last_available_date": min(info["last_date"], latest).isoformat(),
        }
        for table, info in coverage.items()
    }
    forecast_target = latest + timedelta(days=1)
    return {
        **DATASET_SCOPE,
        "tables": tables,
        "forecast_target_date": forecast_target.isoformat() if TEST_START <= forecast_target <= TEST_END else None,
        "forecast_supported_period": [TEST_START.isoformat(), TEST_END.isoformat()],
    }


def invoke_market_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"UNKNOWN_MARKET_TOOL: {name}")
    model = TOOL_MODELS[name].model_validate(arguments)
    return TOOL_FUNCTIONS[name](model)
