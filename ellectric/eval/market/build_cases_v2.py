"""Generate stratified evaluation sets (evaluation plan v2, docs/评测方案v2.md).

Two crossed dimensions:

- complexity ``stratum``: L1 single call; L2 independent calls; L3 a later call's argument (a day, a slot)
  comes out of an earlier result; L4 five or more steps (a period longer than one query allows, ranking,
  drill-down, attribution); L5 an L3/L4 case with an injected transient tool failure;
- ``answerability``: A answerable; B partly answerable; C unanswerable; D ambiguous wording that needs a
  stated interpretation; E policy version boundaries.

Templates are filled with random parameters from a split's own date windows, so dev and test never share a
day: dev uses 2024Q1, 2024Q3, 2025Q1, 2025Q3 (forecasts: January 2026), test uses 2024Q2, 2024Q4, 2025Q2,
2025Q4 (forecasts: October-December 2025); policy topics are split as well. The test split is meant to be
generated only after the agent code is frozen, with a seed not used during development.

References come from direct SQL, snapshot prediction files and located policy clauses, never from the agent's
tools. Numbers record the row keys (date, slot) and result field they come from where that is meaningful, so
scoring can check provenance; values the agent has to compute are marked ``evidence: false`` and their inputs
listed under ``evidence_values``. Each case carries a ``reference_solution`` (tool calls that reach the
answer); ``validate_cases_v2.py`` runs them to confirm every case is solvable and gradable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Callable

from ellectric.eval.market.build_cases import (
    GRID_2023, GRID_2025, MODELS, OUT, POLICY_TOPICS, RULES_2024, RULES_2025, SPIKE_THRESHOLD, PolicyClauses, cn, d,
    number,
)
from ellectric.eval.market.build_holdout_cases import (
    DATA_TOOLS, EXPLAIN_TOOLS, FORECAST_TOOLS, POLICY_ONLY, PRICE_FACT, STATS, UNAVAILABLE, HoldoutRefs, at, day_names,
    mentions, spec, win,
)
from ellectric.policy.retrieval import document_applicability

SPLITS = {
    "dev": {"quarters": [(2024, 1), (2024, 3), (2025, 1), (2025, 3)], "forecast": ("2026-01-02", "2026-01-14"),
            "policy_topics": ["rules_effective", "spot_price_limits", "price_control", "grid_rules"]},
    "test": {"quarters": [(2024, 2), (2024, 4), (2025, 2), (2025, 4)], "forecast": ("2025-10-02", "2025-12-31"),
             "policy_topics": ["secondary_cap_trigger", "mlt_price_limits", "ancillary_rules", "rules_effective"]},
}
# Any tool that reads the table can supply a value: query_stats aggregates market_15min and the weather table
# (with the source row of each extreme), so it is an alternative to the detail tools. Whether the right value
# was obtained is decided by the keyed number check, not by which tool returned it.
LOAD_FACT = ["query_load", "query_stats"]
WEATHER_FACT = ["query_weather", "query_stats"]
LOAD_FIELDS = {"load_actual_mw": "全网实际负荷", "wind_actual_mw": "风电实际出力", "solar_actual_mw": "光伏实际出力",
               "tie_line_actual_mw": "联络线实际受电"}
OTHER_PROVINCES = ["山西", "江苏", "广东", "甘肃", "浙江"]


class Skip(Exception):
    """Parameters gave no unique reference (ties, missing data); draw again."""


def keyed(name: str, value: float, tolerance: float, field: str, day: str | None = None, slot: int | None = None,
          **extra) -> dict:
    keys = {k: v for k, v in (("date", day), ("slot", slot)) if v is not None}
    return number(name, [value], tolerance) | {"field": field, "keys": keys} | extra


def derived(name: str, values: list[float], tolerance: float) -> dict:
    return number(name, values, tolerance) | {"evidence": False}


def prev_label(label: str) -> str:
    h, m = map(int, label.split(":"))
    total = h * 60 + m - 15
    return f"{total // 60:02d}:{total % 60:02d}"


def months_in(start: date, end: date) -> list[tuple[date, date]]:
    out, cursor = [], start
    while cursor <= end:
        nxt = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((cursor, min(end, nxt - timedelta(days=1))))
        cursor = nxt
    return out


class Refs2(HoldoutRefs):
    def row(self, day: str, slot: int) -> dict:
        frame = self.con.execute("SELECT * FROM market_15min WHERE trade_date = ? AND slot_id = ?", [d(day), slot]).fetchdf()
        if len(frame) != 1:
            raise Skip(f"no slot {slot} on {day}")
        return frame.iloc[0].to_dict()

    def daily_unique(self, start: str, end: str, expression: str, how: str) -> dict:
        try:
            return self.daily(start, end, expression, how)
        except ValueError as error:
            raise Skip(str(error)) from error

    def extreme(self, day: str, column: str, how: str) -> dict:
        try:
            return self.column_extreme(day, column, how)
        except ValueError as error:
            raise Skip(str(error)) from error

    def point_extreme(self, start: str, end: str, how: str) -> dict:
        order = "DESC" if how == "max" else "ASC"
        rows = self.con.execute(
            f"SELECT trade_date, slot_id, time_label, rt_price FROM market_15min WHERE trade_date BETWEEN ? AND ?"
            f" AND rt_price IS NOT NULL ORDER BY rt_price {order} LIMIT 2", [d(start), d(end)]).fetchall()
        if rows[0][3] == rows[1][3]:
            raise Skip("price extreme tied")
        day, slot, label, price = rows[0]
        return {"trade_date": day.isoformat(), "slot_id": int(slot), "time_label": label, "value": float(price)}

    def weather(self, start: str, end: str, column: str, how: str) -> dict:
        try:
            return self.weather_extreme(start, end, column, how)
        except (ValueError, IndexError) as error:
            raise Skip(str(error)) from error

    def spikes(self, start: str, end: str) -> dict:
        try:
            return self.spike_day(start, end)
        except ValueError as error:
            raise Skip(str(error)) from error

    def stats(self, start: str, end: str) -> dict:
        out = self.range_stats(d(start), d(end))
        if out["valid"] == 0:
            raise Skip("no prices")
        return out


class Generator:
    def __init__(self, split: str, seed: int) -> None:
        self.split, self.cfg, self.rng = split, SPLITS[split], random.Random(f"{split}:{seed}")
        self.r, self.clauses = Refs2(), PolicyClauses()
        self.days = [d0 for y, q in self.cfg["quarters"] for d0 in self._quarter_days(y, q)]
        self.day_set = set(self.days)
        self.used: set[str] = set()

    # ------------------------------------------------------------------ parameter draws
    @staticmethod
    def _quarter_days(year: int, quarter: int) -> list[date]:
        start = date(year, 3 * quarter - 2, 1)
        end = date(year + (quarter == 4), 1 if quarter == 4 else 3 * quarter + 1, 1) - timedelta(days=1)
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]

    def day(self) -> str:
        return self.rng.choice(self.days).isoformat()

    def quarter(self) -> tuple[date, date]:
        year, q = self.rng.choice(self.cfg["quarters"])
        days = self._quarter_days(year, q)
        return days[0], days[-1]

    def month(self) -> tuple[str, str]:
        first = self.rng.choice([x for x in self.days if x.day == 1])
        last = (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        return first.isoformat(), last.isoformat()

    def span(self, low: int, high: int) -> tuple[str, str]:
        """A run of ``low``..``high`` consecutive days inside one of the split's quarters."""
        q_start, q_end = self.quarter()
        length = self.rng.randint(low, min(high, (q_end - q_start).days + 1))
        start = q_start + timedelta(days=self.rng.randint(0, (q_end - q_start).days + 1 - length))
        return start.isoformat(), (start + timedelta(days=length - 1)).isoformat()

    def phrase(self, options: list[str]) -> tuple[str, int]:
        index = self.rng.randrange(len(options))
        return options[index], index

    @staticmethod
    def period(start: str, end: str) -> str:
        return cn(d(start)) if start == end else f"{cn(d(start))}至{cn(d(end))}"

    @staticmethod
    def as_of_after(end: str, days: int = 1, clock: str = "09:00") -> str:
        return at((d(end) + timedelta(days=days)).isoformat(), clock)

    # ------------------------------------------------------------------ case shell
    def case(self, template: str, stratum: str, answerability: str, question: str, phrasing: int, as_of: str, *,
             allowed: list[str], windows: list[dict], required: list[dict] | None = None,
             numbers: list[dict] | None = None, key_points: list[dict] | None = None,
             solution: list[dict] | None = None, mode: str = "historical_analysis", **extra) -> dict:
        numbers = numbers or []
        key_points = list(key_points or []) + [{"id": f"reports_{n['name']}", "type": "number", "number": n["name"]}
                                               for n in numbers]
        return {"template": template, "stratum": stratum, "category": stratum, "answerability": answerability,
                "question": question, "phrasing": phrasing, "mode": mode, "as_of": as_of, "allowed_tools": allowed,
                "arg_windows": windows, "required_calls": required or [], "required_numbers": numbers,
                "key_points": key_points, "needs_policy": False, "required_policy": [], "task_case": True,
                "reference_solution": solution or [], **extra}

    @staticmethod
    def call(tool: str, **args) -> dict:
        return {"tool": tool, "args": args}

    # ================================================================== L1: one call
    def l1_day_stat(self) -> dict:
        day = self.day()
        stat, word = self.rng.choice([("mean", "日均价"), ("max", "最高价"), ("min", "最低价")])
        s = self.r.stats(day, day)
        q, p = self.phrase([f"{cn(d(day))}山东实时电价的{word}是多少？",
                            f"{d(day).month}月{d(day).day}号（{d(day).year}年）山东现货实时价格{word}多少？"])
        return self.case("l1_day_stat", "L1", "A", q, p, self.as_of_after(day), allowed=DATA_TOOLS,
                         windows=[win(day, day)], required=[spec(STATS, cover=(day, day))],
                         numbers=[number(stat, [s[stat]])], solution=[self.call("query_stats", start_date=day, end_date=day)])

    def l1_slot_price(self) -> dict:
        day, slot = self.day(), self.rng.randrange(96)
        row = self.r.row(day, slot)
        if row["rt_price"] != row["rt_price"] or row["rt_price"] is None:
            raise Skip("missing price")
        label = row["time_label"]
        q, p = self.phrase([f"{cn(d(day))}{prev_label(label)}–{label}这个15分钟时段（时段标签 {label}）的山东实时电价是多少？",
                            f"{cn(d(day))}时段标签为 {label}（即 {prev_label(label)} 到 {label}）的那一刻钟，山东实时电价多少？"])
        return self.case("l1_slot_price", "L1", "A", q, p, self.as_of_after(day), allowed=DATA_TOOLS,
                         windows=[win(day, day)], required=[spec(["query_price"], cover=(day, day))],
                         numbers=[keyed("price", float(row["rt_price"]), 0.01, "rt_price", day, slot)],
                         solution=[self.call("query_price", start_date=day, end_date=day, slot_id=slot)])

    def l1_load_extreme(self) -> dict:
        day = self.day()
        field, label = self.rng.choice(list(LOAD_FIELDS.items()))
        how, word = self.rng.choice([("max", "最大"), ("min", "最小")])
        x = self.r.extreme(day, field, how)
        ask_time = self.rng.random() < 0.5
        q, p = self.phrase([f"{cn(d(day))}山东{label}的{word}值是多少兆瓦？" + ("出现在哪个时段？" if ask_time else ""),
                            f"{cn(d(day))}那天山东{label}{word}到多少 MW？" + ("大概几点？" if ask_time else "")])
        return self.case("l1_load_extreme", "L1", "A", q, p, self.as_of_after(day), allowed=DATA_TOOLS,
                         windows=[win(day, day)], required=[spec(LOAD_FACT, cover=(day, day))],
                         numbers=[keyed(f"{field}_{how}", x["value"], 0.5, field, day)],
                         key_points=[mentions(x["time_label"])] if ask_time else [],
                         solution=[self.call("query_load", start_date=day, end_date=day)])

    def l1_temperature(self) -> dict:
        day = self.day()
        how, word = self.rng.choice([("max", "最高"), ("min", "最低")])
        w = self.r.weather(day, day, "temp_jinan", how)
        q, p = self.phrase([f"{cn(d(day))}济南的{word}气温是多少摄氏度？", f"{cn(d(day))}济南{word}温度几度？"])
        return self.case("l1_temperature", "L1", "A", q, p, self.as_of_after(day), allowed=DATA_TOOLS,
                         windows=[win(day, day)], required=[spec(WEATHER_FACT, cover=(day, day))],
                         numbers=[number(f"temp_{how}", [w["value"]], 0.05) | {"field": "temp_jinan"}],
                         solution=[self.call("query_weather", start_date=day, end_date=day)])

    # ================================================================== L2: independent calls
    def l2_month_means(self) -> dict:
        q_start, q_end = self.quarter()
        months = months_in(q_start, q_end)
        values = [self.r.stats(a.isoformat(), b.isoformat())["mean"] for a, b in months]
        if len(set(round(v, 2) for v in values)) < len(values):
            raise Skip("tied months")
        best = months[max(range(len(values)), key=values.__getitem__)][0]
        names = "、".join(f"{a.month}月" for a, _ in months)
        q, p = self.phrase([f"{q_start.year}年{names}山东实时电价的月均价分别是多少？哪个月最高？",
                            f"{q_start.year}年第{(q_start.month + 2) // 3}季度每个月山东实时电价平均多少？哪个月最贵？"])
        return self.case("l2_month_means", "L2", "A", q, p, self.as_of_after(q_end.isoformat()), allowed=DATA_TOOLS,
                         windows=[win(q_start.isoformat(), q_end.isoformat())],
                         required=[spec(STATS, cover=(a.isoformat(), b.isoformat())) for a, b in months],
                         numbers=[number(f"mean_m{a.month}", [v]) for (a, _), v in zip(months, values)],
                         key_points=[mentions(f"{best.month}月", f"{best.year}-{best.month:02d}")],
                         solution=[self.call("query_stats", start_date=a.isoformat(), end_date=b.isoformat(), group_by="month")
                                   for a, b in months])

    def l2_yoy_window(self) -> dict:
        start, end = self.span(7, 14)
        if d(start).year != 2025:
            raise Skip("need a 2025 window")
        last = (d(start).replace(year=2024).isoformat(), d(end).replace(year=2024).isoformat())
        if (d(start).month, d(start).day) == (2, 29):
            raise Skip("leap day")
        now, before = self.r.stats(start, end)["mean"], self.r.stats(*last)["mean"]
        q, p = self.phrase([f"{self.period(start, end)}山东实时电价均价是多少？和去年同期（{self.period(*last)}）相比变化了多少？",
                            f"{self.period(start, end)}这段时间山东实时电价平均多少？比去年同一时间段涨了还是跌了，差多少？"])
        return self.case("l2_yoy_window", "L2", "A", q, p, self.as_of_after(end), allowed=DATA_TOOLS,
                         windows=[win(start, end), win(*last)],
                         required=[spec(STATS, cover=(start, end)), spec(STATS, cover=last)],
                         numbers=[number("mean_now", [now]), number("mean_last_year", [before]),
                                  derived("change", [now - before, before - now], 0.02)],
                         solution=[self.call("query_stats", start_date=start, end_date=end),
                                   self.call("query_stats", start_date=last[0], end_date=last[1])])

    def l2_price_weather(self) -> dict:
        day = self.day()
        s, w = self.r.stats(day, day), self.r.weather(day, day, "temp_jinan", "min")
        q, p = self.phrase([f"{cn(d(day))}济南最低气温是多少？当天山东实时电价最高到多少？",
                            f"{cn(d(day))}济南最冷的时候几度？那天山东实时电价最高价多少？"])
        return self.case("l2_price_weather", "L2", "A", q, p, self.as_of_after(day), allowed=DATA_TOOLS,
                         windows=[win(day, day)],
                         required=[spec(WEATHER_FACT, cover=(day, day)), spec(STATS, cover=(day, day))],
                         numbers=[number("temp_min", [w["value"]], 0.05) | {"field": "temp_jinan"}, number("max", [s["max"]])],
                         solution=[self.call("query_weather", start_date=day, end_date=day),
                                   self.call("query_stats", start_date=day, end_date=day)])

    def l2_two_days_negative(self) -> dict:
        a, b = sorted(self.rng.sample(self.days, 2))
        a, b = a.isoformat(), b.isoformat()
        na, nb = self.r.stats(a, a)["negative"], self.r.stats(b, b)["negative"]
        if na == nb:
            raise Skip("same count")
        more = a if na > nb else b
        q, p = self.phrase([f"{cn(d(a))}和{cn(d(b))}，山东实时电价各有多少个负电价时段？哪天更多？",
                            f"对比一下{cn(d(a))}、{cn(d(b))}两天山东实时电价的负电价时段数，哪天多？"])
        return self.case("l2_two_days_negative", "L2", "A", q, p, self.as_of_after(b), allowed=DATA_TOOLS,
                         windows=[win(a, a), win(b, b)],
                         required=[spec(PRICE_FACT, cover=(a, a)), spec(PRICE_FACT, cover=(b, b))],
                         numbers=[number("negative_a", [na], 0.5), number("negative_b", [nb], 0.5)],
                         key_points=[mentions(*day_names(more))],
                         solution=[self.call("query_stats", start_date=a, end_date=a),
                                   self.call("query_stats", start_date=b, end_date=b)])

    # ================================================================== L3: dependent hops
    def _l3_window(self) -> tuple[str, str]:
        return self.month() if self.rng.random() < 0.5 else self.span(10, 31)

    def l3_peakday_load(self) -> dict:
        start, end = self._l3_window()
        top = self.r.daily_unique(start, end, "avg(rt_price)", "max")
        load = self.r.extreme(top["trade_date"], "load_actual_mw", "max")
        day = top["trade_date"]
        q, p = self.phrase([f"{self.period(start, end)}，日均实时电价最高的是哪一天？那天山东全网实际负荷最高是多少？",
                            f"{self.period(start, end)}里哪天山东电价（日均）最贵？那天负荷峰值多少兆瓦？"])
        return self.case("l3_peakday_load", "L3", "A", q, p, self.as_of_after(end), allowed=DATA_TOOLS,
                         windows=[win(start, end)],
                         required=[spec(STATS, cover=(start, end)), spec(LOAD_FACT, cover=(day, day))],
                         numbers=[keyed("daily_mean", top["value"], 0.01, "", day),
                                  keyed("load_max", load["value"], 0.5, "load_actual_mw", day)],
                         key_points=[mentions(*day_names(day))], fault_tool="query_load",
                         solution=[self.call("query_stats", start_date=start, end_date=end, group_by="day"),
                                   self.call("query_load", start_date=day, end_date=day)])

    def l3_negday_component(self) -> dict:
        start, end = self._l3_window()
        top = self.r.daily_unique(start, end, "count(*) FILTER (WHERE rt_price < 0)", "max")
        if top["value"] == 0:
            raise Skip("no negative prices")
        day = top["trade_date"]
        field, label = self.rng.choice([("solar_actual_mw", "光伏实际出力"), ("wind_actual_mw", "风电实际出力")])
        x = self.r.extreme(day, field, "max")
        q, p = self.phrase([f"{self.period(start, end)}，负电价时段最多的是哪一天？有多少个？当天{label}最大是多少？",
                            f"{self.period(start, end)}哪天负电价出现得最多（几个时段）？那天{label}峰值多少兆瓦？"])
        return self.case("l3_negday_component", "L3", "A", q, p, self.as_of_after(end), allowed=DATA_TOOLS,
                         windows=[win(start, end)],
                         required=[spec(PRICE_FACT, cover=(start, end)), spec(LOAD_FACT, cover=(day, day))],
                         numbers=[keyed("negative_count", top["value"], 0.5, "", day),
                                  keyed(f"{field}_max", x["value"], 0.5, field, day)],
                         key_points=[mentions(*day_names(day))], fault_tool="query_load",
                         solution=[self.call("query_stats", start_date=start, end_date=end, group_by="day"),
                                   self.call("query_load", start_date=day, end_date=day)])

    def l3_minpoint_renewables(self) -> dict:
        start, end = self.span(7, 31)
        point = self.r.point_extreme(start, end, "min")
        row = self.r.row(point["trade_date"], point["slot_id"])
        day, slot = point["trade_date"], point["slot_id"]
        q, p = self.phrase([f"{self.period(start, end)}山东实时电价的最低点出现在哪天哪个时段？当时风电和光伏实际出力合计是多少？",
                            f"{self.period(start, end)}电价最低的那一刻是什么时候？那一刻风光出力加起来多少 MW？"])
        return self.case("l3_minpoint_renewables", "L3", "A", q, p, self.as_of_after(end), allowed=DATA_TOOLS,
                         windows=[win(start, end)],
                         required=[spec(PRICE_FACT, cover=(start, end)), spec(LOAD_FACT, cover=(day, day))],
                         numbers=[number("min_price", [point["value"]]),
                                  derived("wind_plus_solar", [row["wind_actual_mw"] + row["solar_actual_mw"]], 1.0)],
                         evidence_values=[keyed("wind", row["wind_actual_mw"], 0.5, "wind_actual_mw", day, slot),
                                          keyed("solar", row["solar_actual_mw"], 0.5, "solar_actual_mw", day, slot)],
                         key_points=[mentions(*day_names(day)), mentions(point["time_label"])], fault_tool="query_load",
                         solution=[self.call("query_price", start_date=start, end_date=end, limit=3000),
                                   self.call("query_load", start_date=day, end_date=day)])

    def l3_peakload_price(self) -> dict:
        start, end = self.span(5, 10)
        rows = self.r.con.execute(
            "SELECT trade_date, slot_id, time_label, load_actual_mw FROM market_15min WHERE trade_date BETWEEN ? AND ?"
            " ORDER BY load_actual_mw DESC LIMIT 2", [d(start), d(end)]).fetchall()
        if rows[0][3] == rows[1][3]:
            raise Skip("load tie")
        day, slot, label, load = rows[0][0].isoformat(), int(rows[0][1]), rows[0][2], float(rows[0][3])
        price = self.r.row(day, slot)["rt_price"]
        if price is None or price != price:
            raise Skip("missing price")
        q, p = self.phrase([f"{self.period(start, end)}山东全网实际负荷最高出现在哪天哪个时段？负荷多少？那个时段的实时电价是多少？",
                            f"{self.period(start, end)}负荷最高峰是什么时候、多少兆瓦？当时实时电价多少？"])
        return self.case("l3_peakload_price", "L3", "A", q, p, self.as_of_after(end), allowed=DATA_TOOLS,
                         windows=[win(start, end)],
                         required=[spec(LOAD_FACT, cover=(start, end)), spec(STATS, cover=(day, day))],
                         numbers=[keyed("load_max", load, 0.5, "load_actual_mw", day, slot),
                                  keyed("price_at_peak", float(price), 0.01, "rt_price", day, slot)],
                         key_points=[mentions(*day_names(day)), mentions(label)], fault_tool="query_price",
                         solution=[self.call("query_load", start_date=start, end_date=end, limit=1000),
                                   self.call("query_price", start_date=day, end_date=day, slot_id=slot)])

    def l3_hottest_day_price(self) -> dict:
        start, end = self.month()
        how, word = (("max", "最热") if d(start).month in (6, 7, 8, 9) else ("min", "最冷"))
        w = self.r.weather(start, end, "temp_jinan", how)
        s = self.r.stats(w["day"], w["day"])
        q, p = self.phrase([f"{self.period(start, end)}济南{word}的是哪一天（按小时气温的{('最高' if how == 'max' else '最低')}值）？那天山东实时电价日均价是多少？",
                            f"{self.period(start, end)}济南哪天{word}？那天山东电价平均多少？"])
        return self.case("l3_hottest_day_price", "L3", "A", q, p, self.as_of_after(end), allowed=DATA_TOOLS,
                         windows=[win(start, end)],
                         required=[spec(WEATHER_FACT, cover=(start, end)), spec(STATS, cover=(w["day"], w["day"]))],
                         numbers=[keyed("daily_mean", s["mean"], 0.01, "", w["day"])],
                         evidence_values=[number("temp", [w["value"]], 0.05) | {"field": "temp_jinan"}],
                         key_points=[mentions(*day_names(w["day"]))], fault_tool="query_stats",
                         solution=[self.call("query_weather", start_date=start, end_date=end, limit=1000),
                                   self.call("query_stats", start_date=w["day"], end_date=w["day"])])

    # ================================================================== L4: long chains
    def _l4_span(self) -> tuple[str, str]:
        return self.span(40, 92)

    def l4_negative_drilldown(self) -> dict:
        start, end = self._l4_span()
        segments = months_in(d(start), d(end))
        counts = [self.r.stats(a.isoformat(), b.isoformat())["negative"] for a, b in segments]
        if len(set(counts)) < len(counts):
            raise Skip("tied segments")
        best = segments[max(range(len(counts)), key=counts.__getitem__)]
        top = self.r.daily_unique(best[0].isoformat(), best[1].isoformat(), "count(*) FILTER (WHERE rt_price < 0)", "max")
        day = top["trade_date"]
        q, p = self.phrase([
            f"{self.period(start, end)}期间，按自然月分段（首尾月只算区间内的天数）统计山东实时电价的负电价时段数；"
            f"负电价最多的那个月里，哪一天负电价时段最多、有多少个？请分析那天负电价的可能原因，并说明分析的局限。",
            f"帮我看看{self.period(start, end)}的负电价：每个月（区间内部分）各有多少个负电价时段？最多的那个月里哪天最严重、几个时段？那天为什么会出现负电价？"])
        return self.case("l4_negative_drilldown", "L4", "A", q, p, self.as_of_after(end), allowed=EXPLAIN_TOOLS,
                         windows=[win((d(start) - timedelta(days=7)).isoformat(), end)],
                         required=[spec(PRICE_FACT, cover=(start, end)),
                                   spec(["explain_drivers"], cover=(day, day), max_days=7)],
                         numbers=[number(f"negative_{a.isoformat()[:7]}", [c], 0.5) for (a, _), c in zip(segments, counts)]
                         + [keyed("top_day_negative", top["value"], 0.5, "", day)],
                         key_points=[mentions(*day_names(day)),
                                     {"id": "cause_backed_by_drivers", "type": "cause_cites_tool", "tool": "explain_drivers"},
                                     {"id": "discloses_limitations", "type": "has_limitations"}],
                         fault_tool="explain_drivers",
                         solution=[self.call("query_stats", start_date=a.isoformat(), end_date=b.isoformat(), group_by="day")
                                   for a, b in segments] + [self.call("explain_drivers", start_date=day, end_date=day)])

    def l4_peak_chain(self) -> dict:
        start, end = self._l4_span()
        segments = months_in(d(start), d(end))
        means = [self.r.stats(a.isoformat(), b.isoformat())["mean"] for a, b in segments]
        if len(set(round(m, 2) for m in means)) < len(means):
            raise Skip("tied segments")
        best = segments[max(range(len(means)), key=means.__getitem__)]
        top = self.r.daily_unique(best[0].isoformat(), best[1].isoformat(), "avg(rt_price)", "max")
        day = top["trade_date"]
        load = self.r.extreme(day, "load_actual_mw", "max")
        price = self.r.row(day, load["slot_id"])["rt_price"]
        if price is None or price != price:
            raise Skip("missing price")
        q, p = self.phrase([
            f"{self.period(start, end)}期间，按自然月分段（首尾月只算区间内的天数）计算山东实时电价均价；在均价最高的那段里找出日均价最高的一天，"
            f"给出那天的日均价、全网实际负荷峰值及其出现时段，以及该时段的实时电价。",
            f"{self.period(start, end)}这段时间，哪个月（区间内部分）电价均价最高？那个月里电价最贵的一天日均多少？那天负荷峰值多少、几点？峰值时刻电价多少？"])
        return self.case("l4_peak_chain", "L4", "A", q, p, self.as_of_after(end), allowed=DATA_TOOLS,
                         windows=[win(start, end)],
                         required=[spec(STATS, cover=(start, end)), spec(LOAD_FACT, cover=(day, day))],
                         numbers=[number(f"mean_{a.isoformat()[:7]}", [m]) for (a, _), m in zip(segments, means)]
                         + [keyed("daily_mean", top["value"], 0.01, "", day),
                            keyed("load_max", load["value"], 0.5, "load_actual_mw", day),
                            keyed("price_at_peak", float(price), 0.01, "rt_price", day, load["slot_id"])],
                         key_points=[mentions(*day_names(day)), mentions(load["time_label"])], fault_tool="query_load",
                         solution=[self.call("query_stats", start_date=a.isoformat(), end_date=b.isoformat(), group_by="day")
                                   for a, b in segments] + [self.call("query_load", start_date=day, end_date=day),
                                                            self.call("query_price", start_date=day, end_date=day,
                                                                      slot_id=load["slot_id"])])

    def l4_spike_chain(self) -> dict:
        start, end = self._l4_span()
        segments = months_in(d(start), d(end))
        top = self.r.spikes(start, end)
        day = top["trade_date"]
        w = self.r.weather(day, day, "temp_jinan", "max")
        q, p = self.phrase([
            f"{self.period(start, end)}期间，山东实时电价达到尖峰阈值（{SPIKE_THRESHOLD:g} 元/MWh 及以上）的时段最多的是哪一天？"
            f"那天最高价多少？济南当天最高气温多少？请分析那天价格尖峰的可能原因。",
            f"{self.period(start, end)}哪天价格尖峰（≥{SPIKE_THRESHOLD:g} 元/MWh）最多？当天最高价、济南最高气温各多少？为什么会冲高？"])
        return self.case("l4_spike_chain", "L4", "A", q, p, self.as_of_after(end), allowed=EXPLAIN_TOOLS,
                         windows=[win((d(start) - timedelta(days=7)).isoformat(), end)],
                         required=[spec(PRICE_FACT, cover=(start, end)), spec(WEATHER_FACT, cover=(day, day)),
                                   spec(["explain_drivers"], cover=(day, day), max_days=7)],
                         numbers=[keyed("max_price", top["max"], 0.01, "", day),
                                  number("temp_max", [w["value"]], 0.05) | {"field": "temp_jinan"}],
                         key_points=[mentions(*day_names(day)),
                                     {"id": "cause_backed_by_drivers", "type": "cause_cites_tool", "tool": "explain_drivers"},
                                     {"id": "discloses_limitations", "type": "has_limitations"}],
                         fault_tool="explain_drivers", reference={"spike_day": top, "threshold": SPIKE_THRESHOLD},
                         solution=[self.call("query_price", start_date=a.isoformat(), end_date=b.isoformat(), limit=3000)
                                   for a, b in segments] + [self.call("query_stats", start_date=day, end_date=day),
                                                            self.call("query_weather", start_date=day, end_date=day),
                                                            self.call("explain_drivers", start_date=day, end_date=day)])

    # ================================================================== answerability B-E
    def b_other_province(self) -> dict:
        start, end = self.month()
        other = self.rng.choice(OTHER_PROVINCES)
        s = self.r.stats(start, end)
        q, p = self.phrase([f"{self.period(start, end)}山东和{other}的实时电价月均价分别是多少？",
                            f"{d(start).year}年{d(start).month}月山东、{other}两地现货实时电价平均各多少？"])
        return self.case("b_other_province", "L2", "B", q, p, self.as_of_after(end), allowed=DATA_TOOLS,
                         windows=[win(start, end)], required=[spec(STATS, cover=(start, end))],
                         numbers=[number("mean_shandong", [s["mean"]])],
                         key_points=[mentions(other, anywhere=True), mentions(*UNAVAILABLE, "仅", "只", anywhere=True)],
                         solution=[self.call("query_stats", start_date=start, end_date=end)])

    def b_other_city(self) -> dict:
        day = self.day()
        city = self.rng.choice(["青岛", "烟台", "潍坊", "临沂"])
        w = self.r.weather(day, day, "temp_jinan", "max")
        q, p = self.phrase([f"{cn(d(day))}济南和{city}的最高气温分别是多少？", f"{cn(d(day))}济南、{city}两地最高温各几度？"])
        return self.case("b_other_city", "L2", "B", q, p, self.as_of_after(day), allowed=DATA_TOOLS,
                         windows=[win(day, day)], required=[spec(WEATHER_FACT, cover=(day, day))],
                         numbers=[number("temp_jinan_max", [w["value"]], 0.05) | {"field": "temp_jinan"}],
                         key_points=[mentions(city, anywhere=True), mentions(*UNAVAILABLE, "仅", "只", anywhere=True)],
                         solution=[self.call("query_weather", start_date=day, end_date=day)])

    def b_dayahead_and_realtime(self) -> dict:
        day = self.day()
        s = self.r.stats(day, day)
        q, p = self.phrase([f"{cn(d(day))}山东日前电价和实时电价的均价分别是多少？", f"{cn(d(day))}山东日前、实时两个市场均价各多少？"])
        return self.case("b_dayahead_and_realtime", "L2", "B", q, p, self.as_of_after(day), allowed=DATA_TOOLS,
                         windows=[win(day, day)], required=[spec(STATS, cover=(day, day))],
                         numbers=[number("mean_rt", [s["mean"]])],
                         key_points=[mentions("日前", anywhere=True), mentions(*UNAVAILABLE, "仅", "只", anywhere=True)],
                         solution=[self.call("query_stats", start_date=day, end_date=day)])

    def _refuse(self, template: str, question: str, phrasing: int, as_of: str, window: dict, values: list[str],
                allowed: list[str] = DATA_TOOLS, **extra) -> dict:
        return self.case(template, "L1", "C", question, phrasing, as_of, allowed=allowed, windows=[window],
                         key_points=[mentions(*values, anywhere=True), {"id": "no_numbers", "type": "no_numbers"}], **extra)

    def c_other_province(self) -> dict:
        start, end = self.month()
        other = self.rng.choice(OTHER_PROVINCES)
        q, p = self.phrase([f"{other}电力现货市场{d(start).year}年{d(start).month}月的实时电价均价是多少？",
                            f"{d(start).year}年{d(start).month}月{other}现货电价平均多少？"])
        return self._refuse("c_other_province", q, p, self.as_of_after(end), win(start, end), UNAVAILABLE + ["仅", "只", "山东"])

    def c_after_as_of(self) -> dict:
        day = self.day()
        ask = (d(day) + timedelta(days=self.rng.randint(2, 6))).isoformat()
        q, p = self.phrase([f"{cn(d(ask))}山东实时电价的均价是多少？", f"{cn(d(ask))}山东电价平均多少？"])
        return self._refuse("c_after_as_of", q, p, at(day, "10:00"),
                            win((d(day) - timedelta(days=31)).isoformat(), (d(day) - timedelta(days=1)).isoformat()),
                            UNAVAILABLE + ["晚于", "之后", "未来", "截止"])

    def c_before_database(self) -> dict:
        month = self.rng.randint(1, 12)
        q, p = self.phrase([f"2023年{month}月山东实时电价的月均价是多少？", f"2023年{month}月山东现货电价平均多少？"])
        return self._refuse("c_before_database", q, p, self.as_of_after(self.day()), win("2023-01-01", "2024-01-31"),
                            UNAVAILABLE + ["2024年1月1日", "2024-01-01"])

    def c_dayahead_only(self) -> dict:
        start, end = self.month()
        q, p = self.phrase([f"{d(start).year}年{d(start).month}月山东日前市场的出清均价是多少？",
                            f"{d(start).year}年{d(start).month}月山东日前电价平均多少？"])
        return self._refuse("c_dayahead_only", q, p, self.as_of_after(end), win(start, end),
                            UNAVAILABLE + ["无日前", "没有日前", "不含日前", "仅有实时", "只有实时", "仅包含实时", "只包含实时"])

    def c_forecast_d2(self) -> dict:
        low, high = map(d, self.cfg["forecast"])
        origin = low + timedelta(days=self.rng.randint(0, (high - low).days - 2))
        target = origin + timedelta(days=self.rng.randint(2, 3))
        q, p = self.phrase([f"研判{cn(target)}的山东实时电价，最低价大概多少？", f"预测一下{cn(target)}山东实时电价最低能到多少？"])
        return self._refuse("c_forecast_d2", q, p, at(origin.isoformat()),
                            win((origin - timedelta(days=28)).isoformat(), (origin - timedelta(days=1)).isoformat()),
                            UNAVAILABLE + ["D+1", "次日", "下一个交易日", "仅能", "只能"], allowed=FORECAST_TOOLS,
                            mode="historical_forecast", forecast_target=origin.isoformat())

    def d_recent_week(self) -> dict:
        day = d(self.day())
        if day.weekday() == 0:
            raise Skip("need a mid-week day")
        last7 = ((day - timedelta(days=7)).isoformat(), (day - timedelta(days=1)).isoformat())
        monday = day - timedelta(days=day.weekday() + 7)
        if monday not in self.day_set:
            raise Skip("window leaves the split")
        calendar = (monday.isoformat(), (monday + timedelta(days=6)).isoformat())
        means = [self.r.stats(*last7)["mean"], self.r.stats(*calendar)["mean"]]
        q, p = self.phrase(["最近一周山东实时电价的平均水平是多少？", "这一周左右山东电价平均多少？"])
        return self.case("d_recent_week", "L1", "D", q, p, at(day.isoformat(), "08:30"), allowed=DATA_TOOLS,
                         windows=[win(calendar[0], last7[1])], required=[spec(STATS, cover=(last7[1], last7[1]))],
                         numbers=[number("mean", means)],
                         key_points=[mentions(*day_names(last7[0]), *day_names(calendar[0]), *day_names(last7[1]), anywhere=True)],
                         solution=[self.call("query_stats", start_date=last7[0], end_date=last7[1])])

    def d_clock_time(self) -> dict:
        day, hour = self.day(), self.rng.choice([7, 8, 12, 18, 19, 20, 21])
        label = f"{hour:02d}:00"
        frame = self.r.con.execute("SELECT slot_id, time_label, rt_price FROM market_15min WHERE trade_date = ? AND time_label IN (?, ?)",
                                   [d(day), label, f"{hour:02d}:15"]).fetchall()
        prices = [float(x[2]) for x in frame if x[2] is not None]
        if len(prices) != 2:
            raise Skip("missing slot")
        q, p = self.phrase([f"{cn(d(day))}{'晚上' if hour >= 18 else ('中午' if hour == 12 else '早上')}{hour if hour <= 12 else hour - 12}点山东的实时电价是多少？",
                            f"{cn(d(day))} {hour}点整山东实时电价多少？"])
        return self.case("d_clock_time", "L1", "D", q, p, self.as_of_after(day), allowed=DATA_TOOLS,
                         windows=[win(day, day)], required=[spec(STATS, cover=(day, day))],
                         numbers=[number("price", prices)],
                         key_points=[mentions(label, f"{hour:02d}:15", "时段", "15分钟", "slot", anywhere=True)],
                         solution=[self.call("query_price", start_date=day, end_date=day)])

    def _policy_check(self, path: str, policy_at: str, as_of: str, expected: str) -> None:
        state = document_applicability(self.clauses.meta.get(path), policy_at=policy_at, as_of_date=as_of[:10],
                                       effective_filter=True)
        if state != expected:
            raise Skip(f"{path} on {policy_at}: {state}")

    def e_version(self) -> dict:
        topic = self.rng.choice(self.cfg["policy_topics"])
        prompt, docs = POLICY_TOPICS[topic]
        policy_at = date(2026, 1, 1) + timedelta(days=self.rng.randint(-45, 44))
        paths = list(docs)
        path = paths[0] if policy_at < date(2026, 1, 1) else paths[1]
        as_of = at((policy_at + timedelta(days=self.rng.randint(1, 10))).isoformat())
        self._policy_check(path, policy_at.isoformat(), as_of, "applicable")
        for other in paths:
            if other != path:
                self._policy_check(other, policy_at.isoformat(), as_of, "not_applicable")
        located = self.clauses.locate(path, docs[path])
        q, p = self.phrase([f"{cn(policy_at)}，{prompt}请给出原文依据。", f"按{cn(policy_at)}当天的规定，{prompt.rstrip('？')}？出处是哪份文件？"])
        return self.case("e_version", "L1", "E", q, p, as_of, allowed=POLICY_ONLY, windows=[],
                         required=[spec(["search_policy"], policy_in_force=path)],
                         key_points=[{"id": "cites_applicable_version", "type": "policy_required"}],
                         needs_policy=True, required_policy=[located], wrong_version_paths=[x for x in paths if x != path],
                         policy_at=policy_at.isoformat(),
                         solution=[self.call("search_policy", query=prompt, policy_at=policy_at.isoformat())])

    def e_compare(self) -> dict:
        topic = self.rng.choice([t for t in self.cfg["policy_topics"] if t != "rules_effective"])
        prompt, docs = POLICY_TOPICS[topic]
        old, new = list(docs)
        before = date(2025, 12, 31) - timedelta(days=self.rng.randint(0, 40))
        after = date(2026, 1, 1) + timedelta(days=self.rng.randint(0, 40))
        as_of = at((after + timedelta(days=self.rng.randint(1, 7))).isoformat())
        self._policy_check(old, before.isoformat(), as_of, "applicable")
        self._policy_check(new, after.isoformat(), as_of, "applicable")
        located = [self.clauses.locate(old, docs[old]), self.clauses.locate(new, docs[new])]
        q, p = self.phrase([f"对比{cn(before)}和{cn(after)}分别适用的规定：{prompt}两版有什么不同？请分别给出原文依据。",
                            f"{cn(before)}和{cn(after)}这两天，{prompt.rstrip('？')}？前后两版规定有没有变化，出处分别是什么？"])
        return self.case("e_compare", "L2", "E", q, p, as_of, allowed=POLICY_ONLY, windows=[],
                         required=[spec(["search_policy"], policy_in_force=old), spec(["search_policy"], policy_in_force=new)],
                         key_points=[{"id": "cites_applicable_version", "type": "policy_required"}],
                         needs_policy=True, required_policy=located, policy_at=before.isoformat(),
                         solution=[self.call("search_policy", query=prompt, policy_at=before.isoformat()),
                                   self.call("search_policy", query=prompt, policy_at=after.isoformat())])

    # ------------------------------------------------------------------ drawing
    def draw(self, template: Callable[[], dict], count: int, attempts: int = 60) -> list[dict]:
        out = []
        for _ in range(count * attempts):
            if len(out) == count:
                break
            try:
                case = template()
            except Skip:
                continue
            key = case["template"] + "|" + case["question"] + "|" + case["as_of"]
            if key in self.used:
                continue
            self.used.add(key)
            out.append(case)
        if len(out) < count:
            raise RuntimeError(f"{template.__name__}: only {len(out)}/{count} cases")
        return out


PLAN = {  # template -> cases per split
    "l1_day_stat": 4, "l1_slot_price": 4, "l1_load_extreme": 4, "l1_temperature": 4,
    "l2_month_means": 4, "l2_yoy_window": 4, "l2_price_weather": 4, "l2_two_days_negative": 4,
    "l3_peakday_load": 6, "l3_negday_component": 6, "l3_minpoint_renewables": 6, "l3_peakload_price": 6,
    "l3_hottest_day_price": 6,
    "l4_negative_drilldown": 10, "l4_peak_chain": 10, "l4_spike_chain": 10,
    "b_other_province": 3, "b_other_city": 3, "b_dayahead_and_realtime": 3,
    "c_other_province": 2, "c_after_as_of": 2, "c_before_database": 2, "c_dayahead_only": 2, "c_forecast_d2": 2,
    "d_recent_week": 4, "d_clock_time": 4, "e_version": 6, "e_compare": 4,
}
L5_FROM = {"l3_peakday_load": 2, "l3_negday_component": 2, "l3_minpoint_renewables": 2, "l3_peakload_price": 2,
           "l3_hottest_day_price": 2, "l4_negative_drilldown": 4, "l4_peak_chain": 3, "l4_spike_chain": 3}
# test_v2 is sized from the pilot (plan §9.2): a 15-point paired difference needs about 50 cases per layer, so
# L3/L4/L5 get about 50 each, L1/L2 20 each, and each answerability class 10-15.
TEST_PLAN = PLAN | {
    **{name: 5 for name in PLAN if name.startswith(("l1_", "l2_"))},
    **{name: 10 for name in PLAN if name.startswith("l3_")},
    **{name: 17 for name in PLAN if name.startswith("l4_")},
    **{name: 4 for name in PLAN if name.startswith("b_")},
    **{name: 6 for name in PLAN if name.startswith("d_")},
    "e_version": 7, "e_compare": 5,
}
TEST_L5_FROM = {**{name: 5 for name in L5_FROM if name.startswith("l3_")},
                "l4_negative_drilldown": 9, "l4_peak_chain": 8, "l4_spike_chain": 8}
SIZES = {"dev": (PLAN, L5_FROM), "test": (TEST_PLAN, TEST_L5_FROM)}


def build(split: str, seed: int) -> list[dict]:
    g = Generator(split, seed)
    plan, l5_from = SIZES[split]
    cases = []
    for name, count in plan.items():
        cases += g.draw(getattr(g, name), count)
    for name, count in l5_from.items():
        for base in g.draw(getattr(g, name), count):
            base = base | {"stratum": "L5", "category": "L5", "base_template": base["template"],
                           "faults": [{"tool": base["fault_tool"], "mode": "unavailable", "times": 1}]}
            cases.append(base)
    prefix = {"dev": "V", "test": "T"}[split]
    counters: Counter = Counter()
    for case in cases:
        group = case["stratum"] + case["answerability"]
        counters[group] += 1
        case["id"] = f"{prefix}{group}{counters[group]:02d}"
    for case in cases:
        for s in case["required_calls"]:
            if not set(s["tools"]) & set(case["allowed_tools"]):
                raise ValueError(f"{case['id']} requires {s['tools']} but allows {case['allowed_tools']}")
        for n in case["required_numbers"] + case.get("evidence_values", []):
            if any(v != v for v in n["any_of"]):
                raise ValueError(f"{case['id']} {n['name']} has NaN")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", help="set name (default <split>_v2)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    name = args.name or f"{args.split}_v2"
    cases = build(args.split, args.seed)
    if args.dry_run:
        for c in cases:
            print(c["id"], c["template"], c["as_of"][:16], c["question"])
            print("   numbers:", {n["name"]: [round(v, 3) for v in n["any_of"]] for n in c["required_numbers"]})
        print(Counter((c["stratum"], c["answerability"]) for c in cases))
        return 0
    path = OUT / f"{name}.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; frozen sets are not rebuilt without --overwrite")
    body = json.dumps(cases, ensure_ascii=False, sort_keys=True, default=str)
    payload = {"set": name, "version": 2, "split": args.split, "seed": args.seed,
               "built_at": datetime.now().isoformat(timespec="seconds"), "case_count": len(cases),
               "cases_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
               "spike_threshold_yuan_per_mwh": SPIKE_THRESHOLD,
               "strata": {f"{k[0]}{k[1]}": v for k, v in sorted(Counter((c["stratum"], c["answerability"]) for c in cases).items())},
               "cases": cases}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases) {payload['strata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
