"""Build the held-out generalization set ``holdout_v1``.

formal_v1 was used both to develop the agent and to report results, and its 100 questions come from five
templates with explicit dates inside the forecast test period. This set is written to cover situations
formal_v1 does not, and is frozen before any system answer is seen:

- H  time expressions: relative dates resolved against ``as_of``, named holidays, quarters, point lookups,
     year-over-year comparison, and periods outside the forecast test window (2024 to 2025-09);
- L  other data: load, wind, solar, tie-line, bid space, weather;
- M  multi-hop: an argument of a later call (a day, a slot) comes out of an earlier result;
- F  forecast variants: rolling snapshot, implicit target day, forecast plus history, holiday target;
- P  policy: version boundaries, other documents and clauses, paraphrased or date-less questions,
     version comparison, a date before corpus coverage; policy-only questions may not call data tools;
- R  boundaries and refusals: data after ``as_of``, before the database, another province, day-ahead
     prices that do not exist, a city without weather data, D+2 forecasts, an unsupported forecast period,
     investment advice, missing slots, and an under-specified question.

References come from direct SQL, the saved snapshot predictions, and clauses located in the parsed policy
text, never from the agent's tools. Scoring adds per-case ``required_calls`` (with exact arguments where
they matter), narrower ``allowed_tools``, and key points for refusals; see ``score.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta

import pandas as pd

from ellectric.eval.market.build_cases import (
    ANC_2023, ANC_2025, GRID_2023, GRID_2025, MODELS, OUT, RULES_2024, RULES_2025, SPIKE_THRESHOLD,
    PolicyClauses, Refs, d, number,
)
from ellectric.policy.retrieval import document_applicability

DATA_TOOLS = ["query_price", "query_stats", "query_load", "query_weather", "detect_anomaly", "explain_drivers"]
EXPLAIN_TOOLS = DATA_TOOLS + ["search_policy"]
FORECAST_TOOLS = DATA_TOOLS + ["forecast_price"]
POLICY_ONLY = ["search_policy"]
PRICE_FACT = ["query_stats", "query_price", "detect_anomaly"]
STATS = ["query_stats", "query_price"]
GRID_2026 = "附件1：山东省电力并网运行管理实施细则（2026年修订版）.pdf"
ANC_2026 = "附件2：山东省电力辅助服务管理实施细则（2026年修订版）.pdf"
PARAMS = "山东电力市场参数（试行）.docx"
UNAVAILABLE = ["无法", "不可得", "不可用", "尚未", "未覆盖", "不在", "超出", "没有", "无数据", "缺少", "不支持"]


def at(day: str, clock: str = "00:00") -> str:
    return f"{day}T{clock}:00+08:00"


def spec(tools: list[str], *, cover: tuple[str, str] | None = None, **extra) -> dict:
    out = {"tools": tools, **extra}
    if cover:
        out["cover"] = {"start_date": cover[0], "end_date": cover[1]}
    return out


def win(start: str, end: str) -> dict:
    return {"start_date": start, "end_date": end}


def mentions(*values: str, anywhere: bool = False) -> dict:
    return {"id": "mentions_" + "_".join(values[:1]), "type": "any_text_contains_any" if anywhere else "text_contains_any",
            "values": list(values)}


def day_names(day: str) -> list[str]:
    x = d(day)
    return [day, f"{x.month}月{x.day}日"]


def share_number(name: str, share: float) -> dict:
    # Shares may be reported as a fraction or as a percentage.
    return number(name, [share, share * 100], 0.002, relative=True)


class HoldoutRefs(Refs):
    def column_extreme(self, day: str, column: str, how: str) -> dict:
        order = "DESC" if how == "max" else "ASC"
        value, label, slot = self.one(
            f"SELECT {column}, time_label, slot_id FROM market_15min WHERE trade_date = ? AND {column} IS NOT NULL"
            f" ORDER BY {column} {order}, slot_id LIMIT 1", [d(day)])
        ties = self.one(f"SELECT count(*) FROM market_15min WHERE trade_date = ? AND {column} = ?", [d(day), value])[0]
        if ties != 1:
            raise ValueError(f"{column} {how} on {day} is tied ({ties} slots)")
        return {"value": float(value), "time_label": label, "slot_id": int(slot)}

    def slot_row(self, day: str, label: str) -> dict:
        row = self.con.execute("SELECT * FROM market_15min WHERE trade_date = ? AND time_label = ?", [d(day), label]).fetchdf()
        if len(row) != 1:
            raise ValueError(f"no unique slot {label} on {day}")
        return row.iloc[0].to_dict()

    def next_slot_row(self, day: str, label: str) -> dict:
        slot = int(self.slot_row(day, label)["slot_id"]) + 1
        row = self.con.execute("SELECT * FROM market_15min WHERE trade_date = ? AND slot_id = ?", [d(day), slot]).fetchdf()
        return row.iloc[0].to_dict()

    def daily(self, start: str, end: str, expression: str, how: str) -> dict:
        order = "DESC" if how == "max" else "ASC"
        rows = self.con.execute(
            f"SELECT trade_date, {expression} v FROM market_15min WHERE trade_date BETWEEN ? AND ? GROUP BY 1"
            f" ORDER BY v {order}", [d(start), d(end)]).fetchall()
        if len(rows) > 1 and rows[0][1] == rows[1][1]:
            raise ValueError(f"daily {how} of {expression} tied in {start}..{end}")
        return {"trade_date": rows[0][0].isoformat(), "value": float(rows[0][1])}

    def weather_extreme(self, start: str, end: str, column: str, how: str) -> dict:
        order = "DESC" if how == "max" else "ASC"
        rows = self.con.execute(
            # Same date basis as query_weather (stored timestamp cast to DATE), so the reference matches the tool.
            f"SELECT CAST(CAST(timestamp_raw AS DATE) AS VARCHAR) d, {column} FROM weather_actual_hourly"
            f" WHERE CAST(timestamp_raw AS DATE) BETWEEN ? AND ? AND {column} IS NOT NULL ORDER BY {column} {order}",
            [d(start), d(end)]).fetchall()
        days = {day for day, value in rows if value == rows[0][1]}
        if len(days) != 1:
            raise ValueError(f"weather {how} of {column} spans several days in {start}..{end}")
        return {"day": rows[0][0], "value": float(rows[0][1])}

    def last_negative_day(self, latest: str) -> dict:
        day = self.one("SELECT max(trade_date) FROM market_15min WHERE rt_price < 0 AND trade_date <= ?", [d(latest)])[0]
        count = self.one("SELECT count(*) FROM market_15min WHERE rt_price < 0 AND trade_date = ?", [day])[0]
        return {"trade_date": day.isoformat(), "negative": int(count)}

    def spike_day(self, start: str, end: str) -> dict:
        rows = self.con.execute(
            "SELECT trade_date, count(*) FILTER (WHERE rt_price >= ?) n, max(rt_price) FROM market_15min"
            " WHERE trade_date BETWEEN ? AND ? GROUP BY 1 ORDER BY n DESC", [SPIKE_THRESHOLD, d(start), d(end)]).fetchall()
        if rows[0][1] == 0 or (len(rows) > 1 and rows[0][1] == rows[1][1]):
            raise ValueError(f"no unique spike day in {start}..{end}")
        return {"trade_date": rows[0][0].isoformat(), "spikes": int(rows[0][1]), "max": float(rows[0][2])}

    def model_means(self, target: str) -> list[float]:
        values = []
        for frame in (self.fixed, self.rolling):
            rows = frame.loc[pd.to_datetime(frame["trade_date"]).dt.date == d(target)]
            values += [float(rows[m].mean()) for m in MODELS] + [float(rows[MODELS].mean(axis=1).mean())]
        return values


def case(case_id: str, category: str, question: str, as_of: str, *, allowed: list[str], covers: list[str],
         mode: str = "historical_analysis", windows: list[dict] | None = None, required: list[dict] | None = None,
         numbers: list[dict] | None = None, key_points: list[dict] | None = None, **extra) -> dict:
    numbers = numbers or []
    key_points = list(key_points or []) + [{"id": f"reports_{n['name']}", "type": "number", "number": n["name"]} for n in numbers]
    return {"id": case_id, "category": category, "question": question, "mode": mode, "as_of": as_of,
            "allowed_tools": allowed, "arg_windows": windows or [], "required_calls": required or [],
            "required_numbers": numbers, "key_points": key_points, "needs_policy": False, "required_policy": [],
            "task_case": True, "covers": covers, **extra}


# ----------------------------------------------------------------------------- H: time expressions

def time_cases(r: HoldoutRefs) -> list[dict]:
    out = []
    s = r.range_stats(d("2024-08-01"), d("2024-08-01"))
    out.append(case("XH01", "H", "昨天山东实时电价的均价、最高价和最低价分别是多少？", at("2024-08-02", "09:30"),
                    allowed=DATA_TOOLS, covers=["相对时间:昨天", "2024年数据"],
                    windows=[win("2024-08-01", "2024-08-01")], required=[spec(STATS, cover=("2024-08-01", "2024-08-01"))],
                    numbers=[number("mean", [s["mean"]]), number("max", [s["max"]]), number("min", [s["min"]])],
                    key_points=[mentions(*day_names("2024-08-01"))]))
    s = r.range_stats(d("2025-03-03"), d("2025-03-09"))
    out.append(case("XH02", "H", "上周（周一到周日）山东实时电价一共出现了多少个负电价时段？占有效时段的比例是多少？",
                    at("2025-03-12"), allowed=DATA_TOOLS, covers=["相对时间:上周", "2025年上半年数据"],
                    windows=[win("2025-03-03", "2025-03-09")], required=[spec(PRICE_FACT, cover=("2025-03-03", "2025-03-09"))],
                    numbers=[number("negative_count", [s["negative"]], 0.5), share_number("negative_share", s["negative_share"])]))
    s = r.range_stats(d("2025-05-01"), d("2025-05-07"))
    out.append(case("XH03", "H", "截至信息截止时最近7个完整交易日，山东实时电价的平均水平是多少？", at("2025-05-08"),
                    allowed=DATA_TOOLS, covers=["相对时间:最近7天", "跨节假日"],
                    windows=[win("2025-05-01", "2025-05-07")], required=[spec(STATS, cover=("2025-05-01", "2025-05-07"))],
                    numbers=[number("mean", [s["mean"]])]))
    s = r.range_stats(d("2024-11-01"), d("2024-11-19"))
    day =r.daily("2024-11-01", "2024-11-19", "max(rt_price)", "max")
    out.append(case("XH04", "H", "本月到目前为止，山东实时电价的最高价是多少？出现在哪一天？", at("2024-11-20"),
                    allowed=DATA_TOOLS, covers=["相对时间:本月至今"],
                    windows=[win("2024-11-01", "2024-11-19")], required=[spec(STATS, cover=("2024-11-01", "2024-11-19"))],
                    numbers=[number("max", [s["max"]])], key_points=[mentions(*day_names(day["trade_date"]))]))
    s = r.range_stats(d("2025-06-01"), d("2025-06-30"))
    out.append(case("XH05", "H", "上个月山东实时电价的负电价时段有多少个？占比多少？", at("2025-07-03", "14:00"),
                    allowed=DATA_TOOLS, covers=["相对时间:上个月", "整月"],
                    windows=[win("2025-06-01", "2025-06-30")], required=[spec(PRICE_FACT, cover=("2025-06-01", "2025-06-30"))],
                    numbers=[number("negative_count", [s["negative"]], 0.5), share_number("negative_share", s["negative_share"])]))
    s = r.range_stats(d("2024-10-01"), d("2024-10-07"))
    out.append(case("XH06", "H", "2024年国庆假期期间，山东实时电价的均价和最低价是多少？", at("2024-10-15"),
                    allowed=DATA_TOOLS, covers=["节假日名称:国庆", "2024年数据"],
                    windows=[win("2024-10-01", "2024-10-07")], required=[spec(STATS, cover=("2024-10-01", "2024-10-07"))],
                    numbers=[number("mean", [s["mean"]]), number("min", [s["min"]])]))
    slot = r.lowest_slot_mean(d("2025-01-28"), d("2025-02-04"))
    out.append(case("XH07", "H", "2025年春节假期里，按一天中的时段看，哪个时段的平均实时电价最低？均价多少？", at("2025-02-10"),
                    allowed=DATA_TOOLS, covers=["节假日名称:春节", "按时段统计"],
                    windows=[win("2025-01-28", "2025-02-04")],
                    required=[spec(["query_stats"], cover=("2025-01-28", "2025-02-04"), args={"group_by": "slot"})],
                    numbers=[number("lowest_slot_mean", [slot["mean"]])],
                    key_points=[mentions(slot["time_label"], f"slot {slot['slot_id']}", f"slot_id={slot['slot_id']}")]))
    months = [r.range_stats(d(a), d(b)) for a, b in (("2024-07-01", "2024-07-31"), ("2024-08-01", "2024-08-31"),
                                                       ("2024-09-01", "2024-09-30"))]
    best = max(range(3), key=lambda i: months[i]["mean"])
    best = (["7月", "8月", "9月"][best], f"2024-0{7 + best}")
    out.append(case("XH08", "H", "2024年第三季度每个月的山东实时电价均价分别是多少？哪个月最高？", at("2024-10-10"),
                    allowed=DATA_TOOLS, covers=["季度:超过单次查询上限需拆分", "多月比较"],
                    windows=[win("2024-07-01", "2024-09-30")], required=[spec(STATS, cover=("2024-07-01", "2024-09-30"))],
                    numbers=[number(f"mean_{m}", [x["mean"]]) for m, x in zip(("jul", "aug", "sep"), months)],
                    key_points=[mentions(*best)]))
    months = [r.range_stats(d(a), d(b)) for a, b in (("2025-04-01", "2025-04-30"), ("2025-05-01", "2025-05-31"),
                                                       ("2025-06-01", "2025-06-30"))]
    best = max(range(3), key=lambda i: months[i]["negative"])
    best = (["4月", "5月", "6月"][best], f"2025-0{4 + best}")
    out.append(case("XH09", "H", "2025年4月到6月，哪个月的负电价时段最多？三个月各有多少个？", at("2025-07-05"),
                    allowed=DATA_TOOLS, covers=["多月:拆分查询", "计数比较"],
                    windows=[win("2025-04-01", "2025-06-30")], required=[spec(PRICE_FACT, cover=("2025-04-01", "2025-06-30"))],
                    numbers=[number(f"negative_{m}", [x["negative"]], 0.5) for m, x in zip(("apr", "may", "jun"), months)],
                    key_points=[mentions(*best)]))
    a, b = r.slot_row("2024-07-15", "19:00"), r.next_slot_row("2024-07-15", "19:00")
    out.append(case("XH10", "H", "2024年7月15日晚上19:00山东的实时电价是多少？", at("2024-07-20"),
                    allowed=DATA_TOOLS, covers=["单点查询:具体时刻", "时段口径"],
                    windows=[win("2024-07-15", "2024-07-15")], required=[spec(STATS, cover=("2024-07-15", "2024-07-15"))],
                    arg_constraints={"query_price": {"slot_id": [str(int(a["slot_id"])), str(int(b["slot_id"]))]}},
                    numbers=[number("price_1900", [a["rt_price"], b["rt_price"]])]))
    s24, s25 = r.range_stats(d("2024-12-01"), d("2024-12-14")), r.range_stats(d("2025-12-01"), d("2025-12-14"))
    out.append(case("XH11", "H", "今年12月1日至14日山东实时电价均价是多少？和去年同期相比变化了多少？", at("2025-12-20"),
                    allowed=DATA_TOOLS, covers=["同比:两个窗口", "相对年份"],
                    windows=[win("2024-12-01", "2024-12-14"), win("2025-12-01", "2025-12-14")],
                    required=[spec(STATS, cover=("2025-12-01", "2025-12-14")), spec(STATS, cover=("2024-12-01", "2024-12-14"))],
                    numbers=[number("mean_2025", [s25["mean"]]), number("mean_2024", [s24["mean"]]),
                             number("yoy_change", [s25["mean"] - s24["mean"], s24["mean"] - s25["mean"]])]))
    return out


# ----------------------------------------------------------------------------- L: other data

def data_cases(r: HoldoutRefs) -> list[dict]:
    out = []

    def extreme(case_id, day, column, how, question, covers, as_of, name_time=False):
        x = r.column_extreme(day, column, how)
        kps = [mentions(x["time_label"])] if name_time else []
        return case(case_id, "L", question, as_of, allowed=DATA_TOOLS, covers=covers,
                    windows=[win(day, day)], required=[spec(["query_load"], cover=(day, day))],
                    numbers=[number(f"{column}_{how}", [x["value"]], 0.5)], key_points=kps)

    out.append(extreme("XL01", "2024-08-05", "load_actual_mw", "max", "2024年8月5日山东全网实际负荷最高是多少兆瓦？出现在几点？",
                       ["负荷", "极值时刻"], at("2024-08-10"), name_time=True))
    out.append(extreme("XL02", "2025-03-10", "wind_actual_mw", "max", "2025年3月10日山东风电实际出力的最大值是多少？",
                       ["风电"], at("2025-03-15")))
    out.append(extreme("XL03", "2024-06-10", "solar_actual_mw", "max", "2024年6月10日光伏实际出力最大是多少？在哪个时段？",
                       ["光伏", "极值时刻"], at("2024-06-12"), name_time=True))
    a, b = r.slot_row("2024-12-20", "18:00"), r.next_slot_row("2024-12-20", "18:00")
    out.append(case("XL04", "L", "2024年12月20日18:00，山东实际负荷和预测负荷分别是多少？偏差多大？", at("2024-12-25"),
                    allowed=DATA_TOOLS, covers=["负荷预测偏差", "单点查询", "差值计算"],
                    windows=[win("2024-12-20", "2024-12-20")], required=[spec(["query_load"], cover=("2024-12-20", "2024-12-20"))],
                    arg_constraints={"query_load": {"slot_id": [str(int(a["slot_id"])), str(int(b["slot_id"]))]}},
                    numbers=[number("load_actual", [a["load_actual_mw"], b["load_actual_mw"]], 0.5),
                             number("load_forecast", [a["load_forecast_mw"], b["load_forecast_mw"]], 0.5),
                             number("deviation", [x["load_actual_mw"] - x["load_forecast_mw"] for x in (a, b)]
                                    + [x["load_forecast_mw"] - x["load_actual_mw"] for x in (a, b)], 0.5)]))
    out.append(extreme("XL05", "2025-04-20", "actual_bid_space_mw", "min", "2025年4月20日实际竞价空间的最小值是多少？",
                       ["竞价空间"], at("2025-04-25")))
    w = r.weather_extreme("2024-07-01", "2024-07-31", "temp_jinan", "max")
    out.append(case("XL06", "L", "2024年7月济南的最高气温是多少度？是哪一天？", at("2024-08-05"),
                    allowed=DATA_TOOLS, covers=["气象", "整月小时数据超过默认返回条数"],
                    windows=[win("2024-07-01", "2024-07-31")], required=[spec(["query_weather"], cover=("2024-07-01", "2024-07-31"))],
                    numbers=[number("temp_max", [w["value"]], 0.05)], key_points=[mentions(*day_names(w["day"]))]))
    out.append(extreme("XL07", "2025-01-15", "tie_line_actual_mw", "max", "2025年1月15日山东联络线（外来电）实际受电最大是多少？",
                       ["联络线"], at("2025-01-20")))
    w = r.weather_extreme("2025-01-09", "2025-01-09", "temp_jinan", "min")
    s = r.range_stats(d("2025-01-09"), d("2025-01-09"))
    out.append(case("XL08", "L", "2025年1月9日济南最低气温是多少？当天山东实时电价最高到多少？", at("2025-01-12"),
                    allowed=DATA_TOOLS, covers=["气象+电价:两类工具"],
                    windows=[win("2025-01-09", "2025-01-09")],
                    required=[spec(["query_weather"], cover=("2025-01-09", "2025-01-09")), spec(STATS, cover=("2025-01-09", "2025-01-09"))],
                    numbers=[number("temp_min", [w["value"]], 0.05), number("max", [s["max"]])]))
    return out


# ----------------------------------------------------------------------------- M: multi-hop

def multihop_cases(r: HoldoutRefs) -> list[dict]:
    out = []
    top = r.daily("2025-04-01", "2025-04-30", "count(*) FILTER (WHERE rt_price < 0)", "max")
    solar = r.column_extreme(top["trade_date"], "solar_actual_mw", "max")
    out.append(case("XM01", "M", "2025年4月里负电价时段最多的是哪一天？那天有多少个负电价时段？当天光伏实际出力最大是多少？",
                    at("2025-05-03"), allowed=DATA_TOOLS, covers=["多跳:先找日期再查出力"],
                    windows=[win("2025-04-01", "2025-04-30")],
                    required=[spec(PRICE_FACT, cover=("2025-04-01", "2025-04-30")),
                              spec(["query_load"], cover=(top["trade_date"], top["trade_date"]))],
                    numbers=[number("negative_count", [top["value"]], 0.5), number("solar_max", [solar["value"]], 0.5)],
                    key_points=[mentions(*day_names(top["trade_date"]))]))
    top = r.daily("2024-08-01", "2024-08-31", "avg(rt_price)", "max")
    load = r.column_extreme(top["trade_date"], "load_actual_mw", "max")
    out.append(case("XM02", "M", "2024年8月日均实时电价最高的是哪天？那天的最大实际负荷是多少？", at("2024-09-03"),
                    allowed=DATA_TOOLS, covers=["多跳:日均排名后查负荷"],
                    windows=[win("2024-08-01", "2024-08-31")],
                    required=[spec(STATS, cover=("2024-08-01", "2024-08-31")),
                              spec(["query_load"], cover=(top["trade_date"], top["trade_date"]))],
                    numbers=[number("daily_mean", [top["value"]]), number("load_max", [load["value"]], 0.5)],
                    key_points=[mentions(*day_names(top["trade_date"]))]))
    low = r.daily("2024-10-01", "2024-10-31", "min(rt_price)", "min")
    point = r.column_extreme(low["trade_date"], "rt_price", "min")
    row = r.slot_row(low["trade_date"], point["time_label"])
    out.append(case("XM03", "M", "2024年10月实时电价的最低点出现在哪天哪个时段？当时风电和光伏实际出力合计是多少？",
                    at("2024-11-02"), allowed=DATA_TOOLS, covers=["多跳:先定位时段再查出力", "求和"],
                    windows=[win("2024-10-01", "2024-10-31")],
                    required=[spec(PRICE_FACT, cover=("2024-10-01", "2024-10-31")),
                              spec(["query_load"], cover=(low["trade_date"], low["trade_date"]))],
                    numbers=[number("min_price", [point["value"]]),
                             number("wind_plus_solar", [row["wind_actual_mw"] + row["solar_actual_mw"]], 1.0)],
                    key_points=[mentions(*day_names(low["trade_date"])), mentions(point["time_label"])]))
    last = r.last_negative_day("2024-08-25")
    out.append(case("XM04", "M", "截至信息截止时，山东实时电价最近一次出现负电价是哪一天？那天有几个负电价时段？",
                    at("2024-08-26"), allowed=DATA_TOOLS, covers=["多跳:向前回溯", "相对时间"],
                    windows=[win("2024-06-25", "2024-08-25")],
                    required=[spec(PRICE_FACT, cover=(last["trade_date"], "2024-08-25"))],
                    numbers=[number("negative_count", [last["negative"]], 0.5)],
                    key_points=[mentions(*day_names(last["trade_date"]))], reference={"last_negative": last}))
    spike = r.spike_day("2024-08-01", "2024-08-31")
    out.append(case("XM05", "M", "2024年8月价格尖峰最多的是哪一天？当天最高价多少？可能是什么原因？", at("2024-09-02"),
                    allowed=EXPLAIN_TOOLS, covers=["多跳:先检测再归因", "非测试期归因"],
                    windows=[win("2024-08-01", "2024-08-31")],
                    required=[spec(PRICE_FACT, cover=("2024-08-01", "2024-08-31")),
                              spec(["explain_drivers"], cover=(spike["trade_date"], spike["trade_date"]), max_days=7)],
                    numbers=[number("max_price", [spike["max"]])],
                    key_points=[mentions(*day_names(spike["trade_date"])),
                                {"id": "cause_backed_by_drivers", "type": "cause_cites_tool", "tool": "explain_drivers"},
                                {"id": "discloses_limitations", "type": "has_limitations"}],
                    reference={"spike_day": spike, "threshold": SPIKE_THRESHOLD}))
    return out


# ----------------------------------------------------------------------------- F: forecast variants

def forecast_cases(r: HoldoutRefs) -> list[dict]:
    out = []

    def fc(case_id, target, question, covers, *, mode_arg=None, numbers, extra_required=(), windows=None, **extra):
        args = {"target_date": target} | ({"mode": mode_arg} if mode_arg else {})
        lookback = (d(target) - timedelta(days=28)).isoformat()
        latest = (d(target) - timedelta(days=1)).isoformat()
        return case(case_id, "F", question, at(target), allowed=FORECAST_TOOLS, covers=covers, mode="historical_forecast",
                    windows=windows or [win(lookback, latest)], required=[spec(["forecast_price"], args=args), *extra_required],
                    numbers=numbers, forecast_target=target, **extra)

    f = r.forecast(d("2025-11-20"))
    out.append(fc("XF01", "2025-11-20", "请用滚动重训（rolling）的模型快照研判2025年11月20日山东实时电价，三个模型预测的最低价分别是多少？",
                  ["滚动快照:指定 mode"], mode_arg="rolling",
                  numbers=[number("rolling_model_min", [f["rolling"][f"{m}_min"] for m in MODELS])]))
    f = r.forecast(d("2025-12-16"))
    out.append(fc("XF02", "2025-12-16", "以当前信息截止，研判下一个交易日的山东实时电价，三个模型各自预测的最高价是多少？",
                  ["隐含目标日"],
                  numbers=[number("some_model_max", [f[s][f"{m}_max"] for s in ("fixed", "rolling") for m in MODELS])]))
    actual = r.range_stats(d("2025-12-30"), d("2025-12-30"))
    out.append(fc("XF03", "2026-01-06", "研判2026年1月6日山东实时电价的日均水平（三个模型各自预测的日均价），并和上周同一天的实际日均价比较。",
                  ["预测+历史对比", "相对日期:上周同日"],
                  windows=[win("2025-12-09", "2026-01-05")],
                  extra_required=[spec(STATS, cover=("2025-12-30", "2025-12-30"))],
                  numbers=[number("model_daily_mean", r.model_means("2026-01-06")), number("actual_mean_last_week", [actual["mean"]])]))
    f = r.forecast(d("2026-01-01"))
    out.append(fc("XF04", "2026-01-01", "研判2026年元旦当天的山东实时电价：最低价大概多少？有没有负电价风险？",
                  ["节假日名称作目标日"],
                  numbers=[number("some_model_min", [f[s][f"{m}_min"] for s in ("fixed", "rolling") for m in MODELS])]))
    return out


# ----------------------------------------------------------------------------- P: policy

def policy_cases(clauses: PolicyClauses) -> list[dict]:
    meta = clauses.meta
    out = []

    def check(path, policy_at, as_of, expected):
        state = document_applicability(meta.get(path), policy_at=policy_at, as_of_date=as_of[:10], effective_filter=True)
        if state != expected:
            raise ValueError(f"{path} on {policy_at} (as_of {as_of[:10]}) is {state}, expected {expected}")

    def pol(case_id, question, as_of, required, covers, wrong=(), extra_specs=()):
        policy_at_of = {path: day for path, _, day in required}
        for path, _, day in required:
            check(path, day, as_of, "applicable")
            for other in wrong:
                check(other, day, as_of, "not_applicable")
        located = [clauses.locate(path, keywords) for path, keywords, _ in required]
        return case(case_id, "P", question, as_of, allowed=POLICY_ONLY, covers=covers,
                    required=[spec(["search_policy"], policy_in_force=path) for path in policy_at_of] + list(extra_specs),
                    needs_policy=True, required_policy=located, wrong_version_paths=list(wrong),
                    policy_at=required[0][2],
                    key_points=[{"id": "cites_applicable_version", "type": "policy_required"}])

    out.append(pol("XP01", "2025年12月31日这一天，山东电力市场执行的是哪一版市场规则？依据是什么？", at("2026-01-02"),
                   [(RULES_2024, ["自2024年5月1日起施行"], "2025-12-31")], ["版本边界:新版生效前一天"], wrong=[RULES_2025]))
    out.append(pol("XP02", "2026年10月1日起，山东电力辅助服务管理实施细则用的是哪个版本？自什么时候施行？", at("2026-10-02"),
                   [(ANC_2026, ["本细则自2026年10月1日起施行"], "2026-10-01")], ["版本边界:修订版生效当天", "第三个版本"],
                   wrong=[ANC_2023, ANC_2025]))
    out.append(pol("XP03", "按2026年3月时适用的规则，什么情况下算触发“电力供需紧张”？", at("2026-03-05"),
                   [(RULES_2025, ["电力供需紧张触发条件如下"], "2026-03-01")], ["新条款", "口语化"], wrong=[RULES_2024]))
    out.append(pol("XP04", "2024年9月的时候，电网代理购电的工商业用户，偏差电量是按什么价格结算的？", at("2024-09-12"),
                   [(RULES_2024, ["代理购电工商业用户的偏差电量应按照现货价格结算"], "2024-09-10")], ["新条款", "口语化", "早期版本"]))
    out.append(pol("XP05", "2026年2月，新型储能企业要进入山东电力市场需要满足哪些基本条件？", at("2026-02-16"),
                   [(RULES_2025, ["新型储能企业基本条件"], "2026-02-15")], ["新条款:经营主体条件"]))
    out.append(pol("XP06", "2025年6月，山东集中竞价交易的申报价格最小单位是多少？", at("2025-06-03"),
                   [(PARAMS, ["交易标的申报价格的最小单位设置为0.1元/兆瓦时"], "2025-06-01")], ["配套文件:参数表"]))
    out.append(pol("XP07", "2026年11月适用的山东省电力并网运行管理实施细则是哪一版？何时施行？", at("2026-11-06"),
                   [(GRID_2026, ["本细则自2026年10月1日起施行"], "2026-11-05")], ["第三个版本"], wrong=[GRID_2023, GRID_2025]))
    out.append(pol("XP09", "对比2025年12月和2026年1月分别适用的山东电力市场规则，二级出清限价的第一种触发条件有什么不同？",
                   at("2026-01-10"),
                   [(RULES_2024, ["平均结算均价连续7天超过现货市场申报上限"], "2025-12-15"),
                    (RULES_2025, ["统一结算点电价连续7天超过现货市场申报上限"], "2026-01-05")], ["跨版本对比:两次检索"]))
    out.append(pol("XP10", "现在执行的山东电力辅助服务管理实施细则是哪一版？从什么时候开始施行的？", at("2026-03-10"),
                   [(ANC_2025, ["本细则自2026年1月1日起施行"], "2026-03-10")], ["无日期:按信息截止日判断"],
                   wrong=[ANC_2023]))
    out.append(case("XP08", "P", "2024年3月15日当天，山东现货市场的报价限价和出清限价是怎么规定的？请给出原文依据。",
                    at("2024-03-20"), allowed=POLICY_ONLY, covers=["早于政策库覆盖期:应说明无法确认"],
                    arg_window=win("2024-03-15", "2024-03-15"),
                    required=[spec(["search_policy"], args={"policy_at": "2024-03-15"})],
                    key_points=[mentions("2024年5月1日", "2024-05-01", "未覆盖", "不在覆盖", "覆盖范围", "无法确认", "无法提供",
                                         "没有找到", "未找到", anywhere=True),
                                {"id": "no_policy_basis", "type": "no_policy_basis"}]))
    return sorted(out, key=lambda c: c["id"])


# ----------------------------------------------------------------------------- R: boundaries and refusals

def boundary_cases(r: HoldoutRefs) -> list[dict]:
    out = []
    out.append(case("XR01", "R", "2025年11月12日山东实时电价的均价是多少？", at("2025-11-10"), allowed=DATA_TOOLS,
                    covers=["请求截止时点之后的数据"], windows=[win("2025-10-10", "2025-11-09")],
                    key_points=[mentions("无法", "尚未", "不可得", "晚于", "之后", "未来", "截止", anywhere=True)]))
    out.append(case("XR02", "R", "2023年12月山东实时电价的月均价是多少？", at("2024-06-01"), allowed=DATA_TOOLS,
                    covers=["数据库起点之前"], windows=[win("2023-11-01", "2024-01-31")],
                    key_points=[mentions("2024年1月1日", "2024-01-01", *UNAVAILABLE, anywhere=True),
                                {"id": "no_numbers", "type": "no_numbers"}]))
    out.append(case("XR03", "R", "山西电力现货市场2025年10月的实时电价均价是多少？", at("2025-11-05"), allowed=[],
                    covers=["其他省份:不应调用山东数据工具"],
                    key_points=[mentions("山东", anywhere=True), mentions(*UNAVAILABLE, "仅", "只", anywhere=True),
                                {"id": "no_numbers", "type": "no_numbers"}]))
    f = r.forecast(d("2025-11-23"))
    out.append(case("XR04", "R", "明天（2025年11月23日）山东实时电价怎么样？我该在哪些时段多报量、怎么报价才能多赚钱？",
                    at("2025-11-23"), allowed=FORECAST_TOOLS, mode="historical_forecast", covers=["索要交易建议"],
                    windows=[win("2025-10-26", "2025-11-22")], forecast_target="2025-11-23",
                    required=[spec(["forecast_price"], args={"target_date": "2025-11-23"})],
                    numbers=[number("some_model_min", [f[s][f"{m}_min"] for s in ("fixed", "rolling") for m in MODELS])],
                    key_points=[mentions("投资建议", "交易建议", "报价建议", "仅供参考", "不构成", anywhere=True)]))
    s = r.range_stats(d("2024-01-06"), d("2024-01-06"))
    out.append(case("XR05", "R", "2024年1月6日山东实时电价有多少个负电价时段？占当天有效价格时段的比例是多少？", at("2024-01-10"),
                    allowed=DATA_TOOLS, covers=["含缺失时段:分母应为有效时段"],
                    windows=[win("2024-01-06", "2024-01-06")], required=[spec(PRICE_FACT, cover=("2024-01-06", "2024-01-06"))],
                    numbers=[number("negative_count", [s["negative"]], 0.5), share_number("negative_share", s["negative_share"])],
                    key_points=[mentions("缺失", "缺少", "有效时段", "有效价格", str(s["valid"]), anywhere=True)],
                    reference={"stats": s}))
    out.append(case("XR06", "R", "最近山东电价怎么样？", at("2025-06-16", "08:00"), allowed=DATA_TOOLS,
                    covers=["问题不具体:自行确定合理区间"], windows=[win("2025-05-15", "2025-06-15")],
                    required=[spec(PRICE_FACT, cover=("2025-06-15", "2025-06-15"))],
                    key_points=[mentions("6月", "2025-06"), {"id": "discloses_limitations", "type": "has_limitations"}]))
    out.append(case("XR07", "R", "2025年10月山东日前市场的出清均价是多少？", at("2025-11-01"), allowed=DATA_TOOLS,
                    covers=["不存在的数据口径:日前价格"], windows=[win("2025-10-01", "2025-10-31")],
                    key_points=[mentions("无日前", "没有日前", "不含日前", "不包含日前", "未包含日前", "缺少日前", "无法提供日前",
                                         "仅有实时", "只有实时", "仅包含实时", "只包含实时", "仅提供实时", "只提供实时",
                                         anywhere=True)]))
    out.append(case("XR08", "R", "2024年7月20日青岛的最高气温是多少？", at("2024-07-25"), allowed=DATA_TOOLS,
                    covers=["无数据的城市:只有济南气象"], windows=[win("2024-07-20", "2024-07-20")],
                    key_points=[mentions("济南", anywhere=True)]))
    out.append(case("XR09", "R", "研判2025年11月27日的山东实时电价。", at("2025-11-25"), allowed=FORECAST_TOOLS,
                    mode="historical_forecast", covers=["D+2:只支持次日研判"], windows=[win("2025-10-28", "2025-11-24")],
                    forecast_target="2025-11-25",
                    key_points=[mentions("D+1", "下一个交易日", "次日", "仅能", "只能", "无法", "不支持", anywhere=True)]))
    out.append(case("XR10", "R", "研判2025年8月20日的山东实时电价。", at("2025-08-20"), allowed=FORECAST_TOOLS,
                    mode="historical_forecast", covers=["预测模型不支持的时期"], windows=[win("2025-07-23", "2025-08-19")],
                    forecast_target="2025-08-20", expected_error_tools=["forecast_price"],
                    key_points=[mentions(*UNAVAILABLE, "未能", anywhere=True)]))
    return out


def build() -> list[dict]:
    refs, clauses = HoldoutRefs(), PolicyClauses()
    cases = time_cases(refs) + data_cases(refs) + multihop_cases(refs) + forecast_cases(refs) + policy_cases(clauses) \
        + boundary_cases(refs)
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    for c in cases:
        for s in c["required_calls"]:
            if not set(s["tools"]) & set(c["allowed_tools"]):
                raise ValueError(f"{c['id']} requires {s['tools']} but allows {c['allowed_tools']}")
        for n in c["required_numbers"]:
            if any(v != v for v in n["any_of"]):
                raise ValueError(f"{c['id']} {n['name']} has NaN")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the cases without writing")
    args = parser.parse_args()
    cases = build()
    if args.dry_run:
        for c in cases:
            print(c["id"], c["as_of"][:16], c["question"])
            print("   numbers:", {n["name"]: [round(v, 3) for v in n["any_of"]] for n in c["required_numbers"]},
                  "| required:", [(s["tools"][0], s.get("cover") or s.get("args") or s.get("policy_in_force")) for s in c["required_calls"]])
        return 0
    path = OUT / "holdout_v1.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; frozen sets are not rebuilt without --overwrite")
    body = json.dumps(cases, ensure_ascii=False, sort_keys=True, default=str)
    payload = {"set": "holdout_v1", "version": 1, "built_at": datetime.now().isoformat(timespec="seconds"),
               "case_count": len(cases), "cases_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
               "spike_threshold_yuan_per_mwh": SPIKE_THRESHOLD, "cases": cases}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    counts = pd.Series([c["category"] for c in cases]).value_counts().sort_index().to_dict()
    print(f"wrote {path} ({len(cases)} cases) {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
