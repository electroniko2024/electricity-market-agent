"""Build the frozen market-agent evaluation sets with independently computed references.

References never go through the agent's tools: prices come from direct SQL on
`market_15min`, forecasts from the saved snapshot prediction files, and policy answers from
clauses located in the parsed source text. Run once, review, then freeze (the file records
its own SHA-256); do not regenerate after seeing system answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "market_processed"
OUT = Path(__file__).resolve().parent / "cases"
SPIKE_THRESHOLD = json.loads((DATA / "model_run_2025-10-01" / "models" / "tree_models.json").read_text(encoding="utf-8"))[
    "spike_threshold_yuan_per_mwh"
]
MODELS = ["LightGBM", "XGBoost", "PatchTST"]
ANALYSIS_TOOLS = ["query_price", "query_stats", "query_load", "query_weather", "detect_anomaly", "explain_drivers", "search_policy"]
FORECAST_TOOLS = ANALYSIS_TOOLS + ["forecast_price"]


def d(text: str) -> date:
    return date.fromisoformat(text)


def as_of_after(day: date, days: int = 1) -> str:
    return f"{(day + timedelta(days=days)).isoformat()}T00:00:00+08:00"


def cn(day: date) -> str:
    return f"{day.year}年{day.month}月{day.day}日"


class Refs:
    def __init__(self) -> None:
        self.con = duckdb.connect(str(DATA / "sd_market.duckdb"), read_only=True)
        self.fixed = pd.read_parquet(DATA / "model_run_2025-10-01" / "fixed_test_predictions.parquet")
        self.rolling = pd.read_parquet(DATA / "model_run_rolling_2025-10-01" / "rolling_test_predictions.parquet")

    def one(self, sql: str, params: list) -> tuple:
        return self.con.execute(sql, params).fetchone()

    def range_stats(self, start: date, end: date) -> dict:
        n, valid, mn, mx, mean, neg = self.one(
            "SELECT count(*), count(rt_price), min(rt_price), max(rt_price), avg(rt_price),"
            " sum(CASE WHEN rt_price < 0 THEN 1 ELSE 0 END) FROM market_15min WHERE trade_date BETWEEN ? AND ?",
            [start, end],
        )
        days = (end - start).days + 1
        return {"rows": n, "valid": valid, "expected_slots": days * 96, "missing": days * 96 - valid,
                "min": mn, "max": mx, "mean": mean, "negative": int(neg), "negative_share": neg / valid}

    def lowest_slot_mean(self, start: date, end: date) -> dict:
        slot, label, mean = self.one(
            "SELECT slot_id, any_value(time_label), avg(rt_price) m FROM market_15min WHERE trade_date BETWEEN ? AND ?"
            " GROUP BY slot_id ORDER BY m ASC LIMIT 1", [start, end])
        return {"slot_id": int(slot), "time_label": label, "mean": mean}

    def highest_daily_mean(self, start: date, end: date) -> dict:
        day, mean = self.one(
            "SELECT trade_date, avg(rt_price) m FROM market_15min WHERE trade_date BETWEEN ? AND ?"
            " GROUP BY trade_date ORDER BY m DESC LIMIT 1", [start, end])
        return {"trade_date": day.isoformat(), "mean": mean}

    def forecast(self, target: date) -> dict:
        out = {}
        for name, frame in (("fixed", self.fixed), ("rolling", self.rolling)):
            rows = frame.loc[pd.to_datetime(frame["trade_date"]).dt.date == target]
            if len(rows) != 96:
                raise ValueError(f"{name} forecast missing for {target}")
            spread = rows[MODELS].std(axis=1, ddof=0)
            widest = rows.loc[spread.idxmax()]
            out[name] = {
                **{f"{model}_min": float(rows[model].min()) for model in MODELS},
                **{f"{model}_max": float(rows[model].max()) for model in MODELS},
                "ensemble_min": float(rows[MODELS].mean(axis=1).min()),
                "any_model_negative": bool((rows[MODELS] < 0).any().any()),
                "max_std_slot": int(widest["slot_id"]),
                "max_std_time_label": str(widest["time_label"]),
            }
        return out


def number(name: str, values: list[float], tolerance: float = 0.01, *, relative: bool = False) -> dict:
    return {"name": name, "any_of": [round(float(value), 6) for value in values], "tolerance": tolerance,
            "relative": relative}


def window(start: date, end: date, lookback_days: int = 0) -> dict:
    return {"start_date": (start - timedelta(days=lookback_days)).isoformat(), "end_date": end.isoformat()}


# ----------------------------------------------------------------------------- policy clauses

def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


class PolicyClauses:
    def __init__(self) -> None:
        self.chunks = json.loads((DATA / "policy_chunks.json").read_text(encoding="utf-8"))
        self.meta = {doc["relative_path"]: doc for doc in json.loads(
            (DATA / "policy_metadata.json").read_text(encoding="utf-8"))["documents"]}

    def locate(self, relative_path: str, keywords: list[str]) -> dict:
        hits = [c for c in self.chunks if c["relative_path"] == relative_path
                and "MinerU" not in c["text"] and all(_norm(k) in _norm(c["text"]) for k in keywords)]
        if not hits:
            raise ValueError(f"clause not found in {relative_path}: {keywords}")
        chunk = hits[0]
        return {"relative_path": relative_path, "keywords": keywords, "page": chunk.get("page"),
                "locator": chunk["locator"], "chunk_id": chunk["chunk_id"],
                "document_title": self.meta[relative_path]["title"]}


RULES_2024 = "2024年山东电力市场规则.pdf"
RULES_2025 = "2025年山东电力市场规则（试行）.pdf"
GRID_2023 = "2023年山东省电力并网运行管理实施细则.pdf"
GRID_2025 = "附件1：山东省电力并网运行管理实施细则.pdf"
ANC_2023 = "2023年山东省电力辅助服务管理实施细则.pdf"
ANC_2025 = "附件2：山东省电力辅助服务管理实施细则.pdf"

POLICY_TOPICS = {
    "rules_effective": ("当日适用的山东电力市场规则是哪一版，自何时起施行、有效期多久？", {
        RULES_2024: ["自2024年5月1日起施行"], RULES_2025: ["自2026年1月1日起施行"]}),
    "spot_price_limits": ("现货市场如何设置报价限价和出清限价？", {
        RULES_2024: ["现货市场设定报价限价和出清限价"], RULES_2025: ["现货市场设定报价限价和出清限价"]}),
    "secondary_cap_trigger": ("现货市场在什么条件下执行二级出清限价？", {
        RULES_2024: ["平均结算均价连续7天超过现货市场申报上限"], RULES_2025: ["统一结算点电价连续7天超过现货市场申报上限"]}),
    "price_control": ("市场出清节点电价超过出清价格上限或低于下限时如何处理？", {
        RULES_2024: ["节点电价用出清价格上限代替"], RULES_2025: ["节点电价用出清价格上限代替"]}),
    "mlt_price_limits": ("中长期交易申报价格的上、下限如何设置？", {
        RULES_2024: ["第3.5.4条中长期交易申报价格设置上、下限"], RULES_2025: ["第3.5.4条中长期交易申报价格设置上、下限"]}),
    "grid_rules": ("当日适用的山东省电力并网运行管理实施细则是哪一版，何时施行？", {
        GRID_2023: ["本细则自发布之日起施行"], GRID_2025: ["本细则自2026年1月1日起施行"]}),
    "ancillary_rules": ("当日适用的山东省电力辅助服务管理实施细则是哪一版，何时施行？", {
        ANC_2023: ["本细则自发布之日起施行"], ANC_2025: ["本细则自2026年1月1日起施行"]}),
}


def applicable(topic: str, policy_at: date) -> str:
    docs = list(POLICY_TOPICS[topic][1])
    return docs[0] if policy_at < date(2026, 1, 1) else docs[1]


# ----------------------------------------------------------------------------- case builders

def base(case_id: str, category: str, question: str, mode: str, as_of: str, **extra) -> dict:
    return {"id": case_id, "category": category, "question": question, "mode": mode, "as_of": as_of,
            "required_numbers": [], "key_points": [], "needs_policy": False, "required_policy": [], **extra}


def case_a(refs: Refs, case_id: str, kind: str, start: date, end: date) -> dict:
    stats = refs.range_stats(start, end)
    period = cn(start) if start == end else f"{cn(start)}至{cn(end)}"
    win = window(start, end)
    common = dict(target_window=win, allowed_tools=ANALYSIS_TOOLS, arg_window=win)
    if kind == "neg_share":
        question = f"{period}山东实时电价出现负电价的时段有多少个？占有效时段的比例是多少？"
        nums = [number("negative_count", [stats["negative"]], 0.5),
                number("negative_share", [stats["negative_share"]], 0.0005)]
        groups = [["query_stats", "query_price", "detect_anomaly"]]
    elif kind == "range_stats":
        question = f"{period}山东实时电价的均值、最高价和最低价分别是多少？"
        nums = [number("mean", [stats["mean"]], 0.01), number("max", [stats["max"]], 0.01),
                number("min", [stats["min"]], 0.01)]
        groups = [["query_stats", "query_price"]]
    elif kind == "missing":
        question = f"{period}山东实时电价数据有多少个缺失时段？有效时段有多少个？"
        nums = [number("missing", [stats["missing"]], 0.5), number("valid", [stats["valid"]], 0.5)]
        groups = [["query_stats", "query_price", "detect_anomaly"]]
    elif kind == "lowest_slot":
        slot = refs.lowest_slot_mean(start, end)
        question = f"{period}按日内时段统计，哪个时段的平均实时电价最低？该时段均价是多少？"
        nums = [number("lowest_slot_mean", [slot["mean"]], 0.01)]
        groups = [["query_stats"]]
        common["key_points"] = [{"id": "names_lowest_slot", "type": "text_contains_any",
                                 "values": [slot["time_label"], f"slot {slot['slot_id']}", f"slot_id={slot['slot_id']}"]}]
    elif kind == "highest_day":
        day = refs.highest_daily_mean(start, end)
        question = f"{period}中哪一天的日均实时电价最高？日均价是多少？"
        nums = [number("highest_daily_mean", [day["mean"]], 0.01)]
        groups = [["query_stats"]]
        top = d(day["trade_date"])
        common["key_points"] = [{"id": "names_highest_day", "type": "text_contains_any",
                                 "values": [day["trade_date"], f"{top.month}月{top.day}日"]}]
    else:
        raise ValueError(kind)
    return base(case_id, "A", question, "historical_analysis", as_of_after(end), required_numbers=nums,
                required_tool_groups=groups, reference={"stats": stats}, **common)


def case_b(refs: Refs, case_id: str, target: date) -> dict:
    forecast = refs.forecast(target)
    actual = refs.range_stats(target, target)
    minimums = [forecast[snap][f"{model}_min"] for snap in ("fixed", "rolling") for model in MODELS]
    question = (f"以{cn(target - timedelta(days=1))}24:00为信息截止，研判{cn(target)}山东实时电价："
                "三个模型预测的最低价分别是多少？是否存在负电价风险？模型分歧最大的是哪个时段？")
    return base(case_id, "B", question, "historical_forecast", as_of_after(target, 0),
                target_date=target.isoformat(), allowed_tools=FORECAST_TOOLS,
                arg_window=window(target, target, lookback_days=28), required_tool_groups=[["forecast_price"]],
                forecast_target=target.isoformat(),
                required_numbers=[number("some_model_min", minimums, 0.01)],
                reference={"forecast": forecast},
                posterior={"actual_min": actual["min"], "actual_negative": actual["negative"] > 0})


def case_c(refs: Refs, case_id: str, day: date, kind: str) -> dict:
    stats = refs.range_stats(day, day)
    label = "负电价" if kind == "negative" else "价格尖峰"
    question = f"解释{cn(day)}山东实时电价出现{label}的可能原因，并给出关键数值证据。"
    extreme = number("min_price" if kind == "negative" else "max_price",
                     [stats["min"] if kind == "negative" else stats["max"]], 0.01)
    return base(case_id, "C", question, "historical_analysis", as_of_after(day), allowed_tools=ANALYSIS_TOOLS,
                arg_window=window(day, day, lookback_days=7), target_window=window(day, day),
                required_tool_groups=[["detect_anomaly"], ["explain_drivers"]], required_numbers=[extreme],
                key_points=[
                    {"id": "reports_extreme_price", "type": "number", "number": extreme["name"]},
                    {"id": "names_the_day", "type": "text_contains_any", "values": [day.isoformat(), f"{day.month}月{day.day}日"]},
                    {"id": "cause_backed_by_drivers", "type": "cause_cites_tool", "tool": "explain_drivers"},
                    {"id": "cause_names_supply_component", "type": "cause_mentions_any",
                     "values": ["负荷", "风电", "光伏", "联络线", "外来电", "核电", "自备", "地方电源", "竞价空间", "供给空间"]},
                    {"id": "discloses_limitations", "type": "has_limitations"},
                ],
                reference={"stats": stats, "spike_threshold": SPIKE_THRESHOLD, "kind": kind})


def case_d(clauses: PolicyClauses, case_id: str, topic: str, policy_at: date, as_of_day: date) -> dict:
    prompt, docs = POLICY_TOPICS[topic]
    path = applicable(topic, policy_at)
    required = clauses.locate(path, docs[path])
    wrong = [p for p in docs if p != path]
    return base(case_id, "D", f"{cn(policy_at)}，{prompt}请给出原文依据。", "historical_analysis",
                as_of_after(as_of_day), allowed_tools=ANALYSIS_TOOLS, required_tool_groups=[["search_policy"]],
                policy_at=policy_at.isoformat(), needs_policy=True, required_policy=[required],
                wrong_version_paths=wrong,
                key_points=[{"id": "cites_applicable_version", "type": "policy_required"}])


def case_e(refs: Refs, clauses: PolicyClauses, case_id: str, end: date, with_policy: bool) -> dict:
    start = end - timedelta(days=6)
    target = end + timedelta(days=1)
    stats = refs.range_stats(start, end)
    forecast = refs.forecast(target)
    minimums = [forecast[snap][f"{model}_min"] for snap in ("fixed", "rolling") for model in MODELS]
    question = (f"请生成{cn(start)}至{cn(end)}的山东实时电价周报，说明异常及可能原因，"
                f"并给出以{cn(end)}24:00为截止的{cn(target)}D+1研判。")
    extra = {}
    key_points = [
        {"id": "reports_week_min", "type": "number", "number": "week_min"},
        {"id": "has_anomaly_finding", "type": "has_section", "section": "anomalies"},
        {"id": "cause_backed_by_drivers", "type": "cause_cites_tool", "tool": "explain_drivers"},
        {"id": "outlook_backed_by_forecast", "type": "outlook_cites_tool", "tool": "forecast_price"},
        {"id": "discloses_limitations", "type": "has_limitations"},
    ]
    groups = [["query_stats", "query_price", "detect_anomaly"], ["forecast_price"]]
    if with_policy:
        question += "同时说明该时期出清价格超过上限或低于下限时的处理规则依据。"
        path = applicable("price_control", end)
        extra = {"needs_policy": True, "policy_at": end.isoformat(),
                 "required_policy": [clauses.locate(path, POLICY_TOPICS["price_control"][1][path])],
                 "wrong_version_paths": [p for p in POLICY_TOPICS["price_control"][1] if p != path]}
        key_points.append({"id": "cites_applicable_version", "type": "policy_required"})
        groups.append(["search_policy"])
    return base(case_id, "E", question, "historical_forecast", as_of_after(end), allowed_tools=FORECAST_TOOLS,
                arg_window=window(start, target, lookback_days=21), target_window=window(start, end),
                forecast_target=target.isoformat(), required_tool_groups=groups, key_points=key_points,
                required_numbers=[number("week_min", [stats["min"]], 0.01), number("some_model_min", minimums, 0.01)],
                reference={"stats": stats, "forecast": forecast}, **extra)


def formal(refs: Refs, clauses: PolicyClauses) -> list[dict]:
    cases = []
    a_specs = (
        [("neg_share", x, x) for x in ["2025-10-14", "2025-10-29", "2025-11-16", "2025-12-11", "2025-12-25", "2026-01-09"]]
        + [("range_stats", s, e) for s, e in [("2025-10-06", "2025-10-12"), ("2025-10-20", "2025-10-26"),
                                              ("2025-11-03", "2025-11-09"), ("2025-11-24", "2025-11-30"),
                                              ("2025-12-08", "2025-12-14"), ("2026-01-05", "2026-01-11")]]
        + [("neg_share", s, e) for s, e in [("2025-10-01", "2025-10-31"), ("2025-11-01", "2025-11-30"),
                                            ("2025-12-01", "2025-12-31")]]
        + [("missing", "2025-12-19", "2025-12-21"), ("missing", "2025-12-20", "2025-12-20")]
        + [("lowest_slot", s, e) for s, e in [("2025-10-13", "2025-10-19"), ("2025-11-10", "2025-11-16"),
                                              ("2025-12-01", "2025-12-07"), ("2026-01-01", "2026-01-07")]]
        + [("highest_day", s, e) for s, e in [("2025-10-01", "2025-10-07"), ("2025-10-27", "2025-11-02"),
                                              ("2025-12-15", "2025-12-21"), ("2026-01-08", "2026-01-14")]]
    )
    for index, (kind, start, end) in enumerate(a_specs, 1):
        cases.append(case_a(refs, f"A{index:02d}", kind, d(start), d(end)))
    b_days = ["2025-10-02", "2025-10-09", "2025-10-16", "2025-10-22", "2025-10-29", "2025-11-04", "2025-11-09",
              "2025-11-13", "2025-11-17", "2025-11-21", "2025-11-26", "2025-12-02", "2025-12-06", "2025-12-10",
              "2025-12-16", "2025-12-23", "2025-12-28", "2026-01-03", "2026-01-07", "2026-01-12"]
    cases += [case_b(refs, f"B{i:02d}", d(x)) for i, x in enumerate(b_days, 1)]
    c_specs = [(x, "negative") for x in ["2025-10-18", "2025-10-21", "2025-10-27", "2025-11-02", "2025-11-05",
                                         "2025-11-12", "2025-11-19", "2025-11-24", "2025-12-03", "2025-12-18",
                                         "2025-12-26", "2026-01-01", "2026-01-10", "2026-01-12"]] + \
              [(x, "spike") for x in ["2025-10-01", "2025-10-04", "2025-10-08", "2025-10-15", "2025-10-30", "2025-11-08"]]
    cases += [case_c(refs, f"C{i:02d}", d(x), kind) for i, (x, kind) in enumerate(c_specs, 1)]
    d_specs = [("rules_effective", "2025-12-15", "2026-01-10"), ("rules_effective", "2026-01-10", "2026-01-12"),
               ("spot_price_limits", "2025-11-01", "2026-01-12"), ("spot_price_limits", "2026-01-08", "2026-01-12"),
               ("secondary_cap_trigger", "2025-10-20", "2026-01-12"), ("secondary_cap_trigger", "2026-01-12", "2026-01-13"),
               ("price_control", "2025-12-01", "2026-01-12"), ("price_control", "2026-01-06", "2026-01-12"),
               ("grid_rules", "2025-11-15", "2026-01-12"), ("grid_rules", "2026-01-05", "2026-01-12"),
               ("ancillary_rules", "2025-10-10", "2026-01-12"), ("ancillary_rules", "2026-01-09", "2026-01-12"),
               ("mlt_price_limits", "2025-12-20", "2026-01-12"), ("mlt_price_limits", "2026-01-02", "2026-01-12"),
               ("secondary_cap_trigger", "2025-12-30", "2026-01-12")]
    cases += [case_d(clauses, f"D{i:02d}", topic, d(at), d(asof)) for i, (topic, at, asof) in enumerate(d_specs, 1)]
    e_ends = ["2025-10-07", "2025-10-12", "2025-10-19", "2025-10-26", "2025-11-02", "2025-11-09", "2025-11-16",
              "2025-11-23", "2025-11-30", "2025-12-07", "2025-12-14", "2025-12-21", "2025-12-28", "2026-01-04",
              "2026-01-11", "2025-10-10", "2025-10-31", "2025-11-27", "2025-12-11", "2026-01-13"]
    policy_ends = {"2025-10-12", "2025-10-26", "2025-11-02", "2025-11-23", "2025-12-14", "2025-12-28", "2026-01-11", "2026-01-13"}
    cases += [case_e(refs, clauses, f"E{i:02d}", d(x), x in policy_ends) for i, x in enumerate(e_ends, 1)]
    return cases


def dev(refs: Refs, clauses: PolicyClauses) -> list[dict]:
    return [
        case_a(refs, "devA1", "neg_share", d("2025-10-03"), d("2025-10-03")),
        case_a(refs, "devA2", "range_stats", d("2025-12-22"), d("2025-12-28")),
        case_b(refs, "devB1", d("2025-10-05")),
        case_b(refs, "devB2", d("2026-01-14")),
        case_c(refs, "devC1", d("2025-11-01"), "negative"),
        case_c(refs, "devC2", d("2025-12-22"), "spike"),
        case_d(clauses, "devD1", "rules_effective", d("2025-10-05"), d("2025-10-08")),
        case_d(clauses, "devD2", "grid_rules", d("2026-01-03"), d("2026-01-05")),
        case_e(refs, clauses, "devE1", d("2025-10-05"), False),
        case_e(refs, clauses, "devE2", d("2025-12-31"), True),
    ]


def _write(path: Path, name: str, cases: list[dict]) -> None:
    body = json.dumps(cases, ensure_ascii=False, sort_keys=True)
    payload = {"set": name, "version": 1, "built_at": datetime.now().isoformat(timespec="seconds"),
               "case_count": len(cases), "cases_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
               "spike_threshold_yuan_per_mwh": SPIKE_THRESHOLD, "cases": cases}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    refs, clauses = Refs(), PolicyClauses()
    for name, builder in (("formal_v1", formal), ("dev_v1", dev)):
        path = OUT / f"{name}.json"
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists; frozen sets are not rebuilt without --overwrite")
        cases = builder(refs, clauses)
        _write(path, name, cases)
        counts = pd.Series([c["category"] for c in cases]).value_counts().sort_index().to_dict()
        print(name, len(cases), counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
