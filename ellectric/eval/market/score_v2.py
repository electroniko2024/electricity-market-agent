"""Stage-wise scoring (evaluation plan v2) on top of the frozen v1 ``CaseScorer``.

The v1 main metric (task pass) is unchanged. This module adds diagnostics that say *where* a run failed,
so that architecture effects (which act on evidence gathering) are not drowned by failures in the shared
report stage. Every run gets exactly one stage, checked in order; later checks are skipped once one fails:

1. ``aborted``          the run did not complete;
2. ``evidence``         when tool calls ended, a required tool had not returned, or the observations did not
                        contain the reference values (matched by value and, where the case records them, by
                        row keys and field) or the required policy clauses — independent of the report;
3. ``answerability``    an unanswerable (C) case was answered with numbers / without saying why, or an
                        answerable case was refused although the evidence was in hand;
4. ``self_check``       evidence was in hand but the agent's own report validation rejected the report;
5. ``under_answer``     the report passed validation but a required number is missing: ``empty`` (no
                        numbers at all), ``wrong_row`` (right field bound from another day or slot), or
                        ``wrong_or_missing``;
6. ``key_points``       numbers fine, other key points (dates, slots, citations, disclosures) incomplete;
7. ``pass``             v1 task pass, and every number with row keys bound from the right row.

Cases may carry ``answerability`` (A answerable, B partly, C unanswerable, D ambiguous, E version conflict)
and ``stratum`` (L1-L5); numbers may carry ``keys`` ({"date": ..., "slot": ...}), ``field`` (the result
column the value comes from) and ``evidence: false`` for values the agent must compute (sums, differences),
which are then not looked for in the observations.

Aggregation over repeated runs: per-case mean pass (pass@1), pass^k (all k runs pass), pass@k (any), and
paired comparisons between variants by case-level bootstrap with an exact sign test.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from scipy.stats import binomtest

from ellectric.eval.market.score import HERE, CaseScorer, _metadata, _norm

STAGES = ["aborted", "answerability", "evidence", "self_check", "under_answer", "key_points", "pass"]
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
REFUSAL_WORDS = ["无法", "不可得", "不可用", "尚未", "未覆盖", "超出", "无数据", "不支持", "未能", "不在数据"]
REFUSAL_POINTS = {"any_text_contains_any", "no_numbers", "no_policy_basis", "text_contains_any"}

# Answerability labels for the v1 sets, which predate the label (re-scoring only; the cases are not changed).
_V1_HOLDOUT = {**{c: "C" for c in ("XR01", "XR02", "XR03", "XR07", "XR08", "XR09", "XR10", "XP08")},
               "XR04": "B", "XR06": "D", **{c: "E" for c in ("XP01", "XP02", "XP07", "XP09", "XP10")}}
V1_OVERLAY = {
    "holdout_v1": {case_id: {"answerability": label} for case_id, label in _V1_HOLDOUT.items()},
    "formal_v1": {f"D{i:02d}": {"answerability": "E"} for i in range(1, 16)},
}


# ----------------------------------------------------------------------------- observation leaves

def _row_keys(node: dict) -> dict:
    out: dict[str, Any] = {}
    for name in ("trade_date", "target_date", "timestamp_raw"):
        value = node.get(name)
        if isinstance(value, str) and _DATE.match(value):
            out["date"] = value[:10]
    period = node.get("period")
    if isinstance(period, list) and len(period) == 2 and period[0] == period[1] and isinstance(period[0], str):
        out["date"] = period[0][:10]  # a single-day aggregate (query_stats over one day)
    group = node.get("group_value")
    if isinstance(group, str) and _DATE.match(group):
        out["date"] = group[:10]
    elif isinstance(group, int) and not isinstance(group, bool):
        out["slot"] = group
    if isinstance(node.get("slot_id"), int):
        out["slot"] = node["slot_id"]
    return out


def leaves(node: Any, ctx: dict | None = None, field: str = "") -> Iterator[tuple[str, float, dict]]:
    """Every numeric leaf of a tool result with its field name and the row keys of its enclosing objects."""
    ctx = ctx or {}
    if isinstance(node, dict):
        inner = ctx | _row_keys(node)
        for key, value in node.items():
            yield from leaves(value, inner, key)
    elif isinstance(node, list):
        for value in node:
            yield from leaves(value, ctx, field)
    elif isinstance(node, (int, float)) and not isinstance(node, bool) and node == node:
        yield field, float(node), ctx


def keys_at(document: Any, pointer: str) -> tuple[str, dict] | None:
    """Field name and row keys at a JSON pointer, or None when the pointer does not resolve."""
    ctx, current, field = {}, document, ""
    try:
        for raw in pointer.split("/")[1:]:
            if isinstance(current, dict):
                ctx |= _row_keys(current)
            key = raw.replace("~1", "/").replace("~0", "~")
            current = current[int(key)] if isinstance(current, list) else current[key]
            field = key if not key.isdigit() else field
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return field, ctx


def _close(spec: dict, value: float) -> bool:
    for reference in spec["any_of"]:
        tolerance = spec["tolerance"] * (abs(reference) if spec.get("relative") else 1)
        if math.isclose(value, reference, abs_tol=tolerance + 1e-9):
            return True
    return False


def _keys_ok(spec: dict, ctx: dict) -> bool:
    return all(ctx.get(name) == value for name, value in (spec.get("keys") or {}).items())


# ----------------------------------------------------------------------------- per-run stages

class StageScorer:
    def __init__(self, case: dict, metadata: dict, observable: set[str] | None = None) -> None:
        """``observable``: names of numbers to look for in observations when the case does not mark derived
        values itself (used when re-scoring v1 cases); None means every number not marked ``evidence: false``."""
        self.case = case
        self.v1 = CaseScorer(case, metadata)
        self.observable = observable

    def _evidence_numbers(self) -> list[dict]:
        specs = [n for n in self.case["required_numbers"] if n.get("evidence", True)]
        if self.observable is not None:
            specs = [n for n in specs if n["name"] in self.observable]
        return specs + list(self.case.get("evidence_values", []))

    def evidence(self, observations: list[dict], row: dict) -> dict:
        """Were the tools needed and the reference values in hand when tool calls ended? A value counts as in
        hand if it appears in an observation (with matching row keys and field when the case gives them), or,
        for values without row keys, if the report passed validation with that number right (the agent
        computed it from what it had, e.g. a weekly count summed from daily groups)."""
        ok = [o for o in observations if o["execution_status"] == "ok"]
        pool = [leaf for o in ok for leaf in leaves(o["result"])]
        numbers = {}
        for spec in self._evidence_numbers():
            seen = any(_close(spec, value) and _keys_ok(spec, ctx) and (not spec.get("field") or field == spec["field"])
                       for field, value, ctx in pool)
            # Keyed values must be seen in the right row; unkeyed ones may also have been computed by the agent.
            computed = not spec.get("keys") and row["validation"] == "passed" and row["numbers"].get(spec["name"], False) is True
            numbers[spec["name"]] = seen or computed
        items = [e for o in ok if o["tool_name"] in ("search_policy", "explain_drivers") for e in _policy_items(o["result"])]
        clauses = [any(
            e.get("relative_path") == required["relative_path"]
            and ((required.get("page") is not None and e.get("page") == required["page"])
                 or all(_norm(k) in _norm(e.get("excerpt", "")) for k in required["keywords"]))
            for e in items) for required in self.case.get("required_policy", [])]
        # Tool coverage ignores arguments (the value checks above judge whether the right rows came back).
        used = {o["tool_name"] for o in ok}
        groups = list(self.case.get("required_tool_groups", [])) + [s["tools"] for s in self.case.get("required_calls", [])]
        tools = [bool(set(group) & used) for group in groups]
        checked = list(numbers.values()) + clauses
        acquired = (all(checked) and all(tools)) if checked else None
        return {"numbers": numbers, "clauses": clauses, "tools": tools, "acquired": acquired}

    def attribution(self, report: dict, observations: list[dict]) -> dict:
        """For numbers with row keys: did the report bind the value from the right row? ``wrong_row`` lists
        numbers bound from the right field but another day/slot."""
        by_evidence = {o["evidence_id"]: o.get("result") for o in observations}
        bindings = [b for f in CaseScorer._findings(report) for b in f.get("value_bindings", [])
                    if isinstance(b.get("value"), (int, float)) and not isinstance(b.get("value"), bool)]
        out = {"checked": 0, "correct": 0, "wrong_row": []}
        for spec in self.case["required_numbers"]:
            if not spec.get("keys"):
                continue
            out["checked"] += 1
            located = [(b, keys_at(by_evidence.get(b.get("evidence_id")), b.get("json_pointer", ""))) for b in bindings]
            located = [(b, loc) for b, loc in located if loc is not None]
            if any(_close(spec, float(b["value"])) and _keys_ok(spec, loc[1]) for b, loc in located):
                out["correct"] += 1
            elif spec.get("field") and any(loc[0] == spec["field"] and not _keys_ok(spec, loc[1]) for b, loc in located):
                out["wrong_row"].append(spec["name"])
        return out

    def answerability(self, row: dict, report: dict) -> tuple[bool, str | None]:
        """(ok, kind). C cases: refusal key points must hold, else 'false_answer'. Other cases: a report with no
        numbers (or, for policy-only cases, no citation) that says the answer is unavailable is a
        'false_refusal'; one with no numbers and no such statement is an empty answer, judged later."""
        label = self.case.get("answerability", "A")
        if label == "C":
            points = [p["id"] for p in self.case["key_points"] if p["type"] in REFUSAL_POINTS]
            ok = all(row["key_points"].get(p, True) for p in points)
            return ok, None if ok else "false_answer"
        empty = (not _report_numbers(report)) if self.case["required_numbers"] else             (self.case.get("needs_policy") and not report.get("policy_basis"))
        if empty and any(word in _report_text(report) for word in REFUSAL_WORDS):
            return False, "false_refusal"
        return True, None

    def score(self, record: dict) -> dict:
        row = self.v1.score(record)
        out = {"case_id": self.case["id"], "variant": record["variant"], "run": record["run_index"],
               "stratum": self.case.get("stratum", self.case["category"]),
               "answerability_label": self.case.get("answerability", "A"),
               "pass": bool(row["task_complete"]), "status": record["status"], "tokens": row.get("tokens"),
               "elapsed_ms": row.get("elapsed_ms"), "calls": len(row.get("calls", [])),
               "required_tools_ok": row.get("required_tools_ok", False), "fault_recovered": row.get("fault_recovered")}
        if record["status"] != "completed":
            return out | {"stage": "aborted", "evidence": None, "detail": record.get("error", "")[:200]}
        report, observations = record["report"], record["trace"].get("observations", [])
        evidence = self.evidence(observations, row)
        attribution = self.attribution(report, observations)
        answer_ok, answer_kind = self.answerability(row, report)
        # Provenance-aware grading: a keyed number counts only if bound from the right row (v1 checks the value).
        out["pass"] = out["pass"] and attribution["correct"] == attribution["checked"]
        row = row | {"task_complete": out["pass"]}
        out |= {"evidence": evidence["acquired"], "evidence_detail": evidence, "attribution": attribution,
                "answer_kind": answer_kind, "validation": row["validation"],
                "numbers_ok": all(row["numbers"].values()) if row["numbers"] else None}
        under = None
        if row["task_complete"]:
            stage = "pass"
        elif evidence["acquired"] is False:
            stage = "evidence"
        elif not answer_ok:
            stage = "answerability"
        elif row["validation"] != "passed":
            stage = "self_check"
        elif row["numbers"] and not all(row["numbers"].values()):
            stage = "under_answer"
            under = "empty" if not _report_numbers(report) else ("wrong_row" if attribution["wrong_row"] else "wrong_or_missing")
        else:
            stage = "key_points"
        return out | {"stage": stage, "under_answer_kind": under}


def _report_numbers(report: dict) -> list:
    return [b for f in CaseScorer._findings(report) for b in f.get("value_bindings", []) + f.get("derived_values", [])
            if isinstance(b.get("value"), (int, float)) and not isinstance(b.get("value"), bool) and b.get("value") != 0]


def _report_text(report: dict) -> str:
    parts = [report.get("summary", "")] + [f.get("statement", "") for f in CaseScorer._findings(report)]
    parts += [str(x) for key in ("data_notes", "known_limitations", "unresolved_issues") for x in report.get(key, [])]
    return " ".join(parts)


def _policy_items(result: Any) -> Iterator[dict]:
    """Policy evidence items anywhere in a result (search_policy, or policy evidence nested in explain_drivers)."""
    if isinstance(result, dict):
        if "relative_path" in result and "excerpt" in result:
            yield result
        for value in result.values():
            yield from _policy_items(value)
    elif isinstance(result, list):
        for value in result:
            yield from _policy_items(value)


# ----------------------------------------------------------------------------- aggregation

def load_records(results: Path) -> list[dict]:
    records = []
    for path in sorted(results.glob("E*/run*/*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def observable_numbers(cases: dict[str, dict], records: list[dict]) -> dict[str, set[str]]:
    """For cases that do not mark derived values: a number counts as observable if any run of any variant had
    it in its observations. Values never seen in any observation (sums, differences) are left out of the
    evidence check instead of counting as missing."""
    seen: dict[str, set[str]] = defaultdict(set)
    for record in records:
        case = cases.get(record["case_id"])
        if case is None or record["status"] != "completed":
            continue
        pool = [leaf for o in record["trace"].get("observations", []) if o["execution_status"] == "ok"
                for leaf in leaves(o["result"])]
        for spec in case["required_numbers"]:
            if any(_close(spec, value) and _keys_ok(spec, ctx) for _, value, ctx in pool):
                seen[case["id"]].add(spec["name"])
    return seen


def bootstrap_diff(a: dict[str, float], b: dict[str, float], iterations: int = 10000, seed: int = 7) -> dict:
    """Paired case-level bootstrap of mean(a - b) over the cases both variants ran, plus an exact sign test."""
    ids = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in ids]
    if not ids:
        return {"n": 0}
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iterations))
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {"n": len(ids), "diff": sum(diffs) / len(diffs), "ci95": (boot[int(0.025 * iterations)], boot[int(0.975 * iterations) - 1]),
            "a_better": wins, "b_better": losses,
            "sign_p": binomtest(wins, wins + losses).pvalue if wins + losses else 1.0}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _pct(x: float | None) -> str:
    return "—" if x is None or x != x else f"{x * 100:.1f}%"


def summarize(rows: list[dict], pairs: list[tuple[str, str]], group: str = "stratum") -> str:
    variants = sorted({r["variant"] for r in rows})
    runs = sorted({r["run"] for r in rows})
    lines = [f"运行次数：{len(runs)}（run {', '.join(map(str, runs))}）；记录 {len(rows)} 条。", ""]

    lines += ["## 主指标与稳定性", "",
              "| 变体 | 题数 | pass@1（均值） | 95% CI | pass^k | pass@k | 平均调用 | 平均 token |", "|---|---|---|---|---|---|---|---|"]
    per_case: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_case[r["variant"]][r["case_id"]].append(r["pass"])
    for v in variants:
        cases = per_case[v]
        k = sum(sum(x) for x in cases.values())
        n = sum(len(x) for x in cases.values())
        lo, hi = wilson(k, n)
        vr = [r for r in rows if r["variant"] == v and r["status"] == "completed"]
        tokens = [r["tokens"] for r in vr if r["tokens"]]
        lines.append(f"| {v} | {len(cases)} | {_pct(k / n)}（{k}/{n}） | {_pct(lo)}–{_pct(hi)} | "
                     f"{_pct(sum(all(x) for x in cases.values()) / len(cases))} | {_pct(sum(any(x) for x in cases.values()) / len(cases))} | "
                     f"{sum(r['calls'] for r in vr) / max(len(vr), 1):.1f} | {sum(tokens) / max(len(tokens), 1):.0f} |")
    lines += ["", "pass^k：同一题 k 次运行全部通过的题目比例；pass@k：至少一次通过。CI 为按运行计的 Wilson 区间。"]

    lines += ["", "## 失败阶段（每条记录只记最早失败的一个阶段）", "",
              "| 变体 | " + " | ".join(STAGES) + " |", "|---|" + "---|" * len(STAGES)]
    for v in variants:
        c = Counter(r["stage"] for r in rows if r["variant"] == v)
        lines.append(f"| {v} | " + " | ".join(str(c[s]) for s in STAGES) + " |")

    lines += ["", "## 诊断指标", "",
              "| 变体 | 证据到手率 | 证据到手后通过率 | 误答（C 类） | 误拒 | 自检拒绝 | 空结论 | 数值错或缺 | 取错行 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for v in variants:
        vr = [r for r in rows if r["variant"] == v and r["status"] == "completed"]
        ev = [r for r in vr if r["evidence"] is not None]
        got = [r for r in ev if r["evidence"]]
        c_rows = [r for r in vr if r["answerability_label"] == "C"]
        wrong = [r for r in vr if r["attribution"]["wrong_row"]]
        checked = sum(r["attribution"]["checked"] > 0 for r in vr)
        under = Counter(r["under_answer_kind"] for r in got if r["stage"] == "under_answer")
        n = len(got)
        lines.append(
            f"| {v} | {_pct(n / len(ev) if ev else None)}（{n}/{len(ev)}） | "
            f"{_pct(sum(r['pass'] for r in got) / n if n else None)}（{sum(r['pass'] for r in got)}/{n}） | "
            f"{sum(r['answer_kind'] == 'false_answer' for r in c_rows)}/{len(c_rows)} | "
            f"{sum(r['stage'] == 'answerability' for r in got)}/{n} | "
            f"{sum(r['stage'] == 'self_check' for r in got)}/{n} | "
            f"{under['empty']}/{n} | {under['wrong_or_missing'] + under['wrong_row']}/{n} | "
            f"{len(wrong)}/{checked} |")
    lines += ["", "证据到手率：工具阶段结束时所需工具都已成功返回、观测里已有参考数值（或所需政策条款）的记录比例，"
              "与报告写得如何无关；分母不含没有数值和条款要求的题（如拒答题）。误拒、自检拒绝、空结论、数值错或缺的分母都是证据已到手的记录，"
              "各列互斥（按失败阶段归类）。空结论：报告通过自检但一个数值都没给、也没说明无法回答。"
              "取错行：报告绑定了正确字段但来自别的日期或时段（仅对记录了行键的数值检查）。"]

    groups = sorted({r[group] for r in rows})
    lines += ["", f"## 按{'层' if group == 'stratum' else group}：pass@1 / 证据到手率", "",
              "| 分组 | 题数 | " + " | ".join(variants) + " |", "|---|---|" + "---|" * len(variants)]
    for g in groups:
        cells = []
        for v in variants:
            vr = [r for r in rows if r["variant"] == v and r[group] == g]
            ev = [r for r in vr if r["evidence"] is not None]
            cells.append(f"{_pct(sum(r['pass'] for r in vr) / len(vr) if vr else None)} / "
                         f"{_pct(sum(bool(r['evidence']) for r in ev) / len(ev) if ev else None)}")
        n_cases = len({r["case_id"] for r in rows if r[group] == g})
        lines.append(f"| {g} | {n_cases} | " + " | ".join(cells) + " |")

    if pairs:
        lines += ["", "## 配对比较（按题配对，题内先对多次运行取均值）", "",
                  "| 比较 | 分组 | 题数 | pass 差值 | 95% CI | 前者好/后者好 | 符号检验 p | 证据到手差值 | 95% CI |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for a, b in pairs:
            for g in ["全部"] + groups:
                sub = [r for r in rows if g == "全部" or r[group] == g]
                def mean_by_case(v, key):
                    acc: dict[str, list[float]] = defaultdict(list)
                    for r in sub:
                        if r["variant"] == v and (key == "pass" or r["evidence"] is not None):
                            acc[r["case_id"]].append(float(bool(r[key])))
                    return {c: sum(x) / len(x) for c, x in acc.items()}
                p = bootstrap_diff(mean_by_case(a, "pass"), mean_by_case(b, "pass"))
                e = bootstrap_diff(mean_by_case(a, "evidence"), mean_by_case(b, "evidence"))
                if not p.get("n"):
                    continue
                ecell = (f"{e['diff'] * 100:+.1f}pp | {e['ci95'][0] * 100:+.1f}～{e['ci95'][1] * 100:+.1f}"
                         if e.get("n") else "— | —")
                lines.append(f"| {a} vs {b} | {g} | {p['n']} | {p['diff'] * 100:+.1f}pp | "
                             f"{p['ci95'][0] * 100:+.1f}～{p['ci95'][1] * 100:+.1f} | {p['a_better']}/{p['b_better']} | "
                             f"{p['sign_p']:.3f} | {ecell} |")
    return "\n".join(lines) + "\n"


def score_set(set_name: str, results: Path | None = None, overlay: dict[str, dict] | None = None) -> tuple[list[dict], dict]:
    payload = json.loads((HERE / "cases" / f"{set_name}.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c | (overlay or {}).get(c["id"], {}) for c in payload["cases"]}
    records = [r for r in load_records(results or HERE / "results" / set_name) if r["case_id"] in cases]
    marked = any("evidence" in n for c in cases.values() for n in c["required_numbers"])
    observable = None if marked else observable_numbers(cases, records)
    metadata = _metadata()
    rows = []
    for record in records:
        if record["status"] == "network_error":
            continue
        case = cases[record["case_id"]]
        rows.append(StageScorer(case, metadata, None if observable is None else observable.get(case["id"], set())).score(record))
    return rows, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--pairs", nargs="*", default=["E0:E1", "E0:E2", "E1:E2"])
    parser.add_argument("--group", default="stratum", help="row field to stratify by (stratum or answerability_label)")
    args = parser.parse_args()
    rows, payload = score_set(args.set, args.results, V1_OVERLAY.get(args.set))
    results = args.results or HERE / "results" / args.set
    variants = {r["variant"] for r in rows}
    pairs = [tuple(p.split(":")) for p in args.pairs if set(p.split(":")) <= variants]
    text = f"# 分段评测（v2 判分，题集 {args.set}，sha256 {payload['cases_sha256'][:12]}）\n\n" + summarize(rows, pairs, args.group)
    (results / "summary_v2.md").write_text(text, encoding="utf-8")
    with gzip.open(results / "stages_v2.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, default=str)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
