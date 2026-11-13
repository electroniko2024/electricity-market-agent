"""Score evaluation runs with the frozen definitions; every metric carries numerator and denominator.

Main metrics (per variant and run, then mean and sample std over independent runs):
- tool_call_accuracy  = correct attempts / all attempts
- invalid_call_rate   = invalid attempts / all attempts
- task_completion     = completed C+E cases / C+E cases
- policy_citation     = D cases + policy-marked E cases passing the citation check / those cases

An attempt is any tool call the agent issued (blocked or failed calls stay in the denominator).
Invalid: rejected by schema or the as_of boundary, failed, skipped after a failed dependency,
a tool outside the case's allowed set, or a repeat of an identical successful call.
Correct: valid and its key arguments match the case (date windows, forecast target, policy date, and any
per-case ``arg_constraints``). A failure of a tool listed in the case's ``expected_error_tools`` (the agent
cannot know beforehand that the tool will refuse) is left out of both numerator and denominator.

Cases may also list ``required_calls``: calls that must be made correctly (a date range covered by the union
of correct calls, exact arguments, or a policy date inside a document's effective interval). They feed the
required-call recall and, with ``required_tool_groups``, decide whether the case's tools were complete.
``task_case`` marks cases that count toward task_completion (default: categories C and E).
Runs that died on network errors are listed separately and excluded from all metrics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "market_processed"
DATE_TOOLS = {"query_price", "query_stats", "query_load", "query_weather", "detect_anomaly", "explain_drivers"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _pointer(document: Any, pointer: str) -> Any:
    current = document
    for raw in pointer.split("/")[1:]:
        key = raw.replace("~1", "/").replace("~0", "~")
        current = current[int(key)] if isinstance(current, list) else current[key]
    return current


def _metadata() -> dict[str, dict]:
    payload = json.loads((DATA / "policy_metadata.json").read_text(encoding="utf-8"))
    return {doc["relative_path"]: doc for doc in payload["documents"]}


def _in_force(doc: dict, day: str) -> bool:
    start = doc.get("effective_from") or doc.get("effective_from_not_later_than")
    end = doc.get("effective_to")
    return bool(start) and start <= day and (end is None or day < end)


def _literal_dates(args: dict, as_of: Any) -> dict:
    """Date-range arguments as the tools read them: a month ("2025-08") is a valid bound and stands for its
    first day (start) or its last available day (end). Uses the tools' own expansion, so the judge and the
    tools cannot disagree about what a call covered."""
    from ellectric.service.market_tools import DateRangeInput

    bounds = {name: args[name] for name in ("start_date", "end_date") if name in args}
    if not bounds:
        return args
    expanded = DateRangeInput.expand_month_bounds(bounds | {"as_of": as_of})
    return args | {name: value.isoformat() for name, value in expanded.items()
                   if name in bounds and isinstance(value, date)}


def _days(args: dict) -> set[str]:
    try:
        start, end = date.fromisoformat(str(args.get("start_date"))), date.fromisoformat(str(args.get("end_date")))
    except ValueError:
        return set()
    return {date.fromordinal(n).isoformat() for n in range(start.toordinal(), end.toordinal() + 1)}


def _span_days(args: dict) -> int:
    return len(_days(args)) or 10**6


class CaseScorer:
    def __init__(self, case: dict, metadata: dict[str, dict]) -> None:
        self.case = case
        self.metadata = metadata

    # ------------------------------------------------------------------ tool calls
    def _args_ok(self, tool: str, args: dict) -> bool:
        for name, allowed in self.case.get("arg_constraints", {}).get(tool, {}).items():
            if name in args and str(args[name]) not in allowed:
                return False
        return self._key_args_ok(tool, args)

    def _key_args_ok(self, tool: str, args: dict) -> bool:
        case = self.case
        if tool in DATE_TOOLS:
            windows = case.get("arg_windows") or ([case["arg_window"]] if case.get("arg_window") else [])
            if not windows:
                return True
            start, end = str(args.get("start_date", "")), str(args.get("end_date", ""))
            if not any(w["start_date"] <= start <= end <= w["end_date"] for w in windows):
                return False
            target = case.get("target_window")
            return target is None or (start <= target["end_date"] and end >= target["start_date"])
        if tool == "forecast_price":
            return str(args.get("target_date")) == case.get("forecast_target")
        if tool == "search_policy":
            policy_at = str(args.get("policy_at", ""))
            if case.get("required_policy"):
                return any(_in_force(self.metadata.get(required["relative_path"], {}), policy_at)
                           for required in case["required_policy"])
            window = case.get("arg_window")
            return window is None or window["start_date"] <= policy_at <= window["end_date"]
        return True

    def classify_calls(self, observations: list[dict]) -> list[dict]:
        seen_ok: set[str] = set()
        out = []
        for item in observations:
            args = {k: v for k, v in (item.get("arguments") or {}).items() if k != "as_of"}
            if item["tool_name"] in DATE_TOOLS:
                args = _literal_dates(args, (item.get("arguments") or {}).get("as_of"))
            key = item["tool_name"] + json.dumps(args, sort_keys=True, ensure_ascii=False)
            status = item["execution_status"]
            reason = None
            if item["tool_name"] not in self.case["allowed_tools"]:
                reason = "disallowed_tool"
            elif status == "skipped_dependency_failed":
                reason = "dependency_failed"
            elif status != "ok" and item["tool_name"] in self.case.get("expected_error_tools", []):
                reason = "expected_error"
            elif status != "ok":
                error = item.get("error") or ""
                reason = "as_of_boundary" if "AFTER_AS_OF" in error else (
                    "schema_rejected" if error.startswith("ValidationError") else "tool_error")
            elif key in seen_ok:
                reason = "duplicate"
            if status == "ok":
                seen_ok.add(key)
            valid = reason is None
            out.append({"tool": item["tool_name"], "args": args, "invalid_reason": reason,
                        "correct": valid and self._args_ok(item["tool_name"], args)})
        return out

    def required_calls(self, calls: list[dict]) -> list[bool]:
        """One flag per ``required_calls`` spec: was it satisfied by correct calls?"""
        out = []
        for spec in self.case.get("required_calls", []):
            pool = [c for c in calls if c["correct"] and c["tool"] in spec["tools"]]
            for name, value in spec.get("args", {}).items():
                pool = [c for c in pool if str(c["args"].get(name)) == str(value)]
            if "policy_in_force" in spec:
                doc = self.metadata.get(spec["policy_in_force"], {})
                pool = [c for c in pool if _in_force(doc, str(c["args"].get("policy_at", "")))]
            if "max_days" in spec:
                pool = [c for c in pool if _span_days(c["args"]) <= spec["max_days"]]
            if "cover" in spec:
                covered: set[str] = set()
                for c in pool:
                    covered |= _days(c["args"])
                out.append(_days(spec["cover"]) <= covered)
            else:
                out.append(bool(pool))
        return out

    # ------------------------------------------------------------------ report content
    @staticmethod
    def _findings(report: dict) -> list[dict]:
        # Direct answers (reports written since the answers section exists) count as findings; the reading an
        # answer states in ``basis`` is part of its text. Older reports have no answers and score as before.
        findings = [answer | {"statement": " ".join(filter(None, [answer.get("statement"), answer.get("basis")]))}
                    for answer in report.get("answers", [])]
        findings += list(report.get("anomalies", [])) + list(report.get("causes", []))
        if report.get("outlook"):
            findings.append(report["outlook"])
        return findings

    def _numbers(self, report: dict) -> dict[str, bool]:
        # Derived values (sums, changes, shares) count as reported numbers alongside direct bindings.
        bound = [b["value"] for f in self._findings(report)
                 for b in f.get("value_bindings", []) + f.get("derived_values", [])
                 if isinstance(b.get("value"), (int, float)) and not isinstance(b.get("value"), bool)]
        result = {}
        for spec in self.case["required_numbers"]:
            def match(value: float) -> bool:
                for reference in spec["any_of"]:
                    tolerance = spec["tolerance"] * (abs(reference) if spec.get("relative") else 1)
                    if math.isclose(value, reference, abs_tol=tolerance + 1e-9):
                        return True
                return False
            result[spec["name"]] = any(match(float(value)) for value in bound)
        return result

    def _policy_ok(self, report: dict, observations: list[dict]) -> bool:
        by_evidence = {o["evidence_id"]: o for o in observations}
        cited = []
        for citation in report.get("policy_basis", []):
            observation = by_evidence.get(citation.get("evidence_id"))
            try:
                candidate = _pointer((observation or {}).get("result"), citation.get("json_pointer", ""))
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if isinstance(candidate, dict):
                cited.append(candidate)
        if any(c.get("relative_path") in self.case.get("wrong_version_paths", []) for c in cited):
            return False
        for required in self.case["required_policy"]:
            if not any(
                c.get("relative_path") == required["relative_path"]
                and c.get("evidence_eligible") is True
                and c.get("locator")
                # Page numbers only identify a clause in paginated sources; .docx chunks have page None.
                and ((required["page"] is not None and c.get("page") == required["page"]) or all(_norm(k) in _norm(c.get("excerpt", "")) for k in required["keywords"]))
                for c in cited
            ):
                return False
        return True

    def _key_point(self, point: dict, report: dict, observations: list[dict], numbers: dict, policy_ok: bool) -> bool:
        text = " ".join([report.get("summary", "")] + [f.get("statement", "") for f in self._findings(report)])
        tools_by_evidence = {o["evidence_id"]: o["tool_name"] for o in observations if o["execution_status"] == "ok"}
        kind = point["type"]
        if kind == "number":
            return numbers.get(point["number"], False)
        if kind == "text_contains_any":
            return any(value in text for value in point["values"])
        if kind == "cause_cites_tool":
            return any(tools_by_evidence.get(e) == point["tool"] for f in report.get("causes", []) for e in f.get("evidence_ids", []))
        if kind == "cause_mentions_any":
            return any(value in f.get("statement", "") for f in report.get("causes", []) for value in point["values"])
        if kind == "has_limitations":
            return bool(report.get("known_limitations") or report.get("data_notes")
                        or any(f.get("limitations") for f in self._findings(report)))
        if kind == "has_section":
            return bool(report.get(point["section"]))
        if kind == "outlook_cites_tool":
            outlook = report.get("outlook") or {}
            return any(tools_by_evidence.get(e) == point["tool"] for e in outlook.get("evidence_ids", []))
        if kind == "policy_required":
            return policy_ok
        if kind == "any_text_contains_any":
            # Refusals and disclosures may sit in notes, limitations or unresolved issues, not only findings.
            extra = [report.get("outlook") or {}] + list(report.get("policy_basis", []))
            pieces = [text] + [e.get("statement", "") for e in extra] + [
                str(item) for key in ("data_notes", "known_limitations", "unresolved_issues") for item in report.get(key, [])
            ] + [str(item) for f in self._findings(report) for item in f.get("limitations") or []]
            joined = " ".join(pieces)
            return any(value in joined for value in point["values"])
        if kind == "no_policy_basis":
            return not report.get("policy_basis")
        if kind == "no_numbers":
            # Zero counts ("0 rows returned") document an empty result; they are not an invented answer.
            return not any(isinstance(b.get("value"), (int, float)) and not isinstance(b.get("value"), bool)
                           and b.get("value") != 0
                           for f in self._findings(report)
                           for b in f.get("value_bindings", []) + f.get("derived_values", []))
        raise ValueError(f"unknown key point type {kind}")

    def score(self, record: dict) -> dict:
        case = self.case
        base = {"case_id": case["id"], "category": case["category"], "variant": record["variant"],
                "run_index": record["run_index"], "status": record["status"],
                "task_case": case.get("task_case", case["category"] in {"C", "E"})}
        if record["status"] != "completed":
            # A run error is a failed task with no recorded tool calls; network errors are excluded later.
            return base | {"calls": [], "required_calls": [False] * len(case.get("required_calls", [])),
                           "numbers": {}, "key_points": {}, "validation": "not_run",
                           "policy_ok": False, "task_complete": False, "tokens": None, "elapsed_ms": None,
                           "verified": False, "replans": 0} | self._fault_outcome(record, [])
        report, trace = record["report"], record["trace"]
        observations = trace.get("observations", [])
        calls = self.classify_calls(observations)
        numbers = self._numbers(report)
        validation = (report.get("validation") or {}).get("status")
        policy_ok = bool(case.get("needs_policy")) and validation == "passed" and self._policy_ok(report, observations)
        key_points = {p["id"]: self._key_point(p, report, observations, numbers, policy_ok) for p in case["key_points"]}
        required = self.required_calls(calls)
        groups_ok = all(any(c["correct"] and c["tool"] in group for c in calls)
                        for group in case.get("required_tool_groups", [])) and all(required)
        coverage = sum(key_points.values()) / len(key_points) if key_points else 1.0
        task_complete = (validation == "passed" and groups_ok and all(numbers.values()) and coverage >= 0.8)
        return base | {
            "calls": calls, "required_calls": required, "numbers": numbers, "key_points": key_points, "key_point_coverage": coverage,
            "required_tools_ok": groups_ok, "validation": validation, "policy_ok": policy_ok,
            "task_complete": task_complete, "tokens": trace.get("token_count"), "elapsed_ms": trace.get("elapsed_ms"),
            "verified": bool(report.get("verified")), "replans": trace.get("replan_count", 0),
            "unresolved": len(report.get("unresolved_issues", [])),
        } | self._fault_outcome(record, observations)

    def _fault_outcome(self, record: dict, observations: list[dict]) -> dict:
        """For fault-injection cases: did the injected failure fire, and did a later call of that tool succeed?"""
        faults = self.case.get("faults") or []
        if not faults:
            return {}
        log = record.get("fault_log") or []
        tools = {fault["tool"] for fault in faults}
        fired = all(any(entry["tool"] == tool and entry["injected"] for entry in log) for tool in tools)
        recovered = fired and all(
            any(o["tool_name"] == tool and o["execution_status"] == "ok" for o in observations) for tool in tools
        )
        return {"fault_fired": fired, "fault_recovered": recovered}


def _ratio(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _scored_calls(row: dict) -> list[dict]:
    return [c for c in row["calls"] if c["invalid_reason"] != "expected_error"]


def run_metrics(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["status"] != "network_error"]
    calls = [c for r in scored for c in _scored_calls(r)]
    complex_rows = [r for r in scored if r.get("task_case", r["category"] in {"C", "E"})]
    specs = [flag for r in scored for flag in r.get("required_calls", [])]
    policy_rows = [r for r in scored if r.get("needs_policy")]
    reasons = defaultdict(int)
    for call in calls:
        if call["invalid_reason"]:
            reasons[call["invalid_reason"]] += 1
    completed = [r for r in scored if r["status"] == "completed"]
    return {
        "tool_call_accuracy": _ratio(sum(c["correct"] for c in calls), len(calls)),
        "invalid_call_rate": _ratio(sum(bool(c["invalid_reason"]) for c in calls), len(calls)),
        "task_completion": _ratio(sum(r["task_complete"] for r in complex_rows), len(complex_rows)),
        "policy_citation": _ratio(sum(r["policy_ok"] for r in policy_rows), len(policy_rows)),
        "aux": {
            "cases_scored": len(scored),
            "network_errors_excluded": len(rows) - len(scored),
            "run_errors": sum(r["status"] == "run_error" for r in scored),
            "invalid_reasons": dict(sorted(reasons.items())),
            "report_validation_pass": _ratio(sum(r["validation"] == "passed" for r in scored), len(scored)),
            "verified": _ratio(sum(r["verified"] for r in scored), len(scored)),
            "required_numbers_pass": _ratio(sum(bool(r["numbers"]) and all(r["numbers"].values()) for r in scored if r["numbers"]),
                                            sum(1 for r in scored if r["numbers"])),
            "mean_tool_calls": statistics.fmean([len(r["calls"]) for r in completed]) if completed else None,
            "mean_tokens": statistics.fmean([r["tokens"] for r in completed if r["tokens"] is not None]) if completed else None,
            "mean_elapsed_s": statistics.fmean([r["elapsed_ms"] / 1000 for r in completed if r["elapsed_ms"]]) if completed else None,
            "replan_runs": sum(r["replans"] > 0 for r in completed),
            **({
                "fault_fired": _ratio(sum(r["fault_fired"] for r in scored), len(scored)),
                "fault_recovery": _ratio(sum(r["fault_recovered"] for r in scored if r["fault_fired"]),
                                         sum(r["fault_fired"] for r in scored)),
                "fault_task_completion": _ratio(sum(r["task_complete"] for r in scored if r["fault_fired"]),
                                                sum(r["fault_fired"] for r in scored)),
            } if any("fault_fired" in r for r in scored) else {}),
            **({
                "required_call_recall": _ratio(sum(specs), len(specs)),
                "case_tool_perfect": _ratio(sum(
                    r["status"] == "completed" and r.get("required_tools_ok", False)
                    and all(c["correct"] for c in _scored_calls(r)) for r in scored), len(scored)),
                "expected_errors_excluded": sum(c["invalid_reason"] == "expected_error" for r in scored for c in r["calls"]),
            } if specs else {}),
            "by_category_task_like": {
                cat: _ratio(sum(r["validation"] == "passed" and all(r["numbers"].values()) and r.get("key_point_coverage", 0) >= 0.8
                                and r.get("required_tools_ok", False) for r in scored if r["category"] == cat),
                            sum(1 for r in scored if r["category"] == cat))
                for cat in sorted({r["category"] for r in scored})
            },
        },
    }


def summarize(per_run: dict[tuple[str, int], dict]) -> dict:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for (variant, _run), metrics in sorted(per_run.items()):
        by_variant[variant].append(metrics)
    out = {}
    for variant, runs in by_variant.items():
        entry = {}
        for name in ("tool_call_accuracy", "invalid_call_rate", "task_completion", "policy_citation"):
            values = [m[name]["value"] for m in runs if m[name]["value"] is not None]
            entry[name] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "runs": [{k: m[name][k] for k in ("numerator", "denominator", "value")} for m in runs],
                "pooled": _ratio(sum(m[name]["numerator"] for m in runs), sum(m[name]["denominator"] for m in runs)),
            }
        entry["aux_by_run"] = [m["aux"] for m in runs]
        out[variant] = entry
    return out


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def markdown(summary: dict, cases_sha: str) -> str:
    lines = [f"# 评测结果（题集 sha256 {cases_sha[:12]}）", "",
             "| 变体 | 工具调用准确率 | 无效调用率 | 复杂任务完成率 | 政策引用完整率 | 运行次数 |",
             "|---|---|---|---|---|---|"]
    for variant, entry in sorted(summary.items()):
        cells = []
        for name in ("tool_call_accuracy", "invalid_call_rate", "task_completion", "policy_citation"):
            metric = entry[name]
            std = f" ± {metric['std'] * 100:.1f}" if metric["std"] is not None else ""
            pooled = metric["pooled"]
            cells.append(f"{_pct(metric['mean'])}{std}（{pooled['numerator']}/{pooled['denominator']}）")
        lines.append(f"| {variant} | " + " | ".join(cells) + f" | {len(entry['aux_by_run'])} |")
    fault_rows = [(variant, entry["aux_by_run"]) for variant, entry in sorted(summary.items())
                  if "fault_recovery" in entry["aux_by_run"][0]]
    if fault_rows:
        lines += ["", "## 故障注入", "", "| 变体 | 故障触发 | 故障后补救成功 | 故障题完成率 |", "|---|---|---|---|"]
        for variant, runs in fault_rows:
            cells = []
            for name in ("fault_fired", "fault_recovery", "fault_task_completion"):
                numerator = sum(run[name]["numerator"] for run in runs)
                denominator = sum(run[name]["denominator"] for run in runs)
                cells.append(f"{_pct(numerator / denominator if denominator else None)}（{numerator}/{denominator}）")
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    detail = [(variant, entry["aux_by_run"]) for variant, entry in sorted(summary.items())
              if "required_call_recall" in entry["aux_by_run"][0]]
    if detail:
        def pooled(runs, pick):
            numerator = sum(pick(run)["numerator"] for run in runs)
            denominator = sum(pick(run)["denominator"] for run in runs)
            return numerator, denominator
        lines += ["", "## 工具调用细分", "", "| 变体 | 必需调用召回 | 按题全对（调用全正确且必需调用齐全） |", "|---|---|---|"]
        for variant, runs in detail:
            cells = []
            for name in ("required_call_recall", "case_tool_perfect"):
                n, d = pooled(runs, lambda run: run[name])
                cells.append(f"{_pct(n / d if d else None)}（{n}/{d}）")
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
        cats = sorted(detail[0][1][0]["by_category_task_like"])
        lines += ["", "## 分题型通过", "", "| 变体 | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
        for variant, runs in detail:
            cells = []
            for cat in cats:
                n, d = pooled(runs, lambda run: run["by_category_task_like"].get(cat, {"numerator": 0, "denominator": 0}))
                cells.append(f"{n}/{d}")
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="dev_v1")
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    payload = json.loads((HERE / "cases" / f"{args.set}.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in payload["cases"]}
    results = args.results or HERE / "results" / args.set
    metadata = _metadata()
    rows_by_run: dict[tuple[str, int], list[dict]] = defaultdict(list)
    all_rows = []
    for path in sorted(results.glob("E*/run*/*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            record = json.load(handle)
        case = cases.get(record["case_id"])
        if case is None:
            continue
        row = CaseScorer(case, metadata).score(record) | {"needs_policy": bool(case.get("needs_policy"))}
        rows_by_run[(record["variant"], record["run_index"])].append(row)
        all_rows.append(row)
    per_run = {key: run_metrics(rows) for key, rows in rows_by_run.items()}
    summary = summarize(per_run)
    (results / "summary.json").write_text(json.dumps({"cases_sha256": payload["cases_sha256"], "summary": summary},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    (results / "summary.md").write_text(markdown(summary, payload["cases_sha256"]), encoding="utf-8")
    with (results / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "run", "case", "category", "status", "validation", "calls", "correct_calls",
                         "invalid_calls", "invalid_reasons", "numbers_ok", "key_point_coverage", "policy_ok",
                         "task_complete", "verified", "tokens", "elapsed_s"])
        for r in sorted(all_rows, key=lambda r: (r["variant"], r["run_index"], r["case_id"])):
            writer.writerow([r["variant"], r["run_index"], r["case_id"], r["category"], r["status"], r["validation"],
                             len(r["calls"]), sum(c["correct"] for c in r["calls"]),
                             sum(bool(c["invalid_reason"]) for c in r["calls"]),
                             ";".join(c["invalid_reason"] for c in r["calls"] if c["invalid_reason"]),
                             all(r["numbers"].values()) if r["numbers"] else "", round(r.get("key_point_coverage", 0), 2),
                             r["policy_ok"], r["task_complete"], r["verified"], r["tokens"],
                             round(r["elapsed_ms"] / 1000, 1) if r["elapsed_ms"] else ""])
    print(markdown(summary, payload["cases_sha256"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
