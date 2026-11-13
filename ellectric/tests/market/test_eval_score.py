"""The evaluation scorer applies the frozen metric definitions."""

from __future__ import annotations

import unittest

from ellectric.eval.market.score import CaseScorer, run_metrics

CASE = {
    "id": "C01", "category": "C", "allowed_tools": ["query_stats", "detect_anomaly", "explain_drivers"],
    "arg_window": {"start_date": "2025-10-11", "end_date": "2025-10-18"},
    "target_window": {"start_date": "2025-10-18", "end_date": "2025-10-18"},
    "required_tool_groups": [["detect_anomaly"], ["explain_drivers"]],
    "required_numbers": [{"name": "min_price", "any_of": [-80.0], "tolerance": 0.01, "relative": False}],
    "key_points": [
        {"id": "num", "type": "number", "number": "min_price"},
        {"id": "day", "type": "text_contains_any", "values": ["10月18日"]},
        {"id": "drivers", "type": "cause_cites_tool", "tool": "explain_drivers"},
        {"id": "component", "type": "cause_mentions_any", "values": ["风电"]},
        {"id": "limits", "type": "has_limitations"},
    ],
    "needs_policy": False, "required_policy": [],
}


def obs(evidence, tool, args, status="ok", error=None):
    return {"evidence_id": evidence, "tool_name": tool, "arguments": {**args, "as_of": "x"},
            "execution_status": status, "error": error, "result": {}}


def record(observations, report):
    return {"variant": "E2", "run_index": 1, "status": "completed", "report": report,
            "trace": {"observations": observations, "token_count": 1000, "elapsed_ms": 5000, "replan_count": 0}}


DAY = {"start_date": "2025-10-18", "end_date": "2025-10-18"}
REPORT = {
    "summary": "10月18日出现负电价。",
    "anomalies": [{"statement": "最低价 -80 元/MWh。", "evidence_ids": ["e1"],
                   "value_bindings": [{"evidence_id": "e1", "json_pointer": "/x", "value": -80.0}]}],
    "causes": [{"statement": "风电实际值高于预测。", "evidence_ids": ["e2"], "value_bindings": []}],
    "known_limitations": ["气象仅济南。"],
    "validation": {"status": "passed", "errors": []},
}


class EvalScoreTest(unittest.TestCase):
    def test_call_classification_counts_every_attempt(self) -> None:
        calls = CaseScorer(CASE, {}).classify_calls([
            obs("e1", "detect_anomaly", DAY),
            obs("e2", "explain_drivers", DAY),
            obs("e3", "detect_anomaly", DAY),  # identical repeat
            obs("e4", "query_stats", {"start_date": "2025-10-18", "end_date": "2025-10-25"},
                status="error", error="ValidationError: DATA_AFTER_AS_OF: ..."),
            obs("e5", "forecast_price", {"target_date": "2025-10-19"}),  # not allowed for C
            obs("e6", "query_stats", {"start_date": "2025-10-01", "end_date": "2025-10-05"}),  # misses the day
        ])
        self.assertEqual([c["invalid_reason"] for c in calls],
                         [None, None, "duplicate", "as_of_boundary", "disallowed_tool", None])
        self.assertEqual([c["correct"] for c in calls], [True, True, False, False, False, False])

    def test_complete_case_and_metrics_have_numerators(self) -> None:
        scorer = CaseScorer(CASE, {})
        row = scorer.score(record([obs("e1", "detect_anomaly", DAY), obs("e2", "explain_drivers", DAY)], REPORT))
        self.assertTrue(row["task_complete"])
        self.assertEqual(row["key_point_coverage"], 1.0)
        failed = scorer.score(record([obs("e1", "detect_anomaly", DAY)], REPORT))
        self.assertFalse(failed["task_complete"])  # explain_drivers group missing and drivers key point lost
        metrics = run_metrics([row | {"needs_policy": False}, failed | {"needs_policy": False}])
        self.assertEqual(metrics["task_completion"], {"numerator": 1, "denominator": 2, "value": 0.5})
        self.assertEqual(metrics["tool_call_accuracy"]["denominator"], 3)

    def test_derived_values_count_as_reported_numbers(self) -> None:
        derived = {**REPORT, "anomalies": [{
            "statement": "最低价 -80 元/MWh。", "evidence_ids": ["e1"], "value_bindings": [],
            "derived_values": [{"operation": "difference", "value": -80.0, "operands": [
                {"evidence_id": "e1", "json_pointer": "/a", "value": 20.0},
                {"evidence_id": "e1", "json_pointer": "/b", "value": 100.0}]}]}]}
        self.assertEqual(CaseScorer(CASE, {})._numbers(derived), {"min_price": True})

    def test_network_errors_are_excluded_but_run_errors_count(self) -> None:
        scorer = CaseScorer(CASE, {})
        network = scorer.score({"variant": "E2", "run_index": 1, "status": "network_error"}) | {"needs_policy": False}
        crashed = scorer.score({"variant": "E2", "run_index": 1, "status": "run_error"}) | {"needs_policy": False}
        metrics = run_metrics([network, crashed])
        self.assertEqual(metrics["task_completion"]["denominator"], 1)
        self.assertEqual(metrics["aux"]["network_errors_excluded"], 1)


if __name__ == "__main__":
    unittest.main()


class MonthArgumentScoringTest(unittest.TestCase):
    def test_month_bounds_cover_the_days_the_tool_read(self) -> None:
        """query_stats accepts "2025-08" as a range bound; the judge reads it as the tool does."""
        case = CASE | {"arg_window": {"start_date": "2025-08-01", "end_date": "2025-08-31"}, "target_window": None,
                       "required_tool_groups": [],
                       "required_calls": [{"tools": ["query_stats"], "cover": {"start_date": "2025-08-01", "end_date": "2025-08-31"}}]}
        scorer = CaseScorer(case, {})
        month = {"evidence_id": "e1", "tool_name": "query_stats", "execution_status": "ok", "error": None, "result": {},
                 "arguments": {"start_date": "2025-08", "end_date": "2025-08", "as_of": "2025-09-01T09:00:00+08:00"}}
        calls = scorer.classify_calls([month])
        self.assertEqual((calls[0]["args"]["start_date"], calls[0]["args"]["end_date"]), ("2025-08-01", "2025-08-31"))
        self.assertTrue(calls[0]["correct"])
        self.assertEqual(scorer.required_calls(calls), [True])
        running = month | {"arguments": month["arguments"] | {"as_of": "2025-08-11T09:00:00+08:00"}}
        self.assertEqual(scorer.classify_calls([running])[0]["args"]["end_date"], "2025-08-10")


class DirectAnswerScoringTest(unittest.TestCase):
    def test_numbers_and_text_in_direct_answers_are_scored(self) -> None:
        """A report that states the value in ``answers`` (and nothing in anomalies) is scored like one that
        states it in a finding; the reading given in ``basis`` counts as report text."""
        answer = {"item_id": "q1", "status": "answered", "statement": "最低价 -80 元/MWh。", "basis": "10月18日全天",
                  "evidence_ids": ["e1"], "value_bindings": [{"evidence_id": "e1", "json_pointer": "/x", "value": -80.0}]}
        report = REPORT | {"anomalies": [], "summary": "出现负电价。", "answers": [answer]}
        row = CaseScorer(CASE, {}).score(record(
            [obs("e1", "detect_anomaly", DAY), obs("e2", "explain_drivers", DAY)], report))
        self.assertTrue(row["numbers"]["min_price"])
        self.assertTrue(row["key_points"]["day"])
        self.assertTrue(row["task_complete"])


class FaultInjectionTest(unittest.TestCase):
    FAULT_CASE = {**CASE, "id": "F01", "faults": [{"tool": "explain_drivers", "mode": "unavailable", "times": 1}]}

    def test_injector_fails_only_the_first_call_of_the_faulted_tool(self) -> None:
        from ellectric.eval.market.faults import FaultInjectingToolCall, ToolUnavailableError

        calls = []
        injector = FaultInjectingToolCall(lambda name, args: calls.append(name) or {"tool": name},
                                          self.FAULT_CASE["faults"])
        self.assertEqual(injector("detect_anomaly", {}), {"tool": "detect_anomaly"})
        with self.assertRaises(ToolUnavailableError):
            injector("explain_drivers", {})
        self.assertEqual(injector("explain_drivers", {}), {"tool": "explain_drivers"})
        self.assertEqual(calls, ["detect_anomaly", "explain_drivers"])  # the faulted call never ran
        self.assertEqual([entry["injected"] for entry in injector.log], [False, True, False])

    def test_recovery_needs_a_later_successful_call(self) -> None:
        failed = obs("e9", "explain_drivers", DAY, status="error", error="ToolUnavailableError: ...")
        log = [{"tool": "detect_anomaly", "injected": False}, {"tool": "explain_drivers", "injected": True}]
        scorer = CaseScorer(self.FAULT_CASE, {})
        not_recovered = scorer.score(record([obs("e1", "detect_anomaly", DAY), failed], REPORT) | {"fault_log": log})
        self.assertEqual((not_recovered["fault_fired"], not_recovered["fault_recovered"]), (True, False))
        self.assertFalse(not_recovered["task_complete"])
        recovered = scorer.score(record([obs("e1", "detect_anomaly", DAY), failed, obs("e2", "explain_drivers", DAY)],
                                        REPORT) | {"fault_log": log + [{"tool": "explain_drivers", "injected": False}]})
        self.assertEqual((recovered["fault_fired"], recovered["fault_recovered"]), (True, True))
        metrics = run_metrics([not_recovered, recovered])["aux"]
        self.assertEqual(metrics["fault_recovery"]["numerator"], 1)
        self.assertEqual(metrics["fault_recovery"]["denominator"], 2)


class HoldoutScoringTest(unittest.TestCase):
    """Per-case required calls, argument constraints, expected tool errors and refusal key points."""

    CASE = {
        "id": "XM01", "category": "M", "task_case": True,
        "allowed_tools": ["query_stats", "query_load", "query_price", "forecast_price"],
        "arg_windows": [{"start_date": "2024-12-01", "end_date": "2024-12-14"},
                        {"start_date": "2025-12-01", "end_date": "2025-12-14"}],
        "arg_constraints": {"query_price": {"slot_id": ["75", "76"]}},
        "expected_error_tools": ["forecast_price"], "forecast_target": "2025-08-20",
        "required_calls": [
            {"tools": ["query_stats"], "cover": {"start_date": "2025-12-01", "end_date": "2025-12-14"}},
            {"tools": ["query_load"], "cover": {"start_date": "2024-12-05", "end_date": "2024-12-05"}, "max_days": 3},
        ],
        "required_numbers": [], "key_points": [
            {"id": "refusal", "type": "any_text_contains_any", "values": ["无法"]},
            {"id": "quiet", "type": "no_numbers"},
            {"id": "no_basis", "type": "no_policy_basis"},
        ],
        "needs_policy": False, "required_policy": [],
    }
    REPORT = {"summary": "见说明。", "anomalies": [], "causes": [], "known_limitations": ["该日期数据无法获取。"],
              "validation": {"status": "passed", "errors": []}}

    def test_windows_constraints_and_union_coverage(self) -> None:
        scorer = CaseScorer(self.CASE, {})
        calls = scorer.classify_calls([
            obs("e1", "query_stats", {"start_date": "2025-12-01", "end_date": "2025-12-07"}),
            obs("e2", "query_stats", {"start_date": "2025-12-08", "end_date": "2025-12-14"}),
            obs("e3", "query_stats", {"start_date": "2025-11-25", "end_date": "2025-12-03"}),  # outside both windows
            obs("e4", "query_price", {"start_date": "2024-12-05", "end_date": "2024-12-05", "slot_id": 80}),
            obs("e5", "query_load", {"start_date": "2024-12-01", "end_date": "2024-12-14"}),  # wider than max_days
            obs("e6", "forecast_price", {"target_date": "2025-08-20"}, status="error", error="ValueError: TARGET_OUTSIDE"),
        ])
        self.assertEqual([c["correct"] for c in calls], [True, True, False, False, True, False])
        self.assertEqual(calls[-1]["invalid_reason"], "expected_error")
        self.assertEqual(scorer.required_calls(calls), [True, False])

    def test_metrics_exclude_expected_errors_and_report_recall(self) -> None:
        scorer = CaseScorer(self.CASE, {})
        row = scorer.score(record([
            obs("e1", "query_stats", {"start_date": "2025-12-01", "end_date": "2025-12-14"}),
            obs("e2", "query_load", {"start_date": "2024-12-05", "end_date": "2024-12-05"}),
            obs("e3", "forecast_price", {"target_date": "2025-08-20"}, status="error", error="ValueError: TARGET_OUTSIDE"),
        ], self.REPORT)) | {"needs_policy": False}
        self.assertTrue(row["task_complete"])
        self.assertEqual(row["key_point_coverage"], 1.0)
        metrics = run_metrics([row])
        self.assertEqual(metrics["tool_call_accuracy"], {"numerator": 2, "denominator": 2, "value": 1.0})
        self.assertEqual(metrics["task_completion"]["denominator"], 1)
        self.assertEqual(metrics["aux"]["required_call_recall"]["value"], 1.0)
        self.assertEqual(metrics["aux"]["case_tool_perfect"]["value"], 1.0)
        self.assertEqual(metrics["aux"]["expected_errors_excluded"], 1)

    def test_refusal_key_points_fail_on_numbers_or_citations(self) -> None:
        scorer = CaseScorer(self.CASE, {})
        claimed = {**self.REPORT, "policy_basis": [{"evidence_id": "p1", "statement": "x"}], "anomalies": [
            {"statement": "均价 300。", "evidence_ids": ["e1"], "value_bindings": [{"evidence_id": "e1", "value": 300.0}]}]}
        points = {p["id"]: scorer._key_point(p, claimed, [], {}, False) for p in self.CASE["key_points"]}
        self.assertEqual(points, {"refusal": True, "quiet": False, "no_basis": False})

    def test_zero_counts_are_not_an_answer(self) -> None:
        # "Query returned 0 rows" documents the gap; it must not count as reporting a number.
        empty = {**self.REPORT, "anomalies": [{"statement": "查询返回 0 行。", "evidence_ids": ["e1"],
                                               "value_bindings": [{"evidence_id": "e1", "value": 0}, {"evidence_id": "e1", "value": None}]}]}
        self.assertTrue(CaseScorer(self.CASE, {})._key_point({"id": "quiet", "type": "no_numbers"}, empty, [], {}, False))

    def test_docx_clause_without_page_matches_only_by_keywords(self) -> None:
        case = {**CASE, "required_policy": [{"relative_path": "p.docx", "page": None, "keywords": ["最小单位"]}]}
        observation = {"evidence_id": "p1", "tool_name": "search_policy", "execution_status": "ok", "result": {"data": {"evidence": [
            {"relative_path": "p.docx", "page": None, "locator": "table 1, row 3", "evidence_eligible": True, "excerpt": "其他参数"}]}}}
        report = {"policy_basis": [{"evidence_id": "p1", "json_pointer": "/data/evidence/0"}]}
        self.assertFalse(CaseScorer(case, {})._policy_ok(report, [observation]))


class FrozenHoldoutSetTest(unittest.TestCase):
    """The frozen holdout set is internally consistent and unchanged since it was built."""

    def test_holdout_v1_is_consistent(self) -> None:
        import hashlib
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "eval" / "market" / "cases" / "holdout_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = json.dumps(payload["cases"], ensure_ascii=False, sort_keys=True, default=str)
        self.assertEqual(hashlib.sha256(body.encode("utf-8")).hexdigest(), payload["cases_sha256"])
        for case in payload["cases"]:
            for spec in case["required_calls"]:
                self.assertTrue(set(spec["tools"]) & set(case["allowed_tools"]), case["id"])
            if case["category"] == "P":
                self.assertEqual(case["allowed_tools"], ["search_policy"], case["id"])
            if case["mode"] == "historical_forecast":
                self.assertIn("forecast_target", case, case["id"])
            for point in case["key_points"]:
                if point["type"] == "number":
                    self.assertIn(point["number"], {n["name"] for n in case["required_numbers"]}, case["id"])
