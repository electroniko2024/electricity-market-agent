"""Stage-wise scoring (v2): evidence detection, row-key attribution and stage order."""

from __future__ import annotations

import unittest

from ellectric.eval.market.score_v2 import StageScorer, bootstrap_diff, keys_at, leaves

RESULT = {"data": {"rows": [
    {"trade_date": "2025-04-12", "slot_id": 50, "time_label": "12:45", "solar_actual_mw": 9316.35},
    {"trade_date": "2025-04-17", "slot_id": 50, "time_label": "12:45", "solar_actual_mw": 16994.94},
]}}

CASE = {
    "id": "T1", "category": "L3", "stratum": "L3", "answerability": "A",
    "allowed_tools": ["query_load"], "arg_windows": [], "required_calls": [{"tools": ["query_load"]}],
    "required_numbers": [{"name": "solar_max", "any_of": [9316.35], "tolerance": 0.5, "relative": False,
                          "keys": {"date": "2025-04-12"}, "field": "solar_actual_mw"}],
    "key_points": [{"id": "reports_solar_max", "type": "number", "number": "solar_max"}],
    "needs_policy": False, "required_policy": [], "task_case": True,
}


def record(report, result=RESULT, status="ok"):
    return {"variant": "E0", "run_index": 1, "status": "completed", "report": report,
            "trace": {"observations": [{"evidence_id": "e1", "tool_name": "query_load", "arguments": {},
                                        "execution_status": status, "error": None, "result": result}],
                      "token_count": 1, "elapsed_ms": 1}}


def report(value, row, status="passed"):
    return {"summary": "", "validation": {"status": status}, "anomalies": [{
        "statement": f"{value}", "evidence_ids": ["e1"],
        "value_bindings": [{"evidence_id": "e1", "json_pointer": f"/data/rows/{row}/solar_actual_mw", "value": value}]}]}


class StageScoringTest(unittest.TestCase):
    def test_leaves_carry_row_keys(self) -> None:
        found = [(f, v, c) for f, v, c in leaves(RESULT) if f == "solar_actual_mw"]
        self.assertEqual(found[0], ("solar_actual_mw", 9316.35, {"date": "2025-04-12", "slot": 50}))
        self.assertEqual(keys_at(RESULT, "/data/rows/1/solar_actual_mw"), ("solar_actual_mw", {"date": "2025-04-17", "slot": 50}))
        self.assertIsNone(keys_at(RESULT, "/data/rows/9/x"))

    def test_pass_and_wrong_row(self) -> None:
        scorer = StageScorer(CASE, {})
        self.assertEqual(scorer.score(record(report(9316.35, 0)))["stage"], "pass")
        wrong = scorer.score(record(report(16994.94, 1)))
        self.assertEqual((wrong["stage"], wrong["under_answer_kind"]), ("under_answer", "wrong_row"))
        self.assertTrue(wrong["evidence"])

    def test_evidence_needs_the_keyed_row(self) -> None:
        other_day = {"data": {"rows": [{"trade_date": "2025-04-13", "slot_id": 50, "solar_actual_mw": 9316.35}]}}
        row = StageScorer(CASE, {}).score(record(report(9316.35, 0), result=other_day))
        self.assertFalse(row["evidence"])
        self.assertEqual(row["stage"], "evidence")

    def test_self_check_and_refusal_stages(self) -> None:
        scorer = StageScorer(CASE, {})
        self.assertEqual(scorer.score(record(report(9316.35, 0, status="failed")))["stage"], "self_check")
        refused = {"summary": "无法确定当日光伏出力。", "validation": {"status": "passed"}, "anomalies": []}
        self.assertEqual(scorer.score(record(refused))["stage"], "answerability")
        empty = {"summary": "光伏出力较高。", "validation": {"status": "passed"}, "anomalies": []}
        row = scorer.score(record(empty))
        self.assertEqual((row["stage"], row["under_answer_kind"]), ("under_answer", "empty"))

    def test_bootstrap_is_paired(self) -> None:
        out = bootstrap_diff({"a": 1.0, "b": 1.0, "c": 0.0}, {"a": 0.0, "b": 1.0, "c": 0.0}, iterations=500)
        self.assertEqual((out["n"], out["a_better"], out["b_better"]), (3, 1, 0))
        self.assertAlmostEqual(out["diff"], 1 / 3)


class DevV2SetTest(unittest.TestCase):
    """The generated dev_v2 set is unchanged since it was built and internally consistent."""

    def test_dev_v2_is_consistent(self) -> None:
        import hashlib
        import json
        from collections import Counter
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "eval" / "market" / "cases" / "dev_v2.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = json.dumps(payload["cases"], ensure_ascii=False, sort_keys=True, default=str)
        self.assertEqual(hashlib.sha256(body.encode("utf-8")).hexdigest(), payload["cases_sha256"])
        ids = [c["id"] for c in payload["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        strata = Counter(c["stratum"] for c in payload["cases"])
        self.assertGreaterEqual(min(strata[s] for s in ("L3", "L4")), 30)
        for case in payload["cases"]:
            for spec in case["required_calls"]:
                self.assertTrue(set(spec["tools"]) & set(case["allowed_tools"]), case["id"])
            if case["answerability"] != "C":
                self.assertTrue(case["reference_solution"], case["id"])
            else:
                self.assertFalse(case["required_numbers"], case["id"])
            if case["stratum"] == "L5":
                self.assertEqual(case["faults"][0]["tool"], case["fault_tool"], case["id"])
            for number in case["required_numbers"] + case.get("evidence_values", []):
                self.assertTrue(set(number.get("keys", {})) <= {"date", "slot"}, case["id"])


if __name__ == "__main__":
    unittest.main()
