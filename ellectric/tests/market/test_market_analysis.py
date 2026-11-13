"""Offline graph tests for replanning, traces, and evidence validation."""

from __future__ import annotations

import copy
import os
import re
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone

from ellectric.llm.market_analysis import MarketAnalysisAgent as _MarketAnalysisAgent
from ellectric.llm.market_analysis import (
    _canonicalize_bindings,
    _canonicalize_policy_ids,
    _compact_for_prompt,
    _has_numeric_claim,
    _json_pointer,
    _prompt_observations,
    _thinking_request_fields,
)


def _with_default_scope(json_call):
    """For stubs that predate the scope step: answer ScopeDraft with one answerable item, and give a report
    written without ``answers`` a direct answer to it that cites the first evidence id in the prompt."""
    if json_call is None:
        return None

    def call(name, schema, prompt):
        if name == "ScopeDraft":
            return {"items": [{"item": "问题", "answer_type": "text", "status": "answerable", "note": ""}]}
        payload = json_call(name, schema, prompt)
        if name == "MarketReportDraft" and isinstance(payload, dict) and "answers" not in payload:
            match = re.search(r"ev_[0-9a-f]{6,}", prompt)
            answer = {"item_id": "q1", "status": "answered" if match else "unanswerable",
                      "statement": "已按所取数据作答。" if match else "没有取到数据。",
                      "evidence_ids": [match.group(0)] if match else []}
            payload = {**payload, "answers": [answer]}
        return payload

    return call


class MarketAnalysisAgent(_MarketAnalysisAgent):
    """The agent with an empty data catalog (no database needed) and the stub adapter above."""

    def __init__(self, *, json_call=None, catalog=lambda _as_of: {}, **kwargs):
        super().__init__(json_call=_with_default_scope(json_call), catalog=catalog, **kwargs)


def _cited_report(prompt: str) -> dict:
    """A minimal valid report: one qualitative finding citing the first evidence id the prompt shows."""
    match = re.search(r"ev_[0-9a-f]{6,}", prompt)
    causes = [{"category": "other", "statement": "已核查所取数据。", "evidence_ids": [match.group(0)]}] if match else []
    return {"summary": "完成核查。", "anomalies": [], "causes": causes, "unresolved_issues": []}


class MarketAnalysisGraphTest(unittest.TestCase):
    def test_replans_once_then_validates_report_against_observation(self) -> None:
        plan_calls = 0
        reflection_calls = 0
        tool_calls: list[str] = []

        def json_call(name, _schema, prompt):
            nonlocal plan_calls, reflection_calls
            if name == "PlanDraft":
                plan_calls += 1
                if plan_calls == 1:
                    return {
                        "steps": [
                            {
                                "step_id": "s1",
                                "tool_name": "query_stats",
                                "arguments": {
                                    "start_date": "2025-10-01",
                                    "end_date": "2025-10-01",
                                    "metric": "rt_price",
                                    "group_by": "day",
                                },
                                "depends_on": [],
                                "rationale": "先查当日价格统计",
                            }
                        ]
                    }
                return {
                    "steps": [
                        {
                            "step_id": "s2",
                            "tool_name": "query_price",
                            "arguments": {
                                "start_date": "2025-10-01",
                                "end_date": "2025-10-01",
                                "slot_id": 0,
                                "limit": 10,
                            },
                            "depends_on": ["s1"],
                            "rationale": "补查统计代表点的原始价格",
                        }
                    ]
                }
            if name == "ReflectionDraft":
                reflection_calls += 1
                if reflection_calls == 1:
                    return {
                        "sufficient_evidence": False,
                        "issues": [
                            {
                                "code": "NEED_SOURCE_POINT",
                                "description": "还需抽查原始价格点。",
                                "repairable": True,
                                "related_step_ids": ["s1"],
                            }
                        ],
                    }
                return {"sufficient_evidence": True, "issues": []}
            if name == "MarketReportDraft":
                return {
                    "summary": "已完成指定交易日的实时价格核查。",
                    "anomalies": [],
                    "causes": _cited_report(prompt)["causes"],
                    "outlook": None,
                    "policy_basis": [],
                    "data_notes": [],
                    "unresolved_issues": [],
                }
            raise AssertionError(f"Unexpected model schema: {name}")

        def tool_call(name, arguments):
            tool_calls.append(name)
            if name == "query_stats":
                return {
                    "tool": name,
                    "status": "ok",
                    "data": {"mean_price": 42.5},
                    "limitations": [],
                }
            return {
                "tool": name,
                "status": "ok",
                "data": {"rows": [{"slot_id": arguments["slot_id"], "rt_price": 42.5}]},
                "limitations": [],
            }

        agent = MarketAnalysisAgent(json_call=json_call, tool_call=tool_call)
        result = agent.analyze(
            question="核查 2025-10-01 的实时价格",
            as_of=datetime(2025, 10, 2, tzinfo=timezone.utc),
            mode="historical_analysis",
        )

        self.assertEqual(result["trace"]["replan_count"], 1)
        self.assertEqual(result["trace"]["tool_call_count"], 2)
        self.assertEqual(tool_calls, ["query_stats", "query_price"])
        self.assertEqual(result["report"]["validation"]["status"], "passed")
        self.assertTrue(result["report"]["verified"])
        self.assertEqual(
            [event["type"] for event in result["trace"]["events"]].count("reflection"),
            2,
        )

    def test_numeric_report_claim_requires_matching_json_pointer(self) -> None:
        observation = {
            "evidence_id": "ev_one",
            "result": {"data": {"mean_price": 12.5}},
        }
        valid = {
            "summary": "统计完成。",
            "anomalies": [],
            "causes": [
                {
                    "statement": "均价为 12.5。",
                    "evidence_ids": ["ev_one"],
                    "value_bindings": [
                        {"evidence_id": "ev_one", "json_pointer": "/data/mean_price", "value": 12.5}
                    ],
                }
            ],
            "policy_basis": [],
        }
        self.assertEqual(MarketAnalysisAgent._validate_report(valid, [observation]), [])

        valid["causes"][0]["value_bindings"][0]["value"] = 10.0
        errors = MarketAnalysisAgent._validate_report(valid, [observation])
        self.assertTrue(any("数值与 evidence 不一致" in error for error in errors))
        valid["causes"][0]["value_bindings"][0]["value"] = 12.5
        valid["summary"] = "核查了 1 个交易日。"
        errors = MarketAnalysisAgent._validate_report(valid, [observation])
        self.assertTrue(any("summary 不得包含未绑定的数值" in error for error in errors))


class DerivedValueTest(unittest.TestCase):
    OBSERVATION = {"evidence_id": "ev_d", "result": {"data": {"negative": 22, "total": 96, "today": 310.5, "prior": 280.25}}}

    @staticmethod
    def _operand(field: str, value: float) -> dict:
        return {"evidence_id": "ev_d", "json_pointer": f"/data/{field}", "value": value}

    def _report(self, statement: str, derived: list[dict], bindings: list[dict] | None = None) -> dict:
        return {
            "summary": "统计完成。",
            "anomalies": [{"category": "other", "statement": statement, "evidence_ids": ["ev_d"],
                           "value_bindings": bindings or [], "derived_values": derived}],
            "causes": [],
            "policy_basis": [],
        }

    def test_ratio_and_difference_are_recomputed_and_bind_statement_numbers(self) -> None:
        report = self._report(
            "负价时段 22 个，占比 22.9%；均价较前日上涨 30.25 元/MWh。",
            [
                {"operation": "ratio", "operands": [self._operand("negative", 22), self._operand("total", 96)], "value": 0.2292},
                {"operation": "difference", "operands": [self._operand("today", 310.5), self._operand("prior", 280.25)], "value": 30.25},
            ],
            bindings=[self._operand("negative", 22)],
        )
        self.assertEqual(MarketAnalysisAgent._validate_report(report, [self.OBSERVATION]), [])

    def test_sum_without_direct_bindings_counts_as_numeric_binding(self) -> None:
        report = self._report(
            "两日均价合计 590.75 元/MWh。",
            [{"operation": "sum", "operands": [self._operand("today", 310.5), self._operand("prior", 280.25)], "value": 590.75}],
        )
        self.assertEqual(MarketAnalysisAgent._validate_report(report, [self.OBSERVATION]), [])

    def test_wrong_result_is_rejected_and_number_stays_unbound(self) -> None:
        report = self._report(
            "均价较前日上涨 40 元/MWh。",
            [{"operation": "difference", "operands": [self._operand("today", 310.5), self._operand("prior", 280.25)], "value": 40}],
        )
        errors = MarketAnalysisAgent._validate_report(report, [self.OBSERVATION])
        self.assertTrue(any("计算结果不一致" in error for error in errors))
        self.assertTrue(any("['40']" in error for error in errors))

    def test_operand_must_match_evidence_and_ratio_needs_two_operands(self) -> None:
        bad_operand = self._report(
            "占比 0.25。",
            [{"operation": "ratio", "operands": [self._operand("negative", 24), self._operand("total", 96)], "value": 0.25}],
        )
        self.assertTrue(any("数值与 evidence 不一致" in error
                            for error in MarketAnalysisAgent._validate_report(bad_operand, [self.OBSERVATION])))
        zero = {"evidence_id": "ev_z", "result": {"data": {"zero": 0, "n": 22}}}
        divide_by_zero = self._report(
            "比值 1.0。",
            [{"operation": "ratio", "operands": [{"evidence_id": "ev_z", "json_pointer": "/data/n", "value": 22},
                                                 {"evidence_id": "ev_z", "json_pointer": "/data/zero", "value": 0}],
              "value": 1.0}],
        )
        divide_by_zero["anomalies"][0]["evidence_ids"] = ["ev_z"]
        self.assertTrue(any("分母（最后一个）不为 0" in error
                            for error in MarketAnalysisAgent._validate_report(divide_by_zero, [zero])))

    def test_ratio_numerator_is_the_sum_of_all_but_the_last_operand(self) -> None:
        # Two numerator parts and a denominator: (22 + 22) / 96 = 0.4583.
        report = self._report(
            "两段合计占比 0.4583。",
            [{"operation": "ratio", "operands": [self._operand("negative", 22), self._operand("negative", 22),
                                                 self._operand("total", 96)], "value": 0.4583}],
        )
        self.assertEqual(MarketAnalysisAgent._validate_report(report, [self.OBSERVATION]), [])


class CompactViewStatBindingTest(unittest.TestCase):
    # 30 rows (longer than the prompt list limit), rt_price missing in 4 of them.
    ROWS = [{"slot": index, "rt_price": (None if index % 8 == 0 else float(index))} for index in range(30)]
    OBSERVATION = {"evidence_id": "ev_rows", "result": {"tool": "query_price", "data": {"rows": ROWS}}}

    def _report(self, statement: str, pointer: str, value: float) -> dict:
        return {
            "summary": "统计完成。",
            "anomalies": [{"category": "other", "statement": statement, "evidence_ids": ["ev_rows"],
                           "value_bindings": [{"evidence_id": "ev_rows", "json_pointer": pointer, "value": value}]}],
            "causes": [],
            "policy_basis": [],
        }

    def _validate(self, report: dict) -> list[str]:
        observations = [self.OBSERVATION]
        return MarketAnalysisAgent._validate_report(_canonicalize_bindings(report, observations), observations)

    def test_numeric_range_count_shown_in_prompt_binds(self) -> None:
        compact = _compact_for_prompt(self.OBSERVATION["result"])
        count = compact["data"]["rows"]["numeric_ranges"]["rt_price"]["count"]
        self.assertEqual(count, 26)
        for pointer in ("/data/rows/numeric_ranges/rt_price/count", "/data/rows/rt_price/count"):
            self.assertEqual(self._validate(self._report(f"有效时段 {count} 个。", pointer, count)), [], pointer)
        self.assertEqual(self._validate(self._report("共 30 个时段。", "/data/rows/length", 30)), [])

    def test_wrong_count_is_still_rejected(self) -> None:
        errors = self._validate(self._report("有效时段 30 个。", "/data/rows/numeric_ranges/rt_price/count", 30))
        self.assertTrue(any("数值与 evidence 不一致" in error for error in errors), errors)

    def test_unknown_field_stays_invalid(self) -> None:
        errors = self._validate(self._report("有效时段 26 个。", "/data/rows/numeric_ranges/da_price/count", 26))
        self.assertTrue(any("JSON Pointer 无效" in error for error in errors), errors)


class PolicyIdCanonicalizationTest(unittest.TestCase):
    OBSERVATIONS = [
        {"evidence_id": "ev_rows", "result": {"tool": "query_price", "data": {"min": -80.0}}},
        {
            "evidence_id": "ev_policy",
            "result": {
                "tool": "search_policy",
                "data": {
                    "evidence": [
                        {"evidence_id": "pol_aaaa", "evidence_eligible": False, "locator": "第1条", "page": 2},
                        {"evidence_id": "pol_bbbb", "evidence_eligible": True, "locator": "第5条", "page": 7},
                    ]
                },
            },
        },
    ]

    def test_candidate_ids_are_rewritten_to_observation_id_and_position(self) -> None:
        report = {
            "summary": "依据政策条款。",
            "summary_evidence_ids": ["pol_bbbb", "ev_rows", "ev_policy"],
            "anomalies": [{"category": "other", "statement": "条款在第 7 页。", "evidence_ids": ["pol_bbbb"],
                           "value_bindings": [{"evidence_id": "pol_bbbb", "json_pointer": "/page", "value": 7}]}],
            "causes": [],
            "policy_basis": [{"evidence_id": "pol_bbbb", "json_pointer": "/data/evidence/0", "statement": "文件第5条"}],
        }
        canonical = _canonicalize_policy_ids(report, self.OBSERVATIONS)
        self.assertEqual(canonical["policy_basis"][0]["evidence_id"], "ev_policy")
        self.assertEqual(canonical["policy_basis"][0]["json_pointer"], "/data/evidence/1")
        self.assertEqual(canonical["summary_evidence_ids"], ["ev_policy", "ev_rows"])
        self.assertEqual(canonical["anomalies"][0]["evidence_ids"], ["ev_policy"])
        self.assertEqual(canonical["anomalies"][0]["value_bindings"][0]["json_pointer"], "/data/evidence/1/page")
        self.assertEqual(MarketAnalysisAgent._validate_report(canonical, self.OBSERVATIONS), [])

    def test_candidates_embedded_in_explain_drivers_resolve(self) -> None:
        embedded = {
            "evidence_id": "ev_drivers",
            "result": {"tool": "explain_drivers", "data": {"policy_evidence": copy.deepcopy(self.OBSERVATIONS[1]["result"])}},
        }
        report = {"summary": "政策。", "anomalies": [], "causes": [],
                  "policy_basis": [{"evidence_id": "pol_bbbb", "json_pointer": "/data/policy_evidence/data/evidence/1",
                                    "statement": "文件第5条"}]}
        canonical = _canonicalize_policy_ids(report, [embedded])
        self.assertEqual(canonical["policy_basis"][0]["evidence_id"], "ev_drivers")
        self.assertEqual(MarketAnalysisAgent._validate_report(canonical, [embedded]), [])
        # A pointer into a non-policy part of the result is not a policy citation.
        canonical["policy_basis"][0]["json_pointer"] = "/data"
        self.assertTrue(any("未核验" in error for error in MarketAnalysisAgent._validate_report(canonical, [embedded])))

    def test_ineligible_candidate_still_fails_after_rewrite(self) -> None:
        report = {"summary": "政策。", "anomalies": [], "causes": [],
                  "policy_basis": [{"evidence_id": "pol_aaaa", "json_pointer": "", "statement": "文件第1条"}]}
        canonical = _canonicalize_policy_ids(report, self.OBSERVATIONS)
        self.assertEqual(canonical["policy_basis"][0]["json_pointer"], "/data/evidence/0")
        errors = MarketAnalysisAgent._validate_report(canonical, self.OBSERVATIONS)
        self.assertTrue(any("未核验" in error for error in errors))


class BindingRepairTest(unittest.TestCase):
    """Rounded binding values, bindings cited against the wrong observation, counts and ordinals."""

    DETECT = {"evidence_id": "ev_detect", "result": {"tool": "detect_anomaly", "data": {"events": [
        {"points": 31, "min_price": -80.0}, {"points": 24, "min_price": -50.5},
        {"price_yuan_per_mwh": 1283.604, "threshold_yuan_per_mwh": 753.0},
    ]}}}
    DRIVERS = {"evidence_id": "ev_drivers", "result": {"tool": "explain_drivers", "data": {"events": [
        {"event": {"x": 1}}, {"event": {"x": 2}},
        {"event": {"previous_price_yuan_per_mwh": 730.602, "current_price_yuan_per_mwh": 1283.604,
                   "change_yuan_per_mwh": 553.0020000000001, "threshold_yuan_per_mwh": 300.0}},
    ]}}}
    STATS = {"evidence_id": "ev_stats", "result": {"data": {"mean_price_yuan_per_mwh": 365.974038690}}}

    @staticmethod
    def _report(statement: str, bindings: list[dict], evidence_ids: list[str]) -> dict:
        return {"summary": "统计完成。", "causes": [], "policy_basis": [],
                "anomalies": [{"category": "other", "statement": statement, "evidence_ids": evidence_ids,
                               "value_bindings": bindings}]}

    def _errors(self, report: dict, observations: list[dict]) -> list[str]:
        return MarketAnalysisAgent._validate_report(_canonicalize_bindings(report, observations), observations)

    def test_binding_rounded_to_its_written_decimals_passes(self) -> None:
        report = self._report("均价 365.974 元/MWh。",
                              [{"evidence_id": "ev_stats", "json_pointer": "/data/mean_price_yuan_per_mwh", "value": 365.974}],
                              ["ev_stats"])
        self.assertEqual(self._errors(report, [self.STATS]), [])
        report["anomalies"][0]["value_bindings"][0]["value"] = 365.98
        report["anomalies"][0]["statement"] = "均价 365.98 元/MWh。"
        self.assertTrue(any("数值与 evidence 不一致" in e for e in self._errors(report, [self.STATS])))

    def test_binding_cited_against_wrong_observation_moves_to_the_one_holding_it(self) -> None:
        observations = [self.DETECT, self.DRIVERS]
        report = self._report(
            "由 730.602 元/MWh 升至 1283.604 元/MWh，变化 553.002 元/MWh，超过阈值 300.0 元/MWh。",
            [{"evidence_id": "ev_detect", "json_pointer": "/data/events/2/previous_price_yuan_per_mwh", "value": 730.602},
             {"evidence_id": "ev_detect", "json_pointer": "/data/events/2/current_price_yuan_per_mwh", "value": 1283.604},
             {"evidence_id": "ev_detect", "json_pointer": "/data/events/2/change_yuan_per_mwh", "value": 553.002},
             {"evidence_id": "ev_detect", "json_pointer": "/data/events/2/threshold_yuan_per_mwh", "value": 300.0}],
            ["ev_detect"],
        )
        self.assertEqual(self._errors(report, observations), [])
        bindings = report["anomalies"][0]["value_bindings"]
        self.assertEqual(bindings[0], {"evidence_id": "ev_drivers", "value": 730.602,
                                       "json_pointer": "/data/events/2/event/previous_price_yuan_per_mwh"})
        self.assertIn("ev_drivers", report["anomalies"][0]["evidence_ids"])

    def test_value_found_nowhere_is_still_rejected(self) -> None:
        report = self._report("现价 999.9 元/MWh。",
                              [{"evidence_id": "ev_detect", "json_pointer": "/data/events/2/current_price_yuan_per_mwh",
                                "value": 999.9}], ["ev_detect"])
        self.assertTrue(any("JSON Pointer 无效" in e for e in self._errors(report, [self.DETECT, self.DRIVERS])))

    def test_counts_of_cited_lists_and_ordinals_are_bound(self) -> None:
        report = self._report(
            "共 3 段事件，第 1 段 31 个时段，第 2 段 24 个时段（slot_id 0-6）。",
            [{"evidence_id": "ev_detect", "json_pointer": "/data/events/0/points", "value": 31},
             {"evidence_id": "ev_detect", "json_pointer": "/data/events/1/points", "value": 24}],
            ["ev_detect"],
        )
        self.assertEqual(self._errors(report, [self.DETECT]), [])
        # A count the cited list does not support, or an ordinal of an uncited item, stays unbound.
        report["anomalies"][0]["statement"] = "共 4 段事件，第 5 段 31 个时段。"
        errors = self._errors(report, [self.DETECT])
        self.assertTrue(any("['4', '5']" in e for e in errors))


class PromptCompactionTest(unittest.TestCase):
    def _observation(self) -> dict:
        rows = [{"slot_id": index, "time_label": f"{index:02d}:15", "rt_price": float(index * 10 - 300)} for index in range(96)]
        return {
            "evidence_id": "ev_rows",
            "step_id": "s1",
            "tool_name": "query_price",
            "arguments": {"start_date": "2025-10-01", "end_date": "2025-10-01"},
            "execution_status": "ok",
            "error": None,
            "result": {"tool": "query_price", "status": "ok", "data": {"total_rows": 96, "rows": rows}},
        }

    def test_long_lists_are_compacted_with_pointers_into_full_result(self) -> None:
        observation = self._observation()
        compact = _prompt_observations([observation])[0]
        view = compact["result"]["data"]["rows"]
        self.assertTrue(view["_compacted_list"])
        self.assertEqual(view["length"], 96)
        self.assertEqual(len(view["head"]), 3)
        self.assertEqual(view["omitted_items"], 93)
        self.assertNotIn("slot_id", view["numeric_ranges"])
        price = view["numeric_ranges"]["rt_price"]
        self.assertEqual(price["min"], -300.0)
        self.assertEqual(price["max"], 650.0)
        full = observation["result"]
        self.assertEqual(_json_pointer(full, price["min_pointer"]), price["min"])
        self.assertEqual(_json_pointer(full, price["max_pointer"]), price["max"])
        head = view["head"][1]
        self.assertEqual(
            _json_pointer(full, head["_pointer"]),
            {key: value for key, value in head.items() if key != "_pointer"},
        )
        self.assertEqual(_json_pointer(full, f"{head['_pointer']}/rt_price"), head["rt_price"])
        self.assertEqual(compact["result"]["data"]["total_rows"], 96)
        self.assertEqual(len(full["data"]["rows"]), 96, "原始 observation 不得被修改")

    def test_compacted_pointer_passes_report_validation_against_full_result(self) -> None:
        observation = self._observation()
        price = _prompt_observations([observation])[0]["result"]["data"]["rows"]["numeric_ranges"]["rt_price"]
        report = {
            "summary": "当日存在负价时段。",
            "anomalies": [
                {
                    "statement": "当日最低实时价为负。",
                    "evidence_ids": ["ev_rows"],
                    "value_bindings": [
                        {"evidence_id": "ev_rows", "json_pointer": price["min_pointer"], "value": price["min"]}
                    ],
                }
            ],
            "causes": [],
            "policy_basis": [],
        }
        self.assertEqual(MarketAnalysisAgent._validate_report(report, [observation]), [])

    def test_short_lists_and_strings_are_kept_or_truncated(self) -> None:
        self.assertEqual(_compact_for_prompt([1, 2, 3]), [1, 2, 3])
        text = _compact_for_prompt("x" * 2000)
        self.assertTrue(text.startswith("x" * 500))
        self.assertIn("原长 2000", text)

    def test_graph_prompts_receive_compacted_observations(self) -> None:
        prompts: dict[str, str] = {}

        def json_call(name, _schema, prompt):
            prompts[name] = prompt
            if name == "PlanDraft":
                return {
                    "steps": [
                        {
                            "step_id": "s1",
                            "tool_name": "query_price",
                            "arguments": {"start_date": "2025-10-01", "end_date": "2025-10-01"},
                            "depends_on": [],
                            "rationale": "查当日价格",
                        }
                    ]
                }
            if name == "ReflectionDraft":
                return {"sufficient_evidence": True, "issues": []}
            return {"summary": "已核查。", "anomalies": [], "causes": [], "policy_basis": []}

        full_result = self._observation()["result"]
        agent = MarketAnalysisAgent(json_call=json_call, tool_call=lambda _name, _args: full_result)
        result = agent.analyze(
            question="核查 2025-10-01 的实时价格",
            as_of=datetime(2025, 10, 2, tzinfo=timezone.utc),
            mode="historical_analysis",
        )
        for name in ("ReflectionDraft", "MarketReportDraft"):
            self.assertIn("_compacted_list", prompts[name])
            self.assertNotIn('"time_label": "50:15"', prompts[name])
        self.assertEqual(len(result["trace"]["observations"][0]["result"]["data"]["rows"]), 96)


class RunContextBindingTest(unittest.TestCase):
    def test_run_as_of_replaces_llm_supplied_as_of(self) -> None:
        seen: list[dict] = []

        def json_call(schema_name, _schema, prompt):
            if schema_name == "PlanDraft":
                self.assertIn("latest_available_trade_date=2025-10-01", prompt)
                return {
                    "steps": [
                        {
                            "step_id": "s1",
                            "tool_name": "query_stats",
                            "arguments": {
                                "start_date": "2025-10-01",
                                "end_date": "2025-10-01",
                                "as_of": "2030-01-01T00:00:00+08:00",
                            },
                            "depends_on": [],
                            "rationale": "查价格",
                        }
                    ]
                }
            if schema_name == "ReflectionDraft":
                return {"sufficient_evidence": True, "issues": []}
            return {"summary": "已查询价格。", "summary_evidence_ids": []}

        def tool_call(name, arguments):
            seen.append(arguments)
            return {"tool": name, "status": "ok", "data": {}}

        as_of = datetime(2025, 10, 2, tzinfo=timezone(timedelta(hours=8)))
        agent = MarketAnalysisAgent(json_call=json_call, tool_call=tool_call)
        result = agent.analyze(question="查价格", as_of=as_of, mode="historical_analysis")

        self.assertEqual(seen[0]["as_of"], as_of.isoformat())
        self.assertEqual(result["trace"]["plan"][0]["arguments"]["as_of"], as_of.isoformat())

    def test_argument_validation_failure_is_repairable(self) -> None:
        agent = MarketAnalysisAgent(
            json_call=lambda *_args: {"sufficient_evidence": False, "issues": []}
        )
        state = {
            "question": "q",
            "as_of": "2025-10-02T00:00:00+08:00",
            "mode": "historical_analysis",
            "events": [],
            "observations": [
                {
                    "evidence_id": "ev_a",
                    "step_id": "s1",
                    "tool_name": "query_price",
                    "arguments": {},
                    "execution_status": "error",
                    "result": None,
                    "error": "ValidationError: DATA_AFTER_AS_OF",
                },
                {
                    "evidence_id": "ev_b",
                    "step_id": "s2",
                    "tool_name": "forecast_price",
                    "arguments": {},
                    "execution_status": "error",
                    "result": None,
                    "error": "FileNotFoundError: missing",
                },
            ],
        }
        issues = {issue["related_step_ids"][0]: issue for issue in agent._reflector(state)["reflection"]["issues"]}
        self.assertTrue(issues["s1"]["repairable"])
        self.assertFalse(issues["s2"]["repairable"])


class CompactedPointerTest(unittest.TestCase):
    def test_compacted_view_pointer_is_rewritten_to_full_result_path(self) -> None:
        observations = [
            {
                "evidence_id": "ev_rows",
                "result": {"data": {"rows": [{"rt_price": float(index)} for index in range(20)]}},
            }
        ]
        report = {
            "summary": "已核对价格。",
            "anomalies": [
                {
                    "category": "anomaly",
                    "statement": "第二个时点价格。",
                    "evidence_ids": ["ev_rows"],
                    "value_bindings": [
                        {"evidence_id": "ev_rows", "json_pointer": "/data/rows/head/1/item/rt_price", "value": 1.0},
                        {"evidence_id": "ev_rows", "json_pointer": "/data/rows/head/9/item/missing", "value": 1.0},
                    ],
                }
            ],
        }
        report["anomalies"][0]["value_bindings"].append(
            {"evidence_id": "ev_rows", "json_pointer": "/data/rows/2/item/rt_price", "value": 2.0}
        )
        canonical = _canonicalize_bindings(report, observations)
        self.assertEqual(canonical["anomalies"][0]["value_bindings"][2]["json_pointer"], "/data/rows/2/rt_price")
        bindings = canonical["anomalies"][0]["value_bindings"]
        self.assertEqual(bindings[0]["json_pointer"], "/data/rows/1/rt_price")
        self.assertEqual(bindings[1]["json_pointer"], "/data/rows/head/9/item/missing")
        errors = MarketAnalysisAgent._validate_report(canonical, observations)
        self.assertEqual(len(errors), 1)
        self.assertIn("/data/rows/head/9/item/missing", errors[0])

    def test_dates_and_times_are_not_numeric_claims(self) -> None:
        self.assertFalse(_has_numeric_claim("2025-10-01 至 2025年10月7日，10 月 1 日 08:45 出现负电价。"))
        self.assertTrue(_has_numeric_claim("10 月 1 日共检测到 13 个异常事件。"))
        self.assertTrue(_has_numeric_claim("在2025-10-01出现负电价，最低-80元。"))
        self.assertFalse(_has_numeric_claim("给出 D+1 研判；依据鲁监能市场规〔2025〕63号附件1第3.5.4条，P99 阈值。"))
        self.assertTrue(_has_numeric_claim("共 3 个高分歧时段。"))
        self.assertFalse(_has_numeric_claim("适用《山东省电力并网运行管理实施细则（2023）》（鲁监能市场〔2023〕53号 附件1）。"))
        self.assertTrue(_has_numeric_claim("最低价（2023 元）。"))


class LimitationSeparationTest(unittest.TestCase):
    """Disclosed inherent limits must not block `verified`; gaps must."""

    def _run(self, issues, report_unresolved=None):
        def json_call(name, _schema, _prompt):
            if name == "PlanDraft":
                return {
                    "steps": [
                        {
                            "step_id": "s1",
                            "tool_name": "search_policy",
                            "arguments": {"query": "负电价", "policy_at": "2025-10-01"},
                            "depends_on": [],
                            "rationale": "查政策候选",
                        }
                    ]
                }
            if name == "ReflectionDraft":
                return {"sufficient_evidence": True, "issues": issues}
            if name == "MarketReportDraft":
                return {
                    "summary": "政策候选未核验，不作为依据。",
                    "known_limitations": ["气象仅济南单点。"],
                    "unresolved_issues": report_unresolved or [],
                }
            raise AssertionError(f"Unexpected model schema: {name}")

        def tool_call(name, _arguments):
            return {"tool": name, "status": "unverified_candidates", "data": {}, "limitations": []}

        agent = MarketAnalysisAgent(json_call=json_call, tool_call=tool_call)
        return agent.analyze(
            question="负电价有政策依据吗",
            as_of=datetime(2025, 10, 2, tzinfo=timezone.utc),
            mode="historical_analysis",
        )["report"]

    def test_limitations_are_disclosed_and_do_not_block_verified(self) -> None:
        report = self._run(
            [
                {
                    "code": "SLOT_SEMANTICS",
                    "description": "槽位区间语义未确认。",
                    "repairable": False,
                    "kind": "limitation",
                }
            ]
        )
        self.assertEqual(report["validation"]["status"], "passed")
        self.assertEqual(report["unresolved_issues"], [])
        self.assertIn("槽位区间语义未确认。", report["known_limitations"])
        self.assertIn("气象仅济南单点。", report["known_limitations"])
        # The deterministic unverified-policy issue is a limitation, not a gap.
        self.assertTrue(any("政策候选" in item for item in report["known_limitations"]))
        self.assertTrue(report["verified"])

    def test_issue_without_kind_is_a_gap(self) -> None:
        report = self._run(
            [{"code": "MISSING_DAY", "description": "缺少目标日数据。", "repairable": False}]
        )
        self.assertEqual(report["unresolved_issues"], ["缺少目标日数据。"])
        self.assertFalse(report["verified"])

    def _reflect_on_drivers(self, uncovered):
        agent = MarketAnalysisAgent(
            json_call=lambda *_: {"sufficient_evidence": True, "issues": []},
            tool_call=lambda *_: {},
        )
        observation = {
            "step_id": "s1",
            "evidence_id": "ev_d",
            "tool_name": "explain_drivers",
            "arguments": {},
            "elapsed_ms": 1.0,
            "error": None,
            "execution_status": "ok",
            "result": {
                "tool": "explain_drivers",
                "status": "ok",
                "data": {
                    "event_count": 3,
                    "detected_event_count": 13,
                    "event_type_coverage": {"uncovered_price_event_types": uncovered},
                },
            },
        }
        return agent._reflector(
            {"question": "q", "as_of": "2025-10-08T00:00:00+08:00", "mode": "historical_analysis",
             "observations": [observation]}
        )

    def test_driver_sampling_covering_all_types_is_a_limitation(self) -> None:
        update = self._reflect_on_drivers([])
        self.assertEqual(update["unresolved_issues"], [])
        self.assertTrue(update["reflection"]["sufficient_evidence"])
        self.assertEqual(len(update["known_limitations"]), 1)

    def test_driver_uncovered_event_type_is_a_repairable_gap(self) -> None:
        update = self._reflect_on_drivers(["price_jump_candidate"])
        self.assertTrue(update["reflection"]["repairable"])
        self.assertFalse(update["reflection"]["sufficient_evidence"])
        self.assertIn("price_jump_candidate", update["unresolved_issues"][0])

    def test_report_side_gap_blocks_verified(self) -> None:
        report = self._run([], report_unresolved=["结论与原始数据不一致。"])
        self.assertIn("结论与原始数据不一致。", report["unresolved_issues"])
        self.assertFalse(report["verified"])


class AgentVariantTest(unittest.TestCase):
    REPORT = {"summary": "完成核查。", "anomalies": [], "causes": [], "unresolved_issues": []}

    @staticmethod
    def _tool(name, _arguments):
        return {"tool": name, "status": "ok", "data": {"mean_price": 1.0}, "limitations": []}

    def test_e0_react_loop_calls_tools_one_at_a_time_then_reports(self) -> None:
        turns = iter([
            {"action": "call_tool", "tool_name": "query_stats", "arguments": {
                "start_date": "2025-10-01", "end_date": "2025-10-01"}},
            {"action": "call_tool", "tool_name": "no_such_tool", "arguments": {}},
            {"action": "finish"},
        ])
        seen = []

        def json_call(name, _schema, prompt):
            seen.append(name)
            return next(turns) if name == "ReactDraft" else _cited_report(prompt)

        def tool_call(name, arguments):
            if name == "no_such_tool":
                raise ValueError("UNKNOWN_MARKET_TOOL: no_such_tool")
            return self._tool(name, arguments)

        result = MarketAnalysisAgent(json_call=json_call, tool_call=tool_call, variant="E0").analyze(
            question="q", as_of=datetime(2025, 10, 2, tzinfo=timezone.utc), mode="historical_analysis")
        self.assertEqual(seen, ["ReactDraft", "ReactDraft", "ReactDraft", "MarketReportDraft"])
        self.assertEqual([o["execution_status"] for o in result["trace"]["observations"]], ["ok", "error"])
        self.assertEqual(result["trace"]["variant"], "E0")
        self.assertNotIn("PlanDraft", seen)

    def test_e1_skips_reflection(self) -> None:
        seen = []

        def json_call(name, _schema, prompt):
            seen.append(name)
            if name == "PlanDraft":
                return {"steps": [{"step_id": "s1", "tool_name": "query_stats", "arguments": {
                    "start_date": "2025-10-01", "end_date": "2025-10-01"}, "depends_on": [], "rationale": "r"}]}
            return _cited_report(prompt)

        result = MarketAnalysisAgent(json_call=json_call, tool_call=self._tool, variant="E1").analyze(
            question="q", as_of=datetime(2025, 10, 2, tzinfo=timezone.utc), mode="historical_analysis")
        self.assertEqual(seen, ["PlanDraft", "MarketReportDraft"])
        self.assertEqual(result["report"]["validation"]["status"], "passed")
        self.assertFalse(result["report"]["verified"])  # no reflection, so never verified

    def test_e3_turns_effective_filter_off_inside_tool_calls(self) -> None:
        from ellectric.service.market_tools import POLICY_EFFECTIVE_FILTER

        states = []
        agent = MarketAnalysisAgent(json_call=lambda *_: {}, tool_call=lambda *_: states.append(
            POLICY_EFFECTIVE_FILTER.get()) or {}, variant="E3")
        agent.tool_call("search_policy", {})
        self.assertEqual(states, [False])
        self.assertTrue(POLICY_EFFECTIVE_FILTER.get())


class StepReferenceTest(unittest.TestCase):
    """A plan step can take an argument from an earlier step's result ("$ref:s1:..."), bound before it runs."""

    REPORT = {"summary": "完成核查。", "anomalies": [], "causes": [], "unresolved_issues": []}
    PLAN = {"steps": [
        {"step_id": "s1", "tool_name": "query_stats", "arguments": {
            "start_date": "2025-02-01", "end_date": "2025-02-28", "group_by": "day"}, "depends_on": [],
         "rationale": "定位日均价最高的一天"},
        {"step_id": "s2", "tool_name": "query_load", "arguments": {
            "start_date": "$ref:s1:日均价最高的那一天", "end_date": "$ref:s1:日均价最高的那一天"},
         "depends_on": [], "rationale": "单查那一天的负荷"},
    ]}

    def _run(self, bound):
        seen, calls = [], []

        def json_call(name, _schema, prompt):
            seen.append((name, prompt))
            if name == "PlanDraft":
                return self.PLAN
            if name == "BoundArguments":
                return {"arguments": bound}
            return _cited_report(prompt)

        def tool_call(name, arguments):
            calls.append((name, arguments))
            return {"tool": name, "status": "ok", "data": {"groups": [
                {"group_value": "2025-02-27", "mean_price": 300.0}, {"group_value": "2025-02-28", "mean_price": 316.3}]},
                "limitations": []}

        result = MarketAnalysisAgent(json_call=json_call, tool_call=tool_call, variant="E1").analyze(
            question="2月哪天最贵？那天负荷峰值？", as_of=datetime(2025, 3, 1, 9, tzinfo=timezone(timedelta(hours=8))),
            mode="historical_analysis")
        return result, seen, calls

    def test_reference_is_bound_from_the_referenced_result(self) -> None:
        result, seen, calls = self._run({"start_date": "2025-02-28", "end_date": "2025-02-28"})
        self.assertEqual([name for name, _ in seen], ["PlanDraft", "BoundArguments", "MarketReportDraft"])
        self.assertIn("2025-02-28", seen[1][1])  # the binder sees the referenced step's result
        self.assertEqual(calls[1][0], "query_load")
        self.assertEqual((calls[1][1]["start_date"], calls[1][1]["end_date"]), ("2025-02-28", "2025-02-28"))
        self.assertIn("as_of", calls[1][1])
        plan = result["trace"]["plan"]
        self.assertEqual(plan[1]["depends_on"], ["s1"])  # added from the reference
        self.assertEqual(plan[1]["planned_arguments"]["start_date"], "$ref:s1:日均价最高的那一天")
        self.assertIn("argument_binding", [event["type"] for event in result["trace"]["events"]])

    def test_unresolved_reference_fails_the_step_without_calling_the_tool(self) -> None:
        result, _seen, calls = self._run({"start_date": "$ref:s1:日均价最高的那一天", "end_date": "2025-02-28"})
        self.assertEqual([name for name, _ in calls], ["query_stats"])
        failed = result["trace"]["observations"][1]
        self.assertEqual(failed["execution_status"], "error")
        self.assertTrue(failed["error"].startswith("ARGUMENT_BINDING_FAILED"))

    def test_reference_must_point_at_an_earlier_step(self) -> None:
        from ellectric.llm.market_analysis import PlanDraft

        plan = PlanDraft.model_validate({"steps": [
            {"step_id": "s1", "tool_name": "query_load", "arguments": {"start_date": "$ref:s2:那一天"},
             "depends_on": [], "rationale": "r"},
            {"step_id": "s2", "tool_name": "query_stats", "arguments": {}, "depends_on": [], "rationale": "r"}]})
        with self.assertRaises(ValueError):
            MarketAnalysisAgent._validate_plan(plan, set())


class CompactionRowKeyTest(unittest.TestCase):
    def test_range_extremes_carry_their_row_keys(self) -> None:
        groups = [{"group_value": f"2025-02-{day:02d}", "mean_price": float(day)} for day in range(1, 29)]
        ranges = _compact_for_prompt({"groups": groups})["groups"]["numeric_ranges"]["mean_price"]
        self.assertEqual(ranges["max_pointer"], "/groups/27/mean_price")
        self.assertEqual(ranges["max_row"], {"group_value": "2025-02-28"})
        self.assertEqual(ranges["min_row"], {"group_value": "2025-02-01"})


class TimeBudgetTest(unittest.TestCase):
    def test_waiting_for_the_request_quota_is_not_charged(self) -> None:
        import time as _time

        agent = MarketAnalysisAgent(json_call=lambda *_: {}, tool_call=lambda *_: {})
        started = _time.monotonic() - 200.0
        agent.rate_limit_wait_seconds = 150.0
        self.assertLess(agent._work_seconds(started), 60.0)
        agent.rate_limit_wait_seconds = 0.0
        self.assertGreater(agent._work_seconds(started), 180.0)


class SchemaRetryTest(unittest.TestCase):
    def test_off_schema_output_gets_one_corrective_retry(self) -> None:
        prompts = []
        outputs = iter([{"summary": "s", "extra_field": 1}, {"summary": "s"}])

        def json_call(_name, _schema, prompt):
            prompts.append(prompt)
            return next(outputs)

        agent = MarketAnalysisAgent(json_call=json_call, tool_call=lambda *_: {})
        from ellectric.llm.market_analysis import MarketReportDraft

        draft = agent._json("MarketReportDraft", MarketReportDraft, "write")
        self.assertEqual(draft.summary, "s")
        self.assertIn("extra_field", prompts[1])
        self.assertIn("known_limitations，不是 limitations", prompts[1])
        self.assertEqual(agent.schema_retry_count, 1)

    def test_limitations_key_is_read_as_known_limitations(self) -> None:
        from pydantic import ValidationError

        from ellectric.llm.market_analysis import MarketReportDraft

        draft = MarketReportDraft.model_validate(
            {"summary": "s", "known_limitations": ["a"], "limitations": ["b", "a"]}
        )
        self.assertEqual(draft.known_limitations, ["a", "b"])
        self.assertNotIn("limitations", draft.model_dump())
        with self.assertRaises(ValidationError):
            MarketReportDraft.model_validate({"summary": "s", "notes": ["x"]})  # other extra keys still fail

    def test_second_failure_still_raises(self) -> None:
        agent = MarketAnalysisAgent(json_call=lambda *_: {"bad": 1}, tool_call=lambda *_: {})
        from ellectric.llm.market_analysis import MarketReportDraft

        with self.assertRaises(ValueError):
            agent._json("MarketReportDraft", MarketReportDraft, "write")


class RateLimitTest(unittest.TestCase):
    def setUp(self) -> None:
        import ellectric.llm.market_analysis as module

        self.module = module
        module._RATE_TIMES.clear()

    def tearDown(self) -> None:
        self.module._RATE_TIMES.clear()

    def test_sliding_window_blocks_when_full(self) -> None:
        sleeps = []
        clock = [100.0]

        def fake_sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        with mock.patch.dict(os.environ, {"ELLECTRIC_LLM_MAX_RPM": "2"}), mock.patch.object(
            self.module.time, "monotonic", side_effect=lambda: clock[0]
        ), mock.patch.object(self.module.time, "sleep", side_effect=fake_sleep):
            self.assertEqual(self.module._acquire_rate_slot(), 0.0)
            self.assertEqual(self.module._acquire_rate_slot(), 0.0)
            waited = self.module._acquire_rate_slot()
        self.assertGreaterEqual(waited, 60.0)
        self.assertEqual(len(sleeps), 1)

    def test_disabled_by_default(self) -> None:
        with mock.patch.dict(os.environ, {"ELLECTRIC_LLM_MAX_RPM": "0"}):
            for _ in range(50):
                self.assertEqual(self.module._acquire_rate_slot(), 0.0)

    def test_http_429_is_retried_then_succeeds(self) -> None:
        import io
        import json as json_module
        import urllib.error

        from ellectric.llm.market_analysis import MarketReportDraft

        body = json_module.dumps(
            {"choices": [{"message": {"content": json_module.dumps({"summary": "s"})}}], "usage": {"total_tokens": 3}}
        ).encode()

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        calls = []

        def fake_urlopen(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)
            return Response(body)

        agent = MarketAnalysisAgent(tool_call=lambda *_: {})
        with mock.patch.dict(os.environ, {"ELLECTRIC_LLM_API_KEY": "test-key", "ELLECTRIC_LLM_MAX_RPM": "0"}), mock.patch.object(
            self.module.urllib.request, "urlopen", side_effect=fake_urlopen
        ), mock.patch.object(self.module.time, "sleep"):
            draft = agent._json_once("MarketReportDraft", MarketReportDraft, "write")
        self.assertEqual(draft.summary, "s")
        self.assertEqual(len(calls), 2)
        self.assertEqual(agent.rate_limit_retry_count, 1)

    def test_requests_carry_output_token_cap(self) -> None:
        import io
        import json as json_module

        from ellectric.llm.market_analysis import MarketReportDraft

        sent = []
        body = json_module.dumps({"choices": [{"message": {"content": json_module.dumps({"summary": "s"})}}]}).encode()

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout):
            sent.append(json_module.loads(request.data.decode("utf-8")))
            return Response(body)

        agent = MarketAnalysisAgent(tool_call=lambda *_: {})
        with mock.patch.dict(os.environ, {"ELLECTRIC_LLM_API_KEY": "test-key", "ELLECTRIC_LLM_MAX_RPM": "0"}), mock.patch.object(
            self.module.urllib.request, "urlopen", side_effect=fake_urlopen
        ):
            os.environ.pop("ELLECTRIC_LLM_MAX_TOKENS", None)
            agent._json_once("MarketReportDraft", MarketReportDraft, "write")
        self.assertEqual(sent[0]["max_tokens"], 4000)

    def test_reasoning_tokens_are_counted(self) -> None:
        import io
        import json as json_module

        from ellectric.llm.market_analysis import MarketReportDraft

        body = json_module.dumps({
            "choices": [{"message": {"content": json_module.dumps({"summary": "s"})}}],
            "usage": {"total_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 7}},
        }).encode()

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        agent = MarketAnalysisAgent(tool_call=lambda *_: {})
        with mock.patch.dict(os.environ, {"ELLECTRIC_LLM_API_KEY": "test-key", "ELLECTRIC_LLM_MAX_RPM": "0"}), mock.patch.object(
            self.module.urllib.request, "urlopen", side_effect=lambda request, timeout: Response(body)
        ):
            agent._json_once("MarketReportDraft", MarketReportDraft, "write")
        self.assertEqual(agent.reasoning_token_count, 7)

    def test_http_429_gives_up_after_retries(self) -> None:
        import urllib.error

        from ellectric.llm.market_analysis import MarketReportDraft

        def always_429(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

        agent = MarketAnalysisAgent(tool_call=lambda *_: {})
        with mock.patch.dict(os.environ, {"ELLECTRIC_LLM_API_KEY": "test-key", "ELLECTRIC_LLM_MAX_RPM": "0"}), mock.patch.object(
            self.module.urllib.request, "urlopen", side_effect=always_429
        ), mock.patch.object(self.module.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                agent._json_once("MarketReportDraft", MarketReportDraft, "write")
        self.assertEqual(agent.rate_limit_retry_count, len(self.module.RATE_LIMIT_RETRY_DELAYS))

    def test_http_400_is_retried_once_with_same_body(self) -> None:
        import io
        import json as json_module
        import urllib.error

        from ellectric.llm.market_analysis import MarketReportDraft

        body = json_module.dumps({"choices": [{"message": {"content": json_module.dumps({"summary": "s"})}}]}).encode()

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        sent = []

        def fake_urlopen(request, timeout):
            sent.append(request.data)
            if len(sent) == 1:
                raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)
            return Response(body)

        agent = MarketAnalysisAgent(tool_call=lambda *_: {})
        with mock.patch.dict(os.environ, {"ELLECTRIC_LLM_API_KEY": "test-key", "ELLECTRIC_LLM_MAX_RPM": "0"}), mock.patch.object(
            self.module.urllib.request, "urlopen", side_effect=fake_urlopen
        ):
            draft = agent._json_once("MarketReportDraft", MarketReportDraft, "write")
        self.assertEqual(draft.summary, "s")
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0], sent[1])
        self.assertEqual(agent.bad_request_retry_count, 1)

    def test_http_400_gives_up_after_one_retry(self) -> None:
        import urllib.error

        from ellectric.llm.market_analysis import MarketReportDraft

        calls = []

        def always_400(request, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)

        agent = MarketAnalysisAgent(tool_call=lambda *_: {})
        with mock.patch.dict(os.environ, {"ELLECTRIC_LLM_API_KEY": "test-key", "ELLECTRIC_LLM_MAX_RPM": "0"}), mock.patch.object(
            self.module.urllib.request, "urlopen", side_effect=always_400
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                agent._json_once("MarketReportDraft", MarketReportDraft, "write")
        self.assertEqual(len(calls), 1 + self.module.BAD_REQUEST_RETRIES)

    def test_markdown_output_is_counted_and_retried_with_non_json_hint(self) -> None:
        import io
        import json as json_module

        from ellectric.llm.market_analysis import MarketReportDraft

        replies = [
            {"choices": [{"message": {"content": "# 周报\n\n本周电价..."}, "finish_reason": "length"}],
             "usage": {"total_tokens": 4100}},
            {"choices": [{"message": {"content": json_module.dumps({"summary": "s"})}}], "usage": {"total_tokens": 900}},
        ]

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        sent = []

        def fake_urlopen(request, timeout):
            sent.append(json_module.loads(request.data.decode("utf-8")))
            return Response(json_module.dumps(replies[len(sent) - 1]).encode())

        agent = MarketAnalysisAgent(tool_call=lambda *_: {})
        with mock.patch.dict(os.environ, {"ELLECTRIC_LLM_API_KEY": "test-key", "ELLECTRIC_LLM_MAX_RPM": "0"}), mock.patch.object(
            self.module.urllib.request, "urlopen", side_effect=fake_urlopen
        ):
            draft = agent._json("MarketReportDraft", MarketReportDraft, "write")
        self.assertEqual(draft.summary, "s")
        self.assertIn("Markdown", sent[0]["messages"][0]["content"])
        retry_prompt = sent[1]["messages"][1]["content"]
        self.assertIn("finish_reason=length", retry_prompt)
        self.assertIn("根本不是 JSON", retry_prompt)
        self.assertEqual(agent.non_json_output_count, 1)
        self.assertEqual(agent._last_token_count, 5000)  # the failed reply's tokens still count


class StatementCheckTest(unittest.TestCase):
    def _observation(self):
        return {
            "evidence_id": "ev_d",
            "result": {
                "data": {
                    "contributions": [
                        {"component": "load", "actual_vs_forecast": "above", "bid_space_effect": "increase",
                         "actual_minus_forecast_bid_space_contribution_mw": 5558.087},
                        {"component": "wind", "actual_vs_forecast": "above", "bid_space_effect": "decrease",
                         "actual_minus_forecast_bid_space_contribution_mw": -1152.13},
                    ]
                }
            },
        }

    def _errors(self, statement, bindings):
        report = {
            "summary": "概述。",
            "anomalies": [],
            "causes": [{"category": "cause", "statement": statement, "evidence_ids": ["ev_d"],
                        "value_bindings": bindings}],
        }
        return MarketAnalysisAgent._validate_report(report, [self._observation()])

    def _wind(self):
        return {"evidence_id": "ev_d", "json_pointer": "/data/contributions/1/actual_minus_forecast_bid_space_contribution_mw",
                "value": -1152.13}

    def test_unbound_statement_number_is_rejected(self) -> None:
        errors = self._errors("风电实际值高于预测，贡献 -1152.13 MW，另有 42 个时段。", [self._wind()])
        self.assertTrue(any("[\'42\']" in error for error in errors), errors)

    def test_rounded_and_unsigned_numbers_are_accepted(self) -> None:
        self.assertEqual(self._errors("风电实际值高于预测，形成约 1152 MW 负贡献。", [self._wind()]), [])

    def test_wrong_direction_word_is_rejected(self) -> None:
        errors = self._errors("风电实际值较预测低 1152.13 MW。", [self._wind()])
        self.assertTrue(any("方向描述与证据不符" in error and "wind" in error for error in errors), errors)

    def test_wrong_contribution_sign_word_is_rejected(self) -> None:
        errors = self._errors("风电实际值高于预测 1152.13 MW，构成正贡献。", [self._wind()])
        self.assertTrue(any("负贡献" in error for error in errors), errors)

    def test_percent_and_identifiers_are_handled(self) -> None:
        binding = {"evidence_id": "ev_d", "json_pointer": "/data/contributions/0/actual_minus_forecast_bid_space_contribution_mw",
                   "value": 5558.087}
        self.assertEqual(self._errors("负荷实际值高于预测 5558.087 MW，属 P99 以上的 D+1 偏差。", [binding]), [])


class ReportMarkdownTest(unittest.TestCase):
    def test_markdown_lists_findings_bindings_and_limitations(self) -> None:
        from ellectric.llm.report_markdown import render_report_markdown

        text = render_report_markdown(
            {
                "summary": "概述。",
                "anomalies": [{"statement": "最低价 -80 元/MWh。", "value_bindings": [
                    {"evidence_id": "ev_a", "json_pointer": "/data/min", "value": -80}], "limitations": []}],
                "causes": [],
                "known_limitations": ["气象仅济南。"],
                "unresolved_issues": [],
                "verified": True,
                "validation": {"status": "passed", "errors": []},
            },
            request={"question": "q", "as_of": "2025-10-08T00:00:00+08:00", "mode": "historical_analysis"},
        )
        self.assertIn("**已验证**", text)
        self.assertIn("`-80` ← `ev_a` `/data/min`", text)
        self.assertIn("## 已知限制\n\n- 气象仅济南。", text)
        self.assertIn("## 未解决问题\n\n- 无", text)


class ThinkingConfigTest(unittest.TestCase):
    def test_default_disables_thinking(self) -> None:
        self.assertEqual(_thinking_request_fields("disabled"), {"thinking": {"type": "disabled"}})

    def test_effort_levels_map_to_reasoning_effort(self) -> None:
        self.assertEqual(_thinking_request_fields("low"), {"reasoning_effort": "low"})
        self.assertEqual(_thinking_request_fields("default"), {})

    def test_unknown_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _thinking_request_fields("minimal")


class StatementNumberExtractionTest(unittest.TestCase):
    """Date and clock-time references are stripped without swallowing the numbers written next to them."""

    def test_month_followed_by_a_count_keeps_the_count(self) -> None:
        from ellectric.llm.market_analysis import _statement_numbers

        self.assertEqual(_statement_numbers("2025年2月716个，2025年3月（截至3月16日）344个"), ["716", "344"])
        self.assertEqual(_statement_numbers("2024年8月76个、2024年9月107个"), ["76", "107"])
        self.assertEqual(_statement_numbers("2024年7月18日至9月30日合计 240 个"), ["240"])
        self.assertEqual(_statement_numbers("2024-02-16 共 23 个时段"), ["23"])

    def test_full_timestamps_are_references(self) -> None:
        from ellectric.llm.market_analysis import _statement_numbers

        self.assertEqual(_statement_numbers("最低 -5.2 degC，出现在 2024-03-01 15:00:00+08。"), ["-5.2"])
        self.assertEqual(_statement_numbers("08:00+08:00 至 17:00 共 37 个时段"), ["37"])


class BoundRowConsistencyTest(unittest.TestCase):
    """A value written next to a day/time must be bound from the row of that day/time."""

    OBSERVATION = {
        "evidence_id": "ev_w",
        "result": {"data": {"group_by": "slot", "min_value": 320.55, "groups": [
            {"group_value": 39, "mean_value": 491.92,
             "min_at": {"trade_date": "2025-09-25", "slot_id": 39, "time_label": "10:00", "wind_actual_mw": 491.92}},
            {"group_value": 49, "mean_value": 320.55,
             "min_at": {"trade_date": "2025-09-25", "slot_id": 49, "time_label": "12:30", "wind_actual_mw": 320.55}},
        ]}},
    }
    DAYS = {
        "evidence_id": "ev_d",
        "result": {"data": {"group_by": "day", "groups": [
            {"group_value": "2024-08-26", "negative_price_slots": 24},
            {"group_value": "2024-08-27", "negative_price_slots": 30},
        ]}},
    }

    def _errors(self, statement, *bindings):
        report = {"summary": "概述。", "anomalies": [], "causes": [{
            "category": "cause", "statement": statement, "evidence_ids": sorted({b["evidence_id"] for b in bindings}),
            "value_bindings": list(bindings)}]}
        return MarketAnalysisAgent._validate_report(report, [self.OBSERVATION, self.DAYS])

    @staticmethod
    def _bind(evidence_id, pointer, value):
        return {"evidence_id": evidence_id, "json_pointer": pointer, "value": value}

    def test_value_from_another_slot_is_rejected(self) -> None:
        errors = self._errors("2025-09-25 10:00 风电实际出力为 320.55 MW。", self._bind("ev_w", "/data/groups/1/mean_value", 320.55))
        self.assertTrue(any("时刻 12:30" in error for error in errors), errors)
        self.assertEqual(self._errors("2025-09-25 10:00 风电实际出力为 491.92 MW。",
                                      self._bind("ev_w", "/data/groups/0/mean_value", 491.92)), [])

    def test_slot_mention_counts_as_its_clock_time(self) -> None:
        self.assertEqual(self._errors("最低价时段（slot 39）风电出力 491.92 MW。",
                                      self._bind("ev_w", "/data/groups/0/min_at/wind_actual_mw", 491.92)), [])
        errors = self._errors("最低价时段（slot 39）风电出力 320.55 MW。",
                              self._bind("ev_w", "/data/groups/1/min_at/wind_actual_mw", 320.55))
        self.assertTrue(any("时刻 12:30" in error for error in errors), errors)

    def test_value_from_another_day_is_rejected(self) -> None:
        errors = self._errors("2024年8月27日负电价时段 24 个。", self._bind("ev_d", "/data/groups/0/negative_price_slots", 24))
        self.assertTrue(any("日期 2024-08-26" in error for error in errors), errors)
        self.assertEqual(self._errors("8月27日负电价时段 30 个，8月26日 24 个。",
                                      self._bind("ev_d", "/data/groups/1/negative_price_slots", 30),
                                      self._bind("ev_d", "/data/groups/0/negative_price_slots", 24)), [])

    def test_rowless_values_and_unwritten_operands_are_not_checked(self) -> None:
        # Totals have no row; an operand that only feeds a derived value is not a claim about the named day.
        self.assertEqual(self._errors("2025-09-25 10:00 前后风电当日最低 320.55 MW。",
                                      self._bind("ev_w", "/data/min_value", 320.55)), [])
        report = {"summary": "概述。", "anomalies": [], "causes": [{
            "category": "cause", "statement": "8月27日负电价时段 30 个，比前一日多 6 个。", "evidence_ids": ["ev_d"],
            "value_bindings": [self._bind("ev_d", "/data/groups/1/negative_price_slots", 30)],
            "derived_values": [{"operation": "difference", "value": 6, "operands": [
                self._bind("ev_d", "/data/groups/1/negative_price_slots", 30),
                self._bind("ev_d", "/data/groups/0/negative_price_slots", 24)]}]}]}
        self.assertEqual(MarketAnalysisAgent._validate_report(report, [self.DAYS]), [])

    def test_extreme_sits_at_its_located_row(self) -> None:
        day = {"evidence_id": "ev_x", "result": {"data": {
            "period": ["2024-01-16", "2024-01-16"], "max_price_yuan_per_mwh": 1300.0, "min_value": 2.0,
            "max_at": {"trade_date": "2024-01-16", "slot_id": 40, "time_label": "10:15", "rt_price": 1300.0},
            "min_at": {"timestamp_raw": "2024-01-16 12:00:00+08", "temp_jinan": 2.0}}}}

        def errors(statement, field, value):
            report = {"summary": "概述。", "anomalies": [], "causes": [{
                "category": "cause", "statement": statement, "evidence_ids": ["ev_x"],
                "value_bindings": [self._bind("ev_x", f"/data/{field}", value)]}]}
            return MarketAnalysisAgent._validate_report(report, [day])

        wrong = errors("2024-01-16 当日最高实时价格为 1300.0 元/MWh，出现在 11:15。", "max_price_yuan_per_mwh", 1300.0)
        self.assertTrue(any("时刻 10:15" in error for error in wrong), wrong)
        self.assertEqual(errors("2024-01-16 当日最高价 1300.0 元/MWh，出现在 10:15。", "max_price_yuan_per_mwh", 1300.0), [])
        # A clock time belongs to its own clause: the minimum below is given without a time.
        self.assertEqual(errors("2024-01-16 最高价出现在 10:15；当日最低气温为 2.0 摄氏度。", "min_value", 2.0), [])

    def test_day_range_names_every_day_in_it(self) -> None:
        self.assertEqual(self._errors("8月26日至27日负电价时段分别为 24、30 个。",
                                      self._bind("ev_d", "/data/groups/0/negative_price_slots", 24),
                                      self._bind("ev_d", "/data/groups/1/negative_price_slots", 30)), [])
        errors = self._errors("8月27日至28日负电价时段分别为 24、30 个。",
                              self._bind("ev_d", "/data/groups/0/negative_price_slots", 24))
        self.assertTrue(any("日期 2024-08-26" in error for error in errors), errors)


class EmptyReportTest(unittest.TestCase):
    """Data came back, so the answer must be stated as findings, or the report must say why it cannot be."""

    OBSERVATION = {"evidence_id": "ev_s", "execution_status": "ok",
                   "result": {"status": "ok", "data": {"max_value": 82684.51}}}

    def _errors(self, **fields):
        report = {"summary": "负荷最高出现在 1 月 14 日傍晚。", "anomalies": [], "causes": []} | fields
        return MarketAnalysisAgent._validate_report(report, [self.OBSERVATION])

    def test_report_without_findings_is_rejected(self) -> None:
        self.assertTrue(any("没有任何发现" in error for error in self._errors()))

    def test_unresolved_or_cited_reports_are_accepted(self) -> None:
        self.assertEqual(self._errors(unresolved_issues=["区间内负荷数据缺失，无法确定峰值"]), [])
        finding = {"category": "other", "statement": "负荷峰值为 82684.51 MW。", "evidence_ids": ["ev_s"],
                   "value_bindings": [{"evidence_id": "ev_s", "json_pointer": "/data/max_value", "value": 82684.51}]}
        self.assertEqual(self._errors(causes=[finding]), [])
        unusable = dict(self.OBSERVATION, result={"status": "unverified_candidates", "data": {}})
        self.assertEqual(MarketAnalysisAgent._validate_report(
            {"summary": "政策候选未核验。", "anomalies": [], "causes": []}, [unusable]), [])


class PlanArgumentCheckTest(unittest.TestCase):
    """Literal plan arguments are validated against the tool schema at planning time, with one corrective retry."""

    REPORT = {"summary": "完成核查。", "anomalies": [], "causes": [], "unresolved_issues": []}

    @staticmethod
    def _plan(start: str, end: str) -> dict:
        return {"steps": [{"step_id": "s1", "tool_name": "query_stats", "arguments": {
            "start_date": start, "end_date": end, "group_by": "day"}, "depends_on": [], "rationale": "按日统计"}]}

    def _run(self, plans):
        seen, calls = [], []

        def json_call(name, _schema, prompt):
            seen.append((name, prompt))
            if name == "PlanDraft":
                return plans.pop(0)
            return _cited_report(prompt)

        def tool_call(name, arguments):
            calls.append((name, arguments))
            return {"tool": name, "status": "ok", "data": {"groups": []}, "limitations": []}

        result = MarketAnalysisAgent(json_call=json_call, tool_call=tool_call, variant="E1").analyze(
            question="2月每天的负电价时段数？", as_of=datetime(2025, 3, 1, 9, tzinfo=timezone(timedelta(hours=8))),
            mode="historical_analysis")
        return result, seen, calls

    def test_invalid_literal_argument_is_corrected_before_execution(self) -> None:
        result, seen, calls = self._run([self._plan("2025-02-30", "2025-02-30"), self._plan("2025-02-01", "2025-02-28")])
        self.assertEqual([name for name, _ in seen], ["PlanDraft", "PlanDraft", "MarketReportDraft"])
        self.assertIn("s1 query_stats 参数不合法", seen[1][1])
        self.assertIn("'2025-02-30'", seen[1][1])
        self.assertEqual(len(calls), 1)  # the invalid version never ran
        self.assertEqual(calls[0][1]["start_date"], "2025-02-01")
        self.assertEqual(result["trace"]["schema_retry_count"], 1)

    def test_reference_arguments_are_left_to_the_binder(self) -> None:
        plan = {"steps": [
            {"step_id": "s1", "tool_name": "query_stats", "arguments": {
                "start_date": "2025-02-01", "end_date": "2025-02-28", "group_by": "day"}, "depends_on": [],
             "rationale": "定位"},
            {"step_id": "s2", "tool_name": "query_load", "arguments": {
                "start_date": "$ref:s1:那一天", "end_date": "$ref:s1:那一天"}, "depends_on": [], "rationale": "单查"}]}
        agent = MarketAnalysisAgent(json_call=lambda *_: plan, tool_call=lambda *_: {}, variant="E1")
        from ellectric.llm.market_analysis import PlanDraft

        steps = agent._checked_plan(PlanDraft.model_validate(plan), set(), "2025-03-01T09:00:00+08:00")
        self.assertEqual([step.depends_on for step in steps], [[], ["s1"]])

    def test_plan_with_invalid_arguments_after_the_retry_still_runs(self) -> None:
        """Both plans carry an argument the tool rejects: the run goes on, the step fails in the executor with
        the tool's own error, and a report is written (before, the plan check error ended the whole run)."""
        from ellectric.service.market_tools import TOOL_MODELS

        def json_call(name, _schema, prompt):
            if name == "PlanDraft":
                return self._plan("2025-03-05", "2025-03-05")  # after the last trade date available at as_of
            return _cited_report(prompt)

        def tool_call(name, arguments):
            TOOL_MODELS[name].model_validate(arguments)
            return {"tool": name, "status": "ok", "data": {"groups": []}, "limitations": []}

        result = MarketAnalysisAgent(json_call=json_call, tool_call=tool_call, variant="E1").analyze(
            question="3月5日均价？", as_of=datetime(2025, 3, 1, 9, tzinfo=timezone(timedelta(hours=8))),
            mode="historical_analysis")
        observation = result["trace"]["observations"][0]
        self.assertEqual(observation["execution_status"], "error")
        self.assertIn("DATA_AFTER_AS_OF", observation["error"])
        plan_event = next(event for event in result["trace"]["events"] if event["type"] == "plan")
        self.assertIn("DATA_AFTER_AS_OF", plan_event["argument_errors"])
        self.assertIn("summary", result["report"])

    def test_unusable_plan_is_recorded_and_left_to_the_replanner(self) -> None:
        """A plan that is not runnable at all (duplicate step ids, twice) leaves no plan: the failure becomes a
        repairable issue, and the replanner writes the plan."""
        step = self._plan("2025-02-01", "2025-02-28")["steps"][0]
        plans = [{"steps": [step, step]}, {"steps": [step, step]}, self._plan("2025-02-01", "2025-02-28")]
        seen = []

        def json_call(name, _schema, prompt):
            seen.append(name)
            if name == "PlanDraft":
                return plans.pop(0)
            if name == "ReflectionDraft":
                return {"sufficient_evidence": bool(re.search(r"ev_[0-9a-f]{6,}", prompt)), "issues": []}
            return _cited_report(prompt)

        result = MarketAnalysisAgent(
            json_call=json_call,
            tool_call=lambda name, _arguments: {"tool": name, "status": "ok", "data": {"groups": []}, "limitations": []},
        ).analyze(question="2月每天的负电价时段数？", as_of=datetime(2025, 3, 1, 9, tzinfo=timezone(timedelta(hours=8))),
                  mode="historical_analysis")
        events = [event["type"] for event in result["trace"]["events"]]
        self.assertIn("plan_failed", events)
        first_reflection = next(e for e in result["trace"]["events"] if e["type"] == "reflection")["reflection"]
        self.assertEqual([issue["code"] for issue in first_reflection["issues"]], ["PLAN_INVALID"])
        self.assertEqual(result["trace"]["replan_count"], 1)
        self.assertEqual(result["trace"]["tool_call_count"], 1)
        self.assertEqual(result["report"]["validation"]["status"], "passed")

    def test_invalid_reflection_or_react_output_does_not_end_the_run(self) -> None:
        def json_call(name, _schema, prompt):
            if name == "PlanDraft":
                return self._plan("2025-02-01", "2025-02-28")
            if name in {"ReflectionDraft", "ReactDraft"}:
                return {"not": "the schema"}
            return _cited_report(prompt)

        def tool_call(name, _arguments):
            return {"tool": name, "status": "ok", "data": {"groups": []}, "limitations": []}

        for variant in ("E2", "E0"):
            with self.subTest(variant=variant):
                result = MarketAnalysisAgent(json_call=json_call, tool_call=tool_call, variant=variant).analyze(
                    question="2月每天的负电价时段数？",
                    as_of=datetime(2025, 3, 1, 9, tzinfo=timezone(timedelta(hours=8))), mode="historical_analysis")
                self.assertIn("summary", result["report"])
                self.assertFalse(result["report"]["verified"])


class ScopeAndDirectAnswerTest(unittest.TestCase):
    """The question is checked against the data catalog before any tool call, and the report must answer
    each item the check listed."""

    AS_OF = datetime(2025, 3, 1, 9, tzinfo=timezone(timedelta(hours=8)))
    OBSERVATIONS = [{"evidence_id": "ev_a", "execution_status": "ok",
                     "result": {"status": "ok", "data": {"min_price": -80.0, "min_at": {"time_label": "12:00"}}}}]

    @staticmethod
    def _item(item_id, status="answerable", answer_type="number", note=""):
        return {"item_id": item_id, "item": f"第 {item_id} 项", "answer_type": answer_type, "status": status, "note": note}

    @staticmethod
    def _answer(item_id, **fields):
        return {"item_id": item_id, "status": "answered", "statement": "最低价为 -80 元/MWh。", "evidence_ids": ["ev_a"],
                "value_bindings": [{"evidence_id": "ev_a", "json_pointer": "/data/min_price", "value": -80.0}]} | fields

    def _errors(self, answers, scope):
        report = {"summary": "完成核查。", "answers": answers}
        return _MarketAnalysisAgent._validate_report(report, self.OBSERVATIONS, scope)

    def test_question_outside_the_catalog_is_refused_without_tool_calls(self) -> None:
        seen, tool_calls = [], []

        def json_call(name, _schema, prompt):
            seen.append(name)
            if name == "ScopeDraft":
                self.assertIn('"region": "山东"', prompt)
                return {"items": [{"item": "江苏3月实时均价", "answer_type": "number", "status": "unanswerable",
                                   "note": "数据只有山东，没有江苏"}]}
            if name == "MarketReportDraft":
                self.assertIn('"status": "unanswerable"', prompt)
                return {"answers": [{"item_id": "q1", "status": "unanswerable", "statement": "现有数据只有山东，没有江苏，无法回答。"}],
                        "summary": "数据不包含江苏。"}
            raise AssertionError(name)

        for variant in ("E0", "E2"):
            with self.subTest(variant=variant):
                seen.clear()
                result = _MarketAnalysisAgent(
                    json_call=json_call, tool_call=lambda *args: tool_calls.append(args), variant=variant,
                    catalog=lambda _as_of: {"region": "山东"},
                ).analyze(question="江苏3月实时均价多少？", as_of=self.AS_OF, mode="historical_analysis")
                self.assertEqual(seen, ["ScopeDraft", "MarketReportDraft"])
                self.assertEqual(tool_calls, [])
                self.assertEqual(result["report"]["validation"]["status"], "passed")
                self.assertEqual(result["trace"]["scope"][0]["status"], "unanswerable")

    def test_unusable_scope_output_fails_open(self) -> None:
        def json_call(name, _schema, prompt):
            if name == "ScopeDraft":
                return {"items": []}
            if name == "PlanDraft":
                return {"steps": [{"step_id": "s1", "tool_name": "query_stats", "arguments": {
                    "start_date": "2025-02-01", "end_date": "2025-02-28"}, "depends_on": [], "rationale": "统计"}]}
            report = _cited_report(prompt)
            return report | {"answers": [{"item_id": "q1", "status": "answered", "statement": "已作答。",
                                          "evidence_ids": report["causes"][0]["evidence_ids"]}]}

        result = _MarketAnalysisAgent(
            json_call=json_call, variant="E1", catalog=lambda _as_of: {},
            tool_call=lambda name, _arguments: {"tool": name, "status": "ok", "data": {}, "limitations": []},
        ).analyze(question="2月均价？", as_of=self.AS_OF, mode="historical_analysis")
        self.assertEqual([item["status"] for item in result["trace"]["scope"]], ["answerable"])
        self.assertEqual(result["trace"]["tool_call_count"], 1)
        self.assertEqual(result["report"]["validation"]["status"], "passed")

    def test_scope_keeps_no_method_note_on_answerable_items(self) -> None:
        def json_call(name, _schema, _prompt):
            if name == "ScopeDraft":
                return {"items": [
                    {"item": "最高价", "answer_type": "number", "status": "answerable", "note": "用 slot 79 取数"},
                    {"item": "日前均价", "answer_type": "number", "status": "unanswerable", "note": "没有日前价格"}]}
            raise AssertionError(name)

        agent = _MarketAnalysisAgent(json_call=json_call, catalog=lambda _as_of: {})
        scope = agent._scoper({"question": "问题", "as_of": self.AS_OF.isoformat(), "mode": "historical_analysis"})["scope"]
        self.assertEqual([(item["item_id"], item["note"]) for item in scope], [("q1", ""), ("q2", "没有日前价格")])

    def test_clock_hours_and_sign_conditions_are_not_numeric_claims(self) -> None:
        self.assertFalse(_has_numeric_claim("1月1日7点整、19点30分、7时的电价（rt_price<0 的时段，≥ 0 的时段）"))
        self.assertTrue(_has_numeric_claim("共 96 个时段"))
        self.assertTrue(_has_numeric_claim("价格 <0.5 元"))

    def test_every_scope_item_needs_a_direct_answer(self) -> None:
        scope = [self._item("q1"), self._item("q2")]
        errors = self._errors([self._answer("q1")], scope)
        self.assertEqual(len(errors), 1)
        self.assertIn("缺少 answers[q2]", errors[0])
        self.assertEqual(self._errors([self._answer("q1"), self._answer("q2")], scope), [])
        self.assertIn("不在 scope 中", self._errors([self._answer("q1"), self._answer("q2"), self._answer("q9")], scope)[0])

    def test_answer_must_be_of_the_kind_the_item_asks_for(self) -> None:
        no_number = self._answer("q1", statement="最低价出现在午间。", value_bindings=[])
        self.assertIn("要的是数值", self._errors([no_number], [self._item("q1")])[0])
        when = [self._item("q1", answer_type="date_or_time")]
        self.assertIn("要的是日期或时刻", self._errors([no_number], when)[0])
        self.assertEqual(self._errors([self._answer("q1", statement="最低价 -80 元/MWh 出现在 12:00。")], when), [])
        self.assertEqual(self._errors([no_number], [self._item("q1", answer_type="text")]), [])

    def test_ambiguous_item_is_answered_with_its_reading(self) -> None:
        scope = [self._item("q1", status="ambiguous", note="按 2025-02-22 至 2025-02-28 七个交易日")]
        self.assertIn("basis", self._errors([self._answer("q1")], scope)[0])
        self.assertEqual(self._errors([self._answer("q1", basis="按 2025-02-22 至 2025-02-28")], scope), [])

    def test_item_the_data_lacks_cannot_be_answered_with_other_data(self) -> None:
        scope = [self._item("q1", status="unanswerable", note="没有日前价格"), self._item("q2")]
        errors = self._errors([self._answer("q1"), self._answer("q2")], scope)
        self.assertEqual(len(errors), 1)
        self.assertIn("answers[q1]", errors[0])
        self.assertIn("没有日前价格", errors[0])
        refused = {"item_id": "q1", "status": "unanswerable", "statement": "没有日前价格，无法回答。"}
        self.assertEqual(self._errors([refused, self._answer("q2")], scope), [])

    def test_direct_answers_follow_the_binding_rules_of_findings(self) -> None:
        wrong = self._answer("q1", statement="最低价为 -75 元/MWh。")
        errors = self._errors([wrong], [self._item("q1")])
        self.assertTrue(any(error.startswith("answers[q1] 陈述中的数值") for error in errors), errors)

    def test_markdown_shows_direct_answers(self) -> None:
        from ellectric.llm.report_markdown import render_report_markdown

        text = render_report_markdown({"summary": "完成。", "answers": [
            self._answer("q1", basis="按结束时刻 07:00 的时段"),
            {"item_id": "q2", "status": "unanswerable", "statement": "没有日前价格。"}]})
        self.assertIn("## 直接回答", text)
        self.assertIn("口径：按结束时刻 07:00 的时段", text)
        self.assertIn("（无法回答）没有日前价格。", text)


if __name__ == "__main__":
    unittest.main()
