"""Offline tests for the multi-turn layer: periods, rewrite checks, evidence cache, tiers, session report."""

from __future__ import annotations

import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from ellectric.conversation.cache import CachingToolCall
from ellectric.conversation.periods import mentioned_periods, partial_dates, shift
from ellectric.conversation.presentation import classify_tier, wants_report
from ellectric.conversation.rewrite import Rewriter
from ellectric.conversation.service import ConversationService, ConversationStore, TurnRequest
from ellectric.llm.market_analysis import MarketAnalysisAgent as RealAgent
from ellectric.tests.market.test_market_analysis import MarketAnalysisAgent

AS_OF = datetime(2025, 10, 3, 9, 0, tzinfo=timezone(timedelta(hours=8)))


def _turn(index, question, answer=""):
    return {"index": index, "type": "question", "status": "completed", "standalone_question": question,
            "answers": [{"statement": answer, "status": "answered"}] if answer else []}


class PeriodsTest(unittest.TestCase):
    def test_parses_chinese_iso_and_carried_year(self):
        text = "2025年5月3日至5月10日，以及2024-07-01和2025年8月"
        self.assertEqual(mentioned_periods(text), {"2025-05-03", "2025-05-10", "2024-07-01", "2025-08"})
        self.assertEqual(partial_dates("那7月31日呢？"), {(7, 31)})
        self.assertEqual(mentioned_periods("那7月31日呢？"), set())

    def test_shift(self):
        self.assertEqual(shift("2025-03-01", "shift_days", -1), "2025-02-28")
        self.assertEqual(shift("2025-12", "shift_months", 1), "2026-01")
        self.assertEqual(shift("2025-07-07", "shift_years", -1), "2024-07-07")


def _rewriter(responses):
    calls = []

    def json_call(name, _schema, prompt):
        calls.append(prompt)
        return responses[min(len(calls), len(responses)) - 1]

    return Rewriter(MarketAnalysisAgent(json_call=json_call)), calls


class RewriteTest(unittest.TestCase):
    context = [_turn(1, "2025年5月23日山东实时电价的日均价是多少？", "日均价为 310.5 元/兆瓦时。")]

    def test_first_turn_needs_no_model(self):
        rewriter, calls = _rewriter([{}])
        out = rewriter.rewrite("2025年5月23日电价多少？", [], AS_OF.isoformat())
        self.assertEqual(out["source"], "first_turn")
        self.assertEqual(calls, [])

    def test_declared_shift_is_accepted(self):
        rewriter, _ = _rewriter([{"action": "proceed", "standalone_question": "2025年5月22日山东实时电价的日均价是多少？",
                                  "inherited": [{"slot": "date", "value": "2025-05-22", "from_turn": 1,
                                                 "derivation": "shift_days", "amount": -1, "base": "2025-05-23"}]}])
        out = rewriter.rewrite("那前一天呢？", self.context, AS_OF.isoformat())
        self.assertEqual(out["action"], "proceed")
        self.assertIn("2025年5月22日", out["standalone_question"])

    def test_invented_date_is_sent_back_then_turned_into_clarification(self):
        bad = {"action": "proceed", "standalone_question": "2025年6月1日山东实时电价的日均价是多少？"}
        rewriter, calls = _rewriter([bad, bad])
        out = rewriter.rewrite("那天呢？", self.context, AS_OF.isoformat())
        self.assertEqual(len(calls), 2)
        self.assertIn("2025-06-01", calls[1])
        self.assertEqual((out["action"], out["source"]), ("clarify", "check_failed"))

    def test_wrong_shift_arithmetic_is_rejected(self):
        bad = {"action": "proceed", "standalone_question": "2025年5月21日山东实时电价的日均价是多少？",
               "inherited": [{"slot": "date", "value": "2025-05-21", "from_turn": 1, "derivation": "shift_days",
                              "amount": -1, "base": "2025-05-23"}]}
        rewriter, _ = _rewriter([bad, bad])
        self.assertEqual(rewriter.rewrite("那前一天呢？", self.context, AS_OF.isoformat())["action"], "clarify")

    def test_range_shift_covers_both_endpoints(self):
        context = [_turn(1, "2025年7月7日至2025年7月16日山东实时电价的均价是多少？")]
        rewriter, _ = _rewriter([{"action": "proceed",
                                  "standalone_question": "2024年7月7日至2024年7月16日山东实时电价的均价是多少？",
                                  "inherited": [{"slot": "period", "value": "去年同期", "from_turn": 1,
                                                 "derivation": "shift_years", "amount": -1,
                                                 "base": "2025-07-07至2025-07-16"}]}])
        self.assertEqual(rewriter.rewrite("去年同期呢？", context, AS_OF.isoformat())["action"], "proceed")

    def test_undeclared_shift_gets_a_hint_on_retry(self):
        context = [_turn(1, "2025年7月7日至2025年7月16日山东实时电价的均价是多少？")]
        bad = {"action": "proceed", "standalone_question": "2024年7月7日至2024年7月16日山东实时电价的均价是多少？"}
        rewriter, calls = _rewriter([bad, bad])
        rewriter.rewrite("去年同期呢？", context, AS_OF.isoformat())
        self.assertIn("整个范围写进 base", calls[1])

    def test_year_completion_of_partial_date(self):
        rewriter, _ = _rewriter([{"action": "proceed", "standalone_question": "2025年7月31日山东实时电价的日均价是多少？"}])
        self.assertEqual(rewriter.rewrite("那7月31日呢？", self.context, AS_OF.isoformat())["action"], "proceed")

    def test_two_candidates_force_clarification(self):
        context = [_turn(1, "2025年8月24日和2025年9月5日各有多少个负电价时段？")]
        rewriter, _ = _rewriter([{"action": "proceed", "standalone_question": "2025年8月24日的最高价是多少？",
                                  "references": [{"phrase": "那天", "candidates": ["2025-08-24", "2025-09-05"],
                                                  "resolved": "2025-08-24"}]}])
        out = rewriter.rewrite("那天的最高价是多少？", context, AS_OF.isoformat())
        self.assertEqual((out["action"], out["source"]), ("clarify", "multiple_candidates"))
        self.assertIn("2025-09-05", out["clarification"])
        self.assertEqual(out["standalone_question"], "")


class PluralReferenceTest(unittest.TestCase):
    context = [_turn(1, "2025年8月4日山东实时电价的日均价是多少？"), _turn(2, "2025年7月31日山东实时电价的日均价是多少？")]
    reference = {"phrase": "这两天", "candidates": ["2025年8月4日", "2025年7月31日"], "plural": True}

    def test_plural_reference_proceeds_when_all_named(self):
        rewriter, _ = _rewriter([{"action": "proceed", "references": [self.reference],
                                  "standalone_question": "2025年8月4日和2025年7月31日山东实时电价日均价哪天更高？"}])
        self.assertEqual(rewriter.rewrite("这两天哪天高？", self.context, AS_OF.isoformat())["action"], "proceed")

    def test_plural_reference_missing_one_is_rejected(self):
        bad = {"action": "proceed", "references": [self.reference],
               "standalone_question": "2025年8月4日山东实时电价日均价是多少？"}
        rewriter, calls = _rewriter([bad, bad])
        self.assertEqual(rewriter.rewrite("这两天哪天高？", self.context, AS_OF.isoformat())["action"], "clarify")
        self.assertIn("没有全部写进", calls[1])


class CacheTest(unittest.TestCase):
    def test_hits_are_copies_keyed_by_as_of_and_skip_unavailable(self):
        calls = []

        def base(name, args):
            calls.append(name)
            return {"status": "unavailable" if name == "search_policy" else "ok", "data": {"v": 1}}

        store = {}
        first = CachingToolCall(base, store, "A", "turn1")
        result = first("query_stats", {"start_date": "2025-01-01"})
        result["data"]["v"] = 99  # the caller mutating its copy must not change the cache
        second = CachingToolCall(base, store, "A", "turn2")
        self.assertEqual(second("query_stats", {"start_date": "2025-01-01"})["data"]["v"], 1)
        self.assertEqual(second.hits[0]["fetched_by"], "turn1")
        CachingToolCall(base, store, "B")("query_stats", {"start_date": "2025-01-01"})
        second("search_policy", {"query": "x"})
        second("search_policy", {"query": "x"})
        self.assertEqual(calls, ["query_stats", "query_stats", "search_policy", "search_policy"])


class TierTest(unittest.TestCase):
    def test_report_requests_and_tiers(self):
        self.assertTrue(wants_report("帮我出个报告"))
        self.assertTrue(wants_report("生成一份汇总报告"))
        self.assertFalse(wants_report("5月23日最高负荷多少？"))
        number_scope = {"scope": [{"answer_type": "number"}]}
        self.assertEqual(classify_tier({}, number_scope), "quick")
        self.assertEqual(classify_tier({"causes": [{}]}, number_scope), "analysis")
        self.assertEqual(classify_tier({}, {"scope": [{"answer_type": "text"}]}), "analysis")


def _fake_llm(state):
    """Scope, plan, reflection, report and rewrite stubs for an end-to-end conversation."""

    def call(name, _schema, prompt):
        if name == "ScopeDraft":
            return {"items": [{"item": "日均价", "answer_type": "number", "status": "answerable", "note": ""}]}
        if name == "conversation_rewrite":
            state["rewrites"] += 1
            if "那天" in prompt.rsplit("用户最新一句话：", 1)[1].splitlines()[0]:
                return {"action": "proceed", "standalone_question": "x",
                        "references": [{"phrase": "那天", "candidates": ["2025-10-01", "2025-09-30"]}]}
            return {"action": "proceed", "standalone_question": "2025年10月1日山东实时电价的日均价是多少？请重新核对。",
                    "inherited": [{"slot": "date", "value": "2025-10-01", "from_turn": 1}]}
        if name == "PlanDraft":
            return {"steps": [{"step_id": "s1", "tool_name": "query_stats",
                               "arguments": {"start_date": "2025-10-01", "end_date": "2025-10-01",
                                             "metric": "rt_price", "group_by": "day"},
                               "depends_on": [], "rationale": "查当日统计"}]}
        if name == "ReflectionDraft":
            return {"sufficient_evidence": True, "issues": []}
        if name == "MarketReportDraft":
            evidence = re.search(r"ev_[0-9a-f]{6,}", prompt).group(0)
            binding = {"evidence_id": evidence, "json_pointer": "/data/mean_price", "value": 42.5}
            return {"answers": [{"item_id": "q1", "status": "answered", "statement": "日均价为 42.5 元/兆瓦时。",
                                 "evidence_ids": [evidence], "value_bindings": [binding]}],
                    "summary": "日均价为 42.5 元/兆瓦时。", "summary_evidence_ids": [evidence],
                    "anomalies": [], "causes": [], "unresolved_issues": []}
        if name == "session_overview":
            return {"overview": "两次询问同一天的价格。"}
        raise AssertionError(name)

    return call


class ConversationServiceTest(unittest.TestCase):
    def test_follow_up_reuses_evidence_clarifies_and_merges_report(self):
        state, tools = {"rewrites": 0}, []

        def tool_call(name, args):
            tools.append(name)
            return {"tool": name, "status": "ok", "data": {"mean_price": 42.5}, "limitations": []}

        with tempfile.TemporaryDirectory() as directory:
            service = ConversationService(
                store=ConversationStore(f"{directory}/conv"), trace_dir=f"{directory}/traces", tool_call=tool_call,
                agent_factory=lambda call: RealAgent(json_call=_fake_llm(state), tool_call=call, catalog=lambda _a: {}))
            first = list(service.stream_turn(None, TurnRequest(message="2025年10月1日山东实时电价的日均价是多少？", as_of=AS_OF)))
            conv_id = first[0]["conversation_id"]
            turn1 = first[-2]["turn"]
            self.assertEqual(state["rewrites"], 0)
            self.assertEqual(turn1["status"], "completed")
            self.assertEqual(turn1["presentation"]["tier"], "quick")
            headline = turn1["presentation"]["headline"][0]
            self.assertEqual(headline["refs"], [1])
            citation = turn1["presentation"]["citations"][0]
            self.assertEqual((citation["tool"], citation["value"]), ("query_stats", 42.5))

            turn2 = service.run_turn(conv_id, TurnRequest(message="再核对一次", as_of=AS_OF))
            self.assertEqual(turn2["status"], "completed", turn2)
            self.assertEqual(turn2["type"], "question", turn2.get("rewrite"))
            self.assertEqual(turn2["cache"]["hits"], 1)
            self.assertEqual(tools, ["query_stats"])  # the second turn's call came from the cache

            turn3 = service.run_turn(conv_id, TurnRequest(message="那天最低价呢", as_of=AS_OF))
            self.assertEqual(turn3["type"], "clarify")

            turn4 = service.run_turn(conv_id, TurnRequest(message="帮我出个报告", as_of=AS_OF))
            report = turn4["session_report"]
            self.assertEqual(turn4["type"], "report")
            self.assertEqual(len(report["citations"]), 1)  # the same source used by two turns is listed once
            self.assertIn("## 电价", report["markdown"])
            self.assertIn("两次询问同一天的价格。", report["overview"])

            saved = service.store.load(conv_id)
            self.assertEqual(len(saved["turns"]), 4)
            self.assertNotIn("last_result", saved["turns"][0])


if __name__ == "__main__":
    unittest.main()
