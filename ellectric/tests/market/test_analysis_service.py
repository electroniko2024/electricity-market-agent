"""Analysis endpoints persist traces without storing credentials."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ellectric.service import analysis


class _FakeAgent:
    def analyze(self, *, question, as_of, mode):
        return {
            "run_id": "run_" + "a" * 32,
            "report": {"summary": question, "verified": False},
            "trace": {
                "plan": [],
                "observations": [],
                "reflection": {},
                "events": [],
                "replan_count": 0,
                "report_revision_count": 1,
                "tool_call_count": 0,
                "token_count": 0,
                "elapsed_ms": 1.0,
            },
        }

    def stream_analyze(self, *, question, as_of, mode):
        self.last_result = self.analyze(question=question, as_of=as_of, mode=mode)
        yield {"type": "start", "run_id": self.last_result["run_id"]}
        yield {"type": "done", "run_id": self.last_result["run_id"]}


class AnalysisServiceTest(unittest.TestCase):
    def test_run_persists_trace_and_stream_uses_sse_without_key_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = analysis.AnalysisRequest(
                question="检查历史实时价格",
                as_of=datetime(2025, 10, 2, tzinfo=timezone.utc),
                mode="historical_analysis",
            )
            response = analysis.run_analysis(request, agent=_FakeAgent(), trace_dir=Path(temporary))
            stored = analysis.get_trace(response["trace_id"], trace_dir=Path(temporary))
            self.assertEqual(stored["report"]["summary"], request.question)

            events = list(
                analysis.stream_analysis(request, agent=_FakeAgent(), trace_dir=Path(temporary))
            )
            stream_result = json.loads(events[0].removeprefix("data: ").strip())
            self.assertEqual(stream_result["type"], "start")
            stream_trace = analysis.get_trace("run_" + "a" * 32, trace_dir=Path(temporary))
            self.assertEqual(stream_trace["request"]["mode"], "historical_analysis")
            self.assertNotIn("api_key", json.dumps(stream_trace).lower())

    def test_request_requires_timezone_and_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValueError):
            analysis.AnalysisRequest(
                question="检查价格",
                as_of=datetime(2025, 10, 2),
                mode="historical_analysis",
            )
        with self.assertRaises(ValueError):
            analysis.AnalysisRequest(
                question="检查价格",
                as_of=datetime(2025, 10, 2, tzinfo=timezone.utc),
                mode="historical_analysis",
                hidden="value",
            )


if __name__ == "__main__":
    unittest.main()
