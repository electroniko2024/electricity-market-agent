"""The analysis API exports a stored report as deterministic Markdown."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ellectric.api import server

TRACE = {
    "run_id": "run_" + "a" * 32,
    "request": {"question": "q", "as_of": "2025-10-08T00:00:00+08:00", "mode": "historical_analysis"},
    "report": {"summary": "概述。", "anomalies": [], "causes": [], "known_limitations": ["气象仅济南。"],
               "unresolved_issues": [], "verified": True, "validation": {"status": "passed", "errors": []}},
}


class ReportMarkdownRouteTest(unittest.TestCase):
    def test_markdown_export_and_missing_trace(self) -> None:
        client = TestClient(server.app)
        with patch.object(server, "get_trace", lambda trace_id: TRACE if trace_id == TRACE["run_id"] else None):
            response = client.get(f"/trace/{TRACE['run_id']}/report.md")
            missing = client.get("/trace/run_" + "b" * 32 + "/report.md")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/markdown"))
        self.assertIn("**已验证**", response.text)
        self.assertIn("- 气象仅济南。", response.text)
        self.assertEqual(missing.status_code, 404)


class ConversationRouteTest(unittest.TestCase):
    def test_turn_streams_without_raw_results_and_unknown_conversation_is_404(self) -> None:
        import json
        import tempfile

        from ellectric.conversation.service import ConversationService, ConversationStore

        class Fake(ConversationService):
            def stream_turn(self, conversation_id, request):
                yield {"type": "rewrite", "action": "clarify", "standalone_question": "", "clarification": "哪天？",
                       "inherited": []}
                yield {"type": "turn", "turn": {"type": "clarify"}, "result": {"trace": "secret raw"}}

        with tempfile.TemporaryDirectory() as directory:
            fake = Fake(store=ConversationStore(directory))
            with patch.object(server, "_conversations", fake):
                client = TestClient(server.app)
                conversation_id = client.post("/conversations").json()["conversation_id"]
                response = client.post(f"/conversations/{conversation_id}/turns",
                                       json={"message": "那天呢", "as_of": "2025-10-08T00:00:00+08:00"})
                missing = client.post("/conversations/conv_" + "0" * 32 + "/turns",
                                      json={"message": "x", "as_of": "2025-10-08T00:00:00+08:00"})
                no_report = client.get(f"/conversations/{conversation_id}/report.md")
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual([e["type"] for e in events], ["rewrite", "turn"])
        self.assertNotIn("result", events[1])
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(no_report.status_code, 404)


if __name__ == "__main__":
    unittest.main()
