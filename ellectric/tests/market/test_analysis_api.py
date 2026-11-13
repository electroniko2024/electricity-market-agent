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


if __name__ == "__main__":
    unittest.main()
