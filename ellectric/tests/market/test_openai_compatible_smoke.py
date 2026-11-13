"""The endpoint smoke test validates both compatible response shapes locally."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from ellectric.scripts.smoke_openai_compatible_api import run_smoke_test


class _MockEndpointHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.server.auth_header = self.headers.get("Authorization")
        request_size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(request_size))
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        if request.get("tools"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-test",
                        "type": "function",
                        "function": {
                            "name": "confirm_trade_date",
                            "arguments": '{"trade_date":"2025-10-01"}',
                        },
                    }
                ],
            }
        else:
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {"status": "ok", "trade_date": "2025-10-01", "sample_count": 96}
                ),
            }

        payload = {"choices": [{"message": message}]}
        response = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


class OpenAICompatibleSmokeTest(unittest.TestCase):
    def test_tool_call_and_json_mode_without_printing_key(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _MockEndpointHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = run_smoke_test(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="test-model",
                api_key="secret-for-test-only",
                timeout=2,
            )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

        self.assertEqual(server.auth_header, "Bearer secret-for-test-only")
        self.assertEqual(result["tool_call"], "passed")
        self.assertEqual(result["json_object"], "passed")
        self.assertFalse(result["api_key_printed"])
        self.assertNotIn("secret-for-test-only", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
