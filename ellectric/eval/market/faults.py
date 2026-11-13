"""Evaluation-only tool fault injection: a case's ``faults`` make chosen tool calls fail transiently.

Each fault ``{"tool": name, "mode": "unavailable", "times": n}`` makes the first ``n`` calls of that tool raise
a transient service error before the tool runs; later calls run normally. This measures whether a variant
notices a failed step and recovers (E2 reflection + replanning) or reports without the evidence (E1).
"""

from __future__ import annotations

import threading
from typing import Any, Callable

FAULT_MODES = {"unavailable"}


class ToolUnavailableError(RuntimeError):
    """A transient upstream failure, as a data service timeout would surface."""


class FaultInjectingToolCall:
    def __init__(self, tool_call: Callable[[str, dict[str, Any]], dict[str, Any]], faults: list[dict[str, Any]]):
        for fault in faults:
            if fault.get("mode", "unavailable") not in FAULT_MODES:
                raise ValueError(f"unknown fault mode {fault.get('mode')}")
        self._tool_call = tool_call
        self._remaining = {fault["tool"]: int(fault.get("times", 1)) for fault in faults}
        self._lock = threading.Lock()
        self.log: list[dict[str, Any]] = []

    def __call__(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            inject = self._remaining.get(name, 0) > 0
            if inject:
                self._remaining[name] -= 1
            self.log.append({"tool": name, "injected": inject})
        if inject:
            raise ToolUnavailableError(f"{name} 上游数据服务暂时不可用（连接超时），本次未返回结果，请稍后重试")
        return self._tool_call(name, arguments)
