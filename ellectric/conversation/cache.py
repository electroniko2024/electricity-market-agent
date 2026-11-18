"""Conversation-scoped cache of successful tool results, so a follow-up reuses evidence already fetched.

A hit returns a copy of the stored result, so the agent records it as an ordinary observation and every
report check (value bindings, row identity, direction) applies to it unchanged. The key includes ``as_of``:
a turn asked at a different point in time never sees results fetched for another one.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable

ToolCall = Callable[[str, dict[str, Any]], dict[str, Any]]


def cache_key(as_of: str, name: str, arguments: dict[str, Any]) -> str:
    return json.dumps([as_of, name, arguments], ensure_ascii=False, sort_keys=True, default=str)


class CachingToolCall:
    def __init__(self, base: ToolCall, store: dict[str, dict[str, Any]], as_of: str, run_label: str = "") -> None:
        self.base, self.store, self.as_of, self.run_label = base, store, as_of, run_label
        self.hits: list[dict[str, Any]] = []
        self.misses = 0

    def __call__(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        key = cache_key(self.as_of, name, arguments)
        entry = self.store.get(key)
        if entry is not None:
            self.hits.append({"tool_name": name, "arguments": arguments, "fetched_by": entry.get("fetched_by", "")})
            return copy.deepcopy(entry["result"])
        self.misses += 1
        result = self.base(name, arguments)  # errors propagate and are never cached
        if not isinstance(result, dict) or result.get("status", "ok") == "ok":  # "unavailable" may be transient
            self.store[key] = {"result": copy.deepcopy(result), "fetched_by": self.run_label}
        return result
