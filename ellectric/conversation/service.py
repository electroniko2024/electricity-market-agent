"""Multi-turn conversations around the single-turn analysis agent.

A turn goes: report request? -> merged session report. Otherwise rewrite (skipped on the first turn) ->
clarify, or run the unchanged single-turn agent on the standalone question with a conversation-scoped tool
cache -> present the answer by tier. Conversations are stored as one JSON file each; every agent run also
writes its usual trace file, so ``/trace/{run_id}`` works for conversation turns too.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ellectric.conversation.cache import CachingToolCall
from ellectric.conversation.presentation import present_turn, wants_report
from ellectric.conversation.rewrite import Rewriter
from ellectric.conversation.session_report import build_session_report
from ellectric.llm.market_analysis import MarketAnalysisAgent
from ellectric.service.analysis import TRACE_DIR, _write_trace, get_trace
from ellectric.service.market_tools import DATA_DIR, invoke_market_tool

CONVERSATION_DIR = DATA_DIR / "conversations"
_ID = re.compile(r"conv_[0-9a-f]{32}")
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

AgentFactory = Callable[[Callable[[str, dict[str, Any]], dict[str, Any]]], MarketAnalysisAgent]


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    message: str = Field(min_length=1, max_length=4000)
    as_of: datetime
    mode: Literal["historical_analysis", "historical_forecast"] = "historical_analysis"

    @model_validator(mode="after")
    def validate_as_of(self):
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of 必须包含时区偏移量")
        return self


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lock(conversation_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(conversation_id, threading.Lock())


class ConversationStore:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else CONVERSATION_DIR

    def create(self) -> dict[str, Any]:
        conversation = {"schema_version": 1, "conversation_id": f"conv_{uuid.uuid4().hex}", "created_at_utc": _now(),
                        "turns": [], "tool_cache": {}}
        self.save(conversation)
        return conversation

    def load(self, conversation_id: str) -> dict[str, Any] | None:
        if not _ID.fullmatch(conversation_id):
            return None
        path = self.directory / f"{conversation_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, conversation: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{conversation['conversation_id']}.json"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(conversation, handle, ensure_ascii=False, allow_nan=False, default=str)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def public_view(conversation: dict[str, Any]) -> dict[str, Any]:
    """The conversation without its tool cache (raw tool results stay server-side)."""
    return {k: v for k, v in conversation.items() if k != "tool_cache"}


class ConversationService:
    def __init__(self, *, store: ConversationStore | None = None, agent_factory: AgentFactory | None = None,
                 tool_call: Callable[[str, dict[str, Any]], dict[str, Any]] = invoke_market_tool,
                 trace_dir: str | Path | None = None) -> None:
        self.store = store or ConversationStore()
        self.agent_factory = agent_factory or (lambda call: MarketAnalysisAgent(tool_call=call))
        self.tool_call = tool_call
        self.trace_dir = Path(trace_dir) if trace_dir is not None else TRACE_DIR

    # ------------------------------------------------------------------ public API
    def run_turn(self, conversation_id: str | None, request: TurnRequest) -> dict[str, Any]:
        turn = None
        for event in self.stream_turn(conversation_id, request):
            if event["type"] == "turn":
                turn = event["turn"]
        return turn

    def stream_turn(self, conversation_id: str | None, request: TurnRequest) -> Iterator[dict[str, Any]]:
        conversation = self.store.create() if conversation_id is None else self.store.load(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        with _lock(conversation["conversation_id"]):
            conversation = self.store.load(conversation["conversation_id"]) or conversation
            yield {"type": "conversation", "conversation_id": conversation["conversation_id"]}
            turn = {"index": len(conversation["turns"]) + 1, "utterance": request.message.strip(),
                    "as_of": request.as_of.isoformat(), "mode": request.mode, "created_at_utc": _now()}
            if wants_report(turn["utterance"]) and self._answered(conversation):
                yield from self._report_turn(conversation, turn)
            else:
                yield from self._question_turn(conversation, turn, request)
            result = turn.pop("last_result", None)
            conversation["turns"].append(turn)
            self.store.save(conversation)
            yield {"type": "turn", "turn": turn, "result": result}
            yield {"type": "done", "conversation_id": conversation["conversation_id"]}

    # ------------------------------------------------------------------ turn kinds
    @staticmethod
    def _answered(conversation: dict[str, Any]) -> bool:
        return any(t.get("type") == "question" and t.get("status") == "completed" for t in conversation["turns"])

    def _report_turn(self, conversation: dict[str, Any], turn: dict[str, Any]) -> Iterator[dict[str, Any]]:
        traces = {}
        for item in conversation["turns"]:
            if item.get("run_id"):
                payload = get_trace(item["run_id"], trace_dir=self.trace_dir)
                if payload:
                    traces[item["run_id"]] = payload
        agent = self.agent_factory(self.tool_call)
        report = build_session_report(conversation, traces, agent)
        turn |= {"type": "report", "status": "completed", "session_report": report,
                 "token_count": getattr(agent, "_last_token_count", 0)}
        yield {"type": "session_report", "report": report}

    def _question_turn(self, conversation: dict[str, Any], turn: dict[str, Any],
                       request: TurnRequest) -> Iterator[dict[str, Any]]:
        rewriter_agent = self.agent_factory(self.tool_call)
        try:
            rewrite = Rewriter(rewriter_agent).rewrite(turn["utterance"], conversation["turns"], turn["as_of"])
        except Exception as error:  # LLM transport failure: nothing was answered
            turn |= {"type": "question", "status": "error", "error": _safe(error), "standalone_question": ""}
            yield {"type": "error", "message": turn["error"]}
            return
        turn["rewrite"] = rewrite
        yield {"type": "rewrite", "action": rewrite["action"], "standalone_question": rewrite["standalone_question"],
               "clarification": rewrite["clarification"], "inherited": rewrite["inherited"]}
        if rewrite["action"] == "clarify":
            turn |= {"type": "clarify", "status": "completed", "clarification": rewrite["clarification"],
                     "standalone_question": ""}
            return
        turn |= {"type": "question", "standalone_question": rewrite["standalone_question"]}
        cache = CachingToolCall(self.tool_call, conversation["tool_cache"], turn["as_of"], f"turn{turn['index']}")
        agent = self.agent_factory(cache)
        try:
            for event in agent.stream_analyze(question=turn["standalone_question"], as_of=request.as_of,
                                              mode=request.mode):
                if event.get("type") not in {"report", "done"}:
                    yield {"type": "agent_event", "event": event}
        except Exception as error:
            turn |= {"status": "error", "error": _safe(error)}
            yield {"type": "error", "message": turn["error"]}
        result = getattr(agent, "last_result", None)
        turn["cache"] = {"hits": len(cache.hits), "misses": cache.misses,
                         "reused": [{"tool_name": h["tool_name"], "fetched_by": h["fetched_by"]} for h in cache.hits]}
        if result is None:
            turn.setdefault("status", "error")
            return
        _write_trace({"schema_version": 1, "run_id": result["run_id"], "created_at_utc": _now(),
                      "request": {"question": turn["standalone_question"], "as_of": turn["as_of"], "mode": turn["mode"],
                                  "conversation_id": conversation["conversation_id"], "utterance": turn["utterance"]},
                      "report": result["report"], "trace": result["trace"]}, self.trace_dir)
        presentation = present_turn(result["report"], result["trace"])
        turn |= {"status": "completed", "run_id": result["run_id"],
                 "answers": [{"statement": a.get("statement", ""), "status": a.get("status")}
                             for a in result["report"].get("answers") or []],
                 "presentation": presentation,
                 "token_count": result["trace"].get("token_count", 0) + rewrite.get("token_count", 0)}
        turn["last_result"] = result  # removed before saving; kept for in-process callers (evaluation)
        yield {"type": "answer", "presentation": presentation, "run_id": result["run_id"]}


def _safe(error: Exception) -> str:
    message = re.sub(r"(?i)(api[_ -]?key|bearer)\s*[:=]?\s*\S+", r"\1 [redacted]", str(error))
    return f"{type(error).__name__}: {message}"[:500]
