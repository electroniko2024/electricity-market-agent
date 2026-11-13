"""Local analysis execution, event streaming, and durable trace retrieval."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ellectric.llm.market_analysis import MarketAnalysisAgent
from ellectric.llm.report_markdown import render_report_markdown
from ellectric.service.market_tools import DATA_DIR

TRACE_DIR = DATA_DIR / "analysis_runs"


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    question: str = Field(min_length=2, max_length=4000)
    as_of: datetime
    mode: Literal["historical_analysis", "historical_forecast"]

    @model_validator(mode="after")
    def validate_as_of(self):
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of 必须包含时区偏移量")
        return self


def _write_trace(payload: dict[str, Any], trace_dir: str | Path | None = None) -> Path:
    directory = Path(trace_dir) if trace_dir is not None else TRACE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{payload['run_id']}.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, suffix=".json.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _trace_payload(request: AnalysisRequest, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": result["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": request.model_dump(mode="json"),
        "report": result["report"],
        "trace": result["trace"],
    }


def run_analysis(
    request: AnalysisRequest,
    *,
    agent: MarketAnalysisAgent | None = None,
    trace_dir: str | Path | None = None,
) -> dict[str, Any]:
    runner = agent or MarketAnalysisAgent()
    result = runner.analyze(
        question=request.question,
        as_of=request.as_of,
        mode=request.mode,
    )
    _write_trace(_trace_payload(request, result), trace_dir)
    return {
        "trace_id": result["run_id"],
        "report": result["report"],
        "report_markdown": render_report_markdown(
            result["report"], request=request.model_dump(mode="json")
        ),
        "trace_summary": {
            "plan_steps": len(result["trace"]["plan"]),
            "tool_calls": result["trace"]["tool_call_count"],
            "replan_count": result["trace"]["replan_count"],
            "report_revision_count": result["trace"]["report_revision_count"],
            "elapsed_ms": result["trace"]["elapsed_ms"],
        },
    }


def stream_analysis(
    request: AnalysisRequest,
    *,
    agent: MarketAnalysisAgent | None = None,
    trace_dir: str | Path | None = None,
) -> Iterator[str]:
    runner = agent or MarketAnalysisAgent()
    try:
        for event in runner.stream_analyze(
            question=request.question,
            as_of=request.as_of,
            mode=request.mode,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False, allow_nan=False)}\n\n"
    except Exception as error:
        safe_message = re.sub(r"(?i)(api[_ -]?key|bearer)\s*[:=]?\s*\S+", r"\1 [redacted]", str(error))
        yield "data: " + json.dumps(
            {"type": "error", "message": f"{type(error).__name__}: {safe_message}"},
            ensure_ascii=False,
        ) + "\n\n"
    finally:
        result = getattr(runner, "last_result", None)
        if result:
            _write_trace(_trace_payload(request, result), trace_dir)


def get_trace(
    trace_id: str, *, trace_dir: str | Path = TRACE_DIR
) -> dict[str, Any] | None:
    if not re.fullmatch(r"run_[0-9a-f]{32}", trace_id):
        return None
    path = Path(trace_dir) / f"{trace_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") != trace_id:
        return None
    return payload


def health_components() -> dict[str, Any]:
    import importlib.util

    from ellectric.service.market_tools import DATABASE_PATH, POLICY_DOCUMENTS_PATH

    return {
        "market_database": "ready" if DATABASE_PATH.is_file() else "missing",
        "policy_inventory": "ready" if POLICY_DOCUMENTS_PATH.is_file() else "missing",
        "policy_index": "ready"
        if (DATA_DIR / "policy_index" / "policy_index_manifest.json").is_file()
        else "not_built",
        "analysis_runtime": "ready"
        if importlib.util.find_spec("langgraph")
        and importlib.util.find_spec("langchain_core")
        and importlib.util.find_spec("langsmith")
        else "incomplete",
        "llm_key_configured": bool(os.environ.get("ELLECTRIC_LLM_API_KEY")),
    }
