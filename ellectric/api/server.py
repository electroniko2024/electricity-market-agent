"""
山东电力市场分析 Agent — FastAPI 服务
======================================

路由：
- GET  /health                  → 组件健康检查
- POST /analyze                 → 运行一次有边界的历史分析，保存 trace
- POST /analyze/stream          → SSE 流式返回计划、工具调用、反思与最终报告
- GET  /trace/{trace_id}        → 读取已保存的 trace
- GET  /trace/{trace_id}/report.md → 报告的 Markdown 导出
- POST /market-tools/{name}     → 直接调用一个经 Pydantic 校验的市场工具
- GET  /                        → 前端（ellectric/web 构建产物）

启动命令：

  uvicorn ellectric.api.server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ellectric.llm.report_markdown import render_report_markdown
from ellectric.service.analysis import (
    AnalysisRequest,
    get_trace,
    health_components,
    run_analysis,
    stream_analysis,
)
from ellectric.service.market_tools import invoke_market_tool

logger = logging.getLogger(__name__)

VERSION = "1.0.0"


class MarketToolRequest(BaseModel):
    """Arguments for one deterministic Shandong market-domain tool."""

    model_config = ConfigDict(extra="forbid")
    arguments: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="Shandong Electricity Market Agent API",
    description="山东电力市场分析 Agent：规划 → 执行 → 反思 → 报告 → 校验",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "components": health_components()}


@app.post("/analyze")
def analyze(req: AnalysisRequest):
    """Run a bounded historical analysis and persist its trace."""
    try:
        return run_analysis(req)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except ImportError as error:
        missing = getattr(error, "name", None) or "LangGraph runtime"
        raise HTTPException(status_code=503, detail=f"分析运行依赖不可用: {missing}") from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.post("/analyze/stream")
def analyze_stream(req: AnalysisRequest):
    """Stream plan, tools, reflection, and final report as server-sent events."""
    return StreamingResponse(
        stream_analysis(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/trace/{trace_id}")
def analysis_trace(trace_id: str):
    trace = get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="TRACE_NOT_FOUND")
    return trace


@app.get("/trace/{trace_id}/report.md")
def analysis_report_markdown(trace_id: str):
    trace = get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="TRACE_NOT_FOUND")
    return PlainTextResponse(
        render_report_markdown(trace["report"], request=trace.get("request")),
        media_type="text/markdown; charset=utf-8",
    )


@app.post("/market-tools/{name}")
def market_tool(name: str, req: MarketToolRequest):
    """Invoke one Pydantic-validated tool over the standardized Shandong data."""
    try:
        return invoke_market_tool(name, req.arguments)
    except ValueError as error:
        if str(error).startswith("UNKNOWN_MARKET_TOOL:"):
            raise HTTPException(status_code=404, detail=str(error)) from None
        raise HTTPException(status_code=422, detail=str(error)) from None


# Static frontend must be mounted after the API routes so it does not shadow them.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
else:
    logger.warning("静态目录不存在，跳过挂载: %s", _STATIC_DIR)
