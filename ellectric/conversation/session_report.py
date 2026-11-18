"""Merge a conversation's answered turns into one report: grouped by topic, each source listed once.

Every statement in the merged report is a statement an earlier turn's report already passed validation with;
sources are numbered once for the whole report (the same tool call with the same arguments, or the same
policy clause, gets one number however many turns used it). The optional overview written by the model may
not contain digits, so it cannot introduce a number that was not checked.
"""

from __future__ import annotations

import re
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ellectric.conversation.periods import mentioned_periods
from ellectric.conversation.presentation import Citations, present_turn
from ellectric.llm.market_analysis import PlanCheckError

TOPICS = [
    ("政策规则", ("政策", "规定", "规则", "条款", "办法", "文件", "限价")),
    ("负荷与新能源", ("负荷", "出力", "风电", "光伏", "联络线", "新能源")),
    ("天气", ("气温", "温度", "天气", "降水", "风速")),
    ("电价", ("电价", "价格", "均价", "最高价", "最低价", "负电价", "价差")),
]


def topic_of(question: str) -> str:
    for title, words in TOPICS:
        if any(word in question for word in words):
            return title
    return "其他"


class OverviewDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overview: str = Field(min_length=1, max_length=600)


def _overview_prompt(entries: list[dict[str, Any]]) -> str:
    body = "\n".join(f"- {e['question']}" for e in entries)
    return ("下面是一次电力市场分析对话里用户依次问过的问题。用两三句话概述这次对话关注了什么、各问题之间的关系。"
            "不要写任何数字（包括日期和数值），不要下结论，只做串联。\n" + body + "\n只输出 JSON。")


def build_session_report(conversation: dict[str, Any], traces: dict[str, dict[str, Any]], agent: Any | None = None) -> dict[str, Any]:
    """``traces``: run_id -> stored trace payload ({"report": ..., "trace": ...}) for the conversation's turns."""
    citations = Citations(per_source=True)
    topics: dict[str, list[dict[str, Any]]] = {}
    seen_text: set[str] = set()
    entries = []
    for turn in conversation.get("turns", []):
        if turn.get("type") != "question" or turn.get("status") != "completed":
            continue
        payload = traces.get(turn.get("run_id", ""))
        if not payload:
            continue
        shown = present_turn(payload["report"], payload["trace"], citations, tier="quick")
        # quick tier: the direct answers lead (the summary only when there are none); other findings follow.
        items = []
        for item in shown["headline"] + [i for s in shown["sections"] for i in s["items"]]:
            if item["text"] and item["text"] not in seen_text:
                seen_text.add(item["text"])
                items.append(item)
        entry = {"turn": turn["index"], "question": turn["standalone_question"], "items": items,
                 "verified": shown["verified"], "notes": shown["notes"]}
        entries.append(entry)
        topics.setdefault(topic_of(turn["standalone_question"]), []).append(entry)
    periods = sorted({p for e in entries for p in mentioned_periods(e["question"])})
    overview = f"本次会话共回答 {len(entries)} 个问题，涉及{'、'.join(topics) or '无'}。"
    overview_source = "template"
    if agent is not None and len(entries) >= 2:
        def check(draft: OverviewDraft) -> OverviewDraft:
            if re.search(r"\d", draft.overview):
                raise PlanCheckError("overview 不能包含数字")
            return draft

        agent._run_started_at, agent._last_token_count = time.monotonic(), 0
        try:
            overview += agent._json("session_overview", OverviewDraft, _overview_prompt(entries), check=check).overview
            overview_source = "model"
        except (PlanCheckError, ValueError, RuntimeError, TimeoutError):
            pass
    report = {"title": "会话分析报告", "overview": overview, "overview_source": overview_source, "periods": periods,
              "topics": [{"title": title, "entries": items} for title, items in topics.items()],
              "citations": citations.entries}
    report["markdown"] = render_markdown(report)
    return report


def _cite_text(entry: dict[str, Any]) -> str:
    if entry["kind"] == "policy":
        span = f"（{entry.get('effective_from') or '?'} 起）" if entry.get("effective_from") else ""
        page = f" 第{entry['page']}页" if entry.get("page") else ""
        return f"{entry.get('document')}{page}{span}：{entry.get('excerpt', '')[:120]}"
    args = ", ".join(f"{k}={v}" for k, v in (entry.get("arguments") or {}).items() if k != "as_of")
    values = entry.get("values") or ([entry] if "value" in entry else [])
    shown = "；".join(" ".join(str(x) for x in (v.get("row") or {}).values()) + f" {v['field']}={v['value']}"
                     for v in values)
    return f"{entry.get('tool')}({args})" + (f" → {shown.strip()}" if shown else "")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {report['title']}", "", report["overview"], ""]
    for topic in report["topics"]:
        lines += [f"## {topic['title']}", ""]
        for entry in topic["entries"]:
            lines.append(f"**第{entry['turn']}问：{entry['question']}**" + ("" if entry["verified"] else "（未完全核验）"))
            lines += [f"- {i['text']}" + "".join(f"[{n}]" for n in i["refs"]) for i in entry["items"]]
            lines.append("")
    if report["citations"]:
        lines += ["## 出处", ""] + [f"[{c['n']}] {_cite_text(c)}" for c in report["citations"]]
    return "\n".join(lines).rstrip() + "\n"
