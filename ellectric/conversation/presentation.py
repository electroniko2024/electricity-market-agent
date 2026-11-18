"""Answer presentation by question complexity, with a citation marker behind every bound value.

Tiers (the analysis and its checks always run in full; the tier only decides what is shown first):

- ``quick``: every scope item asks for a number or a date/time (at most three) and the report has no causes or
  policy basis: one or two sentences, each value followed by a marker that opens its source row;
- ``analysis``: a short conclusion, with answers, causes, anomalies and policy basis as expandable sections;
- ``report``: the user asked for a report, answered by the merged session report (``session_report``).

Citations are built from the report's own value bindings and policy citations, which the report validator has
already checked against the observations, so a marker never points at anything the agent did not fetch.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ellectric.llm.market_analysis import _bound_row_identity, _full_result_pointer, _resolve_binding_pointer

REPORT_REQUEST = re.compile(r"(出|生成|做|写|整理|给我)(一份|个|一个)?(完整的?|汇总|总结)?(分析)?报告|汇总报告|会话报告|总结一下(这次|前面|以上)?")
QUICK_ANSWER_TYPES = {"number", "date_or_time"}
MAX_QUICK_ITEMS = 3


def wants_report(utterance: str) -> bool:
    return bool(REPORT_REQUEST.search(utterance))


def classify_tier(report: dict[str, Any], trace: dict[str, Any]) -> str:
    scope = trace.get("scope") or []
    simple = bool(scope) and len(scope) <= MAX_QUICK_ITEMS and all(
        item.get("answer_type") in QUICK_ANSWER_TYPES for item in scope)
    if simple and not report.get("causes") and not report.get("policy_basis"):
        return "quick"
    return "analysis"


class Citations:
    """Numbered sources, shared across everything shown for one answer (or one session report).

    ``per_source=False`` (an answer): one number per cited value, so a marker opens exactly that row.
    ``per_source=True`` (the merged report): one number per tool call (same tool, same arguments), with every
    value cited from it listed under that number, so the same evidence appears once however often it is used.
    """

    def __init__(self, per_source: bool = False) -> None:
        self.per_source = per_source
        self.entries: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}
        self._first_of_source: dict[str, int] = {}

    def _add(self, key: str, entry: dict[str, Any]) -> int:
        if key not in self._index:
            self._index[key] = len(self.entries) + 1
            self.entries.append({"n": self._index[key]} | entry)
        return self._index[key]

    @staticmethod
    def source_key(observation: dict[str, Any]) -> str:
        return json.dumps([observation.get("tool_name"), observation.get("arguments")], ensure_ascii=False,
                          sort_keys=True, default=str)

    def value(self, observation: dict[str, Any], pointer: str, value: Any) -> int:
        result = observation.get("result")
        full = _full_result_pointer(pointer)
        try:
            actual = _resolve_binding_pointer(result, pointer)
        except (KeyError, IndexError, TypeError, ValueError):
            actual = value
        if isinstance(actual, float):
            actual = round(actual, 6)
        cited = {"field": full.rsplit("/", 1)[-1], "value": actual, "row": _bound_row_identity(result, full),
                 "pointer": full}
        if self.per_source:
            n = self.source(observation)
            values = self.entries[n - 1].setdefault("values", [])
            if not any(v["pointer"] == full for v in values):
                values.append(cited)
            return n
        n = self._add(self.source_key(observation) + "|" + full,
                      {"kind": "data", "tool": observation.get("tool_name"), "arguments": observation.get("arguments")}
                      | cited)
        self._first_of_source.setdefault(self.source_key(observation), n)
        return n

    def source(self, observation: dict[str, Any]) -> int:
        """A whole-result citation; reuses a value citation of the same call when there is one."""
        key = self.source_key(observation)
        if key in self._first_of_source:
            return self._first_of_source[key]
        n = self._add(key, {"kind": "data", "tool": observation.get("tool_name"),
                            "arguments": observation.get("arguments")})
        self._first_of_source[key] = n
        return n

    def policy(self, observation: dict[str, Any], pointer: str) -> int:
        try:
            item = _resolve_binding_pointer(observation.get("result"), pointer) if pointer else None
        except (KeyError, IndexError, TypeError, ValueError):
            item = None
        if not isinstance(item, dict):
            return self.source(observation)
        key = f"policy|{item.get('relative_path')}|{item.get('page')}|{str(item.get('excerpt', ''))[:80]}"
        return self._add(key, {"kind": "policy", "document": item.get("document_title") or item.get("relative_path"),
                               "relative_path": item.get("relative_path"), "page": item.get("page"),
                               "effective_from": item.get("effective_from"), "effective_to": item.get("effective_to"),
                               "excerpt": str(item.get("excerpt", ""))[:400]})


def _refs(item: dict[str, Any], observations: dict[str, dict[str, Any]], citations: Citations) -> list[int]:
    refs: list[int] = []
    bindings = list(item.get("value_bindings") or [])
    for derived in item.get("derived_values") or []:
        bindings.extend(derived.get("operands") or [])
    for binding in bindings:
        observation = observations.get(binding.get("evidence_id"))
        if observation is not None:
            refs.append(citations.value(observation, binding.get("json_pointer", ""), binding.get("value")))
    if not refs:
        for evidence_id in item.get("evidence_ids") or []:
            if evidence_id in observations:
                refs.append(citations.source(observations[evidence_id]))
    return list(dict.fromkeys(refs))


def _item(text: str, refs: list[int]) -> dict[str, Any]:
    return {"text": text, "refs": refs}


def present_turn(report: dict[str, Any], trace: dict[str, Any], citations: Citations | None = None,
                 tier: str | None = None) -> dict[str, Any]:
    citations = citations or Citations()
    observations = {o["evidence_id"]: o for o in trace.get("observations", []) if o.get("execution_status") == "ok"}
    tier = tier or classify_tier(report, trace)
    answers = [_item(a["statement"], _refs(a, observations, citations)) for a in report.get("answers") or []]
    def summary_refs() -> list[int]:  # only numbered when the summary is shown, so no source goes unreferenced
        return [citations.source(observations[e]) for e in report.get("summary_evidence_ids") or [] if e in observations]

    sections = []
    for title, key in (("异常", "anomalies"), ("成因", "causes")):
        items = [_item(f["statement"], _refs(f, observations, citations)) for f in report.get(key) or []]
        if items:
            sections.append({"title": title, "items": items})
    if report.get("outlook"):
        sections.append({"title": "展望", "items": [_item(report["outlook"]["statement"],
                                                          _refs(report["outlook"], observations, citations))]})
    policy = []
    for cite in report.get("policy_basis") or []:
        observation = observations.get(cite.get("evidence_id"))
        refs = [citations.policy(observation, cite.get("json_pointer", ""))] if observation else []
        policy.append(_item(cite["statement"], refs))
    if policy:
        sections.append({"title": "政策依据", "items": policy})
    notes = list(report.get("data_notes") or []) + list(report.get("known_limitations") or [])
    if report.get("unresolved_issues"):
        notes += [f"未解决：{x}" for x in report["unresolved_issues"]]
    if tier == "quick":
        headline = answers or [_item(report.get("summary", ""), summary_refs())]
        if sections:
            sections = [{"title": "更多内容", "items": [i for s in sections for i in s["items"]]}]
    else:
        if report.get("summary"):
            headline = [_item(report["summary"], summary_refs())]
            if answers:
                sections.insert(0, {"title": "逐项回答", "items": answers})
        else:
            headline = answers
    return {"tier": tier, "headline": headline, "sections": sections, "notes": notes,
            "verified": bool(report.get("verified")), "citations": citations.entries}


def render_text(presentation: dict[str, Any]) -> str:
    """Plain-text form (markers as [n]) for logs, tests and the merged report."""
    def line(item: dict[str, Any]) -> str:
        return item["text"] + "".join(f"[{n}]" for n in item["refs"])

    parts = [line(i) for i in presentation["headline"]]
    for section in presentation["sections"]:
        parts.append(f"\n{section['title']}：")
        parts += [f"- {line(i)}" for i in section["items"]]
    return "\n".join(parts)
