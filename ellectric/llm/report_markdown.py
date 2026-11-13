"""Deterministic Markdown rendering of a validated MarketReport (no LLM involved)."""

from __future__ import annotations

from typing import Any

_SECTION_TITLES = (("anomalies", "异常"), ("causes", "原因"))
_OPERATION_NAMES = {"sum": "合计", "difference": "差值", "ratio": "比值"}


def _finding_lines(finding: dict[str, Any]) -> list[str]:
    lines = [f"- {finding['statement']}"]
    for binding in finding.get("value_bindings", []):
        lines.append(f"  - `{binding['value']}` ← `{binding['evidence_id']}` `{binding['json_pointer']}`")
    for derived in finding.get("derived_values", []):
        operands = "、".join(f"`{operand['value']}`" for operand in derived.get("operands", []))
        lines.append(f"  - `{derived['value']}` = {_OPERATION_NAMES.get(derived['operation'], derived['operation'])}（{operands}）")
    for limitation in finding.get("limitations", []):
        lines.append(f"  - 限制：{limitation}")
    return lines


def _bullets(items: list[str], empty: str) -> list[str]:
    return [f"- {item}" for item in items] if items else [f"- {empty}"]


def render_report_markdown(report: dict[str, Any], *, request: dict[str, Any] | None = None) -> str:
    validation = report.get("validation", {})
    status = "已验证" if report.get("verified") else "未验证"
    lines = ["# 山东电力市场分析报告", ""]
    if request:
        lines += [
            f"- 问题：{request.get('question', '')}",
            f"- 信息截止：{request.get('as_of', '')}",
            f"- 模式：{request.get('mode', '')}",
        ]
    lines += [
        f"- 状态：**{status}**（报告校验 {validation.get('status', 'not_run')}）",
        "",
        "## 摘要",
        "",
        report.get("summary", ""),
        "",
    ]
    answers = report.get("answers", [])
    if answers:
        lines += ["## 直接回答", ""]
        for answer in answers:
            prefix = "（无法回答）" if answer.get("status") == "unanswerable" else ""
            lines += _finding_lines(answer | {"statement": prefix + answer["statement"]})
            if answer.get("basis"):
                lines.append(f"  - 口径：{answer['basis']}")
        lines.append("")
    for key, title in _SECTION_TITLES:
        findings = report.get(key, [])
        lines += [f"## {title}", ""]
        if findings:
            for finding in findings:
                lines += _finding_lines(finding)
        else:
            lines.append("- 无")
        lines.append("")
    if report.get("outlook"):
        lines += ["## D+1 研判", "", *_finding_lines(report["outlook"]), ""]
    lines += ["## 政策依据", ""]
    citations = report.get("policy_basis", [])
    lines += (
        [f"- {item['statement']}（`{item['evidence_id']}`）" for item in citations]
        if citations
        else ["- 无经核验的政策依据"]
    )
    lines += ["", "## 数据说明", "", *_bullets(report.get("data_notes", []), "无"), ""]
    lines += ["## 已知限制", "", *_bullets(report.get("known_limitations", []), "无"), ""]
    lines += ["## 未解决问题", "", *_bullets(report.get("unresolved_issues", []), "无"), ""]
    errors = validation.get("errors", [])
    if errors:
        lines += ["## 校验错误", "", *_bullets(errors, "无"), ""]
    return "\n".join(lines).rstrip() + "\n"
