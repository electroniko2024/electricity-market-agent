"""Plan-Execute-Reflection market analysis graph with evidence-bound reports."""

from __future__ import annotations

import collections
import http.client
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ellectric.service.market_tools import TOOL_MODELS, data_catalog, invoke_market_tool, latest_available_trade_date

MAX_PLAN_STEPS = 6
MAX_TOOL_CALLS = 12
MAX_REPLANS = 3
MAX_REPORT_REVISIONS = 2
MAX_RUN_SECONDS = 180
MAX_TOTAL_TOKENS = 130000
MAX_PROMPT_LIST_ITEMS = 10
PROMPT_LIST_HEAD = 3
MAX_PROMPT_STRING_CHARS = 500
MAX_OBSERVATION_PROMPT_CHARS = 8000
MAX_OBSERVATIONS_PROMPT_CHARS = 30000
DEEP_PROMPT_LIST_ITEMS = 3
DEEP_PROMPT_STRING_CHARS = 200
# Replanning stops once half of either budget is spent, so the reporter and its revisions still fit.
REPLAN_BUDGET_FRACTION = 0.5
DEFAULT_LLM_TIMEOUT_SECONDS = 45
# Caps runaway generations (seen: 11k-token reports at temperature 0); truncated JSON goes to the schema retry.
DEFAULT_LLM_MAX_TOKENS = 4000
REASONING_EFFORTS = {"low", "medium", "high"}


def _thinking_request_fields(setting: str | None = None) -> dict[str, Any]:
    """Map ELLECTRIC_LLM_THINKING to relay request fields.

    "disabled" sends only ``thinking={"type":"disabled"}`` to the aispot relay (deepseek-v4.1-flash).

    Measured 2026-11-09 evening with the real ``_json_once`` request body (temperature 0,
    json_object, schema prompt), 3 prompts x 3 calls each:
    - ``thinking`` alone: 9/9 succeeded, 0 reasoning.
    - ``thinking`` + ``reasoning_effort="none"``: 5/9 got HTTP 400 ("expected one of
      off|low|medium|high|xhigh|max"); one prompt failed 3/3, so retrying the same body does not help.
    - ``reasoning_effort="none"`` alone: 4/9 still reasoned.
    The relay's behaviour changed during the day (earlier, some prompts got HTTP 400 without
    ``reasoning_effort="none"``); re-measure if 400s or reasoning_content come back.
    Ignored by the relay: top-level ``enable_thinking``, ``reasoning={"effort":"none"}``,
    ``reasoning={"enabled":false}``, and ``reasoning_effort="minimal"``.
    ``usage.reasoning_tokens`` is sometimes 0 while reasoning_content is present, so check that
    field when verifying.
    """
    value = (setting if setting is not None else os.environ.get("ELLECTRIC_LLM_THINKING", "disabled"))
    value = value.strip().lower()
    if value in {"disabled", "off", "none", "false", "0"}:
        return {"thinking": {"type": "disabled"}}
    if value in REASONING_EFFORTS:
        return {"reasoning_effort": value}
    if value in {"default", "enabled", "on"}:
        return {}
    raise ValueError(
        "ELLECTRIC_LLM_THINKING 只能是 disabled / low / medium / high / default"
    )


_RATE_LOCK = threading.Lock()
_RATE_TIMES: collections.deque[float] = collections.deque()
RATE_LIMIT_RETRY_DELAYS = (30.0,)
# The aispot relay occasionally returns a transient HTTP 400 that succeeds when the same body is resent.
BAD_REQUEST_RETRIES = 1


def _llm_max_rpm() -> float:
    """Optional process-wide request cap for relays with a requests-per-minute limit (0 = off)."""
    raw = os.environ.get("ELLECTRIC_LLM_MAX_RPM", "0") or "0"
    rpm = float(raw)
    if not math.isfinite(rpm) or rpm < 0:
        raise ValueError("ELLECTRIC_LLM_MAX_RPM 必须是非负数")
    return rpm


def _acquire_rate_slot() -> float:
    """Block until a sliding 60 s window has room; returns seconds waited."""
    rpm = _llm_max_rpm()
    if rpm <= 0:
        return 0.0
    waited = 0.0
    while True:
        with _RATE_LOCK:
            now = time.monotonic()
            while _RATE_TIMES and now - _RATE_TIMES[0] >= 60.0:
                _RATE_TIMES.popleft()
            if len(_RATE_TIMES) < rpm:
                _RATE_TIMES.append(now)
                return waited
            delay = 60.0 - (now - _RATE_TIMES[0]) + 0.05
        time.sleep(delay)
        waited += delay


def _llm_max_tokens() -> int:
    raw = os.environ.get("ELLECTRIC_LLM_MAX_TOKENS", str(DEFAULT_LLM_MAX_TOKENS))
    value = int(raw)
    if value <= 0:
        raise ValueError("ELLECTRIC_LLM_MAX_TOKENS 必须是正整数")
    return value


def _llm_timeout_seconds() -> float:
    raw = os.environ.get("ELLECTRIC_LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS))
    timeout = float(raw)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("ELLECTRIC_LLM_TIMEOUT_SECONDS 必须是正数")
    return timeout

ToolName = Literal[
    "query_price",
    "query_stats",
    "query_load",
    "query_weather",
    "forecast_price",
    "detect_anomaly",
    "explain_drivers",
    "search_policy",
]


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^s[0-9]{1,2}$")
    tool_name: ToolName
    arguments: dict[str, Any]
    depends_on: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=500)


class PlanCheckError(ValueError):
    """A parsed plan that is structurally or semantically invalid (bad references, invalid tool arguments).

    ``steps`` is set when the structure is sound and only tool arguments are invalid: such a plan can still
    run, and the invalid steps then fail in the executor like any other tool error."""

    def __init__(self, message: str, steps: list["PlanStep"] | None = None) -> None:
        super().__init__(message)
        self.steps = steps


class ScopeItem(BaseModel):
    """One thing the question asks for, checked against the data catalog before any tool is called."""

    model_config = ConfigDict(extra="forbid")

    item: str = Field(min_length=1, max_length=300)
    # What a complete answer to this item must contain; the report validator enforces it.
    answer_type: Literal["number", "date_or_time", "text"]
    status: Literal["answerable", "ambiguous", "unanswerable"]
    # unanswerable: what the catalog lacks; ambiguous: the reading adopted (dates, slot, definition).
    note: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _note_required(self):
        if self.status != "answerable" and not self.note.strip():
            raise ValueError("status 为 ambiguous 或 unanswerable 时必须填写 note")
        return self


class ScopeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ScopeItem] = Field(min_length=1, max_length=8)


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)


class ReactDraft(BaseModel):
    """One ReAct turn for the E0 baseline: call a tool or finish."""

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(default="", max_length=8000)
    action: Literal["call_tool", "finish"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


AgentVariant = Literal["E0", "E1", "E2", "E3"]
VARIANT_DESCRIPTIONS = {
    "E0": "ReAct tool loop, no explicit plan, no reflection",
    "E1": "Plan-Execute, no reflection",
    "E2": "Plan-Execute + Reflection (full system)",
    "E3": "E2 without the policy effective-interval filter (publish-date constraint kept)",
}


class ReflectionIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1000)
    repairable: bool
    # "limitation": disclosed, inherent scope/method limit that does not block the answer.
    # "gap": something that makes a conclusion unsupported or possibly wrong. Defaults to gap (fail closed).
    kind: Literal["limitation", "gap"] = "gap"
    related_step_ids: list[str] = Field(default_factory=list)


class ReflectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient_evidence: bool
    issues: list[ReflectionIssue] = Field(default_factory=list, max_length=20)


class BoundArguments(BaseModel):
    """Literal tool arguments for a plan step whose arguments referenced earlier step results."""

    model_config = ConfigDict(extra="forbid")

    arguments: dict[str, Any]


# A plan argument may be "$ref:<step_id>:<what to take>"; it is resolved right before the step runs.
_STEP_REFERENCE = re.compile(r"^\$ref:(s[0-9]{1,2}):(.+)$", re.S)
PLAN_REFERENCE_NOTE = (
    "参数一般写字面值。某个参数要用前面步骤查出的结果时（例如上一步找到的日期、时段），把它写成字符串 "
    "\"$ref:<step_id>:<要取的量>\"，如 \"$ref:s1:日均价最高的那一天\"，并把该步骤写进 depends_on；"
    "执行到这一步时，系统会根据那一步的结果把它换成具体值。要某一天、某个时段的值时，"
    "先用一步定位，再单独查那一天或那个时段，不要把整个区间的明细拉回来再从中查找。"
)


def _step_references(value: Any) -> set[str]:
    if isinstance(value, str):
        match = _STEP_REFERENCE.match(value.strip())
        return {match.group(1)} if match else set()
    if isinstance(value, dict):
        return set().union(*(_step_references(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_step_references(item) for item in value)) if value else set()
    return set()


def _contains_reference_marker(value: Any) -> bool:
    return "$ref:" in json.dumps(value, ensure_ascii=False, default=str)


class ValueBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    json_pointer: str
    value: int | float | str | bool | None


class DerivedValue(BaseModel):
    """A number computed from bound evidence values (total, change, share); the validator recomputes it."""

    model_config = ConfigDict(extra="forbid")

    # difference = first operand minus the rest; ratio = sum of all but the last operand / the last
    # (two operands: first / second), written as a decimal, not ×100.
    operation: Literal["sum", "difference", "ratio"]
    operands: list[ValueBinding] = Field(min_length=2, max_length=20)
    value: float


class ReportFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["anomaly", "cause", "trend", "data_quality", "other"]
    statement: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=10)
    value_bindings: list[ValueBinding] = Field(default_factory=list, max_length=20)
    derived_values: list[DerivedValue] = Field(default_factory=list, max_length=10)
    limitations: list[str] = Field(default_factory=list, max_length=20)


class PolicyCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    # Points at one candidate inside the search_policy result, e.g. /data/evidence/0.
    # Not pattern-restricted here: a wrong pointer is a validation error the reporter can revise.
    json_pointer: str = Field(default="", max_length=200)
    statement: str = Field(min_length=1, max_length=1000)


class ReportAnswer(BaseModel):
    """The direct answer to one scope item: the value asked for, or why it cannot be given."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=r"^q[0-9]{1,2}$")
    status: Literal["answered", "unanswerable"]
    statement: str = Field(min_length=1, max_length=2000)
    # The reading the answer uses (date range, slot, definition); required for ambiguous items.
    basis: str = Field(default="", max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    value_bindings: list[ValueBinding] = Field(default_factory=list, max_length=20)
    derived_values: list[DerivedValue] = Field(default_factory=list, max_length=10)


class MarketReportDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # First on purpose: models fill fields in schema order, and a report whose only finding slots are
    # "anomalies" and "causes" left plain questions ("what was the lowest price") without a place to answer.
    answers: list[ReportAnswer] = Field(default_factory=list, max_length=8)
    summary: str = Field(min_length=1, max_length=4000)
    summary_evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    anomalies: list[ReportFinding] = Field(default_factory=list, max_length=30)
    causes: list[ReportFinding] = Field(default_factory=list, max_length=30)
    outlook: ReportFinding | None = None
    policy_basis: list[PolicyCitation] = Field(default_factory=list, max_length=20)
    data_notes: list[str] = Field(default_factory=list, max_length=30)
    known_limitations: list[str] = Field(default_factory=list, max_length=30)
    unresolved_issues: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="before")
    @classmethod
    def _accept_limitations_key(cls, data: Any) -> Any:
        """forecast_price results carry a "limitations" list and models copy that key; read it as known_limitations."""
        if isinstance(data, dict) and "limitations" in data:
            data = dict(data)
            extra = data.pop("limitations")
            merged = list(data.get("known_limitations") or []) + (extra if isinstance(extra, list) else [extra])
            data["known_limitations"] = list(dict.fromkeys(str(item) for item in merged))[:30]
        return data


class AgentState(TypedDict, total=False):
    run_id: str
    question: str
    as_of: str
    mode: Literal["historical_analysis", "historical_forecast"]
    started_at: float
    scope: list[dict[str, Any]]
    plan_error: str
    plan: list[dict[str, Any]]
    pending_steps: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    reflection: dict[str, Any]
    unresolved_issues: list[str]
    known_limitations: list[str]
    replan_count: int
    report_revision_count: int
    report: dict[str, Any] | None
    report_errors: list[str]
    validation: dict[str, Any]
    events: list[dict[str, Any]]
    tool_call_count: int


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    raise ValueError("LLM 返回内容不是文本")


def _json_pointer(value: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer 必须以 / 开头")
    current = value
    for raw in pointer[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        current = current[int(key)] if isinstance(current, list) else current[key]
    return current


_RANGE_STATS = ("count", "min", "max")


def _compact_view_stat(value: Any, pointer: str) -> Any:
    """Resolve a statistic that exists only in the compacted prompt view, recomputed from the full list.

    ``<list>/length`` is the list length; ``<list>/numeric_ranges/<field>/<count|min|max>`` (also written
    without the ``numeric_ranges`` segment) is computed over the items whose ``field`` is numeric, exactly as
    ``_numeric_range`` built it; ``_value`` means the list's scalar items.
    """
    tokens = pointer[1:].split("/") if pointer.startswith("/") else []
    if len(tokens) >= 2 and tokens[-1] == "length":
        node = _json_pointer(value, "/" + "/".join(tokens[:-1])) if len(tokens) > 1 else value
        if isinstance(node, list):
            return len(node)
    if len(tokens) >= 3 and tokens[-1] in _RANGE_STATS:
        stat, field = tokens[-1], tokens[-2].replace("~1", "/").replace("~0", "~")
        base = tokens[:-2]
        if base and base[-1] == "numeric_ranges":
            base = base[:-1]
        node = _json_pointer(value, "/" + "/".join(base)) if base else value
        if isinstance(node, list):
            if field == "_value":
                numbers = [item for item in node if _is_number(item)]
            else:
                numbers = [item[field] for item in node if isinstance(item, dict) and _is_number(item.get(field))]
            if numbers:
                return {"count": len(numbers), "min": min(numbers), "max": max(numbers)}[stat]
    raise KeyError(pointer)


def _resolve_binding_pointer(value: Any, pointer: str) -> Any:
    """A binding's value: a real path into the full result, else a compacted-view statistic."""
    try:
        return _json_pointer(value, pointer)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        try:
            return _compact_view_stat(value, pointer)
        except (KeyError, IndexError, TypeError, ValueError):
            raise error from None


# Paths the model writes against the compacted view instead of the full result:
# ``.../head/i/...`` and ``.../i/item/...`` (older view shape) both mean item ``i`` of the original list.
_COMPACT_VIEW_SEGMENTS = (
    re.compile(r"/head/(\d+)(?=/|$)"),
    re.compile(r"/(\d+)/item(?=/|$)"),
)


def _full_result_pointer(pointer: str) -> str:
    for pattern in _COMPACT_VIEW_SEGMENTS:
        pointer = pattern.sub(lambda match: "/" + match.group(1), pointer)
    return pointer


def _report_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Every evidence-bound statement of a report; direct answers are held to the same binding rules."""
    findings = list(report.get("answers", [])) + list(report.get("anomalies", [])) + list(report.get("causes", []))
    if report.get("outlook"):
        findings.append(report["outlook"])
    return findings


def _finding_bindings(finding: dict[str, Any]) -> list[dict[str, Any]]:
    """Direct value bindings plus the operands of every derived value."""
    bindings = list(finding.get("value_bindings", []))
    for derived in finding.get("derived_values", []):
        bindings.extend(derived.get("operands", []))
    return bindings


def _derived_result(operation: str, values: list[float]) -> float | None:
    if operation == "sum":
        return sum(values)
    if operation == "difference":
        return values[0] - sum(values[1:])
    if operation == "ratio" and len(values) >= 2 and values[-1] != 0:
        # Shares such as "negative slots across several events / all slots": numerator parts, then denominator.
        return sum(values[:-1]) / values[-1]
    return None


def _value_decimals(value: float) -> int:
    text = repr(float(value))
    if "e" in text or "E" in text:
        return 9
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


_POLICY_CANDIDATE_POINTER = re.compile(r"^/data/evidence/\d+")
_POLICY_CITATION_POINTER = re.compile(r"^(.*)/data/evidence/\d+$")


def _search_policy_results(value: Any, pointer: str = "") -> list[tuple[str, dict[str, Any]]]:
    """search_policy results inside an observation result: the result itself, or one embedded by
    another tool (explain_drivers carries its candidates at /data/policy_evidence)."""
    if isinstance(value, dict):
        if value.get("tool") == "search_policy":
            return [(pointer, value)]
        found = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                found += _search_policy_results(item, f"{pointer}/{_pointer_token(key)}")
        return found
    if isinstance(value, list):
        return [hit for index, item in enumerate(value) if isinstance(item, (dict, list))
                for hit in _search_policy_results(item, f"{pointer}/{index}")]
    return []


def _policy_candidate(result: Any, pointer: str) -> dict[str, Any] | None:
    """The candidate a policy citation points at, only if it sits in a search_policy result's evidence list."""
    match = _POLICY_CITATION_POINTER.match(pointer or "")
    if not match:
        return None
    try:
        container = _json_pointer(result, match.group(1)) if match.group(1) else result
        candidate = _json_pointer(result, pointer)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not isinstance(container, dict) or container.get("tool") != "search_policy" or not isinstance(candidate, dict):
        return None
    return candidate


def _canonicalize_policy_ids(report: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace a policy candidate's own ``pol_`` id with the observation id and candidate pointer.

    Candidates inside a search_policy result carry an ``evidence_id`` of their own, which models cite in
    place of the observation id. The mapping is exact (one candidate, one position); a pol_ id that occurs
    in several observations resolves to an eligible occurrence first.
    """
    locations: dict[str, tuple[str, str, bool]] = {}
    for observation in observations:
        for base, policy_result in _search_policy_results(observation.get("result")):
            data = policy_result.get("data")
            evidence = data.get("evidence") if isinstance(data, dict) else None
            for index, candidate in enumerate(evidence if isinstance(evidence, list) else []):
                if not isinstance(candidate, dict) or not isinstance(candidate.get("evidence_id"), str):
                    continue
                location = (
                    observation["evidence_id"],
                    f"{base}/data/evidence/{index}",
                    candidate.get("evidence_eligible") is True,
                )
                current = locations.get(candidate["evidence_id"])
                if current is None or (location[2] and not current[2]):
                    locations[candidate["evidence_id"]] = location
    if not locations:
        return report

    def observation_ids(ids: list[Any]) -> list[Any]:
        return list(dict.fromkeys(locations[item][0] if item in locations else item for item in ids))

    for citation in report.get("policy_basis", []):
        location = locations.get(citation.get("evidence_id"))
        if location:
            citation["evidence_id"], citation["json_pointer"] = location[0], location[1]
    if "summary_evidence_ids" in report:
        report["summary_evidence_ids"] = observation_ids(report["summary_evidence_ids"])
    for finding in _report_findings(report):
        if "evidence_ids" in finding:
            finding["evidence_ids"] = observation_ids(finding["evidence_ids"])
        for binding in _finding_bindings(finding):
            location = locations.get(binding.get("evidence_id"))
            if not location:
                continue
            binding["evidence_id"] = location[0]
            pointer = binding.get("json_pointer", "")
            if pointer.startswith(location[1] + "/"):
                continue
            if not pointer.startswith("/data/") or _POLICY_CANDIDATE_POINTER.match(pointer):
                # Field paths written relative to the candidate itself (e.g. /page) are re-rooted on it.
                rest = _POLICY_CANDIDATE_POINTER.sub("", pointer)
                binding["json_pointer"] = location[1] + (rest if rest.startswith("/") or not rest else "/" + rest)
    return report


def _canonicalize_bindings(report: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Rewrite pointers written against the compacted prompt view to full-result paths.

    ``head[i]`` is always item ``i`` of the original list, so the rewrite is exact; a pointer that still
    does not resolve is left as written for the validator to reject.
    """
    results = {item["evidence_id"]: item.get("result") for item in observations}
    for finding in _report_findings(report):
        for binding in _finding_bindings(finding):
            pointer = binding.get("json_pointer", "")
            candidate = _full_result_pointer(pointer)
            if candidate == pointer or binding.get("evidence_id") not in results:
                continue
            try:
                _json_pointer(results[binding["evidence_id"]], candidate)
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            binding["json_pointer"] = candidate
    for finding in _report_findings(report):
        for binding in _finding_bindings(finding):
            relocated = _relocated_binding(binding, results)
            if relocated is None:
                continue
            binding["evidence_id"], binding["json_pointer"] = relocated
            if "evidence_ids" in finding and relocated[0] not in finding["evidence_ids"]:
                finding["evidence_ids"].append(relocated[0])
    return report


def _numeric_leaves(value: Any, pointer: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _numeric_leaves(item, f"{pointer}/{_pointer_token(key)}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _numeric_leaves(item, f"{pointer}/{index}")
    elif _is_number(value):
        yield pointer, value


def _relocated_binding(binding: dict[str, Any], results: dict[str, Any]) -> tuple[str, str] | None:
    """Where a numeric binding really is, when it was cited against the wrong observation or path.

    Models sometimes cite the right field and value under another tool's observation (detect_anomaly's
    /data/events/2 instead of explain_drivers' /data/events/2/event). If the cited location does not hold
    the value, the binding moves to a location whose field name is the same and whose value matches; among
    several such locations the one sharing the longest pointer prefix wins, then the earliest observation.
    """
    value, pointer = binding.get("value"), binding.get("json_pointer", "")
    if not _is_number(value) or not pointer.startswith("/"):
        return None
    try:
        if _same_value(_resolve_binding_pointer(results.get(binding.get("evidence_id")), pointer), value):
            return None
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    field = pointer.rsplit("/", 1)[-1]
    best: tuple[tuple[int, int, int], tuple[str, str]] | None = None
    for order, (evidence_id, result) in enumerate(results.items()):
        for leaf_pointer, actual in _numeric_leaves(result):
            if leaf_pointer.rsplit("/", 1)[-1] != field or not _same_value(actual, value):
                continue
            prefix = len(os.path.commonprefix([leaf_pointer, pointer]))
            rank = (int(math.isclose(float(actual), float(value), rel_tol=1e-9, abs_tol=1e-9)), prefix, -order)
            if best is None or rank > best[0]:
                best = (rank, (evidence_id, leaf_pointer))
    return best[1] if best else None


def _list_facts(finding: dict[str, Any], results: dict[str, Any]) -> tuple[list[float], set[int]]:
    """Lengths of the lists a finding's bindings point into (verifiable counts such as "共 2 段"),
    and the item positions they cite, 0- and 1-based (ordinals such as "第 1 段", "第 38 号时段")."""
    lengths: list[float] = []
    positions: set[int] = set()
    for binding in _finding_bindings(finding):
        node = results.get(binding.get("evidence_id"))
        for token in (binding.get("json_pointer") or "").split("/")[1:]:
            if isinstance(node, list) and token.isdigit() and int(token) < len(node):
                lengths.append(float(len(node)))
                positions.update({int(token), int(token) + 1})
                node = node[int(token)]
            elif isinstance(node, dict):
                node = node.get(token.replace("~1", "/").replace("~0", "~"))
            else:
                break
    return lengths, positions


_ORDINAL_PATTERN = re.compile(r"第\s*(\d+)\s*(?=[段个号次处]|时段)")


def _pointer_token(key: Any) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# Fields that identify which row a list item is (date, slot, group, event); shown next to range extremes.
_ROW_KEY_FIELDS = ("trade_date", "target_date", "group_value", "time_label", "slot_id", "timestamp_raw",
                   "event_id", "event_type")


def _row_keys_of(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {key: item[key] for key in _ROW_KEY_FIELDS if key in item and not isinstance(item[key], (dict, list))}


def _numeric_range(
    indexed: list[tuple[Any, int]],
    pointer_of: Callable[[int], str],
    keys_of: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    low = min(indexed)
    high = max(indexed)
    out = {
        "count": len(indexed),
        "min": low[0],
        "min_pointer": pointer_of(low[1]),
        "max": high[0],
        "max_pointer": pointer_of(high[1]),
    }
    # Without the row's keys the model has to count list positions to learn which day/slot an extreme is.
    if keys_of is not None:
        for name, index in (("min_row", low[1]), ("max_row", high[1])):
            keys = keys_of(index)
            if keys:
                out[name] = keys
    return out


def _head_entry(item: Any, item_pointer: str) -> dict[str, Any]:
    if isinstance(item, dict):
        return {"_pointer": item_pointer, **item}
    return {"_pointer": item_pointer, "value": item}


def _compact_for_prompt(
    value: Any,
    pointer: str = "",
    *,
    max_items: int = MAX_PROMPT_LIST_ITEMS,
    max_chars: int = MAX_PROMPT_STRING_CHARS,
    numeric_ranges: bool = True,
) -> Any:
    """Shrink a tool result for LLM prompts; long lists keep JSON Pointers into the full result."""
    def nested(item: Any, item_pointer: str) -> Any:
        return _compact_for_prompt(
            item, item_pointer, max_items=max_items, max_chars=max_chars, numeric_ranges=numeric_ranges
        )

    if isinstance(value, dict):
        return {key: nested(item, f"{pointer}/{_pointer_token(key)}") for key, item in value.items()}
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]}…[截断，原长 {len(value)} 字符]"
    if not isinstance(value, list):
        return value
    if len(value) <= max_items:
        return [nested(item, f"{pointer}/{index}") for index, item in enumerate(value)]
    head = min(PROMPT_LIST_HEAD, max_items)
    compact: dict[str, Any] = {
        "_compacted_list": True,
        "pointer": pointer,
        "length": len(value),
        # Each head entry is the item itself plus its "_pointer", so "_pointer" + "/field" is a valid path.
        "head": [_head_entry(nested(value[index], f"{pointer}/{index}"), f"{pointer}/{index}") for index in range(head)],
        "omitted_items": len(value) - head,
    }
    if not numeric_ranges:
        return compact
    ranges: dict[str, Any] = {}
    scalars = [(item, index) for index, item in enumerate(value) if _is_number(item)]
    if scalars:
        ranges["_value"] = _numeric_range(scalars, lambda index: f"{pointer}/{index}")
    keys = dict.fromkeys(key for item in value if isinstance(item, dict) for key in item)
    for key in keys:
        if str(key).endswith("_id"):
            continue
        indexed = [
            (item[key], index)
            for index, item in enumerate(value)
            if isinstance(item, dict) and _is_number(item.get(key))
        ]
        if indexed:
            token = _pointer_token(key)
            ranges[str(key)] = _numeric_range(
                indexed,
                lambda index, token=token: f"{pointer}/{index}/{token}",
                lambda index: _row_keys_of(value[index]),
            )
    if ranges:
        compact["numeric_ranges"] = ranges
    return compact


def _compact_observation_result(result: Any, *, deep: bool = False) -> Any:
    compact = _compact_for_prompt(result)
    if not deep and len(json.dumps(compact, ensure_ascii=False)) <= MAX_OBSERVATION_PROMPT_CHARS:
        return compact
    # Nested evidence bundles (e.g. several explained events) still overflow: keep only list heads,
    # since per-list numeric ranges with pointers cost more text than short lists themselves.
    return _compact_for_prompt(
        result,
        max_items=DEEP_PROMPT_LIST_ITEMS,
        max_chars=DEEP_PROMPT_STRING_CHARS,
        numeric_ranges=False,
    )


def _prompt_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """LLM view of observations; the untrimmed results stay in state and the trace."""
    view = _prompt_observation_view(observations, deep=False)
    if len(json.dumps(view, ensure_ascii=False)) > MAX_OBSERVATIONS_PROMPT_CHARS:
        # Every reflector/replanner/reporter call resends all observations; keep the total bounded.
        view = _prompt_observation_view(observations, deep=True)
    return view


def _prompt_observation_view(observations: list[dict[str, Any]], *, deep: bool) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": observation["evidence_id"],
            "step_id": observation["step_id"],
            "tool_name": observation["tool_name"],
            "arguments": {key: value for key, value in observation["arguments"].items() if key != "as_of"},
            "execution_status": observation["execution_status"],
            "error": observation.get("error"),
            "result": _compact_observation_result(observation.get("result"), deep=deep),
        }
        for observation in observations
    ]


PROMPT_COMPACTION_NOTE = (
    "observations 的 result 已为提示压缩：较长列表以 _compacted_list 表示，只列前几项（每项带 _pointer）"
    "及各数值字段的 min/max、对应 JSON Pointer 和所在行的日期/时段（min_row/max_row）；完整结果保存在 trace 中，压缩不是数据缺失。"
    "query_stats 的 group_extremes 直接给出各统计量最大、最小的分组（及并列数）。"
    "未列出的行在这里看不到、也不能绑定：需要某一行（某日、某时段）的值时，用带日期或 slot_id 的窄查询取那一行，"
    "不要拿极值或相邻行代替。"
)

# Dates and clock times are references, not numeric claims that need a value binding.
# A day is only part of a date with 日/号 or an ISO separator, and no date part may run into the next digits:
# in "2025年2月716个" the date is 2025年2月 and 716 is a claim.
_DATE_TIME_PATTERN = re.compile(
    r"\d{4}\s*[-/]\s*\d{1,2}(?!\d)(?:\s*[-/]\s*\d{1,2}(?!\d))?"
    r"|\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*[日号])?)?"
    r"|\d{1,2}\s*月(?:\s*\d{1,2}\s*[日号])?"
    r"|\d{1,2}\s*日"  # a bare 号 is a document number (〔2025〕63号), not a day
    # clock time with optional seconds and UTC offset, e.g. 15:00, 15:00:00+08, 08:00+08:00
    r"|\d{1,2}:\d{2}(?::\d{2})?(?:[+-]\d{2}(?::?\d{2})?)?(?!\d)"
    # clock hour in words: 7点, 7点整, 7点半, 19点30分, 7时 (but not 96 个时段)
    r"|\d{1,2}\s*(?:点|时(?!段))(?:整|半|\s*\d{1,2}\s*分)?"
)


def _has_numeric_claim(text: str) -> bool:
    # Dates, times, identifiers (D+1, P99) and clause locators are not numeric claims.
    return bool(_statement_numbers(text))


# Identifiers and units that contain digits but are not numeric claims (P99, D+1, bge-m3, 15 分钟).
_NON_CLAIM_NUMBER_PATTERN = re.compile(
    r"slot(?:_id)?\s*[=:：]?\s*\d+(?:\s*[-~–—至到]\s*\d+)?|槽位\s*\d+(?:\s*[-~–—至到]\s*\d+)?|[A-Za-z]+\s*[-+]?\s*\d+|\d+\s*分钟"
    # document numbers and clause locators: 〔2025〕63号, [2024]24号, 附件1, 第3.5.4条, 第22章, 第38页,
    # and the edition year in a document title: 《……实施细则（2023）》, （2023年版）
    r"|[〔\[［]\s*\d{4}\s*[〕\]］]\s*\d+\s*号|附件\s*\d+|第\s*[\d.]+\s*[条章节款项页]"
    r"|[（(]\s*(?:19|20)\d{2}\s*(?:年版?|版|年修订版?)?\s*[）)]"
    # a comparison with zero states a sign condition ("rt_price<0", "≥0"); there is no evidence value to bind
    r"|[<>＜＞≤≥]\s*=?\s*0(?![\d.])"
)
_STATEMENT_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?")


def _statement_numbers(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = _NON_CLAIM_NUMBER_PATTERN.sub(" ", _DATE_TIME_PATTERN.sub(" ", text))
    return [match.replace("−", "-") for match in _STATEMENT_NUMBER_PATTERN.findall(text)]


def _number_is_bound(text_value: str, bound: list[float]) -> bool:
    """True when the written number is a rounding of a bound value (sign kept, or the
    magnitude written with direction words, or a ratio written as a percentage)."""
    written = float(text_value)
    decimals = len(text_value.split(".")[1]) if "." in text_value else 0
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    for value in bound:
        candidates = {value, abs(value), value * 100, abs(value) * 100}
        if any(abs(candidate - written) <= tolerance for candidate in candidates):
            return True
    return False


# Row identity named in a statement: clock times ("10:00"), slots ("slot 39"), and calendar days.
_STATEMENT_CLOCK_PATTERN = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?!\d)")
_STATEMENT_SLOT_PATTERN = re.compile(r"slot(?:_id)?\s*[:=]?\s*(\d+)", re.I)
_STATEMENT_DAY_PATTERNS = (
    re.compile(r"\d{4}\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})(?!\d)"),
    re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]"),
)
_ISO_DAY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _clock(hour: Any, minute: Any) -> str:
    hour, minute = int(hour), int(minute)
    return "00:00" if (hour, minute) == (24, 0) else f"{hour:02d}:{minute:02d}"


def _slot_clock(slot_id: int) -> str:
    """A 15-minute slot is labelled by its end time (slot 0 -> 00:15)."""
    minutes = (slot_id + 1) * 15
    return _clock(minutes // 60, minutes % 60)


_DAY_TOKEN = r"(?:\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}(?!\d)|(?:\d{4}\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*[日号]?)"
_DAY_RANGE_PATTERN = re.compile(
    rf"({_DAY_TOKEN})\s*(?:至|到|~|～|—|–)\s*({_DAY_TOKEN}|\d{{1,2}}\s*[日号])"
)


def _month_day(token: str, month_hint: int | None = None) -> tuple[int, int] | None:
    numbers = [int(n) for n in re.findall(r"\d+", token)]
    if re.match(r"^\d{4}", token.strip()):
        numbers = numbers[1:]
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    if len(numbers) == 1 and month_hint is not None:
        return month_hint, numbers[0]
    return None


def _range_days(first: tuple[int, int], last: tuple[int, int]) -> set[tuple[int, int]]:
    """Every (month, day) from first to last inclusive; a range may run over a year end."""
    try:
        start = date(2024, *first)  # a leap year, so 2月29日 is a valid day
        end = date(2024 if last >= first else 2025, *last)
    except ValueError:
        return set()
    if (end - start).days > 400:
        return set()
    return {((start + timedelta(days=i)).month, (start + timedelta(days=i)).day) for i in range((end - start).days + 1)}


def _statement_row_keys(text: str) -> tuple[set[str], set[tuple[int, int]]]:
    """Clock times (slots converted to their clock time) and (month, day) pairs a statement names; a day range
    ("12月19日至21日") names every day in it."""
    clocks = {_clock(*match) for match in _STATEMENT_CLOCK_PATTERN.findall(text)}
    clocks |= {_slot_clock(int(slot)) for slot in _STATEMENT_SLOT_PATTERN.findall(text)}
    plain = _STATEMENT_CLOCK_PATTERN.sub(" ", text)
    days = {(int(month), int(day)) for pattern in _STATEMENT_DAY_PATTERNS for month, day in pattern.findall(plain)}
    for first_token, last_token in _DAY_RANGE_PATTERN.findall(plain):
        first = _month_day(first_token)
        last = _month_day(last_token, first[0] if first else None)
        if first and last:
            days |= _range_days(first, last)
    return clocks, days


def _row_identity(row: dict[str, Any], group_by: Any) -> dict[str, Any]:
    """Day and clock time a result row stands for (its date/time keys, or its group value)."""
    identity: dict[str, Any] = {}
    for name in ("trade_date", "target_date"):
        value = row.get(name)
        if isinstance(value, str) and _ISO_DAY_PATTERN.match(value):
            identity["day"] = value[:10]
    stamp = row.get("timestamp_raw")
    if isinstance(stamp, str) and _ISO_DAY_PATTERN.match(stamp):
        identity["day"] = stamp[:10]
        if re.match(r"^.{10}[ T]\d{2}:\d{2}", stamp):
            identity["clock"] = _clock(stamp[11:13], stamp[14:16])
    group = row.get("group_value")
    if isinstance(group, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", group):
        identity["day"] = group
    elif isinstance(group, int) and not isinstance(group, bool) and group_by == "slot":
        identity["clock"] = _slot_clock(group)
    if isinstance(row.get("slot_id"), int) and not isinstance(row.get("slot_id"), bool):
        identity["clock"] = _slot_clock(row["slot_id"])
    label = row.get("time_label")
    if isinstance(label, str) and re.fullmatch(r"\d{1,2}:\d{2}", label):
        identity["clock"] = _clock(*label.split(":"))
    return identity


def _bound_row_identity(result: Any, pointer: str) -> dict[str, Any]:
    """Day and clock time of the row that directly holds the value at ``pointer`` (empty when it has none)."""
    current, parent, group_by = result, None, None
    try:
        for raw in pointer.split("/")[1:]:
            if isinstance(current, dict):
                group_by = current.get("group_by", group_by)
            parent = current
            key = raw.replace("~1", "/").replace("~0", "~")
            current = current[int(key)] if isinstance(current, list) else current[key]
    except (KeyError, IndexError, TypeError, ValueError):
        return {}
    if not isinstance(parent, dict):
        return {}
    identity = _row_identity(parent, group_by)
    # An extreme (max_price, min_value, ...) sits at the row its max_at / min_at names.
    leaf = pointer.rsplit("/", 1)[-1]
    located = parent.get(f"{leaf[:3]}_at") if leaf[:4] in ("max_", "min_") else None
    if isinstance(located, dict):
        identity |= _row_identity(located, None)
    return identity


def _row_mismatch_errors(finding: dict[str, Any], results: dict[str, Any]) -> list[str]:
    """A value written in the statement must come from a row whose day/time the statement names, when it names any.

    Catches a right field read from the wrong row (e.g. "10:00 风电 320 MW" bound to the 12:30 row)."""
    statement = finding.get("statement", "")
    clocks, days = _statement_row_keys(statement)
    if not clocks and not days:
        return []
    written = _statement_numbers(statement)
    # A clock time attaches to its own clause: "最高 14.1，出现在 00:00；最低 2.0" gives no time for 2.0.
    clauses = [clause for clause in re.split(r"[；;。]", statement) if clause.strip()]
    errors = []
    for binding in _finding_bindings(finding):
        value = binding.get("value")
        if not _is_number(value) or not any(_number_is_bound(number, [float(value)]) for number in written):
            continue  # only values the statement actually writes (not operands of a derived value)
        identity = _bound_row_identity(results.get(binding.get("evidence_id")), binding.get("json_pointer", ""))
        own = [c for c in clauses if any(_number_is_bound(n, [float(value)]) for n in _statement_numbers(c))]
        clause_clocks = set().union(*(_statement_row_keys(c)[0] for c in own)) if own else clocks
        problems = []
        if "clock" in identity and clause_clocks and identity["clock"] not in clause_clocks:
            problems.append(f"时刻 {identity['clock']}")
        if "day" in identity and days:
            _, month, day = identity["day"].split("-")
            if (int(month), int(day)) not in days:
                problems.append(f"日期 {identity['day']}")
        if problems:
            errors.append(
                f"数值 {value} 绑定的行（{binding.get('evidence_id')} {binding.get('json_pointer')}）属于"
                f"{'、'.join(problems)}，陈述中没有这个日期/时刻；请绑定陈述所指那一行的值，"
                "若该数值确属另一日期/时刻，在陈述中写明它的日期/时刻或拆成单独的发现"
            )
    return errors


_COMPONENT_WORDS = {
    "load": ("负荷",),
    "tie_line": ("联络线", "外来电"),
    "wind": ("风电",),
    "solar": ("光伏",),
    "nuclear": ("核电",),
    "captive": ("自备",),
    "local_generation": ("地方电源", "地方电厂"),
}
_ABOVE_WORDS = ("高于", "偏高", "多于", "高出", "超出", "大于", "较预测高", "比预测高", "多发")
_BELOW_WORDS = ("低于", "偏低", "少于", "低出", "不足", "小于", "较预测低", "比预测低", "少发")
_EFFECT_WORDS = {"正贡献": "increase", "负贡献": "decrease"}


def _component_segments(statement: str, component: str) -> list[str]:
    """Clauses that start at a mention of `component` and stop at the next sentence end
    or the next mention of another component."""
    other_words = [word for name, words in _COMPONENT_WORDS.items() if name != component for word in words]
    segments = []
    for word in _COMPONENT_WORDS.get(component, ()):
        for match in re.finditer(re.escape(word), statement):
            rest = statement[match.end():]
            stop = len(rest)
            for boundary in re.finditer(r"[。；;]", rest):
                stop = boundary.start()
                break
            for other in other_words:
                position = rest.find(other)
                if 0 <= position < stop:
                    stop = position
            segments.append(rest[:stop])
    return segments


def _first_word(segment: str, words: tuple[str, ...]) -> tuple[int, str] | None:
    hits = [(segment.find(word), word) for word in words if word in segment]
    return min(hits) if hits else None


def _direction_errors(finding: dict[str, Any], observations_by_id: dict[str, Any]) -> list[str]:
    errors = []
    statement = finding.get("statement", "")
    for binding in finding.get("value_bindings", []):
        match = re.match(r"(.*/contributions/\d+)/", binding.get("json_pointer", ""))
        observation = observations_by_id.get(binding.get("evidence_id"))
        if not match or observation is None:
            continue
        try:
            item = _json_pointer(observation.get("result"), match.group(1))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not isinstance(item, dict) or item.get("component") not in _COMPONENT_WORDS:
            continue
        for segment in _component_segments(statement, item["component"]):
            above = _first_word(segment, _ABOVE_WORDS)
            below = _first_word(segment, _BELOW_WORDS)
            if above or below:
                claimed = "above" if below is None or (above and above[0] < below[0]) else "below"
                actual = item.get("actual_vs_forecast")
                if actual in {"above", "below"} and claimed != actual:
                    word = (above if claimed == "above" else below)[1]
                    errors.append(
                        f"方向描述与证据不符：{item['component']} 实际值{'高于' if actual == 'above' else '低于'}预测"
                        f"（{match.group(1)}/actual_vs_forecast），陈述写成“{word}”"
                    )
            for word, effect in _EFFECT_WORDS.items():
                if word in segment and item.get("bid_space_effect") in {"increase", "decrease"} and item["bid_space_effect"] != effect:
                    errors.append(
                        f"方向描述与证据不符：{item['component']} 对竞价空间为"
                        f"{'正' if item['bid_space_effect'] == 'increase' else '负'}贡献"
                        f"（{match.group(1)}/bid_space_effect），陈述写成“{word}”"
                    )
    return list(dict.fromkeys(errors))


def _same_value(actual: Any, expected: Any) -> bool:
    """Equal, or the bound value is the evidence value rounded to the decimals it is written with
    (365.974 for 365.974038690), the same rule statement numbers follow."""
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not (isinstance(expected, (int, float)) and not isinstance(expected, bool)):
            return False
        if math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
            return True
        return abs(float(actual) - float(expected)) <= 0.5 * 10 ** (-_value_decimals(expected)) + 1e-9
    return actual == expected


class MarketAnalysisAgent:
    """A bounded LangGraph workflow; the model can plan but tools remain schema-validated."""

    def __init__(
        self,
        *,
        json_call: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None,
        tool_call: Callable[[str, dict[str, Any]], dict[str, Any]] = invoke_market_tool,
        variant: AgentVariant = "E2",
        catalog: Callable[[datetime], dict[str, Any]] = data_catalog,
    ) -> None:
        if variant not in VARIANT_DESCRIPTIONS:
            raise ValueError(f"unknown agent variant {variant}")
        self.json_call = json_call
        self.catalog = catalog
        self.variant = variant
        self.tool_call = self._without_effective_filter(tool_call) if variant == "E3" else tool_call
        self._compiled_graph = None

    @staticmethod
    def _without_effective_filter(tool_call):
        """E3: run every tool with the policy effective-interval filter off, in whatever thread runs it."""
        from ellectric.service.market_tools import POLICY_EFFECTIVE_FILTER

        def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            token = POLICY_EFFECTIVE_FILTER.set(False)
            try:
                return tool_call(name, arguments)
            finally:
                POLICY_EFFECTIVE_FILTER.reset(token)

        return call

    def _work_seconds(self, started_at: float) -> float:
        """Run time charged to the time budget: wall time minus waiting for the LLM request quota.

        Queueing behind the process-wide RPM cap (and 429 back-off) depends on how many runs share the
        endpoint, not on the agent's own work; charging it let concurrent runs starve each other's budget.
        """
        return time.monotonic() - started_at - getattr(self, "rate_limit_wait_seconds", 0.0)

    def _json(
        self, schema_name: str, schema: type[BaseModel], prompt: str, check: Callable[[Any], Any] | None = None
    ) -> Any:
        """One structured LLM call with a single corrective retry on malformed or off-schema output.

        ``check`` (optional) validates the parsed object beyond its schema and returns the value to use; a
        ``PlanCheckError`` it raises gets the same single corrective retry, with the error shown to the model.
        """
        def attempt(text: str) -> Any:
            parsed = self._json_once(schema_name, schema, text)
            return check(parsed) if check is not None else parsed

        try:
            return attempt(prompt)
        except (ValidationError, ValueError) as error:
            detail = str(error)[:1500]
            self.schema_retry_count = getattr(self, "schema_retry_count", 0) + 1
            if isinstance(error, PlanCheckError):
                hint = "JSON 结构本身合法，只需按上述错误修正对应步骤（参数须符合该工具的 input_schema），其余步骤保持不变。"
            elif "JSON 解析失败" in detail:
                hint = (
                    "上一次输出根本不是 JSON（例如写成了 Markdown 报告，或写得过长被截断）。"
                    "这次第一个字符必须是 {，不要写任何标题、表格或正文段落；"
                    "每个字符串字段保持简短，整体控制在 schema 要求之内。"
                )
            else:
                hint = (
                    "请只输出一个严格符合 schema 的 JSON 对象，不要添加 schema 之外的字段。"
                    f"顶层字段只能是：{', '.join(schema.model_fields)}，字段名一字不差。"
                    + ("固有限制写 known_limitations，不是 limitations。" if "known_limitations" in schema.model_fields else "")
                )
            retry_prompt = f"{prompt}\n\n上一次输出未通过 {schema_name} 的 JSON/schema 校验：{detail}\n{hint}"
            return attempt(retry_prompt)

    def _json_once(self, schema_name: str, schema: type[BaseModel], prompt: str) -> BaseModel:
        if self._work_seconds(getattr(self, "_run_started_at", time.monotonic())) > MAX_RUN_SECONDS:
            raise TimeoutError("analysis run time budget exceeded")
        if getattr(self, "_last_token_count", 0) >= MAX_TOTAL_TOKENS:
            raise RuntimeError("analysis token budget exceeded")
        if self.json_call is not None:
            payload = self.json_call(schema_name, schema.model_json_schema(), prompt)
            return schema.model_validate(payload)
        api_key = os.environ.get("ELLECTRIC_LLM_API_KEY", "")
        if not api_key:
            raise RuntimeError("ELLECTRIC_LLM_API_KEY 未设置；请在 WSL 运行环境中注入密钥")
        base_url = os.environ.get("ELLECTRIC_LLM_BASE_URL", "https://aispot.swj0227.icu/v1")
        model = os.environ.get("ELLECTRIC_LLM_MODEL", "deepseek-v4.1-flash")
        request_body = {
            "model": model,
            "temperature": 0,
            "max_tokens": _llm_max_tokens(),
            "response_format": {"type": "json_object"},
            **_thinking_request_fields(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object that conforms to the supplied JSON schema. "
                        "The first character of your reply must be '{' and the last must be '}'. "
                        "Never write Markdown, headings, tables, code fences or prose outside the JSON, "
                        "even when the user asks for a report or 周报: put that content into the schema fields. "
                        "Keep every string field concise. "
                        "Use only the given observations; do not invent data or policy authority."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Schema {schema_name}:\n"
                        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}\n\n{prompt}"
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        retry_delays = list(RATE_LIMIT_RETRY_DELAYS)
        bad_request_retries = BAD_REQUEST_RETRIES
        while True:
            self.rate_limit_wait_seconds = getattr(self, "rate_limit_wait_seconds", 0.0) + _acquire_rate_slot()
            try:
                with urllib.request.urlopen(request, timeout=_llm_timeout_seconds()) as response_stream:
                    response_payload = json.loads(response_stream.read())
                break
            except urllib.error.HTTPError as error:
                if error.code == 429 and retry_delays:
                    delay = retry_delays.pop(0)
                    self.rate_limit_retry_count = getattr(self, "rate_limit_retry_count", 0) + 1
                    self.rate_limit_wait_seconds = getattr(self, "rate_limit_wait_seconds", 0.0) + delay
                    time.sleep(delay)
                    continue
                if error.code == 400 and bad_request_retries > 0:
                    bad_request_retries -= 1
                    self.bad_request_retry_count = getattr(self, "bad_request_retry_count", 0) + 1
                    continue
                raise RuntimeError(f"LLM 端点返回 HTTP {error.code}") from None
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError,
                    json.JSONDecodeError) as error:  # transport failures, incl. a connection dropped mid-response
                raise RuntimeError(f"LLM 请求失败: {type(error).__name__}") from None
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("LLM 响应缺少 choices")
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("LLM 响应缺少 assistant message")
        usage = response_payload.get("usage")
        if isinstance(usage, dict):
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
                self.reasoning_token_count = getattr(self, "reasoning_token_count", 0) + details["reasoning_tokens"]
            self._last_token_count = getattr(self, "_last_token_count", 0) + int(
                usage.get("total_tokens", usage.get("completion_tokens", 0))
            )
            if self._last_token_count > MAX_TOTAL_TOKENS:
                raise RuntimeError("analysis token budget exceeded")
        text = _content_text(message.get("content"))
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            self.non_json_output_count = getattr(self, "non_json_output_count", 0) + 1
            finish = choices[0].get("finish_reason")
            start = text.lstrip()[:1] or "空"
            raise ValueError(
                f"{schema_name} JSON 解析失败：输出不是 JSON 对象（首字符 {start!r}，finish_reason={finish}）"
            ) from error
        return schema.model_validate(payload)

    @staticmethod
    def _event(state: AgentState, event_type: str, **payload: Any) -> list[dict[str, Any]]:
        return state.get("events", []) + [
            {"event_id": f"evt_{uuid.uuid4().hex[:16]}", "type": event_type, **payload}
        ]

    @staticmethod
    def _tool_catalog() -> dict[str, Any]:
        return {
            name: {"description": function.__doc__ or name, "input_schema": TOOL_MODELS[name].model_json_schema()}
            for name, function in __import__(
                "ellectric.service.market_tools", fromlist=["TOOL_FUNCTIONS"]
            ).TOOL_FUNCTIONS.items()
        }

    @staticmethod
    def _bind_run_context(steps: list[PlanStep], as_of: str) -> list[PlanStep]:
        """Every tool takes ``as_of``; the run's value always replaces anything the LLM supplied."""
        return [
            step.model_copy(update={"arguments": {**step.arguments, "as_of": as_of}})
            for step in steps
        ]

    @staticmethod
    def _cutoff_note(state: AgentState) -> str:
        latest = latest_available_trade_date(datetime.fromisoformat(state["as_of"]))
        return (
            f"latest_available_trade_date={latest.isoformat()}（as_of 时可得的最后完整交易日；"
            "所有 end_date 不得晚于它）。as_of 参数由系统注入，不要自行填写。"
            "historical_forecast 只能预测 target_date=latest_available_trade_date 的下一天。"
            "search_policy 的 policy_at 不得晚于 as_of 日期。"
        )

    def _checked_plan(self, plan: PlanDraft, existing_step_ids: set[str], as_of: str) -> list[PlanStep]:
        """Structure check, run-context binding, and literal arguments validated against each tool's input
        model, so an invalid argument is corrected at planning time instead of failing (and cascading to its
        dependents) at execution. Steps with $ref arguments are validated after binding, in the executor."""
        steps = self._bind_run_context(self._validate_plan(plan, existing_step_ids), as_of)
        problems = []
        for step in steps:
            if _step_references(step.arguments):
                continue
            try:
                TOOL_MODELS[step.tool_name].model_validate(step.arguments)
            except ValidationError as error:
                detail = "; ".join(
                    f"{'.'.join(str(part) for part in item['loc']) or '参数'}: {item['msg']}（输入 {item.get('input')!r}）"
                    for item in error.errors()
                    if item["loc"] != ("as_of",)
                ) or str(error).splitlines()[0]
                problems.append(f"{step.step_id} {step.tool_name} 参数不合法 — {detail}")
        if problems:
            raise PlanCheckError("计划参数校验失败: " + " | ".join(problems), steps=steps)
        return steps

    @staticmethod
    def _validate_plan(plan: PlanDraft, existing_step_ids: set[str]) -> list[PlanStep]:
        steps = plan.steps
        ids = [step.step_id for step in steps]
        if len(set(ids)) != len(ids) or existing_step_ids.intersection(ids):
            raise PlanCheckError("计划 step_id 重复")
        available = set(existing_step_ids)
        checked = []
        for step in steps:
            references = _step_references(step.arguments)
            if not (set(step.depends_on) | references).issubset(available):
                raise PlanCheckError(f"计划依赖和 $ref 引用必须指向此前步骤: {step.step_id}")
            if references - set(step.depends_on):
                step = step.model_copy(update={"depends_on": [*step.depends_on, *sorted(references - set(step.depends_on))]})
            checked.append(step)
            available.add(step.step_id)
        return checked

    def _scoper(self, state: AgentState) -> dict[str, Any]:
        """Before any tool call: split the question into the things it asks for and check each against the
        data catalog. Whether the data can answer a question is decided here, from what the dataset covers,
        instead of being left to whatever a query happens to return."""
        prompt = (
            "取数之前，先判断现有数据能不能回答用户问题。把问题拆成它要求回答的各项："
            "每个要给出的量、要找出的日期或时刻、要解释的事各算一项（“A 和 B 分别是多少”是两项），最多 8 项。"
            "逐项对照 data_catalog 填写：\n"
            "- answer_type：这一项的答案是数值（number）、日期或时刻（date_or_time），还是文字说明（text）。\n"
            "- status=unanswerable：这一项要的地区、市场或价格类型、指标、日期、预测日，data_catalog 明确没有，"
            "或落在可得日期范围之外。note 写明缺的是什么、目录里有的是什么。\n"
            "- status=ambiguous：数据能回答，但问题里的时间范围、时刻或统计口径有不止一种读法。"
            "note 写明采用哪一种读法（具体的日期区间、时段或口径），取最常见的读法。\n"
            "- status=answerable：其余情况，note 留空。拿不准时用 answerable；数据本身的固有限制"
            "（如时段边界语义未确认）不是 unanswerable 的理由。\n"
            "note 只说明读法或缺什么，用问题里的日期和时刻来写；不要写取数方法、工具名或时段编号。\n"
            "data_catalog 没有的东西，不能用目录里别的地区、别的市场、别的日期或别的指标顶替，"
            "也不要把问题改写成能回答的另一个问题。item 用问题原话里的说法。\n"
            f"question={state['question']}\nas_of={state['as_of']}\nmode={state['mode']}\n"
            f"data_catalog={json.dumps(self.catalog(datetime.fromisoformat(state['as_of'])), ensure_ascii=False)}\n"
            "返回 JSON: {\"items\":[{\"item\":\"...\",\"answer_type\":\"number\","
            "\"status\":\"answerable\",\"note\":\"\"}]}"
        )
        try:
            draft = self._json("ScopeDraft", ScopeDraft, prompt)
        except (ValidationError, ValueError) as error:
            # Fail open: without a usable scope the run proceeds as one answerable item.
            scope = [{"item_id": "q1", "item": state["question"][:300], "answer_type": "text",
                      "status": "answerable", "note": ""}]
            return {"scope": scope, "events": self._event(state, "scope", items=scope, error=str(error)[:300])}
        # An answerable item carries no note: how to fetch it is the planner's (or ReAct loop's) decision, and
        # a method written here was followed in place of the tools' own descriptions.
        scope = [
            {"item_id": f"q{index}", **item.model_dump(mode="json")}
            | ({"note": ""} if item.status == "answerable" else {})
            for index, item in enumerate(draft.items, start=1)
        ]
        return {"scope": scope, "events": self._event(state, "scope", items=scope)}

    @staticmethod
    def _scope_note(state: AgentState) -> str:
        scope = state.get("scope") or []
        if not scope:
            return ""
        return (
            "scope 是取数前对问题各项的可答性判断：status=unanswerable 的项，现有数据不包含，不要为它取数，"
            "也不要用别的地区、市场、日期或指标的数据代替；status=ambiguous 的项按 note 写明的读法处理，"
            "读法本身不再算作问题。\n"
            f"scope={json.dumps(scope, ensure_ascii=False)}\n"
        )

    def _after_scope(self, state: AgentState) -> str:
        scope = state.get("scope") or []
        if scope and all(item["status"] == "unanswerable" for item in scope):
            return "reporter"  # nothing the data can answer: no tool calls, the report states why
        return "react" if self.variant == "E0" else "planner"

    def _planner(self, state: AgentState) -> dict[str, Any]:
        prompt = (
            "为用户问题制定 1 到 6 个步骤。只选择工具目录中的工具，参数必须符合其 schema，"
            "先查事实再解释；只有历史预测问题才调用 forecast_price。政策工具可能返回未验证候选。"
            f"{PLAN_REFERENCE_NOTE}explain_drivers 省略 event_ids 时会自动解释区间内"
            "有代表性的价格事件，不要编造或占位 event_id。\n"
            f"question={state['question']}\nas_of={state['as_of']}\nmode={state['mode']}\n"
            f"{self._scope_note(state)}"
            f"{self._cutoff_note(state)}\n"
            f"tools={json.dumps(self._tool_catalog(), ensure_ascii=False)}\n"
            "返回 JSON: {\"steps\":[{\"step_id\":\"s1\",\"tool_name\":...,"
            "\"arguments\":{},\"depends_on\":[],\"rationale\":\"...\"}]}"
        )
        try:
            steps = self._json(
                "PlanDraft", PlanDraft, prompt, check=lambda draft: self._checked_plan(draft, set(), state["as_of"])
            )
        except (ValidationError, ValueError) as error:
            # The corrective retry failed too. A plan whose only fault is invalid tool arguments still runs:
            # those steps fail in the executor with the tool's own error, which reflection can repair or the
            # report can state. Anything else leaves no plan, recorded as an issue instead of ending the run.
            steps = error.steps if isinstance(error, PlanCheckError) else None
            if not steps:
                message = f"规划失败: {str(error)[:500]}"
                return {
                    "plan": [],
                    "pending_steps": [],
                    "plan_error": message,
                    "unresolved_issues": state.get("unresolved_issues", []) + [message],
                    "events": self._event(state, "plan_failed", error=message),
                }
            plan = [step.model_dump(mode="json") for step in steps]
            return {
                "plan": plan,
                "pending_steps": plan,
                "events": self._event(state, "plan", steps=plan, argument_errors=str(error)[:1000]),
            }
        plan = [step.model_dump(mode="json") for step in steps]
        return {
            "plan": plan,
            "pending_steps": plan,
            "events": self._event(state, "plan", steps=plan),
        }

    def _executor(self, state: AgentState) -> dict[str, Any]:
        observations = list(state.get("observations", []))
        events = list(state.get("events", []))
        tool_call_count = state.get("tool_call_count", 0)
        observation_by_step = {item["step_id"]: item for item in observations}
        pending = list(state.get("pending_steps", []))
        completed_plan_steps = list(state.get("plan", []))

        for step in pending:
            if self._work_seconds(state["started_at"]) > MAX_RUN_SECONDS:
                events = self._event(state | {"events": events}, "budget_exhausted", budget="time")
                break
            failed_dependencies = [
                dependency
                for dependency in step.get("depends_on", [])
                if dependency not in observation_by_step
                or observation_by_step[dependency]["execution_status"] != "ok"
            ]
            if failed_dependencies:
                observation = {
                    "observation_id": f"obs_{uuid.uuid4().hex[:16]}",
                    "evidence_id": f"ev_{uuid.uuid4().hex[:16]}",
                    "step_id": step["step_id"],
                    "tool_name": step["tool_name"],
                    "arguments": step["arguments"],
                    "execution_status": "skipped_dependency_failed",
                    "result": None,
                    "elapsed_ms": 0,
                    "error": f"依赖步骤失败: {failed_dependencies}",
                }
                observations.append(observation)
                observation_by_step[step["step_id"]] = observation
                continue
            if tool_call_count >= MAX_TOOL_CALLS:
                events = self._event(state | {"events": events}, "budget_exhausted", budget="tool_calls")
                break
            if _step_references(step["arguments"]):
                try:
                    bound = self._bind_step_arguments(state, step, observation_by_step)
                except (ValidationError, ValueError, TimeoutError, RuntimeError) as error:
                    observation = {
                        "observation_id": f"obs_{uuid.uuid4().hex[:16]}",
                        "evidence_id": f"ev_{uuid.uuid4().hex[:16]}",
                        "step_id": step["step_id"],
                        "tool_name": step["tool_name"],
                        "arguments": step["arguments"],
                        "execution_status": "error",
                        "result": None,
                        "elapsed_ms": 0,
                        "error": f"ARGUMENT_BINDING_FAILED: {str(error)[:300]}",
                    }
                    observations.append(observation)
                    observation_by_step[step["step_id"]] = observation
                    continue
                events.append(
                    {
                        "event_id": f"evt_{uuid.uuid4().hex[:16]}",
                        "type": "argument_binding",
                        "step_id": step["step_id"],
                        "planned_arguments": step["arguments"],
                        "arguments": bound,
                    }
                )
                step = {**step, "planned_arguments": step["arguments"], "arguments": bound}
                completed_plan_steps = [
                    step if item.get("step_id") == step["step_id"] else item for item in completed_plan_steps
                ]
            tool_call_count += 1
            started = time.monotonic()
            events.append(
                {
                    "event_id": f"evt_{uuid.uuid4().hex[:16]}",
                    "type": "tool_call",
                    "step_id": step["step_id"],
                    "tool_name": step["tool_name"],
                    "arguments": step["arguments"],
                }
            )
            try:
                result = self.tool_call(step["tool_name"], step["arguments"])
                observation = {
                    "observation_id": f"obs_{uuid.uuid4().hex[:16]}",
                    "evidence_id": f"ev_{uuid.uuid4().hex[:16]}",
                    "step_id": step["step_id"],
                    "tool_name": step["tool_name"],
                    "arguments": step["arguments"],
                    "execution_status": "ok",
                    "result": result,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                    "error": None,
                }
                observation_by_step[step["step_id"]] = observation
                observations.append(observation)
                events.append(
                    {
                        "event_id": f"evt_{uuid.uuid4().hex[:16]}",
                        "type": "tool_result",
                        "step_id": step["step_id"],
                        "evidence_id": observation["evidence_id"],
                        "result": result,
                        "elapsed_ms": observation["elapsed_ms"],
                    }
                )
            except Exception as error:  # tool errors are preserved in the trace without secret material
                observation = {
                    "observation_id": f"obs_{uuid.uuid4().hex[:16]}",
                    "evidence_id": f"ev_{uuid.uuid4().hex[:16]}",
                    "step_id": step["step_id"],
                    "tool_name": step["tool_name"],
                    "arguments": step["arguments"],
                    "execution_status": "error",
                    "result": None,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                    "error": f"{type(error).__name__}: {error}",
                }
                observation_by_step[step["step_id"]] = observation
                observations.append(observation)
                events.append(
                    {
                        "event_id": f"evt_{uuid.uuid4().hex[:16]}",
                        "type": "tool_result",
                        "step_id": step["step_id"],
                        "execution_status": "error",
                        "error": observation["error"],
                    }
                )

        return {
            "observations": observations,
            "pending_steps": [],
            "tool_call_count": tool_call_count,
            "events": events,
            "plan": completed_plan_steps,
        }

    def _bind_step_arguments(
        self, state: AgentState, step: dict[str, Any], observation_by_step: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Replace ``$ref`` arguments with literal values read from the referenced steps' results."""
        sources = [observation_by_step[step_id] for step_id in sorted(_step_references(step["arguments"]))]
        planned = {key: value for key, value in step["arguments"].items() if key != "as_of"}
        prompt = (
            "把计划步骤里形如 $ref:<step_id>:<要取的量> 的参数，按所引用步骤的结果换成具体的字面值；"
            "其余参数原样保留。只能使用下面给出的结果，不要猜测；结果里找不到所需的值时不要编造，"
            "在 arguments 里保留原来的 $ref 字符串。参数必须符合该工具的 input_schema。"
            "日期参数取自某个月分组时，起止日期用该分组的 first_date / last_date（只含查询区间内的天）。\n"
            f"question={state['question']}\n{self._cutoff_note(state)}\n"
            f"tool={step['tool_name']}\n"
            f"input_schema={json.dumps(TOOL_MODELS[step['tool_name']].model_json_schema(), ensure_ascii=False)}\n"
            f"planned_arguments={json.dumps(planned, ensure_ascii=False)}\nrationale={step.get('rationale', '')}\n"
            f"{PROMPT_COMPACTION_NOTE}\n"
            f"referenced_results={json.dumps(_prompt_observations(sources), ensure_ascii=False)}\n"
            "返回 JSON: {\"arguments\":{...}}"
        )
        draft = self._json("BoundArguments", BoundArguments, prompt)
        arguments = {key: value for key, value in draft.arguments.items() if key != "as_of"}
        if _contains_reference_marker(arguments):
            raise ValueError("引用的步骤结果里找不到所需的值")
        return {**arguments, "as_of": state["as_of"]}

    def _reflector(self, state: AgentState) -> dict[str, Any]:
        prompt = (
            "检查数据时效、覆盖、预测条件、工具失败和政策支持。区分能通过不同参数/补充工具修复的问题，"
            "与数据缺失、版本未知等不可修复问题。不要要求重复成功且参数相同的步骤。"
            "repairable=true 仅当存在一个尚未执行过的具体工具调用（不同工具或不同参数）能解决该问题；"
            "数据口径、时间对齐未验证、气象仅济南、政策未核验、提示压缩等固有限制一律 repairable=false。"
            "kind 区分问题性质：kind=limitation 指数据范围或方法上的固有限制（如区间语义未确认、"
            "气象仅济南单点、政策未核验、截止时点后数据不可得、提示压缩），在报告中注明后不影响回答问题；"
            "kind=gap 指会让某个结论缺少支撑或可能出错的问题（如工具失败、回答问题必需的数据缺失、"
            "证据相互矛盾、结果与问题时段不符）。拿不准时用 gap。repairable=true 的问题必须是 gap。"
            "explain_drivers 的 event_type_coverage 由系统单独检查，不要为代表事件抽样或事件覆盖另报问题。"
            "sufficient_evidence 表示现有证据能否在注明限制的前提下回答问题。\n"
            f"question={state['question']}\nas_of={state['as_of']}\nmode={state['mode']}\n"
            f"{self._scope_note(state)}"
            f"{PROMPT_COMPACTION_NOTE}\n"
            f"observations={json.dumps(_prompt_observations(state.get('observations', [])), ensure_ascii=False)}\n"
            f"prior_issues={json.dumps(state.get('unresolved_issues', []), ensure_ascii=False)}\n"
            f"prior_limitations={json.dumps(state.get('known_limitations', []), ensure_ascii=False)}\n"
            "返回 JSON: {\"sufficient_evidence\":false,\"issues\":[{\"code\":...,"
            "\"description\":...,\"repairable\":false,\"kind\":\"gap\",\"related_step_ids\":[]}]}"
        )
        try:
            draft = self._json("ReflectionDraft", ReflectionDraft, prompt)
        except (ValidationError, ValueError):
            # Fail closed without ending the run: no model-reported issues, evidence not confirmed sufficient.
            draft = ReflectionDraft(sufficient_evidence=False)
        issues = [issue.model_dump(mode="json") for issue in draft.issues]
        deterministic = []
        if state.get("plan_error") and not state.get("plan"):
            deterministic.append(
                {
                    "code": "PLAN_INVALID",
                    "description": state["plan_error"],
                    "repairable": True,  # the replanner can write the plan afresh
                    "kind": "gap",
                    "related_step_ids": [],
                }
            )
        for observation in state.get("observations", []):
            if observation["execution_status"] != "ok":
                error_text = observation.get("error") or "tool execution failed"
                deterministic.append(
                    {
                        "code": "TOOL_EXECUTION_FAILED",
                        "description": error_text,
                        # Bad arguments (schema or as_of cutoff) can be fixed by a replanned call.
                        "repairable": observation["execution_status"] == "error"
                        and error_text.startswith(("ValidationError", "ARGUMENT_BINDING_FAILED")),
                        "kind": "gap",
                        "related_step_ids": [observation["step_id"]],
                    }
                )
            result = observation.get("result") or {}
            coverage = (result.get("data") or {}).get("event_type_coverage")
            if result.get("tool") == "explain_drivers" and coverage:
                uncovered = coverage.get("uncovered_price_event_types") or []
                data = result["data"]
                if uncovered:
                    deterministic.append(
                        {
                            "code": "DRIVER_EVENT_TYPES_UNCOVERED",
                            "description": f"以下价格事件类型没有驱动解释：{uncovered}；"
                            "可用 detect_anomaly 结果中这些类型的 event_id 通过 event_ids 补查。",
                            "repairable": True,
                            "kind": "gap",
                            "related_step_ids": [observation["step_id"]],
                        }
                    )
                elif data.get("event_count", 0) < data.get("detected_event_count", 0):
                    deterministic.append(
                        {
                            "code": "DRIVER_EVENTS_SAMPLED",
                            "description": "驱动解释按事件类型抽取代表事件，已覆盖全部价格事件类型；"
                            "同类其余事件未逐一归因。",
                            "repairable": False,
                            "kind": "limitation",
                            "related_step_ids": [observation["step_id"]],
                        }
                    )
            if result.get("tool") == "search_policy" and result.get("status") != "ok":
                deterministic.append(
                    {
                        "code": "POLICY_EVIDENCE_UNVERIFIED",
                        "description": "本地政策候选未完成版本和适用期核验，不能支持政策结论。",
                        "repairable": False,
                        # Fails closed elsewhere: the report validator rejects any policy_basis citing it.
                        "kind": "limitation",
                        "related_step_ids": [observation["step_id"]],
                    }
                )
        known = {(item["code"], tuple(item["related_step_ids"])) for item in issues}
        for item in deterministic:
            key = (item["code"], tuple(item["related_step_ids"]))
            if key not in known:
                issues.append(item)
                known.add(key)
        for issue in issues:
            if issue["repairable"]:
                issue["kind"] = "gap"
        gaps = [issue for issue in issues if issue["kind"] == "gap"]
        limitations = [issue for issue in issues if issue["kind"] == "limitation"]
        repairable = any(issue["repairable"] for issue in issues)
        reflection = {
            "sufficient_evidence": draft.sufficient_evidence and not gaps,
            "issues": issues,
            "repairable": repairable,
        }
        return {
            "reflection": reflection,
            "unresolved_issues": [issue["description"] for issue in gaps],
            "known_limitations": [issue["description"] for issue in limitations],
            "events": self._event(state, "reflection", reflection=reflection),
        }

    def _replanner(self, state: AgentState) -> dict[str, Any]:
        existing_ids = {step["step_id"] for step in state.get("plan", [])}
        prompt = (
            "补充或修正步骤以处理可修复问题。最多 6 个新增步骤；禁止重复已成功的相同工具与参数，"
            "依赖只能指向已完成步骤或本次较早步骤。"
            f"{PLAN_REFERENCE_NOTE}\n"
            f"question={state['question']}\nas_of={state['as_of']}\nmode={state['mode']}\n"
            f"{self._scope_note(state)}"
            f"{self._cutoff_note(state)}\n"
            f"issues={json.dumps(state['reflection']['issues'], ensure_ascii=False)}\n"
            f"{PROMPT_COMPACTION_NOTE}\n"
            f"completed_observations={json.dumps(_prompt_observations(state.get('observations', [])), ensure_ascii=False)}\n"
            f"existing_step_ids={sorted(existing_ids)}\ntools={json.dumps(self._tool_catalog(), ensure_ascii=False)}\n"
            "返回 PlanDraft JSON。若无合法补查步骤，返回一个参数合法且不重复的最小确认步骤，或说明无法修复。"
        )
        try:
            steps = self._json(
                "PlanDraft", PlanDraft, prompt,
                check=lambda draft: self._checked_plan(draft, existing_ids, state["as_of"]),
            )
        except (ValidationError, ValueError) as error:
            issues = state["unresolved_issues"] + [f"重规划失败: {error}"]
            return {
                "pending_steps": [],
                "unresolved_issues": issues,
                "replan_count": state.get("replan_count", 0) + 1,
            }

        completed_calls = {
            (item["tool_name"], json.dumps(item["arguments"], sort_keys=True, ensure_ascii=False))
            for item in state.get("observations", [])
            if item["execution_status"] == "ok"
        }
        unique_steps = [
            step
            for step in steps
            if (
                step.tool_name,
                json.dumps(step.arguments, sort_keys=True, ensure_ascii=False),
            )
            not in completed_calls
        ]
        if not unique_steps:
            issues = state["unresolved_issues"] + ["重规划未找到新的、合法的补查步骤。"]
        else:
            issues = list(state["unresolved_issues"])
        new_steps = [step.model_dump(mode="json") for step in unique_steps]
        plan = list(state.get("plan", [])) + new_steps
        return {
            "plan": plan,
            "pending_steps": new_steps,
            "replan_count": state.get("replan_count", 0) + 1,
            "unresolved_issues": issues,
            "events": self._event(
                state, "replan", replan_count=state.get("replan_count", 0) + 1, steps=new_steps
            ),
        }

    def _reporter(self, state: AgentState) -> dict[str, Any]:
        revision = ""
        if state.get("report") and state.get("report_errors"):
            # Edit the failed draft rather than regenerating it, so fixes accumulate across revisions.
            revision = (
                "上一版报告未通过校验。在 previous_report 基础上只修正下列错误，其余内容保持不变：\n"
                f"validation_errors={json.dumps(state['report_errors'], ensure_ascii=False)}\n"
                f"previous_report={json.dumps(state['report'], ensure_ascii=False)}\n\n"
            )
        prompt = (
            f"{revision}"
            "根据 observations 起草中文 MarketReport。先写 answers：它是对问题的直接回答，"
            "scope 里的每一项对应一条，item_id 照抄 scope。"
            "status=answered：statement 直接写出这一项要的数值、日期或时刻，evidence_ids 写所依据的 observation；"
            "scope 里 answer_type=number 的项必须给出带 value_binding 的数值（包括 0、比例、计数），"
            "date_or_time 的项必须写出具体日期或时刻。"
            "scope 判为 ambiguous 的项按 note 的读法作答，并在 basis 写明采用的日期区间、时段或口径；"
            "读法不唯一、数据口径有固有限制都不是不回答的理由，给出数值并说明口径。"
            "status=unanswerable：scope 判为 unanswerable 的项必须用它，statement 写明现有数据缺什么、因此不能回答，"
            "不带任何数值，也不用其他地区、市场、日期或指标的数据顶替；"
            "scope 判为可答、但取数失败或数据确实为空的项也用它，statement 写明原因。"
            "问题没有涉及异常和成因时 anomalies、causes 留空，答案只写在 answers 里。\n"
            "summary 不得出现任何阿拉伯数字（日期、时刻除外），"
            "数量用“若干/多个”等表述，只写有 evidence_id 支持的概述并列出 summary_evidence_ids；"
            "数值只能放在 answers/anomalies/causes/outlook 中且必须附 value_binding，"
            "statement 中的数值一律用阿拉伯数字书写，与对应 value_binding 的值一致，不要写成汉字数字；"
            "statement 中出现的每个数值都必须有 value_binding。"
            "forecast_price 的模型最低/最高价用 /data/model_summary 中的字段，不要用列表首行代替极值。"
            "描述供需分项“实际高于/低于预测”时以该分项 actual_vs_forecast 为准，"
            "“正/负贡献”以 bid_space_effect 为准，不要从带符号的贡献值自行推断方向。"
            "使用 observation result 的 JSON Pointer。policy_basis 只能引用 search_policy 结果中 "
            "evidence_eligible=true 的候选，json_pointer 写该候选位置（如 /data/evidence/0；"
            "explain_drivers 结果内嵌的候选写完整路径，如 /data/policy_evidence/data/evidence/0），statement 写明文件名、"
            "条款定位和生效区间；没有合格候选时 policy_basis 必须为空，"
            "并在 known_limitations 说明。known_limitations 放已注明、不影响回答问题的固有限制"
            "（如输入中的 known_limitations）；unresolved_issues 只放会让结论缺少支撑或可能出错的问题"
            "（如输入中的 unresolved_issues），不要把固有限制写进 unresolved_issues。不要提供投资建议。"
            "value_binding 的 json_pointer 指向完整 result：普通字段按其在 result 中的路径书写；"
            "压缩列表的条目用该条目 _pointer 加字段名（如 _pointer=/data/rows/0 → /data/rows/0/rt_price），"
            "或直接使用 numeric_ranges 中的 min_pointer/max_pointer。"
            "value_binding 的 evidence_id 必须是实际包含该字段的那条 observation：不同工具的同名列表结构不同，"
            "例如 explain_drivers 的事件详情在 /data/events/i/event/ 下，detect_anomaly 的事件字段直接在 /data/events/i/ 下。"
            "固有限制的字段名必须写 known_limitations，不要写成 limitations 或其他名字；"
            "顶层只能出现 schema 中列出的字段。"
            "所有 evidence_id（summary_evidence_ids、evidence_ids、value_binding、policy_basis）一律写 observation 的 "
            "evidence_id，不要写 search_policy 候选自带的 pol_ 编号。"
            "statement 中由证据值计算得到的数（合计、差值、变化量、占比/比值）不能放进 value_bindings，"
            "要写入该发现的 derived_values：operation 取 sum/difference/ratio，operands 列出参与计算的 value_binding"
            "（difference 为第一个减去其余；ratio 的最后一个 operand 是分母，其余 operands 之和是分子，"
            "如多段负价事件的 points 依次列出、最后列 total_rows，value 写小数比值而非百分数），"
            "value 写计算结果；校验会用 operands 的证据值复算。\n"
            f"{PROMPT_COMPACTION_NOTE}\n"
            f"question={state['question']}\nas_of={state['as_of']}\nmode={state['mode']}\n"
            f"scope={json.dumps(state.get('scope', []), ensure_ascii=False)}\n"
            f"observations={json.dumps(_prompt_observations(state.get('observations', [])), ensure_ascii=False)}\n"
            f"unresolved_issues={json.dumps(state.get('unresolved_issues', []), ensure_ascii=False)}\n"
            f"known_limitations={json.dumps(state.get('known_limitations', []), ensure_ascii=False)}"
        )
        try:
            draft = self._json("MarketReportDraft", MarketReportDraft, prompt)
        except (RuntimeError, TimeoutError) as error:
            if not state.get("report"):
                raise
            # A revision that runs out of budget keeps the previous draft; validation then fails closed.
            return {
                "report_revision_count": MAX_REPORT_REVISIONS,
                "unresolved_issues": state.get("unresolved_issues", [])
                + [f"报告修订未完成（{error}），保留上一版草稿，校验错误见 validation"],
                "events": self._event(state, "report_revision_skipped", reason=str(error)),
            }
        observations = state.get("observations", [])
        report = _canonicalize_bindings(
            _canonicalize_policy_ids(draft.model_dump(mode="json"), observations), observations
        )
        return {
            "report": report,
            "report_revision_count": state.get("report_revision_count", 0) + 1,
            "events": self._event(
                state,
                "report",
                revision=state.get("report_revision_count", 0) + 1,
                report=report,
            ),
        }

    @staticmethod
    def _answer_errors(report: dict[str, Any], scope: list[dict[str, Any]]) -> list[str]:
        """The report must answer the question item by item: one direct answer per scope item, of the kind
        the item asks for, and no answer at all (rather than substitute data) for what the data lacks."""
        errors: list[str] = []
        known = {item["item_id"]: item for item in scope}
        by_id: dict[str, dict[str, Any]] = {}
        for answer in report.get("answers", []):
            item_id = answer.get("item_id")
            if item_id not in known:
                errors.append(f"answers[{item_id}] 的 item_id 不在 scope 中；只能是 {sorted(known)}")
            elif item_id in by_id:
                errors.append(f"answers[{item_id}] 重复；每个 scope 项只写一条")
            else:
                by_id[item_id] = answer
        for item in scope:
            label = f"answers[{item['item_id']}]（{item['item']}）"
            excluded = item["status"] == "unanswerable"
            answer = by_id.get(item["item_id"])
            if answer is None:
                errors.append(
                    f"缺少 {label}：scope 的每一项都要在 answers 里有一条直接回答；"
                    + ("这一项 status 写 unanswerable 并说明现有数据缺什么" if excluded else
                       "写出这一项要的数值、日期或时刻并绑定证据，确实取不到数据时 status 写 unanswerable 并说明原因")
                )
                continue
            has_number = bool(answer.get("derived_values")) or any(
                _is_number(binding.get("value")) for binding in answer.get("value_bindings", [])
            )
            if excluded:
                if answer.get("status") != "unanswerable" or answer.get("value_bindings") or answer.get("derived_values"):
                    errors.append(
                        f"{label} 在取数前已判定现有数据不包含（{item.get('note', '')}）："
                        "status 写 unanswerable，statement 说明缺什么，不带数值绑定，不用其他数据顶替"
                    )
                continue
            if answer.get("status") == "unanswerable":
                continue  # the data was expected but a tool failed or returned nothing; the statement says so
            if not answer.get("evidence_ids"):
                errors.append(f"{label} status=answered 时 evidence_ids 要列出所依据的 observation")
            statement = answer.get("statement", "")
            if item.get("answer_type") == "number" and not has_number:
                errors.append(
                    f"{label} 要的是数值：statement 写出该数值并附 value_binding（计算得到的数写 derived_values）"
                )
            if item.get("answer_type") == "date_or_time" and not (
                _DATE_TIME_PATTERN.search(statement) or _STATEMENT_SLOT_PATTERN.search(statement)
            ):
                errors.append(f"{label} 要的是日期或时刻：statement 写出具体的日期或时刻")
            if item["status"] == "ambiguous" and not answer.get("basis", "").strip():
                errors.append(
                    f"{label} 的读法不唯一：在 basis 写明作答采用的日期区间、时段或口径（scope 的读法：{item.get('note', '')}）"
                )
        return errors

    @staticmethod
    def _validate_report(
        report: dict[str, Any], observations: list[dict[str, Any]], scope: list[dict[str, Any]] | None = None
    ) -> list[str]:
        errors: list[str] = []
        by_evidence = {item["evidence_id"]: item for item in observations}
        results = {item["evidence_id"]: item.get("result") for item in observations}
        for evidence_id in report.get("summary_evidence_ids", []):
            if evidence_id not in by_evidence:
                errors.append(f"summary 引用了未知 evidence_id {evidence_id}")
        if _has_numeric_claim(report.get("summary", "")):
            errors.append("summary 不得包含未绑定的数值陈述；将数值移到 answers 或带 value_binding 的发现")
        findings = _report_findings(report)
        answer_count = len(report.get("answers", []))
        labels = [f"answers[{answer.get('item_id')}]" for answer in report.get("answers", [])] + [
            f"finding[{index}]" for index in range(len(findings) - answer_count)
        ]
        if scope:
            errors.extend(MarketAnalysisAgent._answer_errors(report, scope))
        elif (
            not findings
            and not report.get("policy_basis")
            and not report.get("unresolved_issues")
            and any(
                item.get("execution_status") == "ok" and (item.get("result") or {}).get("status") == "ok"
                for item in observations
            )
        ):
            # Data came back but the report states nothing: the answer belongs in findings, not only in summary.
            errors.append(
                "报告没有任何发现：问题要求的数值、日期或时段要写成带 value_binding 的发现（summary 不能带数值）；"
                "确实无法回答的部分写入 unresolved_issues 并说明原因"
            )
        # Values a statement number may round to, per finding: bound evidence values plus verified derived values.
        bound_by_finding: list[list[float]] = []
        for finding_index, finding in enumerate(findings):
            for evidence_id in finding.get("evidence_ids", []):
                if evidence_id not in by_evidence:
                    errors.append(f"{labels[finding_index]} 引用了未知 evidence_id {evidence_id}")
            for binding in _finding_bindings(finding):
                observation = by_evidence.get(binding.get("evidence_id"))
                where = f"{binding.get('evidence_id')} {binding.get('json_pointer')}"
                if observation is None:
                    errors.append(f"{labels[finding_index]} 数值绑定引用了未知 evidence_id {where}")
                    continue
                try:
                    actual = _resolve_binding_pointer(observation.get("result"), binding["json_pointer"])
                except (KeyError, IndexError, TypeError, ValueError) as error:
                    errors.append(
                        f"{labels[finding_index]} JSON Pointer 无效 ({where}): 找不到 {error}；"
                        "请使用压缩列表条目的 _pointer 加字段名、numeric_ranges 的 min_pointer/max_pointer，"
                        "或 <列表 pointer>/numeric_ranges/<字段>/count"
                    )
                    continue
                if not _same_value(actual, binding.get("value")):
                    errors.append(
                        f"{labels[finding_index]} 数值与 evidence 不一致 ({where}): "
                        f"绑定值 {binding.get('value')!r}，实际值 {actual!r}"
                    )
            bound = [
                float(binding["value"]) for binding in finding.get("value_bindings", []) if _is_number(binding.get("value"))
            ]
            for derived_index, derived in enumerate(finding.get("derived_values", [])):
                where = f"{labels[finding_index]} derived_values[{derived_index}]"
                operands = [operand.get("value") for operand in derived.get("operands", [])]
                if not all(_is_number(value) for value in operands):
                    errors.append(f"{where} 的 operands 必须都是数值")
                    continue
                computed = _derived_result(derived.get("operation", ""), [float(value) for value in operands])
                if computed is None:
                    errors.append(f"{where} 无法计算：ratio 需要至少两个 operands 且分母（最后一个）不为 0")
                    continue
                claimed = derived.get("value")
                tolerance = 0.5 * 10 ** (-_value_decimals(claimed)) + 1e-9 if _is_number(claimed) else 0
                if not _is_number(claimed) or abs(float(claimed) - computed) > tolerance:
                    errors.append(f"{where} 计算结果不一致：写为 {claimed!r}，按 operands 复算为 {computed:.6g}")
                    continue
                bound.extend([computed, float(claimed)])
            bound.extend(_list_facts(finding, results)[0])
            bound_by_finding.append(bound)
        for citation_index, citation in enumerate(report.get("policy_basis", [])):
            observation = by_evidence.get(citation.get("evidence_id"))
            # The candidate may sit in a search_policy observation or in one embedded by another tool.
            candidate = _policy_candidate((observation or {}).get("result"), citation.get("json_pointer", ""))
            if candidate is None or candidate.get("evidence_eligible") is not True:
                errors.append(
                    f"policy_basis[{citation_index}] 引用了未核验、版本不适用或不存在的政策证据 "
                    f"({citation.get('evidence_id')} {citation.get('json_pointer')})"
                )
            elif not candidate.get("locator"):
                errors.append(f"policy_basis[{citation_index}] 引用的政策片段缺少原文定位")
        for finding_index, finding in enumerate(findings):
            if (
                _has_numeric_claim(finding.get("statement", ""))
                and not finding.get("value_bindings")
                and not finding.get("derived_values")
            ):
                errors.append(f"{labels[finding_index]} 含数字陈述但没有结构化数值绑定")
                continue
            bound = bound_by_finding[finding_index]
            positions = _list_facts(finding, results)[1]
            # "第 N 段/号时段" naming an item the finding cites is a position, not a claimed value.
            statement = _ORDINAL_PATTERN.sub(
                lambda match: " " if int(match.group(1)) in positions else match.group(0), finding.get("statement", "")
            )
            unbound = [number for number in _statement_numbers(statement) if not _number_is_bound(number, bound)]
            if unbound:
                errors.append(
                    f"{labels[finding_index]} 陈述中的数值 {unbound} 没有对应的 value_binding；"
                    "为每个数值补绑定（计算得到的数写入 derived_values），或删去该数值"
                )
            errors.extend(
                f"{labels[finding_index]} {error}" for error in _direction_errors(finding, by_evidence)
            )
            errors.extend(f"{labels[finding_index]} {error}" for error in _row_mismatch_errors(finding, results))
        return errors

    def _report_validator(self, state: AgentState) -> dict[str, Any]:
        report = state.get("report") or {}
        errors = self._validate_report(report, state.get("observations", []), state.get("scope"))
        validation = {
            "status": "passed" if not errors else "failed",
            "checks": [
                "evidence_ids_resolve",
                "numeric_bindings_match_observations",
                "statement_numbers_are_bound",
                "every_scope_item_has_a_direct_answer",
                "bound_rows_match_statement_day_and_time",
                "component_direction_matches_evidence",
                "policy_citations_are_eligible",
            ],
            "errors": errors,
        }
        events = self._event(state, "report_validation", validation=validation)
        return {
            "validation": validation,
            "report_errors": errors,
            "events": events,
        }

    def _after_reflection(self, state: AgentState) -> str:
        reflection = state.get("reflection", {})
        within_budget = (
            getattr(self, "_last_token_count", 0) < MAX_TOTAL_TOKENS * REPLAN_BUDGET_FRACTION
            and self._work_seconds(state["started_at"]) < MAX_RUN_SECONDS * REPLAN_BUDGET_FRACTION
        )
        if (
            reflection.get("repairable")
            and within_budget
            and state.get("replan_count", 0) < MAX_REPLANS
            and state.get("tool_call_count", 0) < MAX_TOOL_CALLS
        ):
            return "replanner"
        return "reporter"

    @staticmethod
    def _after_replan(state: AgentState) -> str:
        # Without new steps another reflection would only re-read the same observations.
        return "executor" if state.get("pending_steps") else "reporter"

    @staticmethod
    def _after_validation(state: AgentState) -> str:
        if (
            state.get("validation", {}).get("status") == "failed"
            and state.get("report_revision_count", 0) < MAX_REPORT_REVISIONS
        ):
            return "reporter"
        return "finish"

    def _react(self, state: AgentState) -> dict[str, Any]:
        """E0: decide the next single tool call from the observations so far, or finish."""
        within_budget = (
            state.get("tool_call_count", 0) < MAX_TOOL_CALLS
            and self._work_seconds(state["started_at"]) < MAX_RUN_SECONDS * REPLAN_BUDGET_FRACTION
            and getattr(self, "_last_token_count", 0) < MAX_TOTAL_TOKENS * REPLAN_BUDGET_FRACTION
        )
        if not within_budget:
            return {"pending_steps": [], "events": self._event(state, "react_budget_stop")}
        history = [
            {
                "step_id": item["step_id"],
                "tool_name": item["tool_name"],
                "arguments": {k: v for k, v in item["arguments"].items() if k != "as_of"},
                "execution_status": item["execution_status"],
                "error": item.get("error"),
            }
            for item in state.get("observations", [])
        ]
        prompt = (
            "你是使用工具回答电力市场问题的助手。每一轮只决定下一步：调用一个工具，或在信息足够时结束。"
            "参数必须符合工具 schema，只能是字面值。\n"
            f"question={state['question']}\nas_of={state['as_of']}\nmode={state['mode']}\n"
            f"{self._scope_note(state)}"
            f"{self._cutoff_note(state)}\n"
            f"tools={json.dumps(self._tool_catalog(), ensure_ascii=False)}\n"
            f"calls_so_far={json.dumps(history, ensure_ascii=False)}\n"
            f"{PROMPT_COMPACTION_NOTE}\n"
            f"observations={json.dumps(_prompt_observations(state.get('observations', [])), ensure_ascii=False)}\n"
            "返回 JSON: {\"thought\":\"...\",\"action\":\"call_tool\",\"tool_name\":...,\"arguments\":{}} "
            "或 {\"thought\":\"...\",\"action\":\"finish\"}"
        )
        try:
            draft = self._json("ReactDraft", ReactDraft, prompt)
        except (ValidationError, ValueError) as error:
            # An unusable turn ends tool use; the report is written from what has been observed so far.
            return {"pending_steps": [], "events": self._event(state, "react_finish", error=str(error)[:300])}
        if draft.action == "finish" or not draft.tool_name:
            return {"pending_steps": [], "events": self._event(state, "react_finish", thought=draft.thought)}
        step = {
            "step_id": f"r{len(state.get('plan', [])) + 1}",
            "tool_name": draft.tool_name,
            "arguments": {**draft.arguments, "as_of": state["as_of"]},
            "depends_on": [],
            "rationale": draft.thought[:500] or "react step",
        }
        return {
            "plan": list(state.get("plan", [])) + [step],
            "pending_steps": [step],
            "events": self._event(state, "react_step", step=step),
        }

    @staticmethod
    def _after_react(state: AgentState) -> str:
        return "executor" if state.get("pending_steps") else "reporter"

    def _after_executor(self, _state: AgentState) -> str:
        return {"E0": "react", "E1": "reporter"}.get(self.variant, "reflector")

    def _graph(self):
        if self._compiled_graph is None:
            from langgraph.graph import END, START, StateGraph

            graph = StateGraph(AgentState)
            graph.add_node("executor", self._executor)
            graph.add_node("reporter", self._reporter)
            graph.add_node("report_validator", self._report_validator)
            graph.add_node("scoper", self._scoper)
            graph.add_edge(START, "scoper")
            first = "react" if self.variant == "E0" else "planner"
            graph.add_conditional_edges("scoper", self._after_scope, {first: first, "reporter": "reporter"})
            if self.variant == "E0":
                graph.add_node("react", self._react)
                graph.add_conditional_edges(
                    "react", self._after_react, {"executor": "executor", "reporter": "reporter"}
                )
                graph.add_conditional_edges("executor", self._after_executor, {"react": "react"})
            else:
                graph.add_node("planner", self._planner)
                graph.add_edge("planner", "executor")
            if self.variant == "E1":
                graph.add_conditional_edges("executor", self._after_executor, {"reporter": "reporter"})
            if self.variant in {"E2", "E3"}:
                graph.add_node("reflector", self._reflector)
                graph.add_node("replanner", self._replanner)
                graph.add_conditional_edges("executor", self._after_executor, {"reflector": "reflector"})
                graph.add_conditional_edges(
                    "reflector",
                    self._after_reflection,
                    {"replanner": "replanner", "reporter": "reporter"},
                )
                graph.add_conditional_edges(
                    "replanner",
                    self._after_replan,
                    {"executor": "executor", "reporter": "reporter"},
                )
            graph.add_edge("reporter", "report_validator")
            graph.add_conditional_edges(
                "report_validator",
                self._after_validation,
                {"reporter": "reporter", "finish": END},
            )
            self._compiled_graph = graph.compile()
        return self._compiled_graph

    def analyze(
        self,
        *,
        question: str,
        as_of: datetime,
        mode: Literal["historical_analysis", "historical_forecast"],
    ) -> dict[str, Any]:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of 必须包含时区偏移量")
        if not question.strip():
            raise ValueError("question 不能为空")
        self._last_token_count = 0
        self.schema_retry_count = 0
        self.reasoning_token_count = 0
        self.rate_limit_wait_seconds = 0.0
        self.rate_limit_retry_count = 0
        self.bad_request_retry_count = 0
        self.non_json_output_count = 0
        initial = self._initial_state(question, as_of, mode)
        result = self._graph().invoke(initial, config={"recursion_limit": 40})
        return self._finalize(result, initial)

    def _initial_state(
        self,
        question: str,
        as_of: datetime,
        mode: Literal["historical_analysis", "historical_forecast"],
    ) -> AgentState:
        self._run_started_at = time.monotonic()
        return {
            "run_id": f"run_{uuid.uuid4().hex}",
            "question": question.strip(),
            "as_of": as_of.isoformat(),
            "mode": mode,
            "started_at": time.monotonic(),
            "plan": [],
            "pending_steps": [],
            "observations": [],
            "unresolved_issues": [],
            "known_limitations": [],
            "replan_count": 0,
            "report_revision_count": 0,
            "report_errors": [],
            "events": [],
            "tool_call_count": 0,
        }

    def _finalize(self, result: AgentState, initial: AgentState) -> dict[str, Any]:
        report = dict(result.get("report") or {})
        validation = result.get("validation", {"status": "not_run", "errors": []})
        unresolved = list(result.get("unresolved_issues", []))
        unresolved.extend(report.get("unresolved_issues", []))
        report["unresolved_issues"] = list(dict.fromkeys(unresolved))
        # Inherent limits are disclosed but do not block verification; the report's own
        # unresolved_issues still count, so an LLM cannot downgrade a gap by omission.
        limitations = list(result.get("known_limitations", []))
        limitations.extend(report.get("known_limitations", []))
        report["known_limitations"] = [
            item for item in dict.fromkeys(limitations) if item not in report["unresolved_issues"]
        ]
        report["verified"] = (
            result.get("reflection", {}).get("sufficient_evidence", False)
            and validation.get("status") == "passed"
            and not report["unresolved_issues"]
        )
        report["validation"] = validation
        return {
            "run_id": result["run_id"],
            "report": report,
            "trace": {
                "variant": self.variant,
                "scope": result.get("scope", []),
                "plan": result.get("plan", []),
                "observations": result.get("observations", []),
                "reflection": result.get("reflection", {}),
                "events": result.get("events", []),
                "replan_count": result.get("replan_count", 0),
                "report_revision_count": result.get("report_revision_count", 0),
                "tool_call_count": result.get("tool_call_count", 0),
                "token_count": self._last_token_count,
                "schema_retry_count": getattr(self, "schema_retry_count", 0),
                "rate_limit_wait_seconds": round(getattr(self, "rate_limit_wait_seconds", 0.0), 2),
                "rate_limit_retry_count": getattr(self, "rate_limit_retry_count", 0),
                "bad_request_retry_count": getattr(self, "bad_request_retry_count", 0),
                "non_json_output_count": getattr(self, "non_json_output_count", 0),
                "reasoning_token_count": getattr(self, "reasoning_token_count", 0),
                "elapsed_ms": round((time.monotonic() - initial["started_at"]) * 1000, 2),
            },
        }

    def stream_analyze(
        self,
        *,
        question: str,
        as_of: datetime,
        mode: Literal["historical_analysis", "historical_forecast"],
    ):
        """Yield graph events as they occur and retain the completed result for trace storage."""
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of 必须包含时区偏移量")
        if not question.strip():
            raise ValueError("question 不能为空")
        self._last_token_count = 0
        self.schema_retry_count = 0
        self.reasoning_token_count = 0
        self.rate_limit_wait_seconds = 0.0
        self.rate_limit_retry_count = 0
        self.bad_request_retry_count = 0
        self.non_json_output_count = 0
        initial = self._initial_state(question, as_of, mode)
        yield {"type": "start", "run_id": initial["run_id"]}
        current: AgentState = initial
        emitted_events = 0
        for current in self._graph().stream(
            initial,
            config={"recursion_limit": 40},
            stream_mode="values",
        ):
            events = current.get("events", [])
            for event in events[emitted_events:]:
                yield event
            emitted_events = len(events)
        self.last_result = self._finalize(current, initial)
        yield {
            "type": "report",
            "run_id": self.last_result["run_id"],
            "report": self.last_result["report"],
        }
        yield {"type": "done", "run_id": self.last_result["run_id"]}
