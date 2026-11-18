"""Rewrite a follow-up into a standalone question, or decide that the system must ask back.

The model proposes the rewrite; code decides whether it may be used:

- every date or month the standalone question names must come from the current utterance, from earlier
  turns (questions or answers), or from a declared calendar shift of such a date ("前一天", "去年同期"), or
  be a year-less date in the utterance completed with a year seen in context; anything else is invented and
  the rewrite is sent back once with the error, then replaced by a request for the full date;
- a reference with two or more candidates turns the turn into a clarification, whatever action the model
  chose: the system never picks one of several plausible referents on the user's behalf.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ellectric.conversation.periods import mentioned_periods, partial_dates, shift, years_of
from ellectric.llm.market_analysis import PlanCheckError

MAX_CONTEXT_TURNS = 4
MAX_ANSWER_CHARS = 600


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase: str = Field(min_length=1, max_length=100)
    candidates: list[str] = Field(default_factory=list, max_length=6)
    # True when the phrase means all of its candidates together ("这两天", "两次", "它们"), not one of them.
    plural: bool = False
    resolved: str = Field(default="", max_length=200)


class Inherited(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: Literal["date", "month", "period", "metric", "time_slot", "region", "other"]
    value: str = Field(min_length=1, max_length=200)
    from_turn: int = Field(ge=1)
    derivation: Literal["same", "shift_days", "shift_months", "shift_years", "complete_year"] = "same"
    amount: int = 0
    # The date, month or range ("2025-07-07至2025-07-16") the shift starts from; a range shifts every endpoint.
    base: str = Field(default="", max_length=80)


class RewriteDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["proceed", "clarify"]
    standalone_question: str = Field(default="", max_length=1000)
    references: list[Reference] = Field(default_factory=list, max_length=6)
    inherited: list[Inherited] = Field(default_factory=list, max_length=10)
    clarification: str = Field(default="", max_length=300)


def context_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The last completed turns, reduced to what a rewrite may lean on."""
    out = []
    for turn in turns:
        if turn.get("type") != "question" or turn.get("status") != "completed":
            continue
        answer = " ".join(a.get("statement", "") for a in turn.get("answers", []))[:MAX_ANSWER_CHARS]
        out.append({"turn": turn["index"], "question": turn["standalone_question"], "answer": answer,
                    "periods": sorted(mentioned_periods(turn["standalone_question"]) | mentioned_periods(answer))})
    return out[-MAX_CONTEXT_TURNS:]


def _prompt(utterance: str, context: list[dict[str, Any]], as_of: str) -> str:
    lines = [f"第{c['turn']}轮\n  问题：{c['question']}\n  回答：{c['answer'] or '（无）'}" for c in context]
    return (
        "你是山东电力市场分析助手的对话改写器。把用户的最新一句话改写成一个不依赖上下文、能单独回答的完整问题。\n"
        f"当前时间点 as_of：{as_of}\n\n对话上下文（越往下越新）：\n" + "\n".join(lines) +
        f"\n\n用户最新一句话：{utterance}\n\n"
        "规则：\n"
        "1. 只用上下文中出现过的信息补全省略和指代（日期、月份、时段、指标、地区），不要添加用户没问的内容。\n"
        "2. 相对时间（前一天、下个月、去年同期）以被指代那一轮的日期为基准计算；在 inherited 里写 derivation"
        "（shift_days / shift_months / shift_years）、amount（如前一天为 -1，去年同期为 -1）和 base（基准日期，YYYY-MM-DD 或 YYYY-MM；"
        "日期范围写整个范围，如 2025-07-07至2025-07-16，两端按同一偏移推算）。\n"
        "3. 用户只说了月日没说年份（如“那7月31日呢”），用上下文的年份补全，derivation 写 complete_year。\n"
        "4. 最新一句话本身已经完整（有明确日期和要问的指标）时，是新话题：不继承任何东西，standalone_question 照原意完整复述，inherited 为空。\n"
        "5. 用户纠正上一轮（“不对”“说错了”）时，只替换被纠正的部分，其余沿用上一轮问题。\n"
        "6. 指代（那天、当天、这个时段、它）先结合用户原话里的限定词判断；仍对应多个候选时，把所有候选写进该指代的 candidates，"
        "action 设为 clarify，并在 clarification 里用一句话反问用户要哪一个。只有一个候选时 candidates 只写这一个。"
        "指代本身是复数、指的是全部候选（这两天、两次、它们）时，plural 设为 true，candidates 写全部所指对象，"
        "standalone_question 要把它们全部写出来，不需要反问。\n"
        "7. standalone_question 里的日期一律写完整年份，如“2025年5月23日”；沿用上一轮问题的说法和指标名称。\n"
        "8. inherited 列出每个从上下文继承或推算的值；from_turn 写来自第几轮。\n"
        "只输出 JSON。"
    )


def _account(draft: RewriteDraft, utterance: str, context: list[dict[str, Any]]) -> list[str]:
    """Errors for dates/months in the rewrite that come from neither the utterance, the context, nor a declared
    shift or year completion of them."""
    known = mentioned_periods(utterance) | {p for c in context for p in c["periods"]}
    allowed = set(known)
    errors = []
    # The shifted date is computed here from base + derivation + amount; the model's own arithmetic (its
    # ``value``) is not trusted, so a wrong date in the question is caught by the accounting below.
    for item in draft.inherited:
        if item.derivation in {"shift_days", "shift_months", "shift_years"}:
            bases = mentioned_periods(item.base) or {item.base}
            unknown = sorted(b for b in bases if b not in known)
            if unknown:
                errors.append(f"inherited {item.value}: 基准 {unknown or '（空）'} 不在用户原话或上下文中")
                continue
            shifted = {shift(b, item.derivation, item.amount) for b in bases}
            if None in shifted:
                errors.append(f"inherited {item.value}: 无法对 {item.base} 做 {item.derivation} {item.amount}")
                continue
            allowed |= shifted
    partial = partial_dates(utterance)
    years = years_of(known)
    for period in mentioned_periods(draft.standalone_question) - allowed:
        month, day = int(period[5:7]), (int(period[8:10]) if len(period) == 10 else None)
        if int(period[:4]) in years and (month, day) in partial:
            continue  # a year-less date in the utterance completed with a year from context
        errors.append(f"standalone_question 中的 {period} 既不在用户原话或上下文中，也不是声明过的推算结果")
    if any("standalone_question 中的" in e for e in errors):
        errors.append("推算出来的日期要在 inherited 里声明：derivation、amount 和 base（基准日期；日期范围可把整个范围写进 base，"
                      "如 \"2025-07-07至2025-07-16\"，两端会按同一偏移推算）")
    return errors


def _plural_errors(draft: RewriteDraft) -> list[str]:
    """A plural reference must name every one of its candidates in the standalone question."""
    stated = mentioned_periods(draft.standalone_question)
    errors = []
    for reference in draft.references:
        if reference.plural:
            missing = [c for c in reference.candidates if not mentioned_periods(c) <= stated
                       or (not mentioned_periods(c) and c not in draft.standalone_question)]
            if missing:
                errors.append(f"复数指代“{reference.phrase}”的对象 {missing} 没有全部写进 standalone_question")
    return errors


def _clarify_from_references(draft: RewriteDraft) -> str | None:
    for reference in draft.references:
        if reference.plural:
            continue
        candidates = list(dict.fromkeys(c.strip() for c in reference.candidates if c.strip()))
        if len(candidates) >= 2:
            return draft.clarification.strip() or f"你说的“{reference.phrase}”是指{'，还是'.join(candidates)}？"
    return None


class Rewriter:
    """Runs the rewrite through an agent's structured-LLM transport (same model, budgets and retry)."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def rewrite(self, utterance: str, turns: list[dict[str, Any]], as_of: str) -> dict[str, Any]:
        context = context_turns(turns)
        if not context:
            return {"action": "proceed", "standalone_question": utterance.strip(), "references": [], "inherited": [],
                    "clarification": "", "source": "first_turn", "token_count": 0}

        def check(draft: RewriteDraft) -> RewriteDraft:
            if draft.action == "proceed" and not draft.standalone_question.strip():
                raise PlanCheckError("action 为 proceed 时 standalone_question 不能为空")
            if draft.action == "clarify" and not draft.clarification.strip():
                raise PlanCheckError("action 为 clarify 时 clarification 不能为空")
            if draft.action == "proceed" and _clarify_from_references(draft) is None:
                errors = _account(draft, utterance, context) + _plural_errors(draft)
                if errors:
                    raise PlanCheckError("；".join(errors))
            return draft

        agent = self.agent
        agent._run_started_at = time.monotonic()
        agent._last_token_count = 0
        for name in ("schema_retry_count", "reasoning_token_count", "rate_limit_retry_count",
                     "bad_request_retry_count", "non_json_output_count"):
            setattr(agent, name, 0)
        agent.rate_limit_wait_seconds = 0.0
        try:
            draft = agent._json("conversation_rewrite", RewriteDraft, _prompt(utterance, context, as_of), check=check)
        except (PlanCheckError, ValueError) as error:
            return {"action": "clarify", "standalone_question": "", "references": [], "inherited": [],
                    "clarification": "我没能确定你指的是哪个时间，请把日期说完整一些（例如“2025年5月23日”）。",
                    "source": "check_failed", "error": str(error)[:500], "token_count": agent._last_token_count}
        out = draft.model_dump()
        clarification = _clarify_from_references(draft)
        if clarification is not None and draft.action == "proceed":
            out |= {"action": "clarify", "clarification": clarification, "source": "multiple_candidates"}
        else:
            out["source"] = "model"
        if out["action"] == "clarify":
            out["standalone_question"] = ""
        out["token_count"] = agent._last_token_count
        return out
