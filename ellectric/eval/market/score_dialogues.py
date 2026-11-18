"""Score multi-turn dialogue runs.

Per turn:

- ``clarify`` turns pass when the system asked back and ran no analysis;
- ``proceed`` turns need (1) no clarification, (2) a rewrite naming exactly the gold dates/months
  (``rewrite_ok``; for topic switches this also catches carried-over dates), and (3) an answer that passes the
  single-turn v2 stage scorer on the turn's embedded case (``answer_pass``). ``turn_pass`` = all three.

Reported: turn pass by template and for follow-up turns (turn 2+), dialogue pass (every turn), rewrite
accuracy, clarification precision/recall, cache hits, tokens; with ``--oracle`` results, the pass rate of the
same follow-up turns when the gold standalone question is asked directly (the ceiling for the rewrite).
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from ellectric.conversation.periods import mentioned_periods
from ellectric.eval.market.run_dialogues import has_network_error
from ellectric.eval.market.score import _metadata
from ellectric.eval.market.score_v2 import StageScorer, wilson

HERE = Path(__file__).resolve().parent


def load_records(folder: Path) -> list[dict]:
    out = []
    for path in sorted(folder.glob("*/run*/*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            out.append(json.load(handle))
    return out


def score_records(dialogues: dict[str, dict], records: list[dict]) -> list[dict]:
    metadata, rows = _metadata(), []
    for record in records:
        if has_network_error(record):  # transport failures are re-run, not scored (as in the single-turn runner)
            continue
        dialogue = dialogues[record["dialogue_id"]]
        for index, (gold, got) in enumerate(zip(dialogue["turns"], record["turns"]), 1):
            turn, agent = got.get("turn") or {}, got.get("agent")
            row = {"dialogue_id": dialogue["id"], "turn_id": gold["turn_id"], "turn": index,
                   "template": dialogue["template"], "kind": dialogue["kind"], "variant": record["variant"],
                   "run": record["run_index"], "expect": gold["expect"], "oracle": record.get("oracle", False)}
            asked_back = turn.get("type") == "clarify"
            if gold["expect"] == "clarify":
                row |= {"clarified": asked_back, "turn_pass": asked_back if not row["oracle"] else None}
                rows.append(row)
                continue
            if row["oracle"]:
                rewrite_ok = True
            else:
                stated = mentioned_periods(turn.get("standalone_question", "") or "")
                rewrite_ok = not asked_back and stated == set(gold["gold_periods"])
                row |= {"standalone": turn.get("standalone_question", ""), "stated_periods": sorted(stated),
                        "clarified": asked_back, "rewrite_source": (turn.get("rewrite") or {}).get("source"),
                        "cache_hits": (turn.get("cache") or {}).get("hits", 0),
                        "tier": (turn.get("presentation") or {}).get("tier"), "type": turn.get("type")}
            stage = {"pass": False, "stage": "aborted"}
            if agent is not None and agent.get("status") == "completed":
                stage = StageScorer(gold["case"], metadata).score(agent)
            elif agent is not None:
                stage = {"pass": False, "stage": "aborted", "detail": (agent.get("error") or "")[:200]}
            elif asked_back:
                stage = {"pass": False, "stage": "clarified"}
            tokens = ((agent or {}).get("trace") or {}).get("token_count")
            row |= {"rewrite_ok": rewrite_ok, "answer_pass": bool(stage["pass"]), "stage": stage["stage"],
                    "tokens": tokens, "turn_pass": rewrite_ok and bool(stage["pass"])}
            rows.append(row)
    return rows


def _rate(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return "-"
    k, n = sum(bool(v) for v in values), len(values)
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.0%}, {hi:.0%}]"


def summarize(rows: list[dict], oracle_rows: list[dict] | None = None) -> str:
    live = [r for r in rows if not r["oracle"]]
    follow = [r for r in live if r["turn"] > 1]
    proceed_follow = [r for r in follow if r["expect"] == "proceed"]
    lines = ["## 总体", "",
             f"- 追问轮通过率（第 2 轮起）：{_rate(follow, 'turn_pass')}",
             f"- 全部轮次通过率：{_rate(live, 'turn_pass')}",
             f"- 追问改写正确率（日期/月份完全一致）：{_rate(proceed_follow, 'rewrite_ok')}",
             f"- 追问答案通过率（单轮判分）：{_rate(proceed_follow, 'answer_pass')}"]
    by_dialogue = defaultdict(list)
    for r in live:
        by_dialogue[(r["variant"], r["run"], r["dialogue_id"])].append(r["turn_pass"])
    lines.append(f"- 整段对话通过率：{_rate([{'p': all(v)} for v in by_dialogue.values()], 'p')}")
    should = [r for r in follow if r["expect"] == "clarify"]
    asked = [r for r in follow if r.get("clarified")]
    lines += [f"- 该反问时反问（召回）：{_rate(should, 'clarified')}",
              f"- 反问是否必要（精确）：{_rate([{'p': r['expect'] == 'clarify'} for r in asked], 'p')}"]
    hits = [r.get("cache_hits", 0) for r in proceed_follow]
    tokens = [r["tokens"] for r in live if r.get("tokens")]
    if hits:
        lines.append(f"- 追问轮平均证据复用次数：{sum(hits) / len(hits):.2f}（有复用的轮次 {sum(h > 0 for h in hits)}/{len(hits)}）")
    if tokens:
        lines.append(f"- 每轮平均 token：{sum(tokens) / len(tokens):.0f}")
    if oracle_rows:
        ids = {r["turn_id"] for r in proceed_follow}
        ceiling = [r for r in oracle_rows if r["turn_id"] in ids and r["expect"] == "proceed"]
        lines.append(f"- 上限（同一批追问直接问标准独立问题）：{_rate(ceiling, 'answer_pass')}")
    lines += ["", "## 按模板（追问轮）", "", "| 模板 | 通过 | 改写正确 | 答案通过 |", "|---|---|---|---|"]
    by_template = defaultdict(list)
    for r in follow:
        by_template[r["template"]].append(r)
    for name, items in sorted(by_template.items()):
        lines.append(f"| {name} | {_rate(items, 'turn_pass')} | {_rate(items, 'rewrite_ok')} | {_rate(items, 'answer_pass')} |")
    lines += ["", "## 失败的追问轮", ""]
    for r in follow:
        if not r["turn_pass"]:
            lines.append(f"- {r['turn_id']} {r['template']} expect={r['expect']} type={r.get('type')} "
                         f"rewrite_ok={r.get('rewrite_ok')} stage={r.get('stage')} periods={r.get('stated_periods')} "
                         f"{(r.get('standalone') or '')[:80]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--oracle-results", type=Path)
    args = parser.parse_args()
    payload = json.loads((HERE / "cases" / f"{args.set}.json").read_text(encoding="utf-8"))
    dialogues = {x["id"]: x for x in payload["dialogues"]}
    results = args.results or HERE / "results" / args.set
    records = load_records(results)
    skipped = sum(has_network_error(r) for r in records)
    rows = score_records(dialogues, records)
    oracle_dir = args.oracle_results or HERE / "results" / f"{args.set}_oracle"
    oracle = score_records(dialogues, load_records(oracle_dir)) if oracle_dir.is_dir() else None
    note = f"> 有 {skipped} 段对话因 LLM 网络错误未计分（应以 --retry-network-errors 重跑）。\n\n" if skipped else ""
    text = f"# 多轮评测（题集 {args.set}，sha256 {payload['cases_sha256'][:12]}）\n\n" + note + summarize(rows, oracle)
    (results / "summary.md").write_text(text, encoding="utf-8")
    with gzip.open(results / "turns.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, default=str)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
