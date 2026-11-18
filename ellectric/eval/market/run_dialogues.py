"""Run a multi-turn dialogue set through the conversation layer (or, with --oracle, the gold standalone questions
through the single-turn agent, as the upper bound a perfect rewrite would reach).

One gzip record per (variant, run, dialogue) holds every turn: the stored conversation turn (rewrite, cache
use, presentation) and the agent's full report and trace, so scoring is offline. Completed records are skipped
on restart.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ellectric.conversation.service import ConversationService, ConversationStore, TurnRequest
from ellectric.llm.market_analysis import VARIANT_DESCRIPTIONS, MarketAnalysisAgent
from ellectric.service.market_tools import invoke_market_tool

HERE = Path(__file__).resolve().parent


def load_set(name: str) -> dict:
    return json.loads((HERE / "cases" / f"{name}.json").read_text(encoding="utf-8"))


def record_path(out: Path, variant: str, run_index: int, dialogue_id: str) -> Path:
    return out / variant / f"run{run_index}" / f"{dialogue_id}.json.gz"


NETWORK_MARKERS = ("LLM 请求失败", "LLM 端点返回 HTTP")


def is_network_error(message: str | None) -> bool:
    return bool(message) and any(marker in message for marker in NETWORK_MARKERS)


def has_network_error(record: dict) -> bool:
    return any((t.get("agent") or {}).get("status") == "network_error"
               or is_network_error((t.get("turn") or {}).get("error")) for t in record["turns"])


def _agent_record(turn_id: str, variant: str, run_index: int, result: dict | None, error: str | None) -> dict:
    if result is None:
        return {"case_id": turn_id, "variant": variant, "run_index": run_index,
                "status": "network_error" if is_network_error(error) else "run_error", "error": error or "no result"}
    return {"case_id": turn_id, "variant": variant, "run_index": run_index, "status": "completed",
            "run_id": result["run_id"], "report": result["report"], "trace": result["trace"]}


def run_conversation(dialogue: dict, variant: str, run_index: int, workdir: Path) -> list[dict]:
    service = ConversationService(store=ConversationStore(workdir / "conversations"), trace_dir=workdir / "traces",
                                  agent_factory=lambda call: MarketAnalysisAgent(variant=variant, tool_call=call))
    as_of = datetime.fromisoformat(dialogue["as_of"])
    conversation_id, out = None, []
    for gold in dialogue["turns"]:
        started = time.monotonic()
        turn, result, error = None, None, None
        try:
            for event in service.stream_turn(conversation_id, TurnRequest(message=gold["utterance"], as_of=as_of)):
                if event["type"] == "conversation":
                    conversation_id = event["conversation_id"]
                elif event["type"] == "turn":
                    turn, result = event["turn"], event["result"]
        except Exception as exc:  # recorded and scored as aborted
            error = f"{type(exc).__name__}: {exc}"[:2000] + "\n" + traceback.format_exc(limit=6)[-2000:]
        if turn is not None and turn.get("error"):
            error = turn["error"]
        ran_agent = turn is None or turn.get("type") == "question"
        out.append({"turn_id": gold["turn_id"], "turn": turn, "wall_seconds": round(time.monotonic() - started, 2),
                    "agent": _agent_record(gold["turn_id"], variant, run_index, result, error) if ran_agent else None})
    return out


def run_oracle(dialogue: dict, variant: str, run_index: int) -> list[dict]:
    as_of = datetime.fromisoformat(dialogue["as_of"])
    out = []
    for gold in dialogue["turns"]:
        if gold["expect"] != "proceed":
            out.append({"turn_id": gold["turn_id"], "turn": None, "agent": None})
            continue
        started, result, error = time.monotonic(), None, None
        try:
            result = MarketAnalysisAgent(variant=variant, tool_call=invoke_market_tool).analyze(
                question=gold["gold_question"], as_of=as_of, mode="historical_analysis")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:2000]
        out.append({"turn_id": gold["turn_id"], "turn": None, "wall_seconds": round(time.monotonic() - started, 2),
                    "agent": _agent_record(gold["turn_id"], variant, run_index, result, error)})
    return out


def run_one(dialogue: dict, variant: str, run_index: int, out: Path, oracle: bool) -> dict:
    path = record_path(out, variant, run_index, dialogue["id"])
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as workdir:
        turns = run_oracle(dialogue, variant, run_index) if oracle else \
            run_conversation(dialogue, variant, run_index, Path(workdir))
    record = {"dialogue_id": dialogue["id"], "variant": variant, "run_index": run_index, "oracle": oracle,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "turns": turns,
              "wall_seconds": round(time.monotonic() - started, 2)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, default=str)
    os.replace(temporary, path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True)
    parser.add_argument("--variants", nargs="+", default=["E2"], choices=sorted(VARIANT_DESCRIPTIONS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--first-run", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dialogues", nargs="*")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--retry-network-errors", action="store_true",
                        help="re-run dialogues whose record has a turn that failed on the LLM transport")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not os.environ.get("ELLECTRIC_LLM_API_KEY"):
        raise SystemExit("ELLECTRIC_LLM_API_KEY is not set")
    payload = load_set(args.set)
    dialogues = [x for x in payload["dialogues"] if not args.dialogues or x["id"] in args.dialogues]
    out = args.out or HERE / "results" / (args.set + ("_oracle" if args.oracle else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        "set": args.set, "cases_sha256": payload["cases_sha256"], "oracle": args.oracle,
        "model": os.environ.get("ELLECTRIC_LLM_MODEL", "deepseek-v4.1-flash"),
        "thinking": os.environ.get("ELLECTRIC_LLM_THINKING", "disabled")}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    def pending(dialogue: dict, variant: str, run_index: int) -> bool:
        path = record_path(out, variant, run_index, dialogue["id"])
        if not path.exists():
            return True
        if not args.retry_network_errors:
            return False
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return has_network_error(json.load(handle))

    jobs = [(x, v, r) for r in range(args.first_run, args.first_run + args.runs) for v in args.variants
            for x in dialogues if pending(x, v, r)]
    print(f"{len(jobs)} dialogue jobs", flush=True)
    lock, done = threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, x, v, r, out, args.oracle) for x, v, r in jobs]
        for future in as_completed(futures):
            record = future.result()
            with lock:
                done += 1
                kinds = " ".join(
                    ((t["turn"] or {}).get("type", "-") + ":" + (t["agent"] or {}).get("status", "-"))
                    for t in record["turns"])
                print(f"[{done}/{len(jobs)}] {record['variant']} run{record['run_index']} {record['dialogue_id']} "
                      f"{kinds} {record['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
