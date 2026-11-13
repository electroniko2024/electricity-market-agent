"""Run frozen evaluation cases against agent variants E0-E3.

Every (variant, run, case) is a fresh agent with its own run_id; no LLM response is cached
or replayed, so repeated runs are independent online runs. Completed outputs are skipped,
so an interrupted evaluation resumes where it stopped. Each record keeps the full report
and trace (gzip JSON) for offline re-scoring.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ellectric.eval.market.faults import FaultInjectingToolCall
from ellectric.llm.market_analysis import VARIANT_DESCRIPTIONS, MarketAnalysisAgent
from ellectric.service.market_tools import invoke_market_tool

HERE = Path(__file__).resolve().parent


def load_cases(set_name: str) -> dict:
    return json.loads((HERE / "cases" / f"{set_name}.json").read_text(encoding="utf-8"))


def record_path(out_dir: Path, variant: str, run_index: int, case_id: str) -> Path:
    return out_dir / variant / f"run{run_index}" / f"{case_id}.json.gz"


def run_one(case: dict, variant: str, run_index: int, out_dir: Path) -> dict:
    path = record_path(out_dir, variant, run_index, case["id"])
    started = time.monotonic()
    record = {"case_id": case["id"], "category": case["category"], "variant": variant, "run_index": run_index,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "temperature": 0, "seed": None}
    injector = FaultInjectingToolCall(invoke_market_tool, case["faults"]) if case.get("faults") else None
    try:
        agent = MarketAnalysisAgent(variant=variant, tool_call=injector or invoke_market_tool)
        result = agent.analyze(
            question=case["question"], as_of=datetime.fromisoformat(case["as_of"]), mode=case["mode"])
        record.update(status="completed", run_id=result["run_id"], report=result["report"], trace=result["trace"])
    except Exception as error:  # the failure itself is a scored outcome
        message = str(error)
        network = "LLM 请求失败" in message or "LLM 端点返回 HTTP" in message
        record.update(status="network_error" if network else "run_error",
                      error=f"{type(error).__name__}: {message}"[:2000],
                      traceback=traceback.format_exc(limit=8)[-4000:])
    if injector is not None:
        record["fault_log"] = injector.log
    record["wall_seconds"] = round(time.monotonic() - started, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, default=str)
    os.replace(temporary, path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="dev_v1")
    parser.add_argument("--variants", nargs="+", default=["E2"], choices=sorted(VARIANT_DESCRIPTIONS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--first-run", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cases", nargs="*", help="case ids to run (default: all)")
    parser.add_argument("--retry-network-errors", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not os.environ.get("ELLECTRIC_LLM_API_KEY"):
        raise SystemExit("ELLECTRIC_LLM_API_KEY is not set")

    payload = load_cases(args.set)
    cases = [c for c in payload["cases"] if not args.cases or c["id"] in args.cases]
    out_dir = args.out or HERE / "results" / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "set": args.set, "cases_sha256": payload["cases_sha256"], "variants": VARIANT_DESCRIPTIONS,
        "model": os.environ.get("ELLECTRIC_LLM_MODEL", "deepseek-v4.1-flash"),
        "thinking": os.environ.get("ELLECTRIC_LLM_THINKING", "disabled"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    jobs = []
    for run_index in range(args.first_run, args.first_run + args.runs):
        for variant in args.variants:
            for case in cases:
                path = record_path(out_dir, variant, run_index, case["id"])
                if path.exists():
                    if not args.retry_network_errors:
                        continue
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        if json.load(handle).get("status") != "network_error":
                            continue
                jobs.append((case, variant, run_index))
    print(f"{len(jobs)} jobs ({len(cases)} cases x {args.variants} x runs {args.first_run}..{args.first_run + args.runs - 1})", flush=True)
    lock, done = threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, case, variant, run_index, out_dir) for case, variant, run_index in jobs]
        for future in as_completed(futures):
            record = future.result()
            with lock:
                done += 1
                tokens = (record.get("trace") or {}).get("token_count")
                validation = ((record.get("report") or {}).get("validation") or {}).get("status")
                print(f"[{done}/{len(jobs)}] {record['variant']} run{record['run_index']} {record['case_id']} "
                      f"{record['status']} validation={validation} tokens={tokens} {record['wall_seconds']}s"
                      + (f" {record.get('error', '')[:160]}" if record["status"] != "completed" else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
