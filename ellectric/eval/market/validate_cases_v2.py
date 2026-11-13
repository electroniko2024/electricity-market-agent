"""Check a generated v2 set without any LLM: run each case's reference solution through the real tools and
confirm the scorer finds the evidence (every reference number/clause appears in the resulting observations).

A case fails if a reference call errors, or its evidence is not found — then either the case or the scorer is
wrong, and it must be fixed before any agent run. Values marked ``evidence: false`` (computed by the agent)
are skipped; their inputs are listed under ``evidence_values`` and checked instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from ellectric.eval.market.score import HERE
from ellectric.eval.market.score_v2 import StageScorer
from ellectric.service.market_tools import invoke_market_tool


def run_solution(case: dict) -> list[dict]:
    observations = []
    for index, step in enumerate(case["reference_solution"]):
        args = dict(step["args"])
        if step["tool"] != "search_policy" or "as_of" not in args:
            args["as_of"] = datetime.fromisoformat(case["as_of"])
        try:
            result, status, error = invoke_market_tool(step["tool"], args), "ok", None
        except Exception as exc:  # a failing reference call is a case defect
            result, status, error = None, "error", f"{type(exc).__name__}: {exc}"[:300]
        observations.append({"evidence_id": f"ref{index}", "tool_name": step["tool"], "arguments": step["args"],
                             "execution_status": status, "error": error, "result": result})
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    payload = json.loads((HERE / "cases" / f"{args.set}.json").read_text(encoding="utf-8"))
    failures, failed_ids = 0, []
    for case in payload["cases"]:
        if args.cases and case["id"] not in args.cases:
            continue
        observations = run_solution(case)
        errors = [o for o in observations if o["execution_status"] != "ok"]
        # validation "failed" disables the "agent computed it" fallback: only observed values count here.
        evidence = StageScorer(case, {}).evidence(observations, {"validation": "failed", "numbers": {}})
        ok = not errors and evidence["acquired"] is not False
        if not ok:
            failures += 1
            failed_ids.append(case["id"])
            missing = [n for n, v in evidence["numbers"].items() if not v]
            print(f"FAIL {case['id']} {case['template']}: errors={[o['error'] for o in errors]} missing={missing} "
                  f"clauses={evidence['clauses']} tools={evidence['tools']}")
    total = len([c for c in payload["cases"] if not args.cases or c["id"] in args.cases])
    print(f"{total - failures}/{total} cases solvable by their reference solution")
    if not args.cases:
        # Kept in the set and scored as usual, but reported separately: no known tool path reaches the reference.
        sidecar = HERE / "cases" / f"{args.set}.validation.json"
        sidecar.write_text(json.dumps({"cases_sha256": payload["cases_sha256"], "reference_unsolved": failed_ids},
                                      ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
