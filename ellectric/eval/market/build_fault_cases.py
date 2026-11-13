"""Build the fault-injection set ``fault_v1`` from frozen ``formal_v1`` cases.

Each case reuses a formal C/E case unchanged (question, references, key points) and adds one transient
fault on a tool the case cannot complete without, so a variant passes only if it recovers from the failed
call. Base cases are ones both E1 and E2 completed in formal_v1, so a failure here comes from the fault.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent / "cases"

# (base case, faulted tool): C cases need detect_anomaly and explain_drivers; E cases need forecast_price
# and a cause citing explain_drivers (their price-data group has interchangeable tools, so it is not faulted).
FAULTS = [
    *[(case_id, "explain_drivers") for case_id in ("C01", "C02", "C03", "C04", "C05")],
    *[(case_id, "detect_anomaly") for case_id in ("C06", "C07", "C08", "C09", "C10")],
    *[(case_id, "forecast_price") for case_id in ("E07", "E10", "E12", "E14", "E17")],
    *[(case_id, "explain_drivers") for case_id in ("E13", "E15", "E18", "E19", "E20")],
]


def build(formal: dict) -> list[dict]:
    by_id = {case["id"]: case for case in formal["cases"]}
    cases = []
    for index, (base_id, tool) in enumerate(FAULTS, start=1):
        case = copy.deepcopy(by_id[base_id])
        case.update(id=f"F{index:02d}", base_case=base_id,
                    faults=[{"tool": tool, "mode": "unavailable", "times": 1}])
        cases.append(case)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = OUT / "fault_v1.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; frozen sets are not rebuilt without --overwrite")
    formal = json.loads((OUT / "formal_v1.json").read_text(encoding="utf-8"))
    cases = build(formal)
    body = json.dumps(cases, ensure_ascii=False, sort_keys=True)
    payload = {"set": "fault_v1", "version": 1, "built_at": datetime.now().isoformat(timespec="seconds"),
               "base_set": "formal_v1", "base_cases_sha256": formal["cases_sha256"],
               "case_count": len(cases), "cases_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
               "spike_threshold_yuan_per_mwh": formal["spike_threshold_yuan_per_mwh"], "cases": cases}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
