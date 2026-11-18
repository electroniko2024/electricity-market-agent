"""Record (or check) the hashes of every source file that can change an evaluation result.

A test set is generated only after ``freeze`` writes ``<name>.freeze.json``, with a seed not used during
development, and is run once. ``check`` compares the current tree with a freeze record; any difference means
the frozen numbers no longer describe this code and a new test set is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN_DIRS = ["ellectric/api", "ellectric/conversation", "ellectric/eval/market", "ellectric/llm",
               "ellectric/pipeline", "ellectric/policy", "ellectric/service"]
CASES = ROOT / "ellectric" / "eval" / "market" / "cases"


def source_hashes() -> dict[str, str]:
    files = sorted(p for d in FROZEN_DIRS for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def combined(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("freeze")
    write.add_argument("--name", required=True, help="record name, e.g. test_v4")
    write.add_argument("--purpose", required=True)
    check = sub.add_parser("check")
    check.add_argument("--name", required=True)
    args = parser.parse_args()
    path = CASES / f"{args.name}.freeze.json"
    hashes = source_hashes()
    if args.command == "freeze":
        if path.exists():
            raise SystemExit(f"{path} exists")
        record = {"frozen_at": datetime.now().isoformat(timespec="seconds"), "purpose": args.purpose,
                  "file_count": len(hashes), "combined_sha256": combined(hashes), "files": hashes}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.name}: {len(hashes)} files, {record['combined_sha256'][:12]}")
        return 0
    record = json.loads(path.read_text(encoding="utf-8"))
    changed = sorted(k for k in set(hashes) | set(record["files"]) if hashes.get(k) != record["files"].get(k))
    print("unchanged" if not changed else "changed:\n  " + "\n  ".join(changed))
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
