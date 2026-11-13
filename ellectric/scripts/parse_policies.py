"""Parse user-provided policy files into a metadata manifest and source chunks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from ellectric.policy.parser import parse_policy_directory


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "ellectric" / "data" / "market_processed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=repo_root.parent / "data" / "policy")
    parser.add_argument("--documents-output", type=Path, default=output_dir / "policy_documents.json")
    parser.add_argument("--chunks-output", type=Path, default=output_dir / "policy_chunks.json")
    parser.add_argument("--max-chunk-chars", type=int, default=6000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_chunk_chars < 500:
        parser.error("--max-chunk-chars 至少为 500")
    outputs = [args.documents_output, args.chunks_output]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出已存在；为保护已有文件拒绝覆盖。使用 --overwrite 明确重建: "
            + ", ".join(str(path) for path in existing)
        )

    inventory, chunks = parse_policy_directory(
        args.policy_dir,
        max_chunk_chars=args.max_chunk_chars,
    )
    _write_json(args.documents_output, inventory)
    _write_json(args.chunks_output, chunks)
    print(
        json.dumps(
            {
                "documents_output": str(args.documents_output.resolve()),
                "chunks_output": str(args.chunks_output.resolve()),
                "inventory_status": inventory["inventory_status"],
                **inventory["summary"],
                "policy_coverage_verified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if inventory["summary"]["documents_with_parse_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
