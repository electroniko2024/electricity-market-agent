"""Build local BM25 and Chroma indexes from the parsed policy corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ellectric.policy.retrieval import build_policy_index


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "ellectric" / "data" / "market_processed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=data_dir / "policy_documents.json")
    parser.add_argument("--chunks", type=Path, default=data_dir / "policy_chunks.json")
    parser.add_argument("--index-dir", type=Path, default=data_dir / "policy_index")
    parser.add_argument(
        "--vector-backend",
        choices=["chroma", "json"],
        default=None,
        help="默认 chromadb 可导入时用 chroma，否则 json",
    )
    args = parser.parse_args()

    result = build_policy_index(
        args.documents, args.chunks, args.index_dir, vector_backend=args.vector_backend
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
