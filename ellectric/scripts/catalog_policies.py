"""盘点本地山东电力政策文件，不解析正文或推断效力状态。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ellectric.policy.inventory import catalog_policy_files


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=repo_root.parent / "data" / "policy")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "ellectric" / "data" / "market_processed" / "policy_documents.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在，为保护现有文件拒绝覆盖: {args.output}")

    inventory = catalog_policy_files(args.policy_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), **inventory["summary"], "coverage_verified": False},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
