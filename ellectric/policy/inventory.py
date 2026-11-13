"""
政策文件清单 — Policy Source Inventory
======================================

本模块只盘点用户提供的 PDF、DOC 与 DOCX 文件，不解析正文，也不推断
文号、生效时间、发布机关或正式/草案状态。
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}
PENDING_METADATA_FIELDS = [
    "official_url",
    "document_number",
    "issuer",
    "publish_date",
    "effective_from",
    "effective_to",
    "document_status",
    "replaces",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def catalog_policy_files(policy_dir: str | Path) -> dict[str, Any]:
    """返回可复现的源文件清单，并显式保留尚未提取的政策元数据。"""
    root = Path(policy_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"政策目录不存在: {root}")

    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    digest_by_path = {path: _sha256(path) for path in paths}
    paths_by_digest: dict[str, list[str]] = defaultdict(list)
    for path, digest in digest_by_path.items():
        paths_by_digest[digest].append(path.relative_to(root).as_posix())

    documents = []
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        documents.append(
            {
                "source_name": path.name,
                "relative_path": relative_path,
                "format": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
                "sha256": digest_by_path[path],
                "duplicate_content_paths": sorted(paths_by_digest[digest_by_path[path]])
                if len(paths_by_digest[digest_by_path[path]]) > 1
                else [],
                "official_url": None,
                "document_number": None,
                "issuer": None,
                "publish_date": None,
                "effective_from": None,
                "effective_to": None,
                "document_status": "unknown",
                "replaces": [],
                "parse_status": "not_attempted",
                "metadata_pending": PENDING_METADATA_FIELDS.copy(),
            }
        )

    format_counts = Counter(document["format"] for document in documents)
    duplicate_groups = [
        {"sha256": digest, "paths": sorted(group)}
        for digest, group in sorted(paths_by_digest.items())
        if len(group) > 1
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_status": "file_inventory_only",
        "policy_coverage_verified": False,
        "policy_dir_name": root.name,
        "summary": {
            "document_count": len(documents),
            "format_counts": dict(sorted(format_counts.items())),
            "duplicate_content_group_count": len(duplicate_groups),
            "documents_pending_metadata": len(documents),
        },
        "duplicate_content_groups": duplicate_groups,
        "documents": documents,
    }
