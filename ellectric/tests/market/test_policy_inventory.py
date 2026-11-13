from __future__ import annotations

import hashlib

import pytest

from ellectric.policy.inventory import catalog_policy_files


def test_catalog_records_hashes_unknown_metadata_and_duplicate_contents(tmp_path):
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    first = policy_dir / "a.pdf"
    second = policy_dir / "copy.docx"
    ignored = policy_dir / "notes.txt"
    first.write_bytes(b"same source bytes")
    second.write_bytes(b"same source bytes")
    ignored.write_text("not a policy attachment", encoding="utf-8")

    inventory = catalog_policy_files(policy_dir)

    assert inventory["summary"]["document_count"] == 2
    assert inventory["policy_coverage_verified"] is False
    assert inventory["summary"]["duplicate_content_group_count"] == 1
    assert {item["format"] for item in inventory["documents"]} == {"pdf", "docx"}
    for item in inventory["documents"]:
        assert item["sha256"] == hashlib.sha256(b"same source bytes").hexdigest()
        assert item["document_status"] == "unknown"
        assert item["publish_date"] is None
        assert item["parse_status"] == "not_attempted"
        assert "effective_from" in item["metadata_pending"]


def test_catalog_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        catalog_policy_files(tmp_path / "missing")
