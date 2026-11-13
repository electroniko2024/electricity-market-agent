"""Policy text extraction tests that do not require external parser downloads."""

from __future__ import annotations

import hashlib
import json
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ellectric.policy.parser import parse_policy_directory


def _write_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>第一条 本办法适用于测试。</w:t></w:r></w:p>
    <w:p><w:r><w:t>交易机构应保存原始记录。</w:t></w:r></w:p>
    <w:tbl><w:tr>
      <w:tc><w:p><w:r><w:t>参数</w:t></w:r></w:p></w:tc>
      <w:tc><w:p><w:r><w:t>数值</w:t></w:r></w:p></w:tc>
    </w:tr></w:tbl>
  </w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml.encode("utf-8"))


class PolicyParserTest(unittest.TestCase):
    def test_docx_text_and_table_rows_keep_real_locators_and_unknown_metadata(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            policy_dir = Path(temporary)
            _write_docx(policy_dir / "rules.docx")
            inventory, chunks = parse_policy_directory(policy_dir)

        document = inventory["documents"][0]
        self.assertEqual(inventory["inventory_status"], "text_parsed")
        self.assertEqual(document["parse_status"], "parsed")
        self.assertEqual(document["publish_date"], None)
        self.assertEqual(document["document_status"], "unknown")
        self.assertEqual(len(chunks), 2)
        self.assertIn("第一条", chunks[0]["locator"])
        self.assertIn("交易机构", chunks[0]["text"])
        self.assertEqual(chunks[1]["locator"], "table 1, row 1")
        self.assertIn("参数", chunks[1]["text"])


if __name__ == "__main__":
    unittest.main()
