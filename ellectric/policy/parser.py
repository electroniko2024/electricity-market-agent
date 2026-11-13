"""Extract searchable text from user-provided policy attachments without inferring validity."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

from ellectric.policy.inventory import catalog_policy_files

PARSER_VERSION = "policy-text-v1"
MAX_CHUNK_CHARS = 6000
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_ARTICLE_RE = re.compile(r"^\s*(第[〇零一二三四五六七八九十百千万0-9]+条)(?:[\s.、．:：-]*(.*))?$")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _article_locator(text: str) -> str | None:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    match = _ARTICLE_RE.match(first_line)
    return match.group(1) if match else None


def _split_long_text(text: str, max_chars: int) -> list[str]:
    remaining = text.strip()
    pieces: list[str] = []
    while len(remaining) > max_chars:
        split_at = remaining.rfind("\n", 0, max_chars + 1)
        if split_at < max_chars // 2:
            split_at = max_chars
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _pack_blocks(blocks: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for block in blocks:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        locator = str(block["locator"])
        page = block.get("page")
        atomic = bool(block.get("atomic", False))
        if current and current["locator"] == locator and current.get("page") == page and not atomic:
            candidate = current["text"] + "\n" + text
            if len(candidate) <= max_chars:
                current["text"] = candidate
                continue
        if current:
            packed.append(current)
        pieces = _split_long_text(text, max_chars)
        for index, piece in enumerate(pieces, 1):
            piece_locator = locator if len(pieces) == 1 else f"{locator}, part {index}"
            item = {"locator": piece_locator, "page": page, "text": piece}
            if index < len(pieces):
                packed.append(item)
            else:
                current = item
                if atomic:
                    packed.append(current)
                    current = None
    if current:
        packed.append(current)
    return packed


def _extract_pdf(path: Path, max_chars: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    binary = shutil.which("pdftotext")
    if binary is None:
        raise FileNotFoundError("找不到本机 pdftotext；未安装额外 PDF 解析器")
    source_path = str(path)
    if os.name != "nt" and shutil.which("wslpath"):
        source_path = subprocess.run(
            ["wslpath", "-w", str(path)], check=True, capture_output=True, text=True
        ).stdout.strip()
    result = subprocess.run(
        [binary, "-layout", "-enc", "UTF-8", source_path, "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    pages = [page.strip() for page in result.stdout.split("\f") if page.strip()]
    blocks: list[dict[str, Any]] = []
    current_article: str | None = None
    for page_number, page_text in enumerate(pages, start=1):
        lines = page_text.splitlines()
        sections: list[tuple[str, list[str]]] = []
        section_locator = current_article or f"page {page_number}"
        section_lines: list[str] = []
        for line in lines:
            article = _article_locator(line)
            if article:
                if any(value.strip() for value in section_lines):
                    sections.append((section_locator, section_lines))
                current_article = article
                section_locator = f"{article}, page {page_number}"
                section_lines = [line]
            else:
                section_lines.append(line)
        if any(value.strip() for value in section_lines):
            sections.append((section_locator, section_lines))
        for locator, section in sections:
            blocks.append(
                {"locator": locator, "page": page_number, "text": "\n".join(section)}
            )
    return _pack_blocks(blocks, max_chars), {
        "page_count": len(pages),
        "parse_method": "pdftotext -layout",
        "replacement_character_count": result.stdout.count("\ufffd"),
    }


def _element_text(element: ET.Element) -> str:
    parts: list[str] = []
    for node in element.iter():
        if node.tag == f"{_WORD_NS}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{_WORD_NS}tab":
            parts.append("\t")
        elif node.tag == f"{_WORD_NS}br":
            parts.append("\n")
    return "".join(parts).strip()


def _extract_docx(path: Path, max_chars: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    document = ET.fromstring(xml)
    body = document.find(f"{_WORD_NS}body")
    if body is None:
        raise ValueError("DOCX 正文缺少 w:body")

    blocks: list[dict[str, Any]] = []
    paragraph_number = 0
    table_number = 0
    current_article: str | None = None
    for child in list(body):
        if child.tag == f"{_WORD_NS}p":
            paragraph_number += 1
            text = _element_text(child)
            article = _article_locator(text)
            if article:
                current_article = article
            locator = current_article or f"paragraph {paragraph_number}"
            blocks.append({"locator": locator, "text": text})
        elif child.tag == f"{_WORD_NS}tbl":
            table_number += 1
            for row_number, row in enumerate(child.findall(f"{_WORD_NS}tr"), start=1):
                cells = [
                    _element_text(cell)
                    for cell in row.findall(f"{_WORD_NS}tc")
                ]
                blocks.append(
                    {
                        "locator": f"table {table_number}, row {row_number}",
                        "text": "\t".join(cell for cell in cells if cell),
                        "atomic": True,
                    }
                )
    return _pack_blocks(blocks, max_chars), {
        "page_count": None,
        "parse_method": "OpenXML paragraph and table extraction",
        "paragraph_count": paragraph_number,
        "table_count": table_number,
    }


def _extract_doc(path: Path, max_chars: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if os.name != "nt":
        raise RuntimeError("旧版 DOC 解析需要本机 Microsoft Word；请在 Windows 运行")
    from win32com.client import DispatchEx

    word = DispatchEx("Word.Application")
    document = None
    try:
        word.Visible = False
        word.DisplayAlerts = 0
        document = word.Documents.Open(
            FileName=str(path.resolve()),
            ReadOnly=True,
            AddToRecentFiles=False,
            ConfirmConversions=False,
        )
        positioned: list[tuple[int, dict[str, Any]]] = []
        current_article: str | None = None
        for paragraph_number in range(1, document.Paragraphs.Count + 1):
            paragraph = document.Paragraphs(paragraph_number)
            paragraph_range = paragraph.Range
            if paragraph_range.Information(12):
                continue
            text = str(paragraph_range.Text).strip("\r\x07 \t\n")
            if not text:
                continue
            article = _article_locator(text)
            if article:
                current_article = article
            locator = current_article or f"paragraph {paragraph_number}"
            positioned.append(
                (
                    int(paragraph_range.Start),
                    {"locator": locator, "text": text},
                )
            )
        for table_number in range(1, document.Tables.Count + 1):
            table = document.Tables(table_number)
            for row_number in range(1, table.Rows.Count + 1):
                cells = [
                    str(table.Cell(row_number, col_number).Range.Text).strip("\r\x07 \t\n")
                    for col_number in range(1, table.Columns.Count + 1)
                ]
                positioned.append(
                    (
                        int(table.Rows(row_number).Range.Start),
                        {
                            "locator": f"table {table_number}, row {row_number}",
                            "text": "\t".join(cell for cell in cells if cell),
                            "atomic": True,
                        },
                    )
                )
        positioned.sort(key=lambda item: item[0])
        blocks = [block for _, block in positioned]
        page_count = int(document.ComputeStatistics(2))
        return _pack_blocks(blocks, max_chars), {
            "page_count": page_count,
            "parse_method": "Microsoft Word read-only COM extraction",
            "paragraph_count": int(document.Paragraphs.Count),
            "table_count": int(document.Tables.Count),
        }
    finally:
        if document is not None:
            document.Close(SaveChanges=0)
        word.Quit()


def parse_policy_directory(
    policy_dir: str | Path,
    *,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse local files, retaining unknown legal metadata and precise source locators."""
    root = Path(policy_dir)
    inventory = catalog_policy_files(root)
    chunks: list[dict[str, Any]] = []
    parse_errors = 0

    extractors: dict[str, Callable[[Path, int], tuple[list[dict[str, Any]], dict[str, Any]]]] = {
        "pdf": _extract_pdf,
        "docx": _extract_docx,
        "doc": _extract_doc,
    }
    for document in inventory["documents"]:
        source_path = root / document["relative_path"]
        try:
            document_chunks, detail = extractors[document["format"]](
                source_path, max_chunk_chars
            )
            if not document_chunks:
                document["parse_status"] = "empty_text"
                parse_errors += 1
            else:
                document["parse_status"] = "parsed"
            document.update(detail)
            document["chunk_count"] = len(document_chunks)
            document["text_characters"] = sum(len(chunk["text"]) for chunk in document_chunks)
            document["parser_version"] = PARSER_VERSION
            if detail.get("replacement_character_count", 0):
                document["parse_quality"] = "text_extracted_with_replacement_characters"
            else:
                document["parse_quality"] = "text_extracted"
            for ordinal, chunk in enumerate(document_chunks, start=1):
                text = chunk["text"]
                locator = chunk["locator"]
                chunk_id = _digest(f"{document['sha256']}\n{ordinal}\n{locator}\n{text}")
                chunk_record = {
                    "chunk_id": chunk_id,
                    "document_id": document["sha256"],
                    "source_name": document["source_name"],
                    "relative_path": document["relative_path"],
                    "document_format": document["format"],
                    "ordinal": ordinal,
                    "locator": locator,
                    "page": chunk.get("page"),
                    "text": text,
                    "text_sha256": _digest(text),
                }
                chunks.append(chunk_record)
        except Exception as error:
            document["parse_status"] = "parse_error"
            document["parse_error"] = f"{type(error).__name__}: {error}"
            document["chunk_count"] = 0
            document["text_characters"] = 0
            document["parser_version"] = PARSER_VERSION
            parse_errors += 1

    parsed_count = sum(document["parse_status"] == "parsed" for document in inventory["documents"])
    inventory["inventory_status"] = "text_parsed_with_gaps" if parse_errors else "text_parsed"
    inventory["policy_coverage_verified"] = False
    inventory["summary"].update(
        {
            "documents_parsed": parsed_count,
            "documents_with_parse_errors": parse_errors,
            "chunk_count": len(chunks),
            "documents_pending_metadata": len(inventory["documents"]),
        }
    )
    return inventory, chunks
