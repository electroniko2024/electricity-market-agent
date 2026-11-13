"""Local hybrid policy retrieval must preserve citations and fail closed."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from ellectric.policy.retrieval import _tokens, build_policy_index, search_local_policy


class FakeEmbedder:
    """Bag-of-bigrams embedder so tests need neither Ollama nor model files."""

    model_id = "fake:bigram"
    vocabulary = ["实时", "价格", "上限", "储能", "并网", "申报"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            tokens = _tokens(text)
            vector = [float(tokens.count(term)) + 0.01 for term in self.vocabulary]
            norm = sum(value * value for value in vector) ** 0.5
            vectors.append([value / norm for value in vector])
        return vectors


def reverse_length_reranker(_query: str, passages: list[str]) -> list[float]:
    """Prefers shorter passages, so reranking visibly changes the fused order."""
    return [-float(len(passage)) for passage in passages]


class PolicyRetrievalTest(unittest.TestCase):
    def _corpus(self, root: Path) -> tuple[Path, Path]:
        source_hash = "a" * 64
        documents_path = root / "documents.json"
        chunks_path = root / "chunks.json"
        documents_path.write_text(
            json.dumps(
                {
                    "policy_coverage_verified": False,
                    "documents": [
                        {
                            "sha256": source_hash,
                            "source_name": "local-rules.pdf",
                            "document_status": "unknown",
                            "metadata_pending": ["effective_from", "replaces"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        def chunk(ordinal: int, text: str, page: int, **extra):
            return {
                "chunk_id": str(ordinal) * 64,
                "document_id": source_hash,
                "source_name": "local-rules.pdf",
                "relative_path": "local-rules.pdf",
                "ordinal": ordinal,
                "locator": f"page {page}",
                "page": page,
                "text": text,
                **extra,
            }

        chunks_path.write_text(
            json.dumps(
                [
                    chunk(1, "实时价格上限按照市场规则执行，具体价格上限以适用文件为准，另行公布。", 8),
                    chunk(2, "实时价格上限。", 9),
                    chunk(3, "储能电站并网前需办理并网手续。", 10),
                ]
            ),
            encoding="utf-8",
        )
        return documents_path, chunks_path

    def _run(self, backend: str) -> tuple[dict, dict]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents_path, chunks_path = self._corpus(root)
            index_dir = root / "index"
            manifest = build_policy_index(
                documents_path, chunks_path, index_dir, embedder=FakeEmbedder(), vector_backend=backend
            )
            result = search_local_policy(
                query="实时价格上限",
                policy_at="2025-10-01",
                index_dir=index_dir,
                top_k=2,
                embedder=FakeEmbedder(),
                reranker=reverse_length_reranker,
            )
        return manifest, result

    def _assert_candidates(self, manifest: dict, result: dict) -> None:
        self.assertEqual(manifest["indexed_text_chunk_count"], 3)
        self.assertEqual(manifest["embedding_model"], "fake:bigram")
        self.assertEqual(result["status"], "unverified_candidates")
        self.assertEqual(result["metadata_verification_status"], "missing")
        self.assertEqual(len(result["evidence"]), 2)
        top = result["evidence"][0]
        self.assertEqual(top["page"], 9)  # reranker picked the shorter passage first
        self.assertIsNotNone(top["rerank_score"])
        self.assertFalse(top["evidence_eligible"])
        self.assertIn("effective_from", top["metadata_pending"])

    def test_json_backend_fuses_and_reranks(self) -> None:
        self._assert_candidates(*self._run("json"))

    @unittest.skipUnless(importlib.util.find_spec("chromadb"), "chromadb not installed")
    def test_chroma_backend_fuses_and_reranks(self) -> None:
        manifest, result = self._run("chroma")
        self.assertEqual(manifest["vector_backend"], "chroma")
        self._assert_candidates(manifest, result)

    def test_query_with_different_embedding_model_is_rejected(self) -> None:
        class OtherEmbedder(FakeEmbedder):
            model_id = "fake:other"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents_path, chunks_path = self._corpus(root)
            build_policy_index(
                documents_path, chunks_path, root / "index", embedder=FakeEmbedder(), vector_backend="json"
            )
            with self.assertRaises(ValueError):
                search_local_policy(
                    query="实时价格上限",
                    policy_at="2025-10-01",
                    index_dir=root / "index",
                    embedder=OtherEmbedder(),
                    reranker=reverse_length_reranker,
                )


class PolicyVersionFilterTest(unittest.TestCase):
    """Two versions of the same rule: the filter must pick the one in force on policy_at."""

    OLD = {"relative_path": "old.pdf", "family": "rules", "title": "旧规则", "publish_not_later_than": "2024-04-30",
           "effective_from": "2024-05-01", "effective_to": "2026-01-01"}
    NEW = {"relative_path": "new.pdf", "family": "rules", "title": "新规则", "publish_not_later_than": "2025-12-31",
           "effective_from": "2026-01-01", "effective_to": None}

    def _index(self, root: Path) -> Path:
        documents = root / "documents.json"
        documents.write_text(json.dumps({"documents": [
            {"sha256": name * 64, "source_name": f"{name}.pdf", "metadata_pending": ["effective_from"]}
            for name in ("a", "b")
        ]}), encoding="utf-8")
        chunks = root / "chunks.json"
        chunks.write_text(json.dumps([
            {"chunk_id": "1" * 64, "document_id": "a" * 64, "source_name": "old.pdf", "relative_path": "old.pdf",
             "ordinal": 1, "locator": "第 19.2 条", "page": 238, "text": "实时价格上限 旧规则条款。"},
            {"chunk_id": "2" * 64, "document_id": "b" * 64, "source_name": "new.pdf", "relative_path": "new.pdf",
             "ordinal": 1, "locator": "第 22.2 条", "page": 317, "text": "实时价格上限 新规则条款。"},
        ]), encoding="utf-8")
        build_policy_index(documents, chunks, root / "index", embedder=FakeEmbedder(), vector_backend="json")
        return root / "index"

    def _search(self, policy_at: str, *, as_of_date: str, status: str = "user_confirmed", effective_filter: bool = True):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = self._index(root)
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"verification_status": status, "documents": [self.OLD, self.NEW]}),
                                encoding="utf-8")
            return search_local_policy(
                query="实时价格上限", policy_at=policy_at, index_dir=index, top_k=5, embedder=FakeEmbedder(),
                reranker=reverse_length_reranker, as_of_date=as_of_date, effective_filter=effective_filter,
                metadata_path=metadata,
            )

    def test_filter_returns_only_the_version_in_force(self) -> None:
        before = self._search("2025-12-15", as_of_date="2026-01-14")
        after = self._search("2026-01-05", as_of_date="2026-01-14")
        self.assertEqual([item["source_name"] for item in before["evidence"]], ["old.pdf"])
        self.assertEqual([item["source_name"] for item in after["evidence"]], ["new.pdf"])
        self.assertEqual(before["status"], "ok")
        self.assertTrue(before["evidence"][0]["evidence_eligible"])
        self.assertEqual(before["evidence"][0]["effective_to"], "2026-01-01")

    def test_unpublished_version_is_excluded_by_as_of(self) -> None:
        # as_of 2025-11-01: the new rule is neither in force nor provably published.
        result = self._search("2025-10-15", as_of_date="2025-11-01", effective_filter=False)
        self.assertEqual([item["source_name"] for item in result["evidence"]], ["old.pdf"])
        self.assertIn("new.pdf", result["excluded_documents"])

    def test_disabled_filter_lets_wrong_version_through(self) -> None:
        result = self._search("2025-12-15", as_of_date="2026-01-14", effective_filter=False)
        self.assertEqual(sorted(item["source_name"] for item in result["evidence"]), ["new.pdf", "old.pdf"])
        self.assertTrue(all(item["evidence_eligible"] for item in result["evidence"]))

    def test_pending_metadata_is_never_eligible(self) -> None:
        result = self._search("2025-12-15", as_of_date="2026-01-14", status="pending_user_confirmation")
        self.assertEqual(result["status"], "unverified_candidates")
        self.assertFalse(result["evidence"][0]["evidence_eligible"])

    def test_period_before_coverage_is_never_eligible(self) -> None:
        result = self._search("2024-03-01", as_of_date="2026-01-14", effective_filter=False)
        self.assertTrue(result["evidence"])
        self.assertTrue(all(not item["evidence_eligible"] for item in result["evidence"]))
        self.assertIn("POLICY_PERIOD_NOT_COVERED", result["limitations"][0])


if __name__ == "__main__":
    unittest.main()
