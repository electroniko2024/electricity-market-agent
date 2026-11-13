"""Local hybrid retrieval for policy text; results remain unverified candidates.

BM25 and bge-m3 (served by Ollama) recall independently, Reciprocal Rank Fusion merges
the two rankings, and bge-reranker-v2-m3 reorders the fused pool. Vectors are stored in
a persistent Chroma collection when chromadb is importable, otherwise in a JSON file.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Protocol

TOKEN_RE = re.compile(r"[㐀-鿿]+|[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
COLLECTION_NAME = "policy_chunks"
RRF_K = 60
RECALL_DEPTH = 20
RERANK_POOL = 20
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_RERANKER_PATH = "/mnt/d/资料/bge-reranker-v2-m3"


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


Reranker = Callable[[str, list[str]], list[float]]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        raise ValueError("embedding 向量范数为 0")
    return [value / norm for value in vector]


class OllamaEmbedder:
    """bge-m3 dense embeddings through the local Ollama HTTP API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        batch_size: int = 32,
        timeout: float = 300,
    ) -> None:
        self.model = model or os.environ.get("ELLECTRIC_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.base_url = (base_url or os.environ.get("ELLECTRIC_OLLAMA_URL", DEFAULT_OLLAMA_URL)).rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.model_id = f"ollama:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            request = urllib.request.Request(
                f"{self.base_url}/api/embed",
                data=json.dumps({"model": self.model, "input": batch}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
            except (urllib.error.URLError, TimeoutError) as error:
                raise RuntimeError(
                    f"Ollama embedding 服务不可用 ({self.base_url}, model={self.model}): {error}"
                ) from None
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise RuntimeError("Ollama embedding 响应数量与输入不一致")
            vectors.extend(_normalize([float(value) for value in vector]) for vector in embeddings)
        return vectors


_RERANKER_CACHE: dict[str, "CrossEncoderReranker"] = {}


class CrossEncoderReranker:
    """bge-reranker-v2-m3 cross-encoder loaded from a local directory (no download)."""

    def __init__(self, path: str, max_length: int = 512, batch_size: int = 8) -> None:
        if not Path(path, "config.json").is_file():
            raise FileNotFoundError(f"reranker 模型目录不存在或不完整: {path}")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.path = path
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True
        ).eval()

    def __call__(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        with self._torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                pairs = [[query, passage] for passage in passages[start : start + self.batch_size]]
                batch = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                scores.extend(self.model(**batch).logits.view(-1).float().tolist())
        return scores


# CPU cross-encoder inference already uses every core; concurrent calls only thrash, and
# Chroma's PersistentClient construction is not thread-safe. Both are serialized here.
_MODEL_LOCK = threading.RLock()
_CHROMA_CLIENTS: dict[str, Any] = {}


def default_reranker() -> Reranker:
    path = os.environ.get("ELLECTRIC_RERANKER_PATH", DEFAULT_RERANKER_PATH)
    with _MODEL_LOCK:
        if path not in _RERANKER_CACHE:
            _RERANKER_CACHE[path] = CrossEncoderReranker(path)
        model = _RERANKER_CACHE[path]

    def rerank(query: str, passages: list[str]) -> list[float]:
        with _MODEL_LOCK:
            return model(query, passages)

    return rerank


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    output: list[str] = []
    for match in TOKEN_RE.findall(normalized):
        if "㐀" <= match[0] <= "鿿":
            if len(match) == 1:
                output.append(match)
            else:
                output.extend(match[index : index + 2] for index in range(len(match) - 1))
        else:
            output.append(match)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _chroma_client(path: Path):
    import chromadb

    return chromadb.PersistentClient(
        path=str(path),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def _chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def build_policy_index(
    documents_path: str | Path,
    chunks_path: str | Path,
    index_dir: str | Path,
    *,
    embedder: Embedder | None = None,
    vector_backend: str | None = None,
) -> dict[str, Any]:
    """Build BM25 plus a dense bge-m3 index over parsed policy text chunks."""
    embedder = embedder or OllamaEmbedder()
    documents_payload = json.loads(Path(documents_path).read_text(encoding="utf-8"))
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    documents = documents_payload.get("documents", [])
    documents_by_id = {document["sha256"]: document for document in documents}

    items: list[dict[str, Any]] = []
    for chunk in chunks:
        document = documents_by_id.get(chunk.get("document_id"), {})
        metadata = {
            "source_name": str(chunk.get("source_name", "")),
            "relative_path": str(chunk.get("relative_path", "")),
            "source_sha256": str(chunk.get("document_id", "")),
            "locator": str(chunk.get("locator", "")),
            "ordinal": int(chunk.get("ordinal", 0)),
            "document_status": str(document.get("document_status", "unknown")),
            "metadata_pending": ",".join(document.get("metadata_pending", [])),
            "evidence_status": str(chunk.get("evidence_status", "source_text_unverified")),
            "retrieval_eligible": False,
        }
        page = chunk.get("page")
        if isinstance(page, int):
            metadata["page"] = page
        items.append(
            {
                "id": str(chunk["chunk_id"]),
                "text": str(chunk.get("text", "")),
                "tokens": _tokens(str(chunk.get("text", ""))),
                "metadata": metadata,
            }
        )

    if not items:
        raise ValueError("政策正文没有可索引的文本块")

    source_hashes = sorted(document.get("sha256", "") for document in documents)
    fingerprint_input = "\n".join(
        [embedder.model_id] + source_hashes + sorted(item["id"] for item in items)
    )
    fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
    root = Path(index_dir)
    vectors = embedder.embed([item["text"] for item in items])
    dimension = len(vectors[0])

    backend = vector_backend or ("chroma" if _chroma_available() else "json")
    chroma_relative = f"chroma/{fingerprint}"
    if backend == "chroma":
        client = _chroma_client(root / chroma_relative)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:  # collection absent on first build
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "corpus_fingerprint": fingerprint},
        )
        batch_size = 128
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            collection.add(
                ids=[item["id"] for item in batch],
                embeddings=vectors[start : start + batch_size],
                documents=[item["text"] for item in batch],
                metadatas=[item["metadata"] for item in batch],
            )
        (root / "policy_vectors.json").unlink(missing_ok=True)
    elif backend == "json":
        _write_json(
            root / "policy_vectors.json",
            {
                "schema_version": 2,
                "corpus_fingerprint": fingerprint,
                "embedding_model": embedder.model_id,
                "vectors": {item["id"]: vector for item, vector in zip(items, vectors, strict=True)},
            },
        )
    else:
        raise ValueError(f"未知向量后端: {backend}")

    _write_json(
        root / "policy_bm25.json",
        {"schema_version": 1, "corpus_fingerprint": fingerprint, "items": items},
    )
    manifest = {
        "schema_version": 2,
        "status": "local_candidates_unverified",
        "retrieval_methods": ["BM25", f"dense:{embedder.model_id}", "RRF", "cross_encoder_rerank"],
        "embedding_model": embedder.model_id,
        "embedding_dimension": dimension,
        "vector_backend": backend,
        "corpus_fingerprint": fingerprint,
        "source_document_count": len(documents),
        "indexed_text_chunk_count": len(items),
        "chroma_path": chroma_relative if backend == "chroma" else None,
        "policy_coverage_verified": bool(documents_payload.get("policy_coverage_verified")),
        "evidence_eligible": False,
    }
    _write_json(root / "policy_index_manifest.json", manifest)
    return manifest


def _bm25_ranking(items: list[dict[str, Any]], query_tokens: list[str]) -> list[str]:
    if not query_tokens:
        return []
    tokenized = [item["tokens"] for item in items]
    document_count = len(tokenized)
    average_length = sum(map(len, tokenized)) / document_count if document_count else 0
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    scores: list[tuple[float, str]] = []
    for item, tokens in zip(items, tokenized, strict=True):
        frequencies = Counter(tokens)
        score = 0.0
        for token in set(query_tokens):
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_frequency = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(tokens) / max(average_length, 1)
            )
            score += inverse_frequency * frequency * 2.5 / denominator
        if score:
            scores.append((score, item["id"]))
    scores.sort(key=lambda value: (-value[0], value[1]))
    return [item_id for _, item_id in scores]


def _dense_ranking(
    root: Path, manifest: dict[str, Any], query_vector: list[float], depth: int
) -> list[str]:
    if manifest.get("vector_backend") == "chroma":
        chroma_path = str(root / manifest["chroma_path"])
        with _MODEL_LOCK:
            if chroma_path not in _CHROMA_CLIENTS:
                _CHROMA_CLIENTS[chroma_path] = _chroma_client(Path(chroma_path))
            collection = _CHROMA_CLIENTS[chroma_path].get_collection(COLLECTION_NAME)
            response = collection.query(
                query_embeddings=[query_vector],
                n_results=min(depth, collection.count()),
                include=["distances"],
            )
        return list((response.get("ids") or [[]])[0])
    payload = json.loads((root / "policy_vectors.json").read_text(encoding="utf-8"))
    if payload.get("corpus_fingerprint") != manifest.get("corpus_fingerprint"):
        raise ValueError("政策向量文件与索引 manifest 指纹不匹配")
    scores = [
        (_cosine(query_vector, vector), item_id)
        for item_id, vector in payload.get("vectors", {}).items()
    ]
    scores.sort(key=lambda value: (-value[0], value[1]))
    return [item_id for _, item_id in scores[:depth]]


POLICY_COVERAGE_START = "2024-05-01"
CONFIRMED_METADATA_STATUSES = {"user_confirmed"}


def load_policy_metadata(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def document_applicability(
    meta: dict[str, Any] | None,
    *,
    policy_at: str,
    as_of_date: str | None,
    effective_filter: bool,
) -> str:
    """Classify one document for a query: applicable, not_applicable, not_published or unknown.

    Dates compare as ISO strings. A `*_not_later_than` bound proves a date has passed only when
    the bound itself is on or before the comparison date; otherwise the answer is unknown.
    """
    if meta is None:
        return "unknown"
    if as_of_date is not None:
        published = meta.get("publish_date") or meta.get("publish_not_later_than")
        if published is None:
            return "unknown"
        if published > as_of_date:
            exact_from = meta.get("effective_from")
            # Not yet in force at as_of and not provably published: treat as not published.
            if exact_from and exact_from > as_of_date:
                return "not_published"
            return "unknown"
    if not effective_filter:
        return "applicable"
    exact_from = meta.get("effective_from")
    bound_from = meta.get("effective_from_not_later_than")
    if exact_from:
        if policy_at < exact_from:
            return "not_applicable"
    elif bound_from is None or policy_at < bound_from:
        return "unknown"
    effective_to = meta.get("effective_to")
    if effective_to and policy_at >= effective_to:
        return "not_applicable"
    return "applicable"


def _version_fields(meta: dict[str, Any] | None) -> dict[str, Any]:
    if meta is None:
        return {"document_title": None, "document_number": None, "version_family": None,
                "effective_from": None, "effective_to": None}
    return {
        "document_title": meta.get("title"),
        "document_number": meta.get("document_number"),
        "version_family": meta.get("family"),
        "effective_from": meta.get("effective_from") or (
            f"<= {meta['effective_from_not_later_than']}" if meta.get("effective_from_not_later_than") else None
        ),
        "effective_to": meta.get("effective_to"),
        "official_url": meta.get("official_url"),
    }


_SEARCH_CACHE: dict[tuple, dict[str, Any]] = {}
_SEARCH_CACHE_LIMIT = 256
_RANK_CACHE: dict[tuple, tuple] = {}


def search_local_policy(
    *,
    query: str,
    policy_at: str,
    index_dir: str | Path,
    top_k: int = 5,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    as_of_date: str | None = None,
    effective_filter: bool = True,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return citation-bearing candidates; cached per file version when default models are used."""
    import copy

    root = Path(index_dir)
    cache_key = None
    if embedder is None and reranker is None:
        stamps = tuple(
            path.stat().st_mtime_ns if path.is_file() else 0
            for path in (root / "policy_index_manifest.json", Path(metadata_path) if metadata_path else root / "_none")
        )
        cache_key = (query, policy_at, str(root), top_k, as_of_date, effective_filter, str(metadata_path), stamps)
        if cache_key in _SEARCH_CACHE:
            return copy.deepcopy(_SEARCH_CACHE[cache_key])
    result = _search_local_policy(
        query=query, policy_at=policy_at, root=root, top_k=top_k, embedder=embedder,
        reranker=reranker, as_of_date=as_of_date, effective_filter=effective_filter,
        metadata=load_policy_metadata(metadata_path),
    )
    if cache_key is not None:
        if len(_SEARCH_CACHE) >= _SEARCH_CACHE_LIMIT:
            _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)))
        _SEARCH_CACHE[cache_key] = copy.deepcopy(result)
    return result


def _search_local_policy(
    *,
    query: str,
    policy_at: str,
    root: Path,
    top_k: int,
    embedder: Embedder | None,
    reranker: Reranker | None,
    as_of_date: str | None,
    effective_filter: bool,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return citation-bearing candidates, never validated legal conclusions."""
    manifest_path = root / "policy_index_manifest.json"
    bm25_path = root / "policy_bm25.json"
    if not manifest_path.is_file() or not bm25_path.is_file():
        return {
            "status": "unavailable",
            "query": query,
            "policy_at": policy_at,
            "policy_at_filter_applied": False,
            "evidence": [],
            "limitations": ["POLICY_INDEX_NOT_BUILT"],
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        return {
            "status": "unavailable",
            "query": query,
            "policy_at": policy_at,
            "policy_at_filter_applied": False,
            "evidence": [],
            "limitations": ["POLICY_INDEX_OUTDATED: 请用 build_policy_index.py 以 bge-m3 重建索引"],
        }
    corpus = json.loads(bm25_path.read_text(encoding="utf-8"))
    if corpus.get("corpus_fingerprint") != manifest.get("corpus_fingerprint"):
        raise ValueError("政策 BM25 文件与索引 manifest 指纹不匹配")
    items = corpus.get("items", [])
    if not items:
        return {
            "status": "no_candidates",
            "query": query,
            "policy_at": policy_at,
            "policy_at_filter_applied": False,
            "evidence": [],
            "limitations": ["POLICY_INDEX_EMPTY"],
        }

    embedder_is_default = embedder is None
    embedder = embedder or OllamaEmbedder()
    if embedder.model_id != manifest.get("embedding_model"):
        raise ValueError(
            f"查询 embedding 模型 {embedder.model_id} 与索引模型 {manifest.get('embedding_model')} 不一致"
        )
    top_k = min(max(top_k, 1), 20)
    meta_by_path = {doc["relative_path"]: doc for doc in (metadata or {}).get("documents", [])}
    metadata_confirmed = (metadata or {}).get("verification_status") in CONFIRMED_METADATA_STATUSES
    applicability = {
        item["id"]: document_applicability(
            meta_by_path.get(item["metadata"]["relative_path"]),
            policy_at=policy_at,
            as_of_date=as_of_date,
            effective_filter=effective_filter,
        )
        for item in items
    }
    # Version filtering happens before ranking so excluded versions cannot crowd the pool.
    allowed = {item_id for item_id, state in applicability.items() if state in {"applicable", "unknown"}}
    excluded_documents = sorted(
        {
            item["metadata"]["relative_path"]
            for item in items
            if applicability[item["id"]] in {"not_applicable", "not_published"}
        }
    )
    items_by_id = {item["id"]: item for item in items}
    # Ranking depends only on the query and the allowed set, not on the exact dates, so most
    # queries in one version period reuse it. Only default (deterministic, local) models are cached.
    rank_key = (
        (query, manifest.get("corpus_fingerprint"), frozenset(allowed))
        if embedder_is_default and reranker is None
        else None
    )
    if rank_key is not None and rank_key in _RANK_CACHE:
        bm25_ids, dense_ids, fused, pool, rerank_scores = copy.deepcopy(_RANK_CACHE[rank_key])
    else:
        bm25_ids = [item_id for item_id in _bm25_ranking(items, _tokens(query)) if item_id in allowed][:RECALL_DEPTH]
        dense_ids = [
            item_id
            for item_id in _dense_ranking(root, manifest, embedder.embed([query])[0], RECALL_DEPTH * 10)
            if item_id in allowed
        ][:RECALL_DEPTH]

        fused: dict[str, float] = {}
        for ranking in (bm25_ids, dense_ids):
            for rank, item_id in enumerate(ranking, start=1):
                fused[item_id] = fused.get(item_id, 0.0) + 1 / (RRF_K + rank)
        pool = sorted(fused, key=lambda item_id: (-fused[item_id], item_id))[:RERANK_POOL]
        active_reranker = reranker or default_reranker()
        rerank_scores = dict(zip(pool, active_reranker(query, [items_by_id[item_id]["text"] for item_id in pool]), strict=True)) if pool else {}
        if rank_key is not None:
            with _MODEL_LOCK:
                if len(_RANK_CACHE) >= _SEARCH_CACHE_LIMIT:
                    _RANK_CACHE.pop(next(iter(_RANK_CACHE)))
                _RANK_CACHE[rank_key] = copy.deepcopy((bm25_ids, dense_ids, fused, pool, rerank_scores))
    # Candidates whose version provably applies come first; unknown-applicability text follows.
    ordered_ids = sorted(
        pool,
        key=lambda item_id: (applicability[item_id] != "applicable", -rerank_scores[item_id], item_id),
    )[:top_k]

    bm25_rank = {item_id: rank for rank, item_id in enumerate(bm25_ids, start=1)}
    dense_rank = {item_id: rank for rank, item_id in enumerate(dense_ids, start=1)}
    evidence = []
    for item_id in ordered_ids:
        item = items_by_id[item_id]
        meta = meta_by_path.get(item["metadata"]["relative_path"])
        eligible = metadata_confirmed and applicability[item_id] == "applicable" and policy_at >= POLICY_COVERAGE_START
        evidence.append(
            {
                "evidence_id": f"pol_{item_id[:16]}",
                **_version_fields(meta),
                "applicability_on_policy_at": applicability[item_id],
                "source_name": item["metadata"]["source_name"],
                "source_sha256": item["metadata"]["source_sha256"],
                "relative_path": item["metadata"]["relative_path"],
                "locator": item["metadata"]["locator"],
                "page": item["metadata"].get("page"),
                "excerpt": item["text"][:1600],
                "bm25_rank": bm25_rank.get(item_id),
                "dense_rank": dense_rank.get(item_id),
                "rrf_score": round(fused[item_id], 8),
                "rerank_score": round(rerank_scores[item_id], 6),
                "metadata_pending": [] if metadata_confirmed and meta else (
                    item["metadata"]["metadata_pending"].split(",") if item["metadata"]["metadata_pending"] else []
                ),
                "evidence_eligible": eligible,
            }
        )
    eligible_count = sum(1 for item in evidence if item["evidence_eligible"])
    limitations = ["official URLs were not looked up"]
    if not metadata_confirmed:
        limitations.insert(0, "POLICY_METADATA_PENDING_USER_CONFIRMATION: no candidate is eligible as policy evidence")
    if policy_at < POLICY_COVERAGE_START:
        limitations.insert(0, f"POLICY_PERIOD_NOT_COVERED: corpus coverage starts {POLICY_COVERAGE_START}")
    if any(item["applicability_on_policy_at"] == "unknown" for item in evidence):
        limitations.append("some candidates come from documents whose publish or effective dates are unknown; they are not eligible")
    if not effective_filter:
        limitations.append("EFFECTIVE_INTERVAL_FILTER_DISABLED: candidates may come from versions not in force on policy_at")
    return {
        "status": "ok" if eligible_count else ("unverified_candidates" if evidence else "no_candidates"),
        "query": query,
        "policy_at": policy_at,
        "as_of_date": as_of_date,
        "policy_at_filter_applied": effective_filter,
        "publish_filter_applied": as_of_date is not None,
        "metadata_verification_status": (metadata or {}).get("verification_status", "missing"),
        "excluded_documents": excluded_documents,
        "eligible_evidence_count": eligible_count,
        "evidence_eligible": eligible_count > 0,
        "retrieval_methods": manifest["retrieval_methods"],
        "embedding_model": manifest["embedding_model"],
        "evidence": evidence,
        "limitations": limitations,
    }
