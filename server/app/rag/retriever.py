from rank_bm25 import BM25Okapi
from typing import Any
from app.rag.vectorstore import FaissVectorStore
import numpy as np
import re

class HybridRetriever:
    def __init__(self, vector_store: Any):
        self.vector_store = vector_store
        self.bm25_index = None
        self.texts = []
        self.metadatas = []
        self.ids = []
        self._last_version = -1

    def _tokenize(self, text: str):
        return re.findall(r"\w+", (text or "").lower())

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        scores = scores.astype(float)
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))
        if max_score <= min_score:
            return np.zeros_like(scores, dtype=float)
        return (scores - min_score) / (max_score - min_score)

    def _build_bm25(self):
        # Only rebuild when vector store version changes
        if getattr(self.vector_store, "_version", -1) == self._last_version and self.bm25_index is not None:
            return

        # ask the vector store for all documents and metadatas
        try:
            docs_result = self.vector_store.get_all_documents()
        except Exception:
            # fallback: try older collection-style attribute if present
            data = getattr(self.vector_store, "collection", None)
            if data:
                data = data.get(include=["documents", "metadatas"]) if hasattr(data, "get") else {}
                docs = data.get("documents") or []
                metas = data.get("metadatas") or []
                ids = list(range(len(docs)))
            else:
                docs = []
                metas = []
                ids = []
        else:
            if len(docs_result) == 3:
                ids, docs, metas = docs_result
            else:
                docs, metas = docs_result
                ids = list(range(len(docs)))

        if not docs:
            self.bm25_index = None
            self.texts = []
            self.metadatas = []
            self.ids = []
            self._last_version = getattr(self.vector_store, "_version", -1)
            return

        # flatten if nested
        if isinstance(docs[0], list):
            flat_docs = [d for sub in docs for d in sub if isinstance(d, str)]
        else:
            flat_docs = [d for d in docs if isinstance(d, str)]

        self.texts = flat_docs
        self.metadatas = metas if metas else [{} for _ in self.texts]
        self.ids = list(ids) if ids else list(range(len(self.texts)))

        tokenized = [self._tokenize(t) for t in self.texts]
        if tokenized:
            self.bm25_index = BM25Okapi(tokenized)
        else:
            self.bm25_index = None

        self._last_version = getattr(self.vector_store, "_version", -1)

    def retrieve(self, query: str, top_k=5, candidate_k: int | None = None, source_limit: int = 2):
        # ensure indexes reflect latest data
        self._build_bm25()

        if not self.bm25_index:
            return []

        candidate_k = max(top_k * 10, 100) if candidate_k is None else max(top_k, candidate_k)

        query_tokens = self._tokenize(query)
        bm25_scores = np.array(self.bm25_index.get_scores(query_tokens), dtype=float)
        if bm25_scores.size == 0:
            top_bm25_idx = []
        else:
            top_bm25_idx = np.argsort(bm25_scores)[-candidate_k:][::-1]

        semantic_results = self.vector_store.query(query, top_k=candidate_k)

        # Collect candidates by stable id so lexical and semantic signals merge.
        candidates = {}

        bm25_norm = self._normalize_scores(bm25_scores)
        for i in top_bm25_idx:
            i = int(i)
            if i < 0 or i >= len(self.texts):
                continue
            doc_id = self.ids[i] if i < len(self.ids) else i
            meta = self.metadatas[i] if i < len(self.metadatas) and isinstance(self.metadatas[i], dict) else {}
            candidates[doc_id] = {
                "text": self.texts[i],
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", -1),
                "bm25_score": float(bm25_norm[i]) if bm25_norm.size > 0 else 0.0,
                "semantic_score": 0.0,
                "retrieval_type": "bm25",
            }

        semantic_raw_scores = np.array([float(r.get("score", 0.0)) for r in semantic_results], dtype=float)
        semantic_norm = self._normalize_scores(semantic_raw_scores)
        for idx, r in enumerate(semantic_results):
            text = r.get("text", "")
            source = r.get("source", "unknown")
            chunk_id = r.get("chunk_id", -1)
            doc_id = r.get("id", idx)

            entry = candidates.get(doc_id)
            if entry is None:
                entry = {
                    "text": text,
                    "source": source,
                    "chunk_id": chunk_id,
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                    "retrieval_type": "semantic",
                }
                candidates[doc_id] = entry

            entry["text"] = text or entry["text"]
            entry["source"] = source or entry["source"]
            entry["chunk_id"] = chunk_id if chunk_id is not None else entry["chunk_id"]
            entry["semantic_score"] = max(entry["semantic_score"], float(semantic_norm[idx]) if semantic_norm.size > idx else 0.0)
            if entry["bm25_score"] > 0:
                entry["retrieval_type"] = "hybrid"

        scored = []
        for entry in candidates.values():
            hybrid_score = 0.65 * entry.get("semantic_score", 0.0) + 0.35 * entry.get("bm25_score", 0.0)
            entry["score"] = float(hybrid_score)
            scored.append(entry)

        scored.sort(key=lambda item: item["score"], reverse=True)

        # Keep diversity across sources so one long document doesn't crowd out all evidence.
        diversified = []
        per_source_counts = {}
        for item in scored:
            source = item.get("source", "unknown")
            count = per_source_counts.get(source, 0)
            if count >= source_limit:
                continue
            per_source_counts[source] = count + 1
            diversified.append(item)
            if len(diversified) >= top_k:
                break

        if len(diversified) < top_k:
            seen = {(item.get("source"), item.get("chunk_id")) for item in diversified}
            for item in scored:
                key = (item.get("source"), item.get("chunk_id"))
                if key in seen:
                    continue
                diversified.append(item)
                seen.add(key)
                if len(diversified) >= top_k:
                    break

        return diversified[:top_k]
