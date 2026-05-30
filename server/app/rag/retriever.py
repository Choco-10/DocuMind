from rank_bm25 import BM25Okapi
from typing import Any
from app.rag.vectorstore import FaissVectorStore
import numpy as np

class HybridRetriever:
    def __init__(self, vector_store: Any):
        self.vector_store = vector_store
        self.bm25_index = None
        self.texts = []
        self.metadatas = []
        self._last_version = -1

    def _build_bm25(self):
        # Only rebuild when vector store version changes
        if getattr(self.vector_store, "_version", -1) == self._last_version and self.bm25_index is not None:
            return

        # ask the vector store for all documents and metadatas
        try:
            docs, metas = self.vector_store.get_all_documents()
        except Exception:
            # fallback: try older collection-style attribute if present
            data = getattr(self.vector_store, "collection", None)
            if data:
                data = data.get(include=["documents", "metadatas"]) if hasattr(data, "get") else {}
                docs = data.get("documents") or []
                metas = data.get("metadatas") or []
            else:
                docs = []
                metas = []

        if not docs:
            self.bm25_index = None
            self.texts = []
            self.metadatas = []
            self._last_version = getattr(self.vector_store, "_version", -1)
            return

        # flatten if nested
        if isinstance(docs[0], list):
            flat_docs = [d for sub in docs for d in sub if isinstance(d, str)]
        else:
            flat_docs = [d for d in docs if isinstance(d, str)]

        self.texts = flat_docs
        self.metadatas = metas if metas else [{} for _ in self.texts]

        tokenized = [t.split() for t in self.texts]
        if tokenized:
            self.bm25_index = BM25Okapi(tokenized)
        else:
            self.bm25_index = None

        self._last_version = getattr(self.vector_store, "_version", -1)

    def retrieve(self, query: str, top_k=5):
        # ensure indexes reflect latest data
        self._build_bm25()

        if not self.bm25_index:
            return []

        query_tokens = query.split()
        bm25_scores = np.array(self.bm25_index.get_scores(query_tokens))
        if bm25_scores.size == 0:
            top_bm25_idx = []
        else:
            top_bm25_idx = np.argsort(bm25_scores)[-top_k:][::-1]

        semantic_results = self.vector_store.query(query, top_k=top_k)

        hybrid_results = []
        seen = set()

        for i in top_bm25_idx:
            i = int(i)
            if i < 0 or i >= len(self.texts):
                continue
            text = self.texts[i]
            meta = self.metadatas[i] if i < len(self.metadatas) and isinstance(self.metadatas[i], dict) else {}
            source = meta.get("source", "unknown")
            chunk_id = meta.get("chunk_id", -1)

            key = (text, source, chunk_id)
            if key in seen:
                continue
            seen.add(key)

            hybrid_results.append({
                "text": text,
                "source": source,
                "chunk_id": chunk_id,
                "score": float(bm25_scores[i]) if bm25_scores.size > 0 else 0.0,
                "retrieval_type": "bm25"
            })

        for r in semantic_results:
            text = r.get("text", "")
            source = r.get("source", "unknown")
            chunk_id = r.get("chunk_id", -1)

            key = (text, source, chunk_id)
            if key in seen:
                continue
            seen.add(key)

            hybrid_results.append({
                "text": text,
                "source": source,
                "chunk_id": chunk_id,
                "retrieval_type": "semantic"
            })

        return hybrid_results[:top_k]
