from typing import List, Optional
from pathlib import Path
import sqlite3
import json
import numpy as np
import faiss

from app.rag.embeddings import get_embedding


class FaissVectorStore:
    """FAISS-backed vector store with SQLite metadata.

    Notes:
    - Stores vectors in a FAISS IndexIDMap (persisted to disk).
    - Stores document metadata and text in a SQLite DB under `persist_dir`.
    - Uses GPU index when available (faiss GPU index created from CPU index).
    """

    def __init__(self, persist_dir: Optional[str] = None):
        default_dir = Path(__file__).resolve().parents[2] / "faiss"
        self.persist_dir = Path(persist_dir) if persist_dir else default_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self.persist_dir / "meta.sqlite"
        self._index_path = self.persist_dir / "index.faiss"

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._ensure_tables()

        # load embedding dim by probing embedding model
        sample = get_embedding("test")
        self.dim = len(sample)

        # CPU index (IndexIDMap to accept explicit ids)
        self._cpu_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self._gpu_index = None
        self._gpu_res = None

        if self._index_path.exists():
            try:
                cpu_idx = faiss.read_index(str(self._index_path))
                # keep CPU index
                self._cpu_index = cpu_idx
            except Exception:
                pass

        # try create GPU index
        try:
            self._gpu_res = faiss.StandardGpuResources()
            self._gpu_index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._cpu_index)
        except Exception:
            self._gpu_index = None

        self._version = 0

    def _ensure_tables(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                chunk_id INTEGER,
                stored_filename TEXT,
                text TEXT
            )
            """
        )
        self._conn.commit()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        # L2-normalize for cosine similarity using inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add(self, texts: List[str], source: str, stored_filename: Optional[str] = None):
        if not texts:
            return

        embeddings = [get_embedding(t) for t in texts]
        vecs = np.array(embeddings, dtype="float32")
        vecs = self._normalize(vecs)

        cur = self._conn.cursor()
        ids = []
        for i, text in enumerate(texts):
            cur.execute(
                "INSERT INTO documents (source, chunk_id, stored_filename, text) VALUES (?, ?, ?, ?)",
                (source, i, stored_filename, text),
            )
            ids.append(cur.lastrowid)
        self._conn.commit()

        ids_arr = np.array(ids, dtype="int64")
        try:
            # add to CPU index (with ids)
            self._cpu_index.add_with_ids(vecs, ids_arr)
            # persist cpu index
            faiss.write_index(self._cpu_index, str(self._index_path))
            # rebuild gpu index
            if self._gpu_res is not None:
                try:
                    self._gpu_index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._cpu_index)
                except Exception:
                    self._gpu_index = None
        except Exception:
            # rollback DB rows if index add fails
            cur = self._conn.cursor()
            for _id in ids:
                cur.execute("DELETE FROM documents WHERE id=?", (_id,))
            self._conn.commit()
            raise

        self._version += 1

    def query(self, query: str, top_k: int = 5):
        q_vec = np.array([get_embedding(query)], dtype="float32")
        q_vec = self._normalize(q_vec)

        if self._gpu_index is not None:
            D, I = self._gpu_index.search(q_vec, top_k)
        else:
            D, I = self._cpu_index.search(q_vec, top_k)

        ids = I[0].tolist()
        docs = []
        cur = self._conn.cursor()
        for idx, dist in zip(ids, D[0].tolist()):
            if idx < 0:
                continue
            cur.execute("SELECT source, chunk_id, stored_filename, text FROM documents WHERE id=?", (int(idx),))
            row = cur.fetchone()
            if not row:
                continue
            source, chunk_id, stored_filename, text = row
            meta = {"source": source, "chunk_id": chunk_id}
            if stored_filename:
                meta["stored_filename"] = stored_filename
            docs.append({"text": text, **meta})

        return docs

    def list_documents(self):
        cur = self._conn.cursor()
        cur.execute("SELECT source, COUNT(*) FROM documents GROUP BY source")
        rows = cur.fetchall()
        return [{"source": r[0], "chunks": r[1]} for r in rows]

    def delete_by_source(self, source: str) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT id FROM documents WHERE source=?", (source,))
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return 0

        ids_arr = np.array(ids, dtype="int64")
        try:
            # remove from index
            self._cpu_index.remove_ids(ids_arr)
            faiss.write_index(self._cpu_index, str(self._index_path))
            if self._gpu_res is not None:
                try:
                    self._gpu_index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._cpu_index)
                except Exception:
                    self._gpu_index = None
        except Exception:
            pass

        cur.execute("DELETE FROM documents WHERE source=?", (source,))
        self._conn.commit()
        self._version += 1
        return len(ids)

    def get_stored_filenames_by_source(self, source: str) -> List[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT DISTINCT stored_filename FROM documents WHERE source=? AND stored_filename IS NOT NULL", (source,))
        return [r[0] for r in cur.fetchall() if r[0]]

    def get_all_stored_filenames(self) -> List[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT DISTINCT stored_filename FROM documents WHERE stored_filename IS NOT NULL")
        return [r[0] for r in cur.fetchall() if r[0]]

    def clear_documents(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT id FROM documents")
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return 0

        try:
            ids_arr = np.array(ids, dtype="int64")
            self._cpu_index.remove_ids(ids_arr)
            faiss.write_index(self._cpu_index, str(self._index_path))
            if self._gpu_res is not None:
                try:
                    self._gpu_index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._cpu_index)
                except Exception:
                    self._gpu_index = None
        except Exception:
            pass

        cur.execute("DELETE FROM documents")
        self._conn.commit()
        self._version += 1
        return len(ids)

    def count_documents(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        return cur.fetchone()[0]

    def get_all_documents(self):
        """Return documents and metadatas in a tuple similar to Chroma's get(include=["documents","metadatas"]) output."""
        cur = self._conn.cursor()
        cur.execute("SELECT text, source, chunk_id, stored_filename FROM documents ORDER BY id ASC")
        rows = cur.fetchall()
        documents = [r[0] for r in rows]
        metadatas = []
        for r in rows:
            meta = {"source": r[1], "chunk_id": r[2]}
            if r[3]:
                meta["stored_filename"] = r[3]
            metadatas.append(meta)
        return documents, metadatas
