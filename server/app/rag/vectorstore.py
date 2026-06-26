from typing import List, Optional, Union
from pathlib import Path
import sqlite3
import numpy as np
import faiss

from app.rag.embeddings import get_embedding


class FaissVectorStore:
    def __init__(self, persist_dir: Optional[str] = None):
        default_dir = Path(__file__).resolve().parents[2] / "faiss"
        self.persist_dir = Path(persist_dir) if persist_dir else default_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self.persist_dir / "meta.sqlite"
        self._index_path = self.persist_dir / "index.faiss"

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._ensure_tables()

        sample = get_embedding("test")
        self.dim = len(sample)

        self._cpu_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self._gpu_index = None
        self._gpu_res = None

        if self._index_path.exists():
            try:
                cpu_idx = faiss.read_index(str(self._index_path))
                self._cpu_index = cpu_idx
            except Exception:
                pass

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
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add(self, texts: List[str], source: Union[str, List[str]], stored_filename: Optional[str] = None):
        if not texts:
            return

        embeddings = [get_embedding(t) for t in texts]
        vecs = np.array(embeddings, dtype="float32")
        vecs = self._normalize(vecs)

        cur = self._conn.cursor()
        ids = []
        for i, text in enumerate(texts):
            src_val = source[i] if isinstance(source, list) and i < len(source) else source
            cur.execute(
                "INSERT INTO documents (source, chunk_id, stored_filename, text) VALUES (?, ?, ?, ?)",
                (src_val, i, stored_filename, text),
            )
            ids.append(cur.lastrowid)
        self._conn.commit()

        ids_arr = np.array(ids, dtype="int64")
        try:
            self._cpu_index.add_with_ids(vecs, ids_arr)
            faiss.write_index(self._cpu_index, str(self._index_path))
            if self._gpu_res is not None:
                try:
                    self._gpu_index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._cpu_index)
                except Exception:
                    self._gpu_index = None
        except Exception:
            cur = self._conn.cursor()
            for _id in ids:
                cur.execute("DELETE FROM documents WHERE id=?", (_id,))
            self._conn.commit()
            raise

        self._version += 1

    def batch_add(
        self,
        texts: List[str],
        sources: List[str],
        chunk_ids: List[int],
        stored_filenames: Optional[List[str]] = None,
        batch_size: int = 512
    ):
        if not texts:
            return

        from app.rag.embeddings import get_embeddings

        embeddings = get_embeddings(texts, batch_size=batch_size)
        vecs = np.array(embeddings, dtype="float32")
        vecs = self._normalize(vecs)

        cur = self._conn.cursor()
        ids = []

        for i, (text, source, chunk_id) in enumerate(zip(texts, sources, chunk_ids)):
            filename = stored_filenames[i] if stored_filenames else None
            cur.execute(
                "INSERT INTO documents (source, chunk_id, stored_filename, text) VALUES (?, ?, ?, ?)",
                (source, chunk_id, filename, text),
            )
            ids.append(cur.lastrowid)
        self._conn.commit()

        ids_arr = np.array(ids, dtype="int64")
        try:
            self._cpu_index.add_with_ids(vecs, ids_arr)
            faiss.write_index(self._cpu_index, str(self._index_path))

            if self._gpu_res is not None:
                try:
                    self._gpu_index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._cpu_index)
                except Exception:
                    self._gpu_index = None
        except Exception:
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
            cur.execute("SELECT id, source, chunk_id, stored_filename, text FROM documents WHERE id=?", (int(idx),))
            row = cur.fetchone()
            if not row:
                continue
            _id, source, chunk_id, stored_filename, text = row
            meta = {"id": _id, "source": source, "chunk_id": chunk_id, "score": float(dist)}
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
        cur = self._conn.cursor()
        cur.execute("SELECT id, text, source, chunk_id, stored_filename FROM documents ORDER BY id ASC")
        rows = cur.fetchall()
        ids = [r[0] for r in rows]
        documents = [r[1] for r in rows]
        metadatas = []
        for r in rows:
            meta = {"source": r[2], "chunk_id": r[3]}
            if r[4]:
                meta["stored_filename"] = r[4]
            metadatas.append(meta)
        return ids, documents, metadatas