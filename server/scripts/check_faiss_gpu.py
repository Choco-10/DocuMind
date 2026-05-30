"""Check FAISS GPU availability and verify FaissVectorStore uses GPU.

Run this inside your Windows env after installing FAISS.
"""
from pathlib import Path
import sys
import traceback


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

def run_check():
    print("Checking FAISS and GPU availability...")
    try:
        import faiss
        import numpy as np
        print("faiss version:", getattr(faiss, '__version__', 'unknown'))
    except Exception as e:
        print("Failed to import faiss:", e)
        traceback.print_exc()
        return 2

    # Try GPU resources
    try:
        res = faiss.StandardGpuResources()
        print("StandardGpuResources created.")
        dim = 16
        cpu_index = faiss.IndexFlatIP(dim)
        try:
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print("Index successfully transferred to GPU.")
            xb = np.random.random((10, dim)).astype('float32')
            faiss.normalize_L2(xb)
            gpu_index.add(xb)
            D, I = gpu_index.search(xb[:1], 5)
            print("GPU search OK. ids:", I[0].tolist(), "dists:", D[0].tolist())
        except Exception as e:
            print("Failed to create or use GPU index:", e)
            traceback.print_exc()
            print("Falling back: attempting CPU index test...")
            cpu_index = faiss.IndexFlatIP(dim)
            xb = np.random.random((10, dim)).astype('float32')
            faiss.normalize_L2(xb)
            cpu_index.add(xb)
            D, I = cpu_index.search(xb[:1], 5)
            print("CPU search OK. ids:", I[0].tolist(), "dists:", D[0].tolist())
    except Exception as e:
        print("GPU resources creation failed:", e)
        traceback.print_exc()
        print("Try installing a matching CUDA toolkit or use faiss-cpu if no GPU available.")

    # Check our FaissVectorStore
    try:
        from app.rag.vectorstore import FaissVectorStore
        print("Found FaissVectorStore in project.")
        store = FaissVectorStore(persist_dir='faiss_check')
        gpu_present = bool(getattr(store, '_gpu_index', None))
        print('FaissVectorStore _gpu_index present:', gpu_present)
        print('Persist dir:', store.persist_dir)
        print('Document count (initial):', store.count_documents())
        # quick roundtrip
        store.clear_documents()
        store.add(['hello from check'], source='check_doc')
        print('Document count (after add):', store.count_documents())
        res = store.query('hello', top_k=1)
        print('Query result sample:', res)
    except Exception as e:
        print('Failed to instantiate FaissVectorStore or run operations:', e)
        traceback.print_exc()
        return 3

    print('FAISS GPU check complete.')
    return 0


if __name__ == '__main__':
    code = run_check()
    sys.exit(code)
