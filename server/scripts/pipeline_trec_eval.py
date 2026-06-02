"""TREC-style evaluation using the app's pipeline.

The script stays fully under `server/`:
- Downloads the TREC-COVID dataset into `server/data/trec-covid/` if needed.
- Reuses a persistent FAISS datastore under `server/data/trec-covid/faiss/`.
- Ingests with the app's chunking and retrieves with the app's hybrid retriever.

Usage:
    conda run -n rag-faiss --no-capture-output python server/scripts/pipeline_trec_eval.py --dataset trec-covid --split test --ingest
"""
import argparse
import json
import os
import sys
import time
from typing import Dict
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from app.rag.vectorstore import FaissVectorStore
from app.rag.retriever import HybridRetriever
from app.utils.chunking import semantic_chunk_text

try:
    import pytrec_eval
except Exception:
    pytrec_eval = None


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes:02d}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def _log_stage(stage: int, total_stages: int, title: str, meaning: str) -> None:
    percent = int((stage / total_stages) * 100) if total_stages else 0
    print(f"[{percent:>3}%] {title}")
    print(f"      meaning: {meaning}")


def _server_data_dir() -> Path:
    return SERVER_ROOT / "data" / "trec-covid"


def _dataset_paths() -> tuple[Path, Path, Path]:
    data_dir = _server_data_dir()
    return (
        data_dir / "corpus.jsonl",
        data_dir / "queries.jsonl",
        data_dir / "qrels" / "test.tsv",
    )


def ensure_dataset(dataset: str, split: str) -> tuple[Path, Path, Path]:
    if dataset != "trec-covid":
        raise ValueError("This eval currently supports trec-covid only.")
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")

    corpus_path, queries_path, qrels_path = _dataset_paths()
    if corpus_path.exists() and queries_path.exists() and qrels_path.exists():
        return corpus_path, queries_path, qrels_path

    data_dir = _server_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    downloaded = Path(util.download_and_unzip(url, str(data_dir)))
    loaded_corpus = downloaded / "corpus.jsonl"
    loaded_queries = downloaded / "queries.jsonl"
    loaded_qrels = downloaded / "qrels" / f"{split}.tsv"

    if loaded_corpus.exists() and loaded_queries.exists() and loaded_qrels.exists():
        return loaded_corpus, loaded_queries, loaded_qrels

    # Some BEIR archives unpack to a nested directory. Search one level deeper.
    for candidate in downloaded.rglob("corpus.jsonl"):
        candidate_queries = candidate.parent / "queries.jsonl"
        candidate_qrels = candidate.parent / "qrels" / f"{split}.tsv"
        if candidate.exists() and candidate_queries.exists() and candidate_qrels.exists():
            return candidate, candidate_queries, candidate_qrels

    raise FileNotFoundError("Downloaded dataset is missing corpus.jsonl, queries.jsonl, or qrels/test.tsv")


def ingest_corpus(vector_store: FaissVectorStore, corpus: Dict[str, Dict[str, str]], force: bool = False):
    if vector_store.count_documents() > 0 and not force:
        print(f"Vector store already has {vector_store.count_documents()} chunks; reusing stored embeddings.")
        return

    if force and vector_store.count_documents() > 0:
        print("Clearing existing vectorstore documents...")
        vector_store.clear_documents()

    corpus_ids = list(corpus.keys())
    total = len(corpus_ids)
    ingest_started_at = time.perf_counter()
    
    batch_texts = []
    batch_sources = []
    batch_chunk_ids = []
    batch_docs_limit = 10000

    for i, cid in enumerate(corpus_ids, start=1):
        doc = corpus[cid]
        text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        if not text:
            continue
        chunks = semantic_chunk_text(text)
        if not chunks:
            continue
        
        for chunk_id, chunk_text in enumerate(chunks):
            batch_texts.append(chunk_text)
            batch_sources.append(str(cid))
            batch_chunk_ids.append(chunk_id)
            
        if len(batch_sources) >= batch_docs_limit or i == total:
            if batch_texts:
                vector_store.batch_add(
                    texts=batch_texts,
                    sources=batch_sources,
                    chunk_ids=batch_chunk_ids,
                    batch_size=512
                )
                batch_texts.clear()
                batch_sources.clear()
                batch_chunk_ids.clear()
                
            elapsed = time.perf_counter() - ingest_started_at
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = (total - i) / rate if rate > 0 else 0.0
            print(
                f"Ingested {i}/{total} corpus documents | elapsed {_format_duration(elapsed)} | eta {_format_duration(remaining)}"
            )



def evaluate(run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]):
    if pytrec_eval is None:
        raise RuntimeError("pytrec_eval not installed. Install with `pip install pytrec_eval`")

    metric_names = set()
    for k in (1, 3, 5, 10, 100, 1000):
        metric_names.add(f"ndcg_cut.{k}")
        metric_names.add(f"map_cut.{k}")
        metric_names.add(f"recall.{k}")
        metric_names.add(f"P.{k}")
    metric_names.add("recip_rank")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metric_names)
    results = evaluator.evaluate(run)
    agg = {}
    for qid, scores in results.items():
        for metric, value in scores.items():
            agg.setdefault(metric, []).append(value)

    avg = {metric: sum(values) / len(values) for metric, values in agg.items() if values}
    named = {}
    for k in (1, 3, 5, 10, 100, 1000):
        named[f"NDCG@{k}"] = avg.get(f"ndcg_cut_{k}", 0.0)
        named[f"MAP@{k}"] = avg.get(f"map_cut_{k}", 0.0)
        named[f"Recall@{k}"] = avg.get(f"recall_{k}", 0.0)
        named[f"P@{k}"] = avg.get(f"P_{k}", 0.0)
    named["MRR"] = avg.get("recip_rank", 0.0)
    named["raw"] = avg
    return named


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="trec-covid")
    p.add_argument("--split", default="test")
    p.add_argument("--top_k", type=int, default=1000)
    p.add_argument("--ingest", action="store_true", help="Force ingest the corpus through the pipeline")
    args = p.parse_args()

    corpus_path, queries_path, qrels_path = ensure_dataset(args.dataset, args.split)

    # initialize a reusable server-local vector store
    vector_store = FaissVectorStore(persist_dir=str(SERVER_ROOT / "data" / args.dataset / "faiss"))
    retriever = HybridRetriever(vector_store)

    print("TREC pipeline eval timeline")
    total_stages = 6 if args.ingest else 5
    stage = 1

    _log_stage(stage, total_stages, "Dataset check", "Look for cached TREC-COVID files in server/data and download them only if they are missing.")
    stage += 1
    print(f"Dataset source: {corpus_path.parent}")
    print(f"Estimated dataset setup time: {_format_duration(0)} if cached, or several minutes if download is needed.")

    if args.ingest:
        _log_stage(stage, total_stages, "Corpus ingest", "Chunk each corpus document with the app's semantic chunker and write embeddings into the reusable FAISS store.")
        print("Starting corpus ingest via pipeline chunking + vectorstore...")
        print("Estimated ingest time: tens of minutes on first run, since every document must be embedded and stored.")
        corpus, _, _ = GenericDataLoader(str(corpus_path.parent)).load(split=args.split)
        ingest_corpus(vector_store, corpus, force=True)
        print("Ingest complete.")
        stage += 1

    _log_stage(stage, total_stages, "Load qrels and queries", "Read the ground-truth labels and query text used for scoring retrieval quality.")
    corpus, queries, qrels = GenericDataLoader(str(corpus_path.parent)).load(split=args.split)
    if vector_store.count_documents() == 0:
        stage += 1
        _log_stage(stage, total_stages, "Warm FAISS store", "Reuse the existing FAISS datastore if present; otherwise build it once for future runs.")
        print("Vector store is empty; ingesting once for reuse.")
        print("Estimated warm-up time: same as initial ingest, then much faster on later runs.")
        ingest_corpus(vector_store, corpus, force=False)

    stage += 1
    _log_stage(stage, total_stages, "Run retrieval", "For each query, combine BM25 and dense FAISS results with the app's hybrid retriever.")
    query_started_at = time.perf_counter()

    run = {}
    query_items = list(queries.items())
    total_queries = len(query_items)
    for query_number, (qid, qtext) in enumerate(query_items, start=1):
        retrieved = retriever.retrieve(qtext, top_k=args.top_k, candidate_k=max(1000, args.top_k), source_limit=2)
        # map retrieved chunks to document ids (source)
        doc_scores = {}
        for rank, item in enumerate(retrieved, start=1):
            docid = str(item.get("source") or "")
            # score: use hybrid score but ensure it's a float
            score = float(item.get("score", 0.0))
            if not docid:
                continue
            # keep max score if duplicate sources appear
            doc_scores[docid] = max(doc_scores.get(docid, 0.0), score)

        # ensure there is at least an entry so evaluator can compute per-query metrics
        run[qid] = doc_scores

        if query_number % 10 == 0 or query_number == total_queries:
            elapsed = time.perf_counter() - query_started_at
            rate = query_number / elapsed if elapsed > 0 else 0.0
            remaining = (total_queries - query_number) / rate if rate > 0 else 0.0
            print(
                f"Retrieval progress: {query_number}/{total_queries} queries | elapsed {_format_duration(elapsed)} | eta {_format_duration(remaining)}"
            )

    stage += 1
    _log_stage(stage, total_stages, "Score against qrels", "Compare the retrieved document IDs against the relevance judgments and compute standard IR metrics.")
    print("Estimated scoring time: seconds to a few minutes, depending on the metric set and query count.")
    print("Evaluating retrieval with pytrec_eval...")
    try:
        metrics = evaluate(run, qrels)
    except Exception as e:
        print("Evaluation failed:", e)
        metrics = {}

    stage += 1
    _log_stage(stage, total_stages, "Save metrics", "Write the aggregated scores to a JSON file under server/data for later comparison.")
    # write out results
    out_dir = SERVER_ROOT / "data" / args.dataset / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.split}_pipeline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "dataset": args.dataset, "split": args.split}, fh, indent=2)

    print("Done. Metrics written to:", out_path)


if __name__ == "__main__":
    main()
