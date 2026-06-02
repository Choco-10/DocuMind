from app.rag.vectorstore import FaissVectorStore
from app.rag.retriever import HybridRetriever
from app.memory.redis import RedisMemory
from app.llm.llm_model import generate_answer
from app.utils.chunking import semantic_chunk_text
from app.celery_worker import celery_app

vector_store = FaissVectorStore(persist_dir="faiss")
retriever = HybridRetriever(vector_store)
memory = RedisMemory()

def _recent_history(messages, max_messages: int = 6):
    return messages[-max_messages:]


def _format_context(retrieved_docs):
    lines = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.get("source", "unknown")
        chunk_id = doc.get("chunk_id", -1)
        retrieval_type = doc.get("retrieval_type", "semantic")
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[Doc {idx} | source={source} | chunk={chunk_id} | type={retrieval_type}] {text}")
    return "\n\n".join(lines)

@celery_app.task
def ingest_document(text: str, source: str, stored_filename: str | None = None):
    chunks = semantic_chunk_text(text)
    vector_store.add(chunks, source, stored_filename=stored_filename)
    return {"source": source, "chunks": len(chunks)}

def query_rag(question: str, session_id: str, top_k: int = 5):
    history = memory.get_history(session_id)
    history_messages = _recent_history(history)
    candidate_k = max(100, top_k * 20)
    context_k = max(top_k, 8)
    retrieved = retriever.retrieve(question, top_k=context_k, candidate_k=candidate_k, source_limit=2)

    if not retrieved:
        answer = "No relevant documents found."
        sources = []
    else:
        context = _format_context(retrieved)
        answer = generate_answer(question, context, history_messages=history_messages)
        sources = [{"source": r.get("source", "unknown"), "chunk_id": r.get("chunk_id", -1)} for r in retrieved]

    memory.add_message(session_id, "user", question)
    memory.add_message(session_id, "assistant", answer)
    return {"answer": answer, "sources": sources}