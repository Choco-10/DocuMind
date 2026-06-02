from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from app.rag.pipeline import query_rag, retriever, memory, _format_context
from app.llm.llm_model import stream_answer, sanitize_generated_text

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = Field(default=5, ge=1, le=20)

def _recent_history(messages, max_messages: int = 6):
    return messages[-max_messages:]

@router.post("")
def chat(req: ChatRequest):
    return query_rag(
        question=req.question,
        session_id=req.session_id,
        top_k=req.top_k
    )

@router.post("/stream")
def chat_stream(req: ChatRequest):
    def event_generator():
        history = memory.get_history(req.session_id)
        history_messages = _recent_history(history)
        candidate_k = max(100, req.top_k * 20)
        context_k = max(req.top_k, 8)
        retrieved = retriever.retrieve(req.question, top_k=context_k, candidate_k=candidate_k, source_limit=2)

        if not retrieved:
            answer = "No relevant documents found."
            yield f"data: {json.dumps({'token': answer})}\n\n"
            memory.add_message(req.session_id, "user", req.question)
            memory.add_message(req.session_id, "assistant", answer)
            yield "data: [DONE]\n\n"
            return

        context = _format_context(retrieved)
        sources = [{"source": r.get("source", "unknown"), "chunk_id": r.get("chunk_id", -1)} for r in retrieved]

        parts = []
        for token in stream_answer(req.question, context, history_messages=history_messages):
            parts.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        final_answer = sanitize_generated_text("".join(parts))
        memory.add_message(req.session_id, "user", req.question)
        memory.add_message(req.session_id, "assistant", final_answer)

        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")