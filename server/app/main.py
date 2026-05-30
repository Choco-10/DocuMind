from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import upload, chat
from app.celery_worker import celery_app
from app.rag.pipeline import vector_store, memory

app = FastAPI(title="RAG Server")

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    upload.router,
    prefix="/api/upload",
    tags=["Upload"]
)

app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["Chat"]
)

@app.get("/")
def health_check():
    return {"status": "RAG server running"}

@app.get("/health/live")
def health_live():
    return {"status": "alive"}

@app.get("/health/ready")
def health_ready():
    checks = {
        "redis": False,
        "celery": False,
        "faiss": False,
    }
    details = {}

    # Redis check
    try:
        checks["redis"] = bool(memory.client.ping())
        details["redis"] = "ok"
    except Exception as e:
        details["redis"] = str(e)

    # Celery check (worker responsiveness)
    try:
        ping_result = celery_app.control.inspect(timeout=1).ping()
        checks["celery"] = bool(ping_result)
        details["celery"] = "ok" if ping_result else "no active worker"
    except Exception as e:
        details["celery"] = str(e)

    # FAISS + metadata check
    try:
        doc_count = vector_store.count_documents()
        checks["faiss"] = True
        details["faiss"] = {"status": "ok", "documents_count": doc_count}
    except Exception as e:
        details["faiss"] = str(e)

    ready = all(checks.values())
    return {
        "ready": ready,
        "checks": checks,
        "details": details
    }