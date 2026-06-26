from fastapi import APIRouter, UploadFile, File, HTTPException
from celery.result import AsyncResult
from app.rag.pipeline import ingest_document, vector_store
from app.loaders import load_document
from app.celery_worker import celery_app
import os
import uuid
from pathlib import Path
import re

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"}

IMAGE_MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "webp": "image/webp",
}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    *IMAGE_MIME_TYPES.values(),
}


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "document")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)

    return cleaned or "document"


def _delete_uploaded_files(file_names: list[str]) -> int:
    deleted = 0
    for name in set(file_names):
        if not name:
            continue
        path = Path(UPLOAD_DIR) / os.path.basename(name)
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted

@router.post("")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    original_name = _sanitize_filename(file.filename)

    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max allowed is {MAX_UPLOAD_MB}MB")

    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}__{original_name}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        f.write(raw)

    text = load_document(save_path)
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in document")

    task = ingest_document.delay(text, source=original_name, stored_filename=safe_name)

    return {
        "message": "File uploaded successfully",
        "task_id": task.id,
        "status": "processing"
    }

@router.get("/task/{task_id}")
def task_status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "state": task.state}
    if task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.result)
    return response

@router.get("/documents")
def list_documents():
    return {"documents": vector_store.list_documents()}


@router.delete("/documents/{source}")
def delete_document(source: str):
    stored_files = vector_store.get_stored_filenames_by_source(source)
    deleted_chunks = vector_store.delete_by_source(source)
    if deleted_chunks == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    deleted_files = _delete_uploaded_files(stored_files)

    return {
        "message": "Document removed from context",
        "source": source,
        "deleted_chunks": deleted_chunks,
        "deleted_files": deleted_files,
    }


@router.delete("/documents")
def clear_documents():
    stored_files = vector_store.get_all_stored_filenames()
    deleted_chunks = vector_store.clear_documents()
    deleted_files = _delete_uploaded_files(stored_files)

    for ext in ALLOWED_EXTENSIONS:
        for leftover in Path(UPLOAD_DIR).glob(f"*__*.{ext}"):
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
                deleted_files += 1

    return {
        "message": "All documents removed from context",
        "deleted_chunks": deleted_chunks,
        "deleted_files": deleted_files,
    }