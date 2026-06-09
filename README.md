# DocuMind

RAG-powered document Q&A that lets you upload PDF documents and ask natural-language questions about their content. It uses hybrid search (semantic + BM25) over a FAISS vector index to retrieve relevant chunks, then generates grounded answers using a locally-running LLM.

---

## Features

- **PDF Ingestion** - Upload PDFs, text is extracted and embedded into a vector index for retrieval.
- **Hybrid Search** - Combines semantic and keyword-based retrieval for better coverage across query types.
- **Streaming Q&A** - Retrieved context is passed to a local LLM and answers are streamed token-by-token.
- **Conversation Memory** - Chat history is persisted per session for coherent multi-turn conversations.
- **Document Management** - List, delete, or clear documents from the UI.

---

## How It Works

1. **Ingest** - PDF text is extracted and split into chunks, then embedded and indexed for search.
2. **Retrieve** - A question is searched across both semantic and keyword indexes to find the most relevant chunks.
3. **Generate** - Retrieved context is passed to a local LLM which produces a grounded answer.
4. **Stream** - The answer is streamed token-by-token to the frontend in real-time.

---


## Getting Started

### Prerequisites

- **Python 3.11** (Conda recommended)
- **CUDA-capable GPU** with at least 6 GB VRAM (required for both embeddings and LLM)
- **Redis** running locally (for Celery broker + conversation memory)
- **Node.js 18+** (for the frontend)

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Choco-10/RAG_Project.git
   cd RAG_Project
   ```

2. **Create the Conda environment**
   ```bash
   cd server
   conda env create -f environment.yml
   conda activate rag-faiss
   ```

3. **Start Redis**
   ```bash
   redis-server
   ```

4. **Start the FastAPI server**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   The server will be available at `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

5. **Start the Celery worker** (in a separate terminal)
   ```bash
   cd server
   celery -A app.celery_worker:celery_app worker --loglevel=info --pool=solo
   ```

### Frontend Setup

```bash
cd client
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.
