"""
api.py
------
FastAPI backend that wires together:
    ingestor.py -> embedder.py   (on /upload)
    retriever.py -> generator.py (on /ask)

Run with:
    uvicorn backend.api:app --reload
"""

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.ingestor import process_pdf
from backend.embedder import embed_and_store
from backend.generator import generate_answer

app = FastAPI(title="RAG Document Q&A API")

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DOCUMENTS: dict[str, str] = {}


class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None
    k: int = 5


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accepts a PDF, runs it through ingestor -> embedder, stores chunks in ChromaDB."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = f"{uuid.uuid4().hex[:8]}_{file.filename.replace('.pdf', '')}"
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = process_pdf(save_path)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    stored_count = embed_and_store(chunks, doc_id)
    DOCUMENTS[doc_id] = file.filename

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_stored": stored_count,
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Answers a question, optionally scoped to a single doc_id."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = generate_answer(request.question, doc_id=request.doc_id, k=request.k)
    return result


@app.get("/documents")
async def list_documents():
    """Returns all uploaded documents."""
    return [{"doc_id": doc_id, "filename": name} for doc_id, name in DOCUMENTS.items()]


@app.get("/")
async def root():
    return {"status": "RAG Document Q&A API is running"}
