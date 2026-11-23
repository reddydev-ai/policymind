"""FastAPI surface for PolicyMind.
Endpoints:
  GET  /health            -> liveness
  POST /ingest            -> upload a PDF, ingest it
  POST /ask               -> ask a question, get a cited answer
"""
from __future__ import annotations
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from .config import get_settings
from .ingestion import ingest_pdf
from .security import require_api_key
from .rag import answer_question, Answer

settings = get_settings()
app = FastAPI(title="PolicyMind", version="1.0.0",
              description="Enterprise policy RAG assistant with hybrid search, reranking and cited answers.")


class AskRequest(BaseModel):
    question: str


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.chat_model, "collection": settings.collection}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), _auth: None = Depends(require_api_key)) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    # Bound size, validate file signature, and do not trust client-provided paths.
    MAX_UPLOAD_BYTES = 15 * 1024 * 1024
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 15 MiB limit.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid PDF header.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        count = ingest_pdf(tmp_path, source_name=os.path.basename(file.filename))
    finally:
        os.unlink(tmp_path)
    return IngestResponse(filename=file.filename, chunks_indexed=count)


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest, _auth: None = Depends(require_api_key)) -> Answer:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    return answer_question(req.question)
