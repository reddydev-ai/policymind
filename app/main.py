"""FastAPI surface for PolicyMind.
Endpoints:
  GET  /health            -> liveness
  POST /ingest            -> upload a PDF, ingest it
  POST /ask               -> ask a question, get a cited answer
"""
from __future__ import annotations
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from .config import get_settings
from .ingestion import ingest_pdf
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
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        count = ingest_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)
    return IngestResponse(filename=file.filename, chunks_indexed=count)


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    return answer_question(req.question)
