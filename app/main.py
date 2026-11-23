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
from .security import require_api_key, require_eval_key
from .provenance import safe_source_name
from .rag import answer_question, answer_question_with_evidence, Answer

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
    try:
        source_name = safe_source_name(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="A valid PDF filename is required.") from exc
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
        count = ingest_pdf(tmp_path, source_name=source_name)
    finally:
        os.unlink(tmp_path)
    return IngestResponse(filename=source_name, chunks_indexed=count)


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest, _auth: None = Depends(require_api_key)) -> Answer:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    return answer_question(req.question)


@app.post("/evaluation/ask")
def evaluate_ask(
    req: AskRequest,
    _auth: None = Depends(require_api_key),
    _eval: None = Depends(require_eval_key),
) -> dict:
    """Opt-in, dual-key endpoint: one retrieval and answer-generation pass.

    Returns the exact chunks used to construct the answer, not reference/gold
    contexts. Do not expose outside isolated evaluation environments.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    answer, chunks = answer_question_with_evidence(req.question)
    return {
        "answer": answer.model_dump(),
        "retrieved_contexts": [
            {"text": c.text, "source": c.source, "page": c.page, "score": c.score}
            for c in chunks
        ],
        "generation_mode": "llm" if settings.openai_api_key else "extractive-demo",
        "evidence_origin": "same-invocation-retrieval",
    }
