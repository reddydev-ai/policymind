# PolicyMind — Enterprise Policy RAG Assistant

Upload internal HR / compliance / benefits policy documents, ask plain-English
questions, and get **cited answers with a confidence score** and an honest
**"I don't know"** when the answer isn't supported by the documents.

> PolicyMind is designed for internal policy documents only. It does **not**
> provide legal or immigration advice.

## Why it's built this way

| Decision | Reason |
|---|---|
| **Hybrid search** (dense + sparse) | Dense embeddings handle meaning ("parental leave rules"); sparse BM25 handles exact terms ("clause 7b", "section 4.2"). Pure vector search misses the second kind. |
| **RRF fusion** | Combines the two retrieval branches fairly before reranking. |
| **BGE cross-encoder reranker** | Re-scores fused candidates and keeps the top-5. This is the single biggest precision win over a beginner RAG. |
| **Confidence threshold + "I don't know"** | The system refuses to answer when retrieval is weak, instead of hallucinating. Production-maturity signal. |
| **Page-level citations** | Every answer points back to a real filename + page number. |

## Architecture

```
PDF upload
   │
   ▼  PyMuPDF (clean text extraction, page-aware)
Chunking  (LangChain RecursiveCharacterTextSplitter, 800/150)
   │
   ├─► OpenAI text-embedding-3-large  → dense vector
   └─► FastEmbed BM25                 → sparse vector
   │
   ▼  Qdrant (hybrid collection: dense + sparse)
User query
   │
   ▼  hybrid retrieval (dense top-20 + sparse top-20) → RRF fusion
   ▼  BGE reranker → top-5 chunks
   ▼  confidence check (threshold → "I don't know" fallback)
   ▼  GPT-4o → grounded, cited answer
   ▼  FastAPI /ask  ·  Streamlit UI
```

## Security status

**Portfolio demonstration, not a multi-user enterprise service.** API endpoints `/ask` and `/ingest` require `X-API-Key` matching `POLICYMIND_API_KEY`; unset credentials fail closed. Ingestion checks the PDF header and limits uploads to 15 MiB, but does not provide complete malware scanning, PDF sandboxing, authorization per document, or data retention controls. **Do not expose the standalone Streamlit UI publicly**, since it does not enforce API authentication. Deploy behind TLS and an identity-aware access layer before handling real policies. Citation metadata preserves the original upload basename rather than a temporary server path. Relevance score is a heuristic, **not a calibrated confidence probability**; `grounded=True` does not prove claims are supported.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your OPENAI_API_KEY
set -a; source .env; set +a  # export credentials for this shell (trusted local file only)
```

### Run the API
```bash
uvicorn app.main:app --reload
# Swagger docs at http://localhost:8000/docs
```

Ingest a PDF and ask a question:
```bash
curl -H "X-API-Key: $POLICYMIND_API_KEY" -F "file=@vacation_policy.pdf" http://localhost:8000/ingest
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $POLICYMIND_API_KEY" \
  -d '{"question":"How many vacation days do new employees get?"}'
```

### Run the UI
```bash
streamlit run streamlit_app.py
```

## Deployment
Deploy on **Railway** or **Render** (Dockerfile included). For real scale, point
`QDRANT_URL` at a managed Qdrant instead of the local on-disk store.

## Known limitations
- Local on-disk Qdrant is for demo/small corpora. Use a Qdrant server past ~100k chunks.
- No OCR: scanned/image-only PDFs need an OCR step before ingestion.
- Confidence is a heuristic (top reranker score), not a calibrated probability.
- Single-tenant: no per-user document isolation yet.

## What I would improve next
- Add OCR (e.g. for scanned PDFs) and table-aware parsing.
- Add a feedback loop: thumbs up/down per answer feeding a re-ranking eval set.
- Per-document access control and multi-tenant collections.
- Streaming responses and answer caching.
- Wire **GuardEval** in CI to block deploys when faithfulness/citation scores drop.
