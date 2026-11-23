# PolicyMind — Enterprise Policy RAG Assistant

> **Product-management portfolio brief:** [Problem, users, success metrics, roadmap and limitations](docs/PRODUCT_CASE_STUDY.md). This is an independent prototype; no employer or customer outcome is claimed.


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

**Portfolio demonstration, not a multi-user enterprise service.** API endpoints `/ask` and `/ingest` require `X-API-Key` matching `POLICYMIND_API_KEY`; unset credentials fail closed. Ingestion checks the PDF header and limits uploads to 15 MiB, but does not provide complete malware scanning, PDF sandboxing, authorization per document, or data retention controls. The standalone Streamlit UI **fails closed by default** and requires `POLICYMIND_ENABLE_LOCAL_UI=true` plus the shared `POLICYMIND_API_KEY` to run locally. This is not user-level authentication; **do not expose the UI publicly**. Deploy behind TLS and an identity-aware access layer before handling real policies. Citation metadata uses a sanitized upload basename rather than a temporary server path. Identical PDF re-ingestion uses stable content-addressed chunk IDs to avoid duplicate points for the same file version. **Older indexes are not migrated**; replacing a document with a different version does not yet remove its prior chunks. Rebuild demo indexes before relying on this feature. Relevance score is a heuristic, **not a calibrated confidence probability**; `grounded=True` does not prove claims are supported.

## Live evaluation evidence (isolated deployments only)

GuardEval can now request the **exact retrieved contexts used in the same answer-generation call**. The endpoint `POST /evaluation/ask` is **disabled by default**; it requires all three:

- `POLICYMIND_ENABLE_EVAL_API=true`
- A valid `X-API-Key` matching `POLICYMIND_API_KEY`
- An independent `X-Eval-Key` matching `POLICYMIND_EVAL_KEY`

Only use this endpoint with synthetic or explicitly authorized evaluation documents, in a private environment. It returns retrieved document text and can disclose sensitive information if exposed. For production evaluation, provide audited service-to-service identity, network isolation, access logging and redaction. The endpoint also returns a `generation_mode` flag: `extractive-demo` is not an LLM model-quality measurement.

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
POLICYMIND_ENABLE_LOCAL_UI=true streamlit run streamlit_app.py --server.address=127.0.0.1
# Enter POLICYMIND_API_KEY in the password field; local demo only.
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
