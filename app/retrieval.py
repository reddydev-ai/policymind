"""Retrieval layer: hybrid search (Qdrant RRF) -> optional BGE cross-encoder rerank.

The reranker is used when sentence-transformers/model files are available. If the
reranker cannot load, the system falls back to Qdrant's fused scores so the demo
still runs; this fallback is suitable for local validation, not final production.
"""
from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass

from .config import get_settings
from .embeddings import embed_dense, embed_sparse
from .store import get_client, hybrid_search

settings = get_settings()


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float


@lru_cache
def _reranker():
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        return CrossEncoder(settings.reranker_model)
    except Exception:
        return None


def retrieve(query: str) -> list[RetrievedChunk]:
    client = get_client()
    dense_q = embed_dense([query])[0]
    sparse_q = embed_sparse([query])[0]

    points = hybrid_search(client, dense_q, sparse_q)
    if not points:
        return []

    candidates = [
        {
            "text": p.payload.get("text", ""),
            "source": p.payload.get("source", "unknown"),
            "page": p.payload.get("page", 0),
            "qdrant_score": float(getattr(p, "score", 0.0) or 0.0),
        }
        for p in points
    ]

    reranker = _reranker()
    if reranker is None:
        ranked = sorted(
            (
                RetrievedChunk(text=c["text"], source=c["source"], page=c["page"], score=c["qdrant_score"])
                for c in candidates
            ),
            key=lambda x: x.score,
            reverse=True,
        )
        return ranked[: settings.rerank_top_n]

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(
        (
            RetrievedChunk(text=c["text"], source=c["source"], page=c["page"], score=float(s))
            for c, s in zip(candidates, scores)
        ),
        key=lambda x: x.score,
        reverse=True,
    )
    return ranked[: settings.rerank_top_n]
