"""Embedding helpers.
- Dense: OpenAI text-embedding-3-large when OPENAI_API_KEY is available.
- Dense no-key mode: deterministic hash embeddings so local demos/tests can run.
- Sparse: FastEmbed BM25 when available; deterministic sparse fallback otherwise.

Hybrid retrieval needs both so the system handles semantic and keyword-style
queries. All fallback modes are transparent and intended for local/demo use only.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

from qdrant_client import models

from .config import get_settings

settings = get_settings()


@lru_cache
def _openai():
    if not settings.openai_api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=settings.openai_api_key)
    except Exception:
        return None


@lru_cache
def _sparse_model():
    try:
        from fastembed import SparseTextEmbedding  # type: ignore
        return SparseTextEmbedding(model_name=settings.sparse_model)
    except Exception:
        return None


def _hash_dense(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _hash_sparse(text: str) -> models.SparseVector:
    counts: dict[int, float] = {}
    for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        h = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % 1_000_000
        counts[h] = counts.get(h, 0.0) + 1.0
    if not counts:
        return models.SparseVector(indices=[0], values=[0.0])
    indices = sorted(counts)
    values = [counts[i] for i in indices]
    return models.SparseVector(indices=indices, values=values)


def embed_dense(texts: list[str]) -> list[list[float]]:
    client = _openai()
    if client is None:
        return [_hash_dense(t, settings.embed_dim) for t in texts]
    resp = client.embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    model = _sparse_model()
    if model is None:
        return [_hash_sparse(t) for t in texts]
    out: list[models.SparseVector] = []
    for emb in model.embed(texts):
        out.append(models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist()))
    return out
