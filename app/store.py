"""Qdrant store: creates a hybrid collection (dense + sparse vectors) and exposes
helpers to upsert chunks and run hybrid search with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations
from .provenance import point_id

from qdrant_client import QdrantClient, models

from .config import get_settings

settings = get_settings()


def get_client() -> QdrantClient:
    """Returns a Qdrant client.
    - If QDRANT_URL is set -> connect to a real Qdrant server (production).
    - Otherwise -> local on-disk Qdrant, so the demo runs with zero infra.
    """
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return QdrantClient(path=settings.qdrant_path)


DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def ensure_collection(client: QdrantClient) -> None:
    """Create the hybrid collection if it does not exist."""
    if client.collection_exists(settings.collection):
        return
    client.create_collection(
        collection_name=settings.collection,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=settings.embed_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(
                modifier=models.Modifier.IDF  # enables BM25-style IDF weighting
            )
        },
    )


def upsert_chunks(
    client: QdrantClient,
    dense_vectors: list[list[float]],
    sparse_vectors: list[models.SparseVector],
    payloads: list[dict],
) -> int:
    if not (len(dense_vectors) == len(sparse_vectors) == len(payloads)):
        raise ValueError("Vector and payload counts must match; refusing partial ingestion")
    points = []
    for dvec, svec, payload in zip(dense_vectors, sparse_vectors, payloads):
        points.append(
            models.PointStruct(
                id=point_id(
                    payload["source"], payload["document_sha256"],
                    payload["page"], payload["chunk_index"],
                ),
                vector={DENSE_VECTOR: dvec, SPARSE_VECTOR: svec},
                payload=payload,
            )
        )
    if not points:
        return 0
    # A source filename represents one current document version in this demo.
    # Commit the replacement points first, then remove old versions; if the
    # upsert fails, previously indexed content is preserved.
    client.upsert(collection_name=settings.collection, points=points, wait=True)
    versions = {(p["source"], p["document_sha256"]) for p in payloads}
    if len({source for source, _ in versions}) != len(versions):
        # This should not be called with multiple versions of the same source.
        raise ValueError("One batch cannot contain multiple versions of a source")
    for source, document_sha in versions:
        client.delete(
            collection_name=settings.collection,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))],
                must_not=[models.FieldCondition(
                    key="document_sha256", match=models.MatchValue(value=document_sha)
                )],
            )),
            wait=True,
        )
    return len(points)


def hybrid_search(
    client: QdrantClient,
    dense_query: list[float],
    sparse_query: models.SparseVector,
) -> list[models.ScoredPoint]:
    """Run dense + sparse retrieval in parallel and fuse with RRF.
    This is the technically correct Qdrant hybrid pattern: two prefetch branches
    (dense semantic + sparse keyword) fused before we hand results to the reranker.
    """
    result = client.query_points(
        collection_name=settings.collection,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using=DENSE_VECTOR,
                limit=settings.dense_top_k,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices, values=sparse_query.values
                ),
                using=SPARSE_VECTOR,
                limit=settings.sparse_top_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=max(settings.dense_top_k, settings.sparse_top_k),
        with_payload=True,
    )
    return result.points
