"""Test exact-file reingestion against an in-memory Qdrant collection.

This uses the real qdrant-client implementation, not a mocked store.
"""
import pytest
from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.store import ensure_collection, upsert_chunks


@pytest.fixture
def memory_store():
    client = QdrantClient(":memory:")
    ensure_collection(client)
    yield client
    client.close()


def test_duplicate_reingestion_is_idempotent(memory_store):
    settings = get_settings()
    dense = [[1.0] + [0.0] * (settings.embed_dim - 1)]
    sparse = [models.SparseVector(indices=[10], values=[1.0])]
    payload = [{
        "source": "policy.pdf",
        "document_sha256": "a" * 64,
        "page": 1,
        "chunk_index": 0,
        "text": "Policy language",
    }]
    assert upsert_chunks(memory_store, dense, sparse, payload) == 1
    assert upsert_chunks(memory_store, dense, sparse, payload) == 1
    assert memory_store.count(collection_name=settings.collection, exact=True).count == 1


@pytest.mark.parametrize(("dense", "sparse", "payloads"), [
    ([[]], [], []),
    ([], [models.SparseVector(indices=[1], values=[1.0])], []),
    ([[]], [models.SparseVector(indices=[1], values=[1.0])], []),
])
def test_mismatched_batch_counts_fail_before_storage(memory_store, dense, sparse, payloads):
    with pytest.raises(ValueError, match="counts must match"):
        upsert_chunks(memory_store, dense, sparse, payloads)


def test_new_version_replaces_previous_source_without_affecting_other_files(memory_store):
    settings = get_settings()
    dense = [[1.0] + [0.0] * (settings.embed_dim - 1)]
    sparse = [models.SparseVector(indices=[10], values=[1.0])]

    def upsert(source, digest):
        return upsert_chunks(memory_store, dense, sparse, [{
            "source": source,
            "document_sha256": digest,
            "page": 1,
            "chunk_index": 0,
            "text": "Version " + digest[:6],
        }])

    upsert("unchanged.pdf", "c" * 64)
    upsert("policy.pdf", "a" * 64)
    upsert("policy.pdf", "b" * 64)

    points, _ = memory_store.scroll(collection_name=settings.collection, limit=20, with_payload=True)
    assert len(points) == 2
    assert {(p.payload["source"], p.payload["document_sha256"]) for p in points} == {
        ("policy.pdf", "b" * 64), ("unchanged.pdf", "c" * 64)
    }


def test_mixed_versions_fail_before_writing(memory_store):
    settings = get_settings()
    dense = [[1.0] + [0.0] * (settings.embed_dim - 1)] * 2
    sparse = [models.SparseVector(indices=[10], values=[1.0])] * 2
    payloads = [{
        "source": "policy.pdf", "document_sha256": digest,
        "page": 1, "chunk_index": 0, "text": "text"
    } for digest in ("a" * 64, "b" * 64)]
    with pytest.raises(ValueError, match="multiple versions"):
        upsert_chunks(memory_store, dense, sparse, payloads)
    assert memory_store.count(collection_name=settings.collection, exact=True).count == 0
