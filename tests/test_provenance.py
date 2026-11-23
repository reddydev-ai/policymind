"""Source and idempotent chunk identity tests (no external services)."""
import pytest
from app.provenance import point_id, safe_source_name


@pytest.mark.parametrize(("raw", "expected"), [
    ("policy.pdf", "policy.pdf"),
    ("C:\\Users\\alice\\policy.pdf", "policy.pdf"),
    ("../../annual policy.pdf", "annual policy.pdf"),
    ("../weird\nname.pdf", "weird_name.pdf"),
    ("UPPER.PDF", "UPPER.pdf"),
])
def test_source_name_normalizes_untrusted_paths(raw, expected):
    assert safe_source_name(raw) == expected


@pytest.mark.parametrize("raw", ["", ".", "file.txt", "../../.pdf", "/.pdf"])
def test_invalid_source_fails(raw):
    with pytest.raises(ValueError):
        safe_source_name(raw)


def test_source_name_is_bounded():
    assert len(safe_source_name("a" * 500 + ".pdf")) <= 124


def test_reingestion_of_same_document_has_same_id():
    digest = "a" * 64
    assert point_id("policy.pdf", digest, 1, 0) == point_id("policy.pdf", digest, 1, 0)


@pytest.mark.parametrize(("source", "digest", "page", "chunk"), [
    ("other.pdf", "a" * 64, 1, 0),
    ("policy.pdf", "b" * 64, 1, 0),
    ("policy.pdf", "a" * 64, 2, 0),
    ("policy.pdf", "a" * 64, 1, 1),
])
def test_different_provenance_produces_different_point_id(source, digest, page, chunk):
    assert point_id(source, digest, page, chunk) != point_id("policy.pdf", "a" * 64, 1, 0)


@pytest.mark.parametrize(("digest", "page", "chunk"), [
    ("bad", 1, 0),
    ("a" * 64, 0, 0),
    ("a" * 64, 1, -1),
])
def test_invalid_point_identity_is_rejected(digest, page, chunk):
    with pytest.raises(ValueError):
        point_id("policy.pdf", digest, page, chunk)
