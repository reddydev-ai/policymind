"""Source identification utilities for a single-tenant RAG demo.

File names are untrusted input. Use a safe display name in citations and
content-addressed point IDs to make ingestion of identical files repeatable.
"""
from __future__ import annotations

import re
import uuid


_POINT_NAMESPACE = uuid.UUID("a7b6c8b0-9b9b-4fd8-b047-e322eeb4298a")


def safe_source_name(raw_name: str) -> str:
    """Convert a user supplied PDF basename to safe, bounded citation text."""
    if not raw_name or not isinstance(raw_name, str):
        raise ValueError("A PDF filename is required")
    # Handle both POSIX and Windows separators irrespective of host OS.
    basename = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
    if not basename.lower().endswith(".pdf"):
        raise ValueError("Only PDF filenames are supported")
    stem = basename[:-4]
    stem = re.sub(r"[^A-Za-z0-9._() -]", "_", stem).strip(" ._")
    if not stem:
        raise ValueError("Filename must contain a meaningful stem")
    return stem[:120] + ".pdf"


def point_id(source: str, document_sha256: str, page: int, chunk_index: int) -> str:
    """Stable UUID for one chunk of one exact source-document version."""
    if not source or not re.fullmatch(r"[0-9a-f]{64}", document_sha256):
        raise ValueError("Invalid document identity")
    if page < 1 or chunk_index < 0:
        raise ValueError("Invalid chunk position")
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{source}\x00{document_sha256}\x00{page}\x00{chunk_index}"))
