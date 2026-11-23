"""Document ingestion pipeline.
PDF -> text (PyMuPDF) -> chunks (LangChain splitter, page-aware) -> dense+sparse
embeddings -> Qdrant. Each chunk keeps its source filename and page number so the
RAG layer can return real citations.
"""
from __future__ import annotations
import os
import hashlib

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings
from .embeddings import embed_dense, embed_sparse
from .store import get_client, ensure_collection, upsert_chunks
from .provenance import safe_source_name

settings = get_settings()

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _pages_from_pdf(path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] with page numbers starting at 1."""
    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((i, text))
    return pages


def ingest_pdf(path: str, source_name: str | None = None) -> int:
    """Ingest a single PDF. Returns number of chunks stored."""
    filename = safe_source_name(source_name or os.path.basename(path))
    with open(path, "rb") as document:
        document_sha256 = hashlib.file_digest(document, "sha256").hexdigest()
    client = get_client()
    ensure_collection(client)

    texts: list[str] = []
    payloads: list[dict] = []
    for page_no, page_text in _pages_from_pdf(path):
        for chunk_index, chunk in enumerate(_splitter.split_text(page_text)):
            texts.append(chunk)
            payloads.append({
                "text": chunk,
                "source": filename,
                "page": page_no,
                "chunk_index": chunk_index,
                "document_sha256": document_sha256,
            })

    if not texts:
        return 0

    # Embed in batches to stay within API limits.
    dense: list[list[float]] = []
    sparse = []
    BATCH = 64
    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        dense.extend(embed_dense(batch))
        sparse.extend(embed_sparse(batch))

    return upsert_chunks(client, dense, sparse, payloads)


def ingest_folder(folder: str) -> dict[str, int]:
    """Ingest every PDF in a folder. Returns {filename: chunk_count}."""
    results: dict[str, int] = {}
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".pdf"):
            results[name] = ingest_pdf(os.path.join(folder, name))
    return results
