"""Central configuration for PolicyMind. All secrets come from environment variables."""
import os
from functools import lru_cache
from pydantic import BaseModel


class Settings(BaseModel):
    # --- LLM / embeddings ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    embed_dim: int = int(os.getenv("EMBED_DIM", "3072"))  # 3072 for text-embedding-3-large

    # --- Qdrant ---
    # If QDRANT_URL is empty we use a local on-disk Qdrant (no server needed).
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_path: str = os.getenv("QDRANT_PATH", "./qdrant_data")
    collection: str = os.getenv("QDRANT_COLLECTION", "policymind")

    # --- Retrieval ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "20"))
    sparse_top_k: int = int(os.getenv("SPARSE_TOP_K", "20"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    sparse_model: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

    # Minimum reranker score to consider context "good enough" to answer.
    min_context_score: float = float(os.getenv("MIN_CONTEXT_SCORE", "0.05"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
