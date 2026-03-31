from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
VECTOR_INDEX_PATH = DATA_DIR / "vector_store.faiss"
VECTOR_META_PATH = DATA_DIR / "vector_store_meta.json"
VECTOR_EMB_PATH = DATA_DIR / "vector_store_embeddings.npy"
MEMORY_PATH = DATA_DIR / "memory" / "user_preferences.json"


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Personal Productivity Agent"
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    use_mock_llm: bool = os.getenv("USE_MOCK_LLM", "true").lower() == "true"

    default_city: str = os.getenv("DEFAULT_CITY", "Los Angeles")
    short_memory_window: int = int(os.getenv("SHORT_MEMORY_WINDOW", "8"))
    rag_embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    rag_reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rag_use_faiss: bool = os.getenv("RAG_USE_FAISS", "false").lower() == "true"
    rag_use_transformers: bool = os.getenv("RAG_USE_TRANSFORMERS", "false").lower() == "true"
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "300"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    rag_candidate_k: int = int(os.getenv("RAG_CANDIDATE_K", "20"))


settings = Settings()
