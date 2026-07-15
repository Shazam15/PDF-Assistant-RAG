"""
HuggingFace local embeddings using sentence-transformers.
Loads the model once via singleton pattern for efficiency.
"""
import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from app.config import get_settings
from app.rag.tracing import trace_call

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Singleton embedding model ────────────────────────
_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Get or create the embedding model (singleton).
    Uses a multilingual E5 retrieval model with query/passage prefixes.
    """
    global _embedding_model

    if _embedding_model is None:
        if settings.EMBEDDING_DEVICE == "cpu" and settings.CPU_THREADS:
            torch.set_num_threads(settings.CPU_THREADS)
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": settings.EMBEDDING_BATCH_SIZE,
            },
        )
        logger.info("Embedding model loaded successfully")

    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts into vectors."""
    model = get_embedding_model()
    passages = [_passage_text(text) for text in texts]
    return trace_call(
        "embed_texts",
        lambda: model.embed_documents(passages),
        run_type="embedding",
        metadata={
            "embedding_model": settings.EMBEDDING_MODEL,
            "text_count": len(texts),
        },
    )


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    model = get_embedding_model()
    return trace_call(
        "embed_query",
        lambda: model.embed_query(_query_text(query)),
        run_type="embedding",
        metadata={
            "embedding_model": settings.EMBEDDING_MODEL,
            "query_length": len(query),
        },
    )


def _is_e5_model() -> bool:
    return "e5" in settings.EMBEDDING_MODEL.lower()


def _passage_text(text: str) -> str:
    return f"passage: {text}" if _is_e5_model() else text


def _query_text(query: str) -> str:
    if _is_e5_model():
        return f"query: {query}"
    if "qwen3-embedding" in settings.EMBEDDING_MODEL.lower():
        return (
            "Instruct: Retrieve passages that provide direct evidence for the research question.\n"
            f"Query: {query}"
        )
    return query


def release_embedding_model() -> None:
    global _embedding_model
    _embedding_model = None
