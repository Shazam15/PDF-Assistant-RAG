"""Shared Ollama client construction for every RAG pipeline stage."""

import logging
from typing import Any

from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)

_KEEP_ALIVE_SUFFIXES = {"s": 1, "m": 60, "h": 3600}


def _keep_alive_seconds(value: str) -> int | None:
    """Convert an Ollama ``keep_alive`` duration (e.g. "30m", "-1", "300") to seconds.

    ``ChatOllama`` forwards the raw string straight to Ollama's API, which
    accepts either form. ``OllamaEmbeddings``'s underlying client is stricter
    and only accepts an int, so the shared ``OLLAMA_KEEP_ALIVE`` setting needs
    converting here for the embeddings client specifically.
    """
    text = value.strip().lower()
    if not text:
        return None
    try:
        if text[-1] in _KEEP_ALIVE_SUFFIXES and text[:-1].lstrip("-").isdigit():
            return int(text[:-1]) * _KEEP_ALIVE_SUFFIXES[text[-1]]
        return int(text)
    except ValueError:
        logger.warning("Could not parse OLLAMA_KEEP_ALIVE=%r as a duration; using Ollama's default", value)
        return None


def create_chat_ollama(*, timeout_seconds: int | None = None, **kwargs: Any) -> ChatOllama:
    """Create a consistently configured Ollama chat client.

    ``OLLAMA_BASE_URL`` is explicit for WSL-to-Windows connections. When it is
    blank, the Ollama SDK retains its normal ``OLLAMA_HOST``/localhost behavior.
    """
    settings = get_settings()
    client_kwargs = dict(kwargs.pop("client_kwargs", {}) or {})
    client_kwargs.setdefault(
        "timeout",
        timeout_seconds or settings.LLM_REQUEST_TIMEOUT_SECONDS,
    )

    kwargs.setdefault("model", settings.LLM_MODEL)
    kwargs.setdefault("keep_alive", settings.OLLAMA_KEEP_ALIVE)
    kwargs["client_kwargs"] = client_kwargs
    if settings.OLLAMA_BASE_URL:
        kwargs.setdefault("base_url", settings.OLLAMA_BASE_URL)
    return ChatOllama(**kwargs)


def create_ollama_embeddings(*, timeout_seconds: int | None = None, **kwargs: Any) -> OllamaEmbeddings:
    """Create a consistently configured Ollama embeddings client.

    Mirrors ``create_chat_ollama`` so embedding calls reach the same remote
    Ollama host (e.g. a Windows Xeon/T4 box on the LAN) as chat calls, letting
    a machine that's too weak/old for local sentence-transformers inference
    (no AVX2, no GPU) offload embedding to wherever Ollama already runs.
    """
    settings = get_settings()
    client_kwargs = dict(kwargs.pop("client_kwargs", {}) or {})
    client_kwargs.setdefault(
        "timeout",
        timeout_seconds or settings.LLM_REQUEST_TIMEOUT_SECONDS,
    )

    kwargs.setdefault("model", settings.EMBEDDING_OLLAMA_MODEL)
    keep_alive = _keep_alive_seconds(settings.OLLAMA_KEEP_ALIVE)
    if keep_alive is not None:
        kwargs.setdefault("keep_alive", keep_alive)
    kwargs["client_kwargs"] = client_kwargs
    if settings.OLLAMA_BASE_URL:
        kwargs.setdefault("base_url", settings.OLLAMA_BASE_URL)
    return OllamaEmbeddings(**kwargs)
