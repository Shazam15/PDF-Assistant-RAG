"""Shared Ollama client construction for every RAG pipeline stage."""

from typing import Any

from langchain_ollama import ChatOllama

from app.config import get_settings


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
