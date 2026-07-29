from types import SimpleNamespace
from unittest.mock import MagicMock

from app.rag import llm_client


def test_shared_ollama_client_applies_wsl_endpoint_timeout_and_keep_alive(monkeypatch):
    constructor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(llm_client, "ChatOllama", constructor)
    monkeypatch.setattr(
        llm_client,
        "get_settings",
        lambda: SimpleNamespace(
            LLM_MODEL="qwen3:14b-q4_K_M",
            LLM_REQUEST_TIMEOUT_SECONDS=900,
            OLLAMA_BASE_URL="http://172.20.0.1:11434",
            OLLAMA_KEEP_ALIVE="30m",
        ),
    )

    llm_client.create_chat_ollama(temperature=0, num_ctx=8192)

    assert constructor.call_args.kwargs == {
        "model": "qwen3:14b-q4_K_M",
        "temperature": 0,
        "num_ctx": 8192,
        "keep_alive": "30m",
        "base_url": "http://172.20.0.1:11434",
        "client_kwargs": {"timeout": 900},
    }


def test_shared_ollama_client_allows_stage_timeout_and_explicit_overrides(monkeypatch):
    constructor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(llm_client, "ChatOllama", constructor)
    monkeypatch.setattr(
        llm_client,
        "get_settings",
        lambda: SimpleNamespace(
            LLM_MODEL="default",
            LLM_REQUEST_TIMEOUT_SECONDS=900,
            OLLAMA_BASE_URL="",
            OLLAMA_KEEP_ALIVE="5m",
        ),
    )

    llm_client.create_chat_ollama(
        model="planner",
        keep_alive="1m",
        timeout_seconds=30,
        client_kwargs={"follow_redirects": True},
    )

    assert constructor.call_args.kwargs["model"] == "planner"
    assert constructor.call_args.kwargs["keep_alive"] == "1m"
    assert "base_url" not in constructor.call_args.kwargs
    assert constructor.call_args.kwargs["client_kwargs"] == {
        "follow_redirects": True,
        "timeout": 30,
    }
