from types import SimpleNamespace
from unittest.mock import MagicMock

from app import runtime_doctor


def _settings(**overrides):
    values = {
        "MODEL_PROFILE": "ubuntu_t4",
        "DEVICE": "cpu",
        "EMBEDDING_DEVICE": "cpu",
        "RERANKER_DEVICE": "cpu",
        "EMBEDDING_DIMENSION": 1024,
        "EMBEDDING_INDEX_VERSION": "hierarchical-qwen3-1024-v1",
        "LLM_MODEL": "qwen3:14b-q4_K_M",
        "LLM_CONTEXT_WINDOW": 8192,
        "LLM_MAX_NEW_TOKENS": 3072,
        "LLM_DISABLE_THINKING": True,
        "CPU_THREADS": 28,
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "DATABASE_URL": "postgresql+psycopg://atlas@localhost/atlas",
        "CELERY_ENABLED": True,
        "CELERY_BROKER_URL": "redis://localhost:6379/0",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_doctor_accepts_expected_profile(monkeypatch):
    monkeypatch.setattr(runtime_doctor, "get_settings", _settings)
    assert runtime_doctor._check_profile("ubuntu_t4") is True


def test_doctor_rejects_accidental_llm_override(monkeypatch):
    monkeypatch.setattr(
        runtime_doctor,
        "get_settings",
        lambda: _settings(LLM_MODEL="qwen3:4b-instruct-2507-q4_K_M"),
    )

    assert runtime_doctor._check_profile("ubuntu_t4") is False


def test_doctor_detects_local_tesla_t4(monkeypatch):
    result = MagicMock(
        stdout="Tesla T4, 15360 MiB, 580.65.06\n",
    )
    monkeypatch.setattr(runtime_doctor.subprocess, "run", MagicMock(return_value=result))

    assert runtime_doctor._check_nvidia() is True


def test_doctor_requires_configured_ollama_model(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"models": [{"name": "qwen3:4b"}]}
    monkeypatch.setattr(runtime_doctor, "get_settings", _settings)
    monkeypatch.setattr(runtime_doctor.httpx, "get", MagicMock(return_value=response))

    assert runtime_doctor._check_ollama() is False


def test_doctor_checks_postgres_extensions(monkeypatch):
    connection = MagicMock()
    connection.execute.side_effect = [
        MagicMock(),
        [("vector",), ("unaccent",), ("pg_trgm",), ("uuid-ossp",)],
    ]
    context = MagicMock()
    context.__enter__.return_value = connection
    engine = MagicMock()
    engine.connect.return_value = context
    monkeypatch.setattr(runtime_doctor, "engine", engine)
    monkeypatch.setattr(runtime_doctor, "get_settings", _settings)

    assert runtime_doctor._check_database() is True


def test_doctor_checks_redis_when_celery_is_enabled(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(runtime_doctor, "get_settings", _settings)
    monkeypatch.setattr(
        runtime_doctor.redis.Redis,
        "from_url",
        MagicMock(return_value=client),
    )

    assert runtime_doctor._check_redis() is True
    client.ping.assert_called_once_with()
