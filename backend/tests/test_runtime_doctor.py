from types import SimpleNamespace
from unittest.mock import MagicMock

from app import runtime_doctor


def _settings(**overrides):
    values = {
        "MODEL_PROFILE": "wsl_t4",
        "DEVICE": "cpu",
        "EMBEDDING_DEVICE": "cpu",
        "RERANKER_DEVICE": "cpu",
        "EMBEDDING_DIMENSION": 1024,
        "EMBEDDING_INDEX_VERSION": "hierarchical-qwen3-1024-v1",
        "LLM_MODEL": "qwen3:14b-q4_K_M",
        "CPU_THREADS": 28,
        "OLLAMA_BASE_URL": "http://172.20.0.1:11434",
        "DATABASE_URL": "postgresql+psycopg://atlas@localhost/atlas",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_doctor_accepts_expected_profile(monkeypatch):
    monkeypatch.setattr(runtime_doctor, "get_settings", _settings)
    assert runtime_doctor._check_profile() is True


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
