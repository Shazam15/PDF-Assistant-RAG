import pytest

from app.config import Settings


def test_production_settings_require_secret_key():
    with pytest.raises(ValueError):
        Settings(ENVIRONMENT="production", SECRET_KEY="")


def test_research_gpu_uses_cpu_for_embeddings_and_respects_device_overrides():
    defaults = Settings(_env_file=None, MODEL_PROFILE="research_gpu")
    overridden = Settings(
        _env_file=None,
        MODEL_PROFILE="research_gpu",
        RERANKER_DEVICE="cpu",
        EMBEDDING_BATCH_SIZE=16,
        LLM_MODEL="qwen3:14b-q4_K_M",
        LLM_CONTEXT_WINDOW=8192,
    )

    assert defaults.EMBEDDING_DEVICE == "cpu"
    assert defaults.RERANKER_DEVICE == "cuda"
    assert defaults.EMBEDDING_BATCH_SIZE == 64
    assert overridden.RERANKER_DEVICE == "cpu"
    assert overridden.EMBEDDING_BATCH_SIZE == 16
    assert overridden.LLM_MODEL == "qwen3:14b-q4_K_M"
    assert overridden.LLM_CONTEXT_WINDOW == 8192
