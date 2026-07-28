import pytest

from app.config import Settings


def test_experimental_branch_defaults_to_ubuntu_t4_and_14b_llm():
    settings = Settings(_env_file=None)

    assert settings.MODEL_PROFILE == "ubuntu_t4"
    assert settings.LLM_MODEL == "qwen3:14b-q4_K_M"
    assert settings.LLM_CONTEXT_WINDOW == 8192
    assert settings.LLM_DISABLE_THINKING is True
    assert settings.OLLAMA_BASE_URL == "http://127.0.0.1:11434"


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


def test_local_balanced_profile_keeps_qwen_and_uses_small_mps_batches():
    settings = Settings(_env_file=None, MODEL_PROFILE="local_balanced")

    assert settings.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.EMBEDDING_DIMENSION == 1024
    assert settings.EMBEDDING_DEVICE == "mps"
    assert settings.EMBEDDING_BATCH_SIZE == 4
    assert settings.CPU_THREADS == 4


def test_wsl_t4_profile_reserves_gpu_for_windows_ollama_and_respects_overrides():
    defaults = Settings(_env_file=None, MODEL_PROFILE="wsl_t4")
    overridden = Settings(
        _env_file=None,
        MODEL_PROFILE="wsl_t4",
        CPU_THREADS=20,
        EMBEDDING_BATCH_SIZE=32,
        OLLAMA_BASE_URL="http://172.20.0.1:11434/",
    )

    assert defaults.DEVICE == "cpu"
    assert defaults.EMBEDDING_DEVICE == "cpu"
    assert defaults.RERANKER_DEVICE == "cpu"
    assert defaults.EMBEDDING_BATCH_SIZE == 64
    assert defaults.CPU_THREADS == 28
    assert defaults.LLM_MODEL == "qwen3:14b-q4_K_M"
    assert defaults.LLM_CONTEXT_WINDOW == 8192
    assert defaults.LLM_MAX_NEW_TOKENS == 3072
    assert defaults.LLM_DISABLE_THINKING is True
    assert defaults.OLLAMA_KEEP_ALIVE == "30m"
    assert defaults.RESEARCH_TIMEOUT_SECONDS == 1800
    assert overridden.CPU_THREADS == 20
    assert overridden.EMBEDDING_BATCH_SIZE == 32
    assert overridden.OLLAMA_BASE_URL == "http://172.20.0.1:11434"


def test_ubuntu_t4_profile_uses_local_ollama_and_reserves_gpu_for_llm():
    settings = Settings(_env_file=None, MODEL_PROFILE="ubuntu_t4")

    assert settings.DEVICE == "cpu"
    assert settings.EMBEDDING_DEVICE == "cpu"
    assert settings.RERANKER_DEVICE == "cpu"
    assert settings.CPU_THREADS == 28
    assert settings.LLM_MODEL == "qwen3:14b-q4_K_M"
    assert settings.OLLAMA_BASE_URL == "http://127.0.0.1:11434"
    assert settings.OLLAMA_KEEP_ALIVE == "30m"


def test_redis_cache_configuration_is_loaded_from_env_file_settings():
    settings = Settings(
        _env_file=None,
        REDIS_URL="redis://localhost:6379/0",
        CACHE_TTL=7200,
        CACHE_LRU_MAX_SIZE=256,
    )

    assert settings.REDIS_URL == "redis://localhost:6379/0"
    assert settings.CACHE_TTL == 7200
    assert settings.CACHE_LRU_MAX_SIZE == 256


def test_ollama_base_url_requires_http_transport():
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
        Settings(_env_file=None, OLLAMA_BASE_URL="172.20.0.1:11434")


def test_legacy_docling_switch_maps_to_extraction_mode():
    fast = Settings(_env_file=None, PDF_USE_DOCLING=False)
    quality = Settings(_env_file=None, PDF_USE_DOCLING=True)

    assert fast.PDF_EXTRACTION_MODE == "fast"
    assert quality.PDF_EXTRACTION_MODE == "quality"


def test_long_local_generation_timeouts_are_supported():
    settings = Settings(
        _env_file=None,
        LLM_REQUEST_TIMEOUT_SECONDS=900,
        RESEARCH_TIMEOUT_SECONDS=1800,
        RESEARCH_SYNTHESIS_RESERVE_SECONDS=600,
    )

    assert settings.LLM_REQUEST_TIMEOUT_SECONDS == 900
    assert settings.RESEARCH_TIMEOUT_SECONDS == 1800


def test_timeout_limits_still_reject_unbounded_values():
    with pytest.raises(ValueError, match="LLM_REQUEST_TIMEOUT_SECONDS"):
        Settings(_env_file=None, LLM_REQUEST_TIMEOUT_SECONDS=3601)

    with pytest.raises(ValueError, match="RESEARCH_TIMEOUT_SECONDS"):
        Settings(_env_file=None, RESEARCH_TIMEOUT_SECONDS=7201)
