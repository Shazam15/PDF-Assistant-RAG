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


def test_local_balanced_profile_keeps_qwen_and_uses_small_mps_batches():
    settings = Settings(_env_file=None, MODEL_PROFILE="local_balanced")

    assert settings.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.EMBEDDING_DIMENSION == 1024
    assert settings.EMBEDDING_DEVICE == "mps"
    assert settings.EMBEDDING_BATCH_SIZE == 4
    assert settings.CPU_THREADS == 4


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
