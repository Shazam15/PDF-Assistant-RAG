"""Preflight checks for the supported Xeon/Tesla T4 and LAN-client deployments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

from urllib.parse import urlparse

import httpx
import redis
from sqlalchemy import text

from app.config import get_settings
from app.database import engine


REQUIRED_POSTGRES_EXTENSIONS = {"vector", "unaccent", "pg_trgm", "uuid-ossp"}
T4_PROFILES = {"ubuntu_t4", "wsl_t4"}
# lan_client runs ATLAS on a GPU-less machine that reaches Ollama over the LAN.
SUPPORTED_PROFILES = T4_PROFILES | {"lan_client"}
PROFILE_LABELS = {
    "ubuntu_t4": "Ubuntu/T4",
    "wsl_t4": "WSL/T4",
    "lan_client": "LAN client",
}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _windows_host_from_default_route() -> str | None:
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\bvia\s+(\S+)", result.stdout)
    return match.group(1) if match else None


def _ollama_url(profile: str | None = None) -> str:
    settings = get_settings()
    configured = settings.OLLAMA_BASE_URL or os.getenv("OLLAMA_HOST", "").strip().rstrip(
        "/"
    )
    if configured:
        if not configured.startswith(("http://", "https://")):
            return f"http://{configured}"
        return configured
    if (profile or settings.MODEL_PROFILE) == "wsl_t4":
        windows_host = _windows_host_from_default_route()
        if windows_host:
            return f"http://{windows_host}:11434"
    return "http://127.0.0.1:11434"


def _report(ok: bool, message: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {message}")
    return ok


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _check_profile(expected_profile: str | None = None) -> bool:
    settings = get_settings()
    profile = expected_profile or str(settings.MODEL_PROFILE).lower()
    if profile not in SUPPORTED_PROFILES:
        return _report(False, f"Unsupported runtime profile: {profile}")

    if profile == "lan_client":
        # The remote host owns the models, so the model tags are verified
        # against what Ollama actually serves instead of being pinned here.
        # What this profile does pin is that nothing tries to use a local GPU.
        expected = {
            "MODEL_PROFILE": profile,
            "DEVICE": "cpu",
            "EMBEDDING_DEVICE": "cpu",
            "RERANKER_DEVICE": "cpu",
        }
    else:
        expected = {
            "MODEL_PROFILE": profile,
            "DEVICE": "cpu",
            "EMBEDDING_DEVICE": "cpu",
            "RERANKER_DEVICE": "cpu",
            "EMBEDDING_DIMENSION": 1024,
            "EMBEDDING_INDEX_VERSION": "hierarchical-qwen3-1024-v1",
            "LLM_MODEL": "qwen3:14b-q4_K_M",
            "LLM_CONTEXT_WINDOW": 8192,
            "LLM_MAX_NEW_TOKENS": 3072,
            "LLM_DISABLE_THINKING": True,
        }
    mismatches = [
        f"{field}={getattr(settings, field)!r} (expected {value!r})"
        for field, value in expected.items()
        if getattr(settings, field) != value
    ]
    if mismatches:
        label = "LAN client" if profile == "lan_client" else "T4"
        return _report(False, f"{label} profile mismatch: " + "; ".join(mismatches))

    if profile == "lan_client":
        if not settings.OLLAMA_BASE_URL:
            return _report(
                False,
                "lan_client requires OLLAMA_BASE_URL to name the remote Ollama host "
                "(e.g. http://192.168.1.10:11434)",
            )
        if (urlparse(settings.OLLAMA_BASE_URL).hostname or "") in LOOPBACK_HOSTS:
            _warn(
                f"OLLAMA_BASE_URL={settings.OLLAMA_BASE_URL} points at this machine; "
                "lan_client expects the Ollama host to be a separate box on the LAN"
            )
        if settings.EMBEDDING_BACKEND != "ollama":
            _warn(
                "EMBEDDING_BACKEND=local computes embeddings on this CPU; set it to "
                "'ollama' to offload them to the same host that serves the LLM"
            )
        return _report(
            True,
            (
                f"profile={settings.MODEL_PROFILE}, llm={settings.LLM_MODEL}, "
                f"embeddings={settings.EMBEDDING_BACKEND}, "
                f"CPU threads={settings.CPU_THREADS or 'auto'}"
            ),
        )

    return _report(
        True,
        (
            f"profile={settings.MODEL_PROFILE}, llm={settings.LLM_MODEL}, "
            f"CPU threads={settings.CPU_THREADS}"
        ),
    )


def _check_cpu_features() -> None:
    """Advisory only: lan_client exists precisely for old client CPUs."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return
    flags: set[str] = set()
    for line in cpuinfo.splitlines():
        if line.startswith("flags") and ":" in line:
            flags = set(line.split(":", 1)[1].split())
            break
    if not flags:
        return
    if "avx" not in flags:
        _warn(
            "This CPU reports no AVX support. The prebuilt PyTorch wheels behind the "
            "reranker, the NLI verifier and docling may fault with an illegal "
            "instruction; embeddings and the LLM are already offloaded to Ollama"
        )
    elif "avx2" not in flags:
        _warn(
            "This CPU has AVX but no AVX2. Use the AVX2-compatible frontend toolchain "
            "described in frontend/README-avx2-fallback.md"
        )


def _check_nvidia() -> bool:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _report(False, f"NVIDIA driver or nvidia-smi is unavailable: {exc}")

    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    t4_devices = [device for device in devices if "T4" in device.upper()]
    if not t4_devices:
        available = "; ".join(devices) or "none"
        return _report(False, f"Tesla T4 was not detected (available: {available})")
    return _report(True, f"NVIDIA GPU available: {t4_devices[0]}")


def _check_ollama(profile: str | None = None) -> bool:
    settings = get_settings()
    base_url = _ollama_url(profile)
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return _report(False, f"Ollama is unreachable at {base_url}: {exc}")

    names = {
        str(model.get("name") or model.get("model") or "")
        for model in payload.get("models", [])
        if isinstance(model, dict)
    }
    if settings.LLM_MODEL not in names:
        available = ", ".join(sorted(name for name in names if name)) or "none"
        return _report(
            False,
            (
                f"Ollama responded, but {settings.LLM_MODEL!r} is not installed "
                f"(available: {available})"
            ),
        )
    if settings.EMBEDDING_BACKEND == "ollama" and settings.EMBEDDING_OLLAMA_MODEL not in names:
        available = ", ".join(sorted(name for name in names if name)) or "none"
        return _report(
            False,
            (
                f"Ollama responded, but embedding model {settings.EMBEDDING_OLLAMA_MODEL!r} "
                f"is not installed (available: {available})"
            ),
        )
    return _report(True, f"Ollama at {base_url} provides {settings.LLM_MODEL}")


def _check_embedding_dimension(profile: str | None = None) -> bool:
    """Verify the remote embedding model emits vectors the index can store.

    A tag that looks right can still return a different vector width than
    EMBEDDING_DIMENSION, which only surfaces later as an insert failure or a
    silently unusable index. One probe here is cheaper than a reindex.
    """
    settings = get_settings()
    base_url = _ollama_url(profile)
    try:
        response = httpx.post(
            f"{base_url}/api/embed",
            json={"model": settings.EMBEDDING_OLLAMA_MODEL, "input": "dimension probe"},
            timeout=180,
        )
        response.raise_for_status()
        dimension = len(response.json()["embeddings"][0])
    except Exception as exc:
        return _report(
            False,
            f"Could not embed a probe with {settings.EMBEDDING_OLLAMA_MODEL!r} at {base_url}: {exc}",
        )

    if dimension != settings.EMBEDDING_DIMENSION:
        return _report(
            False,
            (
                f"{settings.EMBEDDING_OLLAMA_MODEL!r} returns {dimension}-dimensional "
                f"vectors but EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION}"
            ),
        )
    return _report(
        True,
        f"Remote embeddings via {settings.EMBEDDING_OLLAMA_MODEL} return {dimension} dimensions",
    )


def _check_database(profile: str | None = None) -> bool:
    settings = get_settings()
    profile = profile or str(settings.MODEL_PROFILE).lower()
    if not settings.DATABASE_URL.startswith("postgresql"):
        if profile == "lan_client":
            # A spare client box may legitimately run the file-backed corpus
            # store instead of hosting PostgreSQL itself.
            _warn(
                "DATABASE_URL does not point to PostgreSQL; ATLAS will use the "
                f"{settings.CORPUS_STORE_BACKEND!r} corpus store without pgvector"
            )
            return True
        return _report(
            False,
            f"DATABASE_URL must point to PostgreSQL for the {profile} profile",
        )
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            extensions = {
                row[0]
                for row in connection.execute(
                    text(
                        "SELECT extname FROM pg_extension "
                        "WHERE extname IN ('vector', 'unaccent', 'pg_trgm', 'uuid-ossp')"
                    ),
                )
            }
    except Exception as exc:
        return _report(False, f"PostgreSQL is unavailable: {exc}")

    missing = REQUIRED_POSTGRES_EXTENSIONS - extensions
    if missing:
        return _report(
            False,
            "PostgreSQL is missing extensions: " + ", ".join(sorted(missing)),
        )
    return _report(True, "PostgreSQL and pgvector extensions are available")


def _check_redis() -> bool:
    settings = get_settings()
    if not settings.CELERY_ENABLED:
        return _report(True, "Celery is disabled; Redis is optional")
    try:
        client = redis.Redis.from_url(
            settings.CELERY_BROKER_URL,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        client.ping()
    except Exception as exc:
        return _report(False, f"Redis is unavailable for Celery: {exc}")
    return _report(True, "Redis is available for Celery document processing")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(SUPPORTED_PROFILES))
    args = parser.parse_args(argv)

    settings = get_settings()
    profile = args.profile or str(settings.MODEL_PROFILE).lower()
    label = PROFILE_LABELS.get(profile, profile)
    print(f"ATLAS {label} runtime check")

    if profile == "wsl_t4":
        project_path = Path.cwd().resolve()
        if str(project_path).startswith("/mnt/"):
            _warn(
                f"Project is under {project_path}; use the WSL filesystem for better I/O performance"
            )
    if profile == "lan_client":
        _check_cpu_features()

    checks = [_check_profile(profile)]
    if profile == "ubuntu_t4":
        checks.append(_check_nvidia())
    checks.append(_check_ollama(profile))
    if settings.EMBEDDING_BACKEND == "ollama":
        checks.append(_check_embedding_dimension(profile))
    checks.extend(
        [
            _check_database(profile),
            _check_redis(),
        ]
    )
    if all(checks):
        print(f"ATLAS is ready to start in {label} mode.")
        return 0
    print("ATLAS preflight failed. Resolve the FAIL entries before starting.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
