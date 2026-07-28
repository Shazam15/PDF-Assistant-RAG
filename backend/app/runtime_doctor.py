"""Preflight checks for the supported Xeon/Tesla T4 deployments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

import httpx
import redis
from sqlalchemy import text

from app.config import get_settings
from app.database import engine


REQUIRED_POSTGRES_EXTENSIONS = {"vector", "unaccent", "pg_trgm", "uuid-ossp"}
T4_PROFILES = {"ubuntu_t4", "wsl_t4"}


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
    if profile not in T4_PROFILES:
        return _report(False, f"Unsupported T4 runtime profile: {profile}")

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
        return _report(False, "T4 profile mismatch: " + "; ".join(mismatches))
    return _report(
        True,
        (
            f"profile={settings.MODEL_PROFILE}, llm={settings.LLM_MODEL}, "
            f"CPU threads={settings.CPU_THREADS}"
        ),
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
    return _report(True, f"Ollama at {base_url} provides {settings.LLM_MODEL}")


def _check_database(profile: str | None = None) -> bool:
    settings = get_settings()
    if not settings.DATABASE_URL.startswith("postgresql"):
        return _report(
            False,
            f"DATABASE_URL must point to PostgreSQL for the {profile or settings.MODEL_PROFILE} profile",
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
    parser.add_argument("--profile", choices=sorted(T4_PROFILES))
    args = parser.parse_args(argv)

    settings = get_settings()
    profile = args.profile or str(settings.MODEL_PROFILE).lower()
    label = "Ubuntu/T4" if profile == "ubuntu_t4" else "WSL/T4"
    print(f"ATLAS {label} runtime check")

    if profile == "wsl_t4":
        project_path = Path.cwd().resolve()
        if str(project_path).startswith("/mnt/"):
            _warn(
                f"Project is under {project_path}; use the WSL filesystem for better I/O performance"
            )

    checks = [_check_profile(profile)]
    if profile == "ubuntu_t4":
        checks.append(_check_nvidia())
    checks.extend(
        [
            _check_ollama(profile),
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
