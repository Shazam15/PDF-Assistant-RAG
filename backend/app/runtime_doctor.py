"""Preflight checks for the native WSL backend and Windows Ollama split."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

import httpx
from sqlalchemy import text

from app.config import get_settings
from app.database import engine


REQUIRED_POSTGRES_EXTENSIONS = {"vector", "unaccent", "pg_trgm", "uuid-ossp"}


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


def _ollama_url() -> str:
    settings = get_settings()
    configured = settings.OLLAMA_BASE_URL or os.getenv("OLLAMA_HOST", "").strip().rstrip(
        "/"
    )
    if configured:
        return configured
    windows_host = _windows_host_from_default_route()
    if windows_host:
        return f"http://{windows_host}:11434"
    return "http://127.0.0.1:11434"


def _report(ok: bool, message: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {message}")
    return ok


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _check_profile() -> bool:
    settings = get_settings()
    expected = {
        "MODEL_PROFILE": "wsl_t4",
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
        return _report(False, "WSL/T4 profile mismatch: " + "; ".join(mismatches))
    return _report(
        True,
        (
            f"profile={settings.MODEL_PROFILE}, llm={settings.LLM_MODEL}, "
            f"CPU threads={settings.CPU_THREADS}"
        ),
    )


def _check_ollama() -> bool:
    settings = get_settings()
    base_url = _ollama_url()
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


def _check_database() -> bool:
    settings = get_settings()
    if not settings.DATABASE_URL.startswith("postgresql"):
        return _report(False, "DATABASE_URL must point to PostgreSQL for the wsl_t4 profile")
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


def main() -> int:
    print("ATLAS WSL/T4 runtime check")
    project_path = Path.cwd().resolve()
    if str(project_path).startswith("/mnt/"):
        _warn(
            f"Project is under {project_path}; use the WSL filesystem for better I/O performance"
        )

    checks = [_check_profile(), _check_ollama(), _check_database()]
    if all(checks):
        print("ATLAS is ready to start in WSL/T4 mode.")
        return 0
    print("ATLAS preflight failed. Resolve the FAIL entries before starting.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
