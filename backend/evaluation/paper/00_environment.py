"""Tabla 2 — captura el entorno de cómputo real de esta máquina.

Registra hardware, versiones de software, configuración efectiva de ATLAS
(la que resulta de .env + MODEL_PROFILE, no la que se cree tener) y los
metadatos del modelo servido por Ollama (cuantización, tamaño, VRAM usada).

    python evaluation/paper/00_environment.py --label A-mac

Con MODEL_PROFILE=lan_client (ATLAS en un cliente, Ollama en el servidor con
la T4) este script se ejecuta en el CLIENTE: describe el cliente, consulta al
servidor por OLLAMA_BASE_URL (modelos, cuantización, VRAM, latencia de red) y
completa CPU/RAM/GPU del servidor con el JSON de 00_server_info.ps1:

    python evaluation/paper/00_environment.py --label B-t4 --server-info evaluation/paper/server_info.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

import _common

SETTINGS_KEYS = (
    "MODEL_PROFILE", "LLM_MODEL", "LLM_CONTEXT_WINDOW", "LLM_MAX_NEW_TOKENS", "LLM_DISABLE_THINKING",
    "DEVICE", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION", "EMBEDDING_DEVICE", "EMBEDDING_BATCH_SIZE",
    "RERANKER_MODEL", "RERANKER_DEVICE", "RERANK_MAX_LENGTH", "CPU_THREADS",
    "CORPUS_STORE_BACKEND", "PDF_EXTRACTION_MODE", "CHUNK_SIZE", "CHUNK_OVERLAP",
    "PARENT_CHUNK_SIZE", "TOP_K_RETRIEVAL", "TOP_K_RERANK", "RESEARCH_MAX_ROUNDS",
    "RESEARCH_MAX_FACETS", "RESEARCH_TIMEOUT_SECONDS", "GRAPH_MAX_RELATIONSHIPS",
    "NLI_MODEL", "LANGFUSE_ENABLED", "EMBEDDING_BACKEND", "EMBEDDING_OLLAMA_MODEL",
    "OLLAMA_BASE_URL", "OLLAMA_KEEP_ALIVE", "AGENT_MAX_ITERATIONS",
)
PACKAGES = (
    "torch", "transformers", "sentence-transformers", "langchain-core", "langchain-ollama",
    "langgraph", "chromadb", "psycopg", "pgvector", "sqlalchemy", "fastapi", "spacy",
    "networkx", "docling", "pymupdf", "ragas", "ollama",
)


def run(cmd: list[str], timeout: int = 20) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
    except Exception:
        return None


def is_wsl() -> bool:
    try:
        return "microsoft" in open("/proc/version", encoding="utf-8").read().lower()
    except OSError:
        return False


def cpu_and_memory() -> Dict[str, Any]:
    import psutil

    info: Dict[str, Any] = {
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_visible_gb": round(psutil.virtual_memory().total / 1024**3, 1),
    }
    system = platform.system()
    if system == "Darwin":
        info["cpu"] = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        memsize = run(["sysctl", "-n", "hw.memsize"])
        info["ram_host_gb"] = round(int(memsize) / 1024**3, 1) if memsize else None
        perf = run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        eff = run(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
        if perf and eff:
            info["core_layout"] = f"{perf} rendimiento + {eff} eficiencia"
        displays = run(["system_profiler", "SPDisplaysDataType"], timeout=60) or ""
        match = re.search(r"Total Number of Cores:\s*(\d+)", displays)
        info["gpu"] = f"GPU integrada Apple ({match.group(1)} núcleos), memoria unificada" if match else "GPU integrada Apple"
    elif system == "Linux":
        try:
            cpuinfo = open("/proc/cpuinfo", encoding="utf-8").read()
            match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            info["cpu"] = match.group(1).strip() if match else platform.processor()
        except OSError:
            info["cpu"] = platform.processor()
    else:
        info["cpu"] = platform.processor()

    if is_wsl() or system == "Windows":
        # WSL only sees the RAM granted in .wslconfig; ask Windows for the host values.
        shell = "powershell.exe" if shutil.which("powershell.exe") else "powershell"
        host_ram = run([shell, "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        host_cpu = run([shell, "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"])
        sockets = run([shell, "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_Processor | Measure-Object).Count"])
        if host_ram and host_ram.strip().isdigit():
            info["ram_host_gb"] = round(int(host_ram.strip()) / 1024**3, 1)
        if host_cpu:
            info["cpu_host"] = host_cpu.strip()
        if sockets and sockets.strip().isdigit():
            info["cpu_sockets_host"] = int(sockets.strip())
    return info


def nvidia() -> Optional[Dict[str, Any]]:
    query = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"])
    if not query:
        return None
    header = run(["nvidia-smi"]) or ""
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", header)
    gpus = []
    for line in query.splitlines():
        name, memory, driver = [part.strip() for part in line.split(",")]
        gpus.append({"name": name, "memory_gb": round(int(memory) / 1024, 1), "driver": driver})
    return {"gpus": gpus, "cuda_driver_version": cuda.group(1) if cuda else None}


def torch_info() -> Dict[str, Any]:
    try:
        import torch

        return {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "threads": torch.get_num_threads(),
        }
    except Exception as exc:
        return {"error": str(exc)}


OLLAMA_URL: Optional[str] = None  # set from settings.OLLAMA_BASE_URL (lan_client) in main()


def ollama_base_url() -> str:
    if OLLAMA_URL:
        return OLLAMA_URL.rstrip("/")
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    if not re.search(r":\d+$", host.split("//", 1)[1]):
        host += ":11434"
    return host.rstrip("/")


def ollama_request(path: str, payload: Optional[dict] = None, timeout: int = 600) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        ollama_base_url() + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def ollama_rtt_ms(samples: int = 20) -> Dict[str, Optional[float]]:
    """Round trip of a trivial request: the network + HTTP floor every LLM and
    embedding call pays when Ollama runs on another machine (lan_client)."""
    import time

    values = []
    for _ in range(samples):
        start = time.perf_counter()
        ollama_request("/api/version", timeout=10)
        values.append((time.perf_counter() - start) * 1000)
    return {"median": round(_common.percentile(values, 50), 2),
            "p95": round(_common.percentile(values, 95), 2), "samples": samples}


def ollama_model_details(model: str) -> Dict[str, Any]:
    show = ollama_request("/api/show", {"model": model})
    tags = ollama_request("/api/tags").get("models", [])
    match = next((m for m in tags if m.get("name") == model or m.get("model") == model), {})
    model_info = show.get("model_info", {}) or {}
    return {
        "model": model,
        "details": show.get("details", {}),
        "native_context_length": next(
            (value for key, value in model_info.items() if key.endswith(".context_length")), None
        ),
        "digest": match.get("digest"),
        "size_gb": round(match.get("size", 0) / 1024**3, 2) if match else None,
    }


def ollama_info(model: str, context_window: int, embedding_model: Optional[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"base_url": ollama_base_url()}
    try:
        info["version"] = ollama_request("/api/version").get("version")
        info["rtt_ms"] = ollama_rtt_ms()
        if embedding_model:
            info["embedding"] = ollama_model_details(embedding_model)
        info.update(ollama_model_details(model))
        # Load the model with the same num_ctx ATLAS requests, then read the CPU/GPU split.
        generation = ollama_request("/api/generate", {
            "model": model, "prompt": "ok", "stream": False,
            "options": {"num_ctx": context_window, "num_predict": 1},
        })
        info["load_seconds_cold_or_warm"] = round(generation.get("load_duration", 0) / 1e9, 2)
        running = ollama_request("/api/ps").get("models", [])
        loaded = next((m for m in running if m.get("name") == model or m.get("model") == model), None)
        if loaded:
            size, vram = loaded.get("size", 0), loaded.get("size_vram", 0)
            info["loaded_size_gb"] = round(size / 1024**3, 2)
            info["loaded_vram_gb"] = round(vram / 1024**3, 2)
            info["gpu_offload_pct"] = round(100 * vram / size, 1) if size else None
            info["loaded_context_length"] = loaded.get("context_length")
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def database_info(settings) -> Dict[str, Any]:
    url = str(settings.DATABASE_URL)
    info: Dict[str, Any] = {"scheme": url.split(":", 1)[0]}
    try:
        from sqlalchemy import text

        from app.database import engine

        with engine.connect() as connection:
            if url.startswith("postgresql"):
                info["server_version"] = connection.execute(text("SHOW server_version")).scalar()
                info["pgvector"] = connection.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ).scalar()
            else:
                info["sqlite_version"] = connection.execute(text("select sqlite_version()")).scalar()
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def package_versions() -> Dict[str, Optional[str]]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def to_markdown(env: Dict[str, Any]) -> str:
    hw, cfg, oll = env["hardware"], env["atlas_settings"], env["ollama"]
    gpu = hw.get("gpu") or "—"
    if env.get("nvidia"):
        gpu = "; ".join(f"{g['name']}, {g['memory_gb']} GB (driver {g['driver']})" for g in env["nvidia"]["gpus"])
        gpu += f"; CUDA (driver) {env['nvidia'].get('cuda_driver_version')}"
    details = oll.get("details", {})
    server = env.get("inference_server") or {}
    remote = bool(cfg.get("OLLAMA_BASE_URL"))
    client = " (cliente ATLAS)" if remote else ""
    if remote:
        gpu = "— (sin GPU en el cliente; inferencia en el servidor)"
    embedding = f"{cfg.get('EMBEDDING_MODEL')} ({cfg.get('EMBEDDING_DIMENSION')} dim., {cfg.get('EMBEDDING_DEVICE')})"
    if cfg.get("EMBEDDING_BACKEND") == "ollama":
        emb = oll.get("embedding", {})
        embedding = (f"{cfg.get('EMBEDDING_OLLAMA_MODEL')} vía Ollama en el servidor "
                     f"({emb.get('details', {}).get('quantization_level', '?')}, {cfg.get('EMBEDDING_DIMENSION')} dim.)")
    store = (
        f"PostgreSQL {env['database'].get('server_version')} + pgvector {env['database'].get('pgvector')} (HNSW)"
        if cfg.get("CORPUS_STORE_BACKEND") == "postgres" else "Chroma (HNSW, coseno)"
    )
    lexical = "tsvector + ts_rank_cd" if cfg.get("CORPUS_STORE_BACKEND") == "postgres" else (
        f"SQLite {env['database'].get('sqlite_version')} FTS5 (BM25)"
    )
    rows = [
        ("CPU / SoC" + client, f"{hw.get('cpu_host') or hw.get('cpu')} — {hw.get('physical_cores')} núcleos físicos / "
                      f"{hw.get('logical_cores')} lógicos" + (f" ({hw['core_layout']})" if hw.get("core_layout") else "")
                      + (f", {hw['cpu_sockets_host']} sockets" if hw.get("cpu_sockets_host") else "")),
        ("Memoria" + client, f"{hw.get('ram_host_gb') or hw.get('ram_visible_gb')} GB"
                    + (f" (visibles para el backend: {hw['ram_visible_gb']} GB)"
                       if hw.get("ram_host_gb") and abs(hw["ram_host_gb"] - hw["ram_visible_gb"]) > 1 else "")),
        ("GPU" + client, gpu),
        *([
            ("Servidor de inferencia: CPU", server.get("cpu", "— (ver 00_server_info.ps1)")),
            ("Servidor de inferencia: memoria", f"{server['ram_gb']} GB" if server.get("ram_gb") else "—"),
            ("Servidor de inferencia: GPU", "; ".join(
                f"{g.get('name')}, {g.get('memory_gb')} GB (driver {g.get('driver')})" for g in server.get("gpus", []))
                + (f"; CUDA (driver) {server['cuda_driver_version']}" if server.get("cuda_driver_version") else "")
                or "—"),
            ("Servidor de inferencia: SO", server.get("os", "—")),
            ("Red cliente → Ollama (RTT)", f"mediana {oll.get('rtt_ms', {}).get('median', '—')} ms, "
                                           f"p95 {oll.get('rtt_ms', {}).get('p95', '—')} ms"),
        ] if remote else []),
        ("Modelo generador", f"{cfg.get('LLM_MODEL')} — {details.get('parameter_size', '?')}, "
                             f"{details.get('quantization_level', '?')}, {details.get('format', '')}"),
        ("Descarga a GPU (Ollama)", f"{oll.get('gpu_offload_pct', '—')} % "
                                    f"({oll.get('loaded_vram_gb', '—')} de {oll.get('loaded_size_gb', '—')} GB)"),
        ("Embeddings", embedding),
        ("Reranker", f"{cfg.get('RERANKER_MODEL')} ({cfg.get('RERANKER_DEVICE')})"),
        ("Almacén vectorial", store),
        ("Índice léxico", lexical),
        ("Extracción de PDF", f"modo {cfg.get('PDF_EXTRACTION_MODE')}"),
        ("Recuperación (TOP_K)", f"{cfg.get('TOP_K_RETRIEVAL')} candidatos / {cfg.get('TOP_K_RERANK')} finales; "
                                 f"máx. {cfg.get('RESEARCH_MAX_FACETS')} preguntas de evidencia"),
        ("Ventana de contexto", f"{cfg.get('LLM_CONTEXT_WINDOW')} tokens (nativa del modelo: {oll.get('native_context_length', '—')})"),
        ("Pensamiento (thinking)", "desactivado" if cfg.get("LLM_DISABLE_THINKING") else "por defecto del modelo"),
        ("Ollama", f"{oll.get('version')}, keep_alive {cfg.get('OLLAMA_KEEP_ALIVE') or 'por defecto del servidor'}"
                   + (f", {server.get('ollama_env')}" if server.get("ollama_env") else "")),
        ("Rama / commit", f"{env.get('git_branch')} / {env['git_commit'][:7]}"),
        ("Python / PyTorch", f"{env['software']['python']} / {env['software']['torch'].get('version')}"),
        ("Sistema operativo", env["software"]["os"]),
    ]
    lines = [f"### Tabla 2 — {env['label']}", "", "| Componente | Valor |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in rows]
    lines += ["", f"Commit: `{env['git_commit']}`" + (" (con cambios locales en backend/app)" if env["git_dirty"] else "")]
    return "\n".join(lines) + "\n"


def settings_remote(settings) -> bool:
    return bool(getattr(settings, "OLLAMA_BASE_URL", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="Nombre de la configuración, p. ej. A-mac o B-t4.")
    parser.add_argument("--skip-ollama", action="store_true", help="No consultar ni cargar el modelo en Ollama.")
    parser.add_argument("--server-info", type=Path,
                        help="JSON del servidor de inferencia generado con 00_server_info.ps1 (perfil lan_client).")
    args = parser.parse_args()

    _common.bootstrap()
    from app.config import get_settings

    settings = get_settings()
    global OLLAMA_URL
    OLLAMA_URL = getattr(settings, "OLLAMA_BASE_URL", "") or None
    wsl = is_wsl()
    embedding_model = (getattr(settings, "EMBEDDING_OLLAMA_MODEL", None)
                       if getattr(settings, "EMBEDDING_BACKEND", "local") == "ollama" else None)
    env: Dict[str, Any] = {
        "label": args.label,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _common.git_commit(),
        "git_dirty": _common.git_dirty(),
        "git_branch": _common.git_branch(),
        "hardware": cpu_and_memory(),
        "nvidia": nvidia(),
        "software": {
            "os": f"{platform.system()} {platform.release()}" + (" (WSL2)" if wsl else ""),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "torch": torch_info(),
            "packages": package_versions(),
        },
        "atlas_settings": {key: getattr(settings, key, None) for key in SETTINGS_KEYS},
        "database": database_info(settings),
        "ollama": {} if args.skip_ollama else ollama_info(
            settings.LLM_MODEL, settings.LLM_CONTEXT_WINDOW, embedding_model),
        "inference_server": (
            json.loads(args.server_info.read_text(encoding="utf-8-sig")) if args.server_info else None
        ),
    }
    out_dir = _common.results_dir(args.label)
    _common.write_json(out_dir / "environment.json", env)
    markdown = to_markdown(env)
    (out_dir / "environment.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    if env["ollama"].get("error"):
        print(f"AVISO Ollama: {env['ollama']['error']} (URL usada: {ollama_base_url()})")
    if settings_remote(settings) and not args.server_info:
        print("AVISO: Ollama es remoto (lan_client). Ejecuta 00_server_info.ps1 en el servidor Windows y "
              "repite con --server-info para completar CPU/RAM/GPU del servidor en la Tabla 2.")
    print(f"Guardado en {out_dir}")


if __name__ == "__main__":
    main()
