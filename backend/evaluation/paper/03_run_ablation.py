"""Tablas 3–7 — ejecuta las condiciones C1–C5 sobre el conjunto de preguntas.

Fase 1 (siempre): rankings a profundidad fija (densa, RRF, reordenada) para
Recall@20 / nDCG@10. No usa el LLM y es determinista.

Fase 2 (salvo --retrieval-only): cada pregunta × condición × repetición pasa
por el código real de ATLAS. Las condiciones C1–C3 desactivan componentes
intercambiando funciones en tiempo de ejecución; C4 es ``simple_rag`` tal
como está en producción y C5 es ``research_rag``. Por ejecución se guardan
respuesta, evidencia entregada al generador, eventos de verificación,
latencia extremo a extremo, tiempo por etapa y tokens de Ollama.

Es reanudable: si se interrumpe, al relanzar salta lo ya registrado.

    python evaluation/paper/03_run_ablation.py --label A-mac --user-email tu@correo --limit 2   # prueba
    python evaluation/paper/03_run_ablation.py --label A-mac --user-email tu@correo --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import _common
import _instrument

NO_EVIDENCE_PREFIXES = (
    "No encontré información suficiente",
    "No fue posible generar una respuesta con citas verificables",
    "No fue posible producir una síntesis",
    "No se recuperó evidencia documental suficiente",
    "La investigación alcanzó su presupuesto temporal",
    "Los documentos recuperados no contienen texto utilizable",
)


def settings_snapshot(settings) -> Dict[str, Any]:
    keys = (
        "MODEL_PROFILE", "LLM_MODEL", "LLM_CONTEXT_WINDOW", "LLM_MAX_NEW_TOKENS", "LLM_DISABLE_THINKING",
        "AGENT_SYNTHESIS_MAX_TOKENS", "EMBEDDING_MODEL", "EMBEDDING_DEVICE", "RERANKER_MODEL",
        "RERANKER_DEVICE", "CORPUS_STORE_BACKEND", "PDF_EXTRACTION_MODE", "TOP_K_RETRIEVAL", "TOP_K_RERANK",
        "RESEARCH_MAX_ROUNDS", "RESEARCH_MAX_FACETS", "RESEARCH_TIMEOUT_SECONDS", "GRAPH_MAX_RELATIONSHIPS",
        "LANGFUSE_ENABLED", "EMBEDDING_BACKEND", "EMBEDDING_OLLAMA_MODEL", "OLLAMA_KEEP_ALIVE",
        "AGENT_MAX_ITERATIONS", "CPU_THREADS",
    )
    snapshot = {key: getattr(settings, key, None) for key in keys}
    snapshot["OLLAMA_REMOTE"] = bool(getattr(settings, "OLLAMA_BASE_URL", ""))
    return snapshot


def run_phase_rankings(questions, user_id: str, path: Path, depth: int) -> None:
    done = {row["qid"] for row in _common.read_jsonl(path)}
    pending = [q for q in questions if q["id"] not in done]
    if not pending:
        print(f"[fase 1] rankings ya calculados ({len(done)})")
        return
    print(f"[fase 1] rankings a profundidad {depth} para {len(pending)} preguntas")
    for index, question in enumerate(pending, 1):
        start = time.perf_counter()
        ranked = _instrument.rankings(question["question"], user_id, depth)
        _common.append_jsonl(path, {
            "qid": question["id"],
            "rankings": {name: [_instrument.chunk_record(c) for c in chunks] for name, chunks in ranked.items()},
            "seconds": round(time.perf_counter() - start, 3),
        })
        print(f"  {index}/{len(pending)} {question['id']} ({time.perf_counter() - start:.1f}s)")


def execute(condition: str, question: str, user_id: str) -> Dict[str, Any]:
    from app.rag import agent

    if condition == "C5":
        decision = agent.RoutingDecision("research_rag", "paper_eval", 0, "research", None)
        result = agent.run_research_agent(
            question=question,
            user_id=user_id,
            document_id=None,
            top_k=agent._effective_retrieval_top_k(decision, None),
            chat_history=None,
            dependencies=agent._research_dependencies(question, None, None),
        )
        return {
            "route": "research_rag",
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "evidence": result.get("evidence", []),
            "coverage": result.get("coverage", {}),
        }

    decision = agent.RoutingDecision("simple_rag", "paper_eval", 0, "quick", None)
    with _instrument.condition(condition):
        result = agent._execute_document_route(decision, question, user_id, None, None, None, None)
    return {
        "route": "simple_rag",
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "evidence": list(_instrument.REC.delivered or []),
        "coverage": {},
    }


def ollama_version(settings) -> str:
    """Version of the Ollama server the backend actually talks to (lan_client or local)."""
    base = getattr(settings, "OLLAMA_BASE_URL", "") or os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/api/version", timeout=10) as response:
            return json.loads(response.read().decode()).get("version", "?")
    except Exception as exc:
        return f"unreachable ({type(exc).__name__})"


def run_one(condition: str, question: Dict[str, Any], repeat: int, user_id: str, label: str) -> Dict[str, Any]:
    import psutil

    rec = _instrument.REC
    rec.reset()
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    error = None
    try:
        output = execute(condition, question["question"], user_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        output = {"route": "research_rag" if condition == "C5" else "simple_rag",
                  "answer": "", "sources": [], "evidence": [], "coverage": {}}
    latency = time.perf_counter() - start

    answer = output["answer"] or ""
    stages = rec.stages
    contexts = [
        chunk.get("context_text") or chunk.get("parent_text") or chunk.get("text", "")
        for chunk in output["evidence"]
    ]
    if rec.graph_context:
        contexts.append(rec.graph_context)
    t0 = start
    return {
        "key": f"{question['id']}|{condition}|{repeat}",
        "label": label,
        "qid": question["id"],
        "condition": condition,
        "repeat": repeat,
        "route": output["route"],
        "question": question["question"],
        "reference": question["reference"],
        "answer": answer,
        "abstained": (not answer.strip()) or answer.strip().startswith(NO_EVIDENCE_PREFIXES),
        "sources": [
            {k: s.get(k) for k in ("source_id", "document_id", "filename", "page", "page_start", "page_end")}
            for s in output["sources"]
        ],
        "evidence": [_instrument.chunk_record(c) for c in output["evidence"]],
        "contexts": contexts,
        "graph_context_chars": len(rec.graph_context),
        "coverage": output["coverage"],
        "rounds": sum(1 for s in stages if s["name"] == "retrieve") if condition == "C5" else 1,
        "repairs": sum(1 for s in stages if s["name"] == "repair"),
        "events": [{**e, "t": round(e["t"] - t0, 3)} for e in rec.events],
        "stages": [{**s, "seconds": round(s["seconds"], 3)} for s in stages],
        "llm_calls": rec.llm_calls,
        "latency_seconds": round(latency, 3),
        "retrieval_seconds": round(rec.retrieve_seconds, 3),
        "graph_seconds": round(rec.graph_seconds, 3),
        "rss_mb": round(psutil.Process().memory_info().rss / 1024**2, 1),
        "error": error,
        "started_at": started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="Configuración: A-mac, B-t4, B-t4-4b...")
    _common.add_user_args(parser)
    parser.add_argument("--dataset", type=Path, default=_common.DEFAULT_DATASET)
    parser.add_argument("--conditions", default="C1,C2,C3,C4,C5")
    parser.add_argument("--repeats", type=int, default=1, help="Repeticiones por pregunta y condición.")
    parser.add_argument("--limit", type=int, help="Solo las primeras N preguntas (para probar).")
    parser.add_argument("--depth", type=int, default=20, help="Profundidad de los rankings de la fase 1.")
    parser.add_argument("--retrieval-only", action="store_true", help="Solo fase 1 (sin LLM).")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=13, help="Semilla del orden aleatorio de ejecución.")
    parser.add_argument("--verbose", action="store_true", help="Logs INFO del backend.")
    parser.add_argument("--expect-ollama", help="Versión de Ollama exigida (p. ej. 0.33.0); la corrida se detiene "
                                                "si el servidor reporta otra, también a mitad de la corrida.")
    args = parser.parse_args()

    _common.bootstrap(verbose=args.verbose)
    from app.config import get_settings

    settings = get_settings()
    _instrument.install()
    user_id = _common.resolve_user_id(args.user_email, args.user_id)
    questions = _common.load_questions(args.dataset, args.limit)
    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(_instrument.CONDITIONS)
    if unknown:
        raise SystemExit(f"Condiciones desconocidas: {sorted(unknown)}")

    out_dir = _common.results_dir(args.label)
    meta_path = out_dir / "run_meta.json"
    meta = {
        "label": args.label,
        "dataset": str(args.dataset),
        "questions": len(questions),
        "conditions": {c: _instrument.CONDITIONS[c] for c in conditions},
        "repeats": args.repeats,
        "seed": args.seed,
        "temperature": 0,
        "git_commit": _common.git_commit(),
        "git_dirty": _common.git_dirty(),
        "settings": settings_snapshot(settings),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if meta_path.exists():
        import json

        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("settings") != meta["settings"]:
            print("AVISO: la configuración efectiva cambió respecto a ejecuciones previas de esta etiqueta:")
            for key, value in meta["settings"].items():
                if old.get("settings", {}).get(key) != value:
                    print(f"   {key}: {old['settings'].get(key)!r} -> {value!r}")
            print("   Usa otra --label si es una configuración distinta.")
    meta["ollama_version"] = ollama_version(settings)
    if args.expect_ollama and meta["ollama_version"] != args.expect_ollama:
        raise SystemExit(f"Ollama reporta {meta['ollama_version']}, se esperaba {args.expect_ollama}. "
                         "No se ejecuta nada para no mezclar versiones.")
    _common.write_json(meta_path, meta)

    run_phase_rankings(questions, user_id, out_dir / "rankings.jsonl", args.depth)
    if args.retrieval_only:
        return

    runs_path = out_dir / "runs.jsonl"
    done = {row["key"] for row in _common.read_jsonl(runs_path) if not row.get("error")}
    plan: List[tuple] = []
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        # Shuffle question and condition order per repeat so neither model
        # warm-up nor Ollama's prefix cache systematically favours one condition.
        order = list(questions)
        rng.shuffle(order)
        for question in order:
            conds = list(conditions)
            rng.shuffle(conds)
            for condition in conds:
                if f"{question['id']}|{condition}|{repeat}" not in done:
                    plan.append((condition, question, repeat))
    if not plan:
        print("[fase 2] nada pendiente")
        return

    if not args.no_warmup:
        print("[fase 2] calentamiento (no se registra): carga de embeddings, reranker y modelo en Ollama")
        warm = run_one("C4", questions[0], -1, user_id, args.label)
        print(f"  calentamiento {warm['latency_seconds']:.1f}s" + (f" ERROR {warm['error']}" if warm["error"] else ""))

    print(f"[fase 2] {len(plan)} ejecuciones pendientes ({len(done)} ya registradas)")
    started = time.perf_counter()
    for index, (condition, question, repeat) in enumerate(plan, 1):
        current = ollama_version(settings)
        if args.expect_ollama and current != args.expect_ollama:
            print(f"DETENIDO: Ollama cambió a {current} (se esperaba {args.expect_ollama}). "
                  "Lo ya registrado es válido; restaura la versión y relanza para continuar.")
            break
        record = run_one(condition, question, repeat, user_id, args.label)
        record["ollama_version"] = current
        _common.append_jsonl(runs_path, record)
        elapsed = time.perf_counter() - started
        eta = elapsed / index * (len(plan) - index)
        status = f"ERROR {record['error']}" if record["error"] else (
            f"{record['latency_seconds']:.1f}s, {len(record['evidence'])} frag., rondas={record['rounds']}, "
            f"rep={record['repairs']}" + (", abstención" if record["abstained"] else "")
        )
        print(f"  {index}/{len(plan)} {question['id']} {condition} r{repeat}: {status} | ETA {eta / 60:.0f} min")
    print(f"Guardado en {runs_path}")


if __name__ == "__main__":
    main()
