"""Faithfulness / Factual correctness / LLM context recall (RAGAS) con juez local.

Puntúa las ejecuciones de ``runs.jsonl`` con RAGAS usando un juez servido por
Ollama (no envía nada a terceros). Para que las cifras de A y B sean
comparables, puntúa AMBAS configuraciones con el MISMO juez y en la MISMA
máquina (la del T4): copia ``results/A-mac/runs.jsonl`` a la PC Windows y
ejecuta este script dos veces.

    pip install "ragas>=0.3,<0.4"
    python evaluation/paper/04_ragas_score.py --label B-t4 --judge-model qwen3:14b-q4_K_M
    python evaluation/paper/04_ragas_score.py --label A-mac --judge-model qwen3:14b-q4_K_M

Reanudable: guarda una fila por ejecución en ``ragas_scores.jsonl``.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import _common

METRIC_ALIASES = {"faithful": "faithfulness", "factual": "factual_correctness", "recall": "context_recall"}


def ollama_base_url() -> str:
    """OLLAMA_BASE_URL of the backend settings (lan_client), else OLLAMA_HOST."""
    try:
        _common.bootstrap()
        from app.config import get_settings

        configured = getattr(get_settings(), "OLLAMA_BASE_URL", "")
        if configured:
            return configured.rstrip("/")
    except Exception:
        pass
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    if ":" not in host.split("//", 1)[1]:
        host += ":11434"
    return host.rstrip("/")


def bounded_contexts(contexts: List[str], max_total: int) -> List[str]:
    total = sum(len(c) for c in contexts)
    if total <= max_total or not contexts:
        return contexts
    per_context = max(400, max_total // len(contexts))
    return [c[:per_context] for c in contexts]


def build_evaluator(args):
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOllama(
        model=args.judge_model,
        base_url=args.ollama_url or ollama_base_url(),
        temperature=0,
        num_ctx=args.judge_ctx,
        num_predict=args.judge_max_tokens,
        # Only Qwen3-style models accept the think switch; others (e.g. gemma3) must not receive it.
        reasoning=False if "qwen3" in args.judge_model.lower() else None,
        client_kwargs={"timeout": args.timeout},
    )
    return LangchainLLMWrapper(llm)


def score_batch(rows: List[Dict[str, Any]], evaluator, args) -> List[Dict[str, Any]]:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list([
        {
            "user_input": row["question"],
            "retrieved_contexts": bounded_contexts(row.get("contexts") or [""], args.max_context_chars) or [""],
            "response": row["answer"] or "",
            "reference": row["reference"],
        }
        for row in rows
    ])
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), FactualCorrectness(), LLMContextRecall()],
        llm=evaluator,
        run_config=RunConfig(timeout=args.timeout, max_workers=args.workers, max_retries=2),
        raise_exceptions=False,
        show_progress=False,
    )
    frame = result.to_pandas()
    scored = []
    for index, row in enumerate(rows):
        values: Dict[str, Any] = {}
        for column in frame.columns:
            name = next((alias for key, alias in METRIC_ALIASES.items() if key in str(column).lower()), None)
            if name:
                value = frame.iloc[index][column]
                values[name] = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
        scored.append({"key": row["key"], "label": row["label"], "qid": row["qid"],
                       "condition": row["condition"], "repeat": row["repeat"], **values})
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="Carpeta results/<label> con runs.jsonl.")
    parser.add_argument("--judge-model", required=True, help="Modelo de Ollama usado como juez.")
    parser.add_argument("--ollama-url", help="Por defecto OLLAMA_BASE_URL de la .env, luego OLLAMA_HOST.")
    parser.add_argument("--conditions", default="C1,C2,C3,C4,C5")
    parser.add_argument("--judge-ctx", type=int, default=16384, help="num_ctx del juez.")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--max-context-chars", type=int, default=40000,
                        help="Recorte total de contextos para no exceder num_ctx del juez.")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="Peticiones concurrentes al juez.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--include-abstentions", action="store_true",
                        help="Puntuar también respuestas de 'evidencia insuficiente'.")
    args = parser.parse_args()

    out_dir = _common.RESULTS_DIR / args.label
    runs = [r for r in _common.read_jsonl(out_dir / "runs.jsonl")
            if not r.get("error") and r["condition"] in args.conditions.split(",")]
    if not args.include_abstentions:
        runs = [r for r in runs if not r.get("abstained")]
    scores_path = out_dir / "ragas_scores.jsonl"
    done = {row["key"] for row in _common.read_jsonl(scores_path)}
    pending = [r for r in runs if r["key"] not in done]
    _common.write_json(out_dir / "ragas_meta.json", {
        "judge_model": args.judge_model,
        "judge_ctx": args.judge_ctx,
        "max_context_chars": args.max_context_chars,
        "include_abstentions": args.include_abstentions,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    if not pending:
        print("Nada pendiente.")
        return
    print(f"Puntuando {len(pending)} ejecuciones con {args.judge_model} ({len(done)} ya puntuadas)")
    evaluator = build_evaluator(args)
    started = time.perf_counter()
    for offset in range(0, len(pending), args.batch):
        batch = pending[offset: offset + args.batch]
        for row in score_batch(batch, evaluator, args):
            _common.append_jsonl(scores_path, row)
        finished = offset + len(batch)
        eta = (time.perf_counter() - started) / finished * (len(pending) - finished)
        print(f"  {finished}/{len(pending)} | ETA {eta / 60:.0f} min")
    print(f"Guardado en {scores_path}")


if __name__ == "__main__":
    main()
