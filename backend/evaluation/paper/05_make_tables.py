"""Genera las Tablas 2–8 del artículo, las pruebas estadísticas y los [VALOR].

Lee ``results/<label>/`` de todas las configuraciones disponibles (copia la
carpeta de la otra máquina a ``results/`` antes de ejecutarlo) y escribe:

    results/tables/tables.md            todas las tablas, listas para pegar
    results/tables/stats.md             Wilcoxon pareado + Holm + tamaño del efecto
    results/tables/values.json          cifras para los [VALOR] del resumen y conclusiones
    results/tables/per_run_metrics.csv  métricas por ejecución (datos crudos)
    results/tables/per_question_metrics.csv

    python evaluation/paper/05_make_tables.py --config-a A-mac --config-b B-t4
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

import _common

CIT = re.compile(r"\[((?:D|W)\d+)\]")
CONDITION_NAMES = {"C1": "C1 Densa", "C2": "C2 + RRF", "C3": "C3 + Reranking", "C4": "C4 + Grafo", "C5": "C5 Completo",
                   "C4M": "C4M + Mediación", "C5S": "C5S Auditoría como señal", "C5M": "C5M Señal + Mediación"}
STAGES = ("understand", "retrieve", "ledger", "audit", "outline", "draft", "verify", "repair", "finalize")
PAPER_COMPARISONS = (("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C3", "C5"), ("C1", "C5"))
# Claim-mediation extensions. Holm is applied to each family separately so the
# paper's adjusted p-values do not change when these conditions are present.
EXTENSION_COMPARISONS = (("C4", "C4M"), ("C5", "C5S"), ("C5S", "C5M"), ("C5", "C5M"), ("C4", "C5M"))
COMPARISONS = PAPER_COMPARISONS + EXTENSION_COMPARISONS


def _is_research(condition: str) -> bool:
    """C5 and its variants run research_rag (see _instrument.is_research)."""
    return condition.startswith("C5")

TEST_METRICS = ("recall_at_20", "ndcg_at_10", "citation_precision", "citation_precision_draft",
                "claim_support", "completeness", "faithfulness", "factual_correctness", "latency_seconds")


# ── Per-run metrics ──────────────────────────────────────────────────────


class ClaimChecker:
    def __init__(self, nli: bool) -> None:
        from app.rag import agent

        self.agent = agent
        self.nli = nli

    def support(self, answer: str, evidence: List[Dict[str, Any]]) -> Optional[float]:
        agent = self.agent
        claims = [s for s in agent._answer_sentences(answer or "") if agent._is_substantive_claim(s)]
        if not claims:
            return None
        texts = {e["source_id"]: e.get("text") or "" for e in evidence if e.get("source_id")}
        supported = 0
        for claim in claims:
            ids = set(CIT.findall(claim))
            if not ids or not ids <= set(texts):
                continue
            premise = "\n".join(texts[i] for i in sorted(ids))
            if agent._claim_support_issue(claim, premise, verify_entailment=self.nli) is None:
                supported += 1
        if self.nli and agent._nli_unavailable:
            raise SystemExit(f"El modelo NLI ({agent.settings.NLI_MODEL}) no pudo cargarse; ejecuta sin --nli.")
        return supported / len(claims)


def run_metrics(run: Dict[str, Any], gold: List[Dict[str, Any]], mode: str, checker: ClaimChecker) -> Dict[str, Any]:
    answer = run.get("answer") or ""
    evidence = run.get("evidence") or []
    source_ids = {s.get("source_id") for s in run.get("sources") or [] if s.get("source_id")}
    cited = CIT.findall(answer)
    valid = [cid for cid in cited if cid in source_ids]
    by_id = {e["source_id"]: e for e in evidence if e.get("source_id")}

    events = run.get("events") or []
    validations = [e for e in events if e["kind"] == "validate_citations"]
    checks = [e for e in events if e["kind"] == "evidence_check" and not e.get("placeholder")]
    verifies = [e for e in events if e["kind"] == "verify"]
    prunes = [e for e in events if e["kind"] == "prune" and e.get("changed")]
    mediations = [e for e in events if e["kind"] == "mediate"]
    first = validations[0] if validations else None
    repair_called = any(e["kind"] == "repair_called" for e in events)

    metrics: Dict[str, Any] = {
        "abstained": bool(run.get("abstained")),
        "citation_precision": (len(valid) / len(cited)) if cited else None,
        "citation_precision_draft": (first["valid"] / first["cited"]) if first and first["cited"] else None,
        "draft_without_citations": bool(first and first["category"] == "no_citations"),
        "claim_support": checker.support(answer, evidence),
        "distinct_documents": len({_common.normalize_text(e.get("filename")) for e in evidence}),
        "evidence_chunks": len(evidence),
        "latency_seconds": run.get("latency_seconds"),
        "rounds": run.get("rounds"),
        "corrective_rounds": max(0, (run.get("rounds") or 1) - 1),
        "repairs": run.get("repairs"),
        # Table 5 flags (any occurrence during the run)
        "v_invented_citation": any(e["category"] == "invented_citation" for e in validations),
        "v_no_citations": any(e["category"] == "no_citations" for e in validations),
        "v_unsupported_number": any(e.get("unsupported_number") for e in checks),
        "v_uncited_claims": any(e.get("uncited_claims") for e in checks),
        "v_prune_synthesis": any(e.get("stage") != "repair" for e in prunes),
        "v_prune_after_repair": any(e.get("stage") == "repair" for e in prunes),
        # Claim mediation (C4M, C5M): claims the model anchored to a verified quote.
        "mediation_flagged": sum(e.get("flagged", 0) for e in mediations),
        "mediation_accepted": sum(e.get("accepted_supported", 0) + e.get("accepted_partial", 0) for e in mediations),
        "mediation_rejected": sum(sum((e.get("rejected") or {}).values()) for e in mediations),
    }
    if _is_research(run["condition"]):
        metrics["v_repair_success"] = repair_called and bool(verifies) and verifies[-1]["issue_count"] == 0
        metrics["v_residual_issues"] = bool(verifies) and verifies[-1]["issue_count"] > 0
    else:
        metrics["v_repair_success"] = (
            len(validations) >= 2 and not run.get("abstained") and not prunes
            and bool(checks) and not checks[-1]["issues"]
        )
        metrics["v_residual_issues"] = bool(prunes) or bool(checks and checks[-1]["issues"])
    metrics["v_any"] = any(metrics[k] for k in (
        "v_invented_citation", "v_no_citations", "v_unsupported_number", "v_uncited_claims",
        "v_prune_synthesis", "v_prune_after_repair", "v_repair_success"))

    if gold:
        delivered = _common.retrieval_metrics(evidence, gold, mode)
        metrics["recall_delivered"] = delivered["recall_at_20"]
        metrics["ndcg_delivered"] = delivered["ndcg_at_10"]
        cited_chunks = [by_id[cid] for cid in set(valid) if cid in by_id]
        metrics["citation_grounded"] = (
            sum(any(_common.unit_matches(c, u, mode) for u in gold) for c in cited_chunks) / len(cited_chunks)
            if cited_chunks else None
        )
        groups = _common.evidence_groups(gold)
        covered = sum(any(_common.unit_matches(c, u, mode) for c in cited_chunks for u in units)
                      for units in groups.values())
        metrics["completeness"] = covered / len(groups)
    if _is_research(run["condition"]) and run.get("coverage"):
        coverage = run["coverage"]
        total = len(coverage.get("supported", [])) + len(coverage.get("missing", []))
        metrics["facet_coverage"] = len(coverage.get("supported", [])) / total if total else None

    calls = run.get("llm_calls") or []
    metrics["completion_tokens"] = sum(c.get("completion_tokens") or 0 for c in calls)
    metrics["prompt_tokens"] = sum(c.get("prompt_tokens") or 0 for c in calls)
    metrics["eval_seconds"] = sum(c.get("eval_seconds") or 0 for c in calls)
    metrics["prompt_eval_seconds"] = sum(c.get("prompt_eval_seconds") or 0 for c in calls)
    metrics["llm_calls"] = len(calls)
    return metrics


# ── Statistics ───────────────────────────────────────────────────────────


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    from scipy.stats import rankdata, wilcoxon as scipy_wilcoxon

    diffs = [y - x for x, y in zip(a, b)]
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    result = {"n": len(diffs), "median_diff": _common.percentile(diffs, 50), "p": 1.0, "r": 0.0}
    if len(nonzero) < 1:
        return result
    ranks = rankdata([abs(d) for d in nonzero])
    positive = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    negative = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    result["r"] = (positive - negative) / (positive + negative)  # matched-pairs rank-biserial
    try:
        result["p"] = float(scipy_wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        pass
    return result


def holm(pvalues: List[float]) -> List[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - position) * pvalues[index]))
        adjusted[index] = running
    return adjusted


# ── Loading ──────────────────────────────────────────────────────────────


def discover_labels(selected: Optional[str]) -> List[str]:
    if selected:
        return [label.strip() for label in selected.split(",")]
    if not _common.RESULTS_DIR.exists():
        return []
    return sorted(p.name for p in _common.RESULTS_DIR.iterdir()
                  if p.is_dir() and p.name != "tables" and (p / "run_meta.json").exists())


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", help="Etiquetas separadas por coma (por defecto todas en results/).")
    parser.add_argument("--config-a", help="Etiqueta de la Configuración A (columna de la Tabla 6).")
    parser.add_argument("--config-b", help="Etiqueta de la Configuración B.")
    parser.add_argument("--dataset", type=Path, help="JSONL con gold (por defecto el usado en la ejecución).")
    parser.add_argument("--match", choices=("page", "quote", "auto"), default="page")
    parser.add_argument("--nli", action="store_true",
                        help="Respaldo de afirmaciones con el clasificador NLI además de la verificación numérica.")
    args = parser.parse_args()

    _common.bootstrap()
    labels = discover_labels(args.labels)
    if not labels:
        raise SystemExit("No hay resultados en evaluation/paper/results/. Ejecuta 03_run_ablation.py primero.")
    metas = {label: json.loads((_common.RESULTS_DIR / label / "run_meta.json").read_text(encoding="utf-8"))
             for label in labels}
    dataset_path = args.dataset or Path(metas[labels[0]]["dataset"])
    if not dataset_path.exists():
        # run_meta stores the path of the machine that ran it; fall back to the local copy.
        dataset_path = _common.HERE / dataset_path.name
        if not dataset_path.exists():
            dataset_path = _common.DEFAULT_DATASET
    questions = {q["id"]: q for q in _common.load_questions(dataset_path)}
    has_gold = any(q.get("gold") for q in questions.values())
    checker = ClaimChecker(args.nli)
    out_dir = _common.RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run: List[Dict[str, Any]] = []
    # (label, condition, qid) -> metric -> [values over repeats]
    perq: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def add(label, condition, qid, metric, value):
        if value is None:
            return
        perq[(label, condition, qid)][metric].append(float(value))

    for label in labels:
        folder = _common.RESULTS_DIR / label
        for row in _common.read_jsonl(folder / "rankings.jsonl"):
            gold = questions.get(row["qid"], {}).get("gold") or []
            if not gold:
                continue
            for channel, condition in (("dense", "C1"), ("rrf", "C2"), ("rerank", "C3"), ("rerank", "C4")):
                ranked = _common.retrieval_metrics(row["rankings"][channel], gold, args.match)
                add(label, condition, row["qid"], "recall_at_20", ranked["recall_at_20"])
                add(label, condition, row["qid"], "ndcg_at_10", ranked["ndcg_at_10"])
        ragas = {r["key"]: r for r in _common.read_jsonl(folder / "ragas_scores.jsonl")}
        for run in _common.read_jsonl(folder / "runs.jsonl"):
            if run.get("error") or run["qid"] not in questions:
                continue
            gold = questions[run["qid"]].get("gold") or []
            metrics = run_metrics(run, gold, args.match, checker)
            for name in ("faithfulness", "factual_correctness", "context_recall"):
                metrics[name] = (ragas.get(run["key"]) or {}).get(name)
            if _is_research(run["condition"]):
                # C5 has no fixed-depth ranking: score the evidence it delivered.
                metrics["recall_at_20"] = metrics.get("recall_delivered")
                metrics["ndcg_at_10"] = metrics.get("ndcg_delivered")
                add(label, run["condition"], run["qid"], "recall_at_20", metrics["recall_at_20"])
                add(label, run["condition"], run["qid"], "ndcg_at_10", metrics["ndcg_at_10"])
            per_run.append({"label": label, "key": run["key"], "qid": run["qid"], "condition": run["condition"],
                            "repeat": run["repeat"], "route": run["route"], **metrics,
                            "stages": run.get("stages") or []})
            for name, value in metrics.items():
                if name.startswith("v_") or name in {"abstained", "draft_without_citations"}:
                    add(label, run["condition"], run["qid"], name, 1.0 if value else 0.0)
                elif isinstance(value, (int, float)) and not isinstance(value, bool) and name not in {"recall_at_20", "ndcg_at_10"}:
                    add(label, run["condition"], run["qid"], name, value)

    # run_meta.json only lists the conditions of the last invocation of 03_run_ablation.py (e.g. a
    # rerun with --conditions C5); report every condition that actually has runs.
    for label in labels:
        present = {row["condition"] for row in per_run if row["label"] == label}
        declared = metas[label]["conditions"]
        metas[label]["conditions"] = {c: declared.get(c, "") if isinstance(declared, dict) else ""
                                      for c in CONDITION_NAMES if c in present or c in declared}

    def q_values(label: str, condition: str, metric: str) -> Dict[str, float]:
        return {qid: mean(values[metric]) for (lab, cond, qid), values in perq.items()
                if lab == label and cond == condition and values.get(metric)}

    def cell(label: str, condition: str, metric: str, digits: int = 3) -> str:
        return _common.fmt_median_iqr(list(q_values(label, condition, metric).values()), digits)

    def mean_cell(label: str, condition: str, metric: str, digits: int = 3) -> str:
        values = list(q_values(label, condition, metric).values())
        return _common.fmt(mean(values), digits) if values else "—"

    md: List[str] = ["# Tablas generadas", "",
                     f"Conjunto: `{dataset_path.name}` ({len(questions)} preguntas). Coincidencia de evidencia: "
                     f"`{args.match}`. Valores: mediana [RIC] sobre preguntas (media de repeticiones por pregunta), "
                     "salvo indicación.", ""]
    if not has_gold:
        md += ["> **Aviso:** el conjunto no tiene campo `gold`; Recall, nDCG y completitud no se pueden calcular. "
               "Ver 02_relevance_judgments.py.", ""]

    # Table 2
    md += ["## Tabla 2. Configuraciones de hardware evaluadas", ""]
    env_rows: Dict[str, Dict[str, str]] = {}
    for label in labels:
        env_md = _common.RESULTS_DIR / label / "environment.md"
        if env_md.exists():
            for line in env_md.read_text(encoding="utf-8").splitlines():
                parts = [p.strip() for p in line.strip("|").split("|")]
                if line.startswith("| ") and len(parts) == 2 and parts[0] not in {"Componente"}:
                    env_rows.setdefault(parts[0], {})[label] = parts[1]
    if env_rows:
        md += [table(["", *labels], [[key, *[vals.get(label, "—") for label in labels]] for key, vals in env_rows.items()]), ""]
    else:
        md += ["_Sin environment.md: ejecuta 00_environment.py en cada máquina._", ""]

    corpus_rows = []
    for label in labels:
        corpus = _common.RESULTS_DIR / label / "corpus.json"
        if corpus.exists():
            s = json.loads(corpus.read_text(encoding="utf-8"))["summary"]
            corpus_rows.append([label, s["documents_ready"], s["pages_total"], s["chunks_total"],
                                f"{s['ocr_pages']} ({_common.fmt(s['ocr_pages_pct'], 2)} %)",
                                ", ".join(f"{k}: {v}" for k, v in s["by_language"].items()),
                                s["domains_count"]])
    if corpus_rows:
        md += ["## §3.10 Corpus", "", table(["Config.", "Documentos", "Páginas", "Fragmentos", "Páginas con OCR",
                                             "Idioma", "Dominios"], corpus_rows), ""]

    values: Dict[str, Any] = {"labels": labels, "n_questions": len(questions)}
    for label in labels:
        conds = metas[label]["conditions"]
        reps = metas[label]["repeats"]
        n_q = len({qid for (lab, _, qid) in perq if lab == label})
        md += [f"---", "", f"# Configuración {label}", "",
               f"LLM `{metas[label]['settings'].get('LLM_MODEL')}`, {reps} repetición(es) por pregunta y condición.", ""]

        # Table 3
        rows3 = [[CONDITION_NAMES[c], cell(label, c, "recall_at_20"), cell(label, c, "ndcg_at_10"),
                  cell(label, c, "recall_delivered"), cell(label, c, "distinct_documents", 1)]
                 for c in CONDITION_NAMES if c in conds or c in {"C1", "C2", "C3"}]
        n_gold = len(q_values(label, "C3", "recall_at_20"))
        md += [f"## Tabla 3. Calidad de recuperación por condición (n = {n_gold} preguntas, mediana [RIC])", "",
               table(["Condición", "Recall@20", "nDCG@10", "Recall (evidencia entregada)", "Documentos distintos"], rows3), "",
               "Recall@20 y nDCG@10 de C1–C4 se calculan sobre rankings de profundidad fija (densa, RRF, reordenada) "
               "antes del corte del grupo líder; C4 comparte ranking con C3 (el grafo no altera los fragmentos). "
               "C5 no tiene un ranking único: se puntúa la evidencia final tras auditoría. 'Evidencia entregada' es "
               "lo que efectivamente recibió el generador.", ""]
        tags = sorted({t for q in questions.values() for t in q.get("tags", [])})
        for tag in tags:
            tagged = {qid for qid, q in questions.items() if tag in q.get("tags", [])}
            rows = []
            for c in ("C1", "C2", "C3"):
                vals = q_values(label, c, "recall_at_20")
                inside = [v for qid, v in vals.items() if qid in tagged]
                outside = [v for qid, v in vals.items() if qid not in tagged]
                rows.append([CONDITION_NAMES[c], f"{_common.fmt_median_iqr(inside)} (n={len(inside)})",
                             f"{_common.fmt_median_iqr(outside)} (n={len(outside)})"])
            md += [f"### Tabla 3b. Recall@20 en preguntas con etiqueta `{tag}` frente al resto", "",
                   table(["Condición", f"`{tag}`", "resto"], rows), ""]

        # Table 4
        rows4 = [[CONDITION_NAMES[c], cell(label, c, "citation_precision"), cell(label, c, "citation_precision_draft"),
                  cell(label, c, "claim_support"), cell(label, c, "completeness"), cell(label, c, "citation_grounded"),
                  cell(label, c, "faithfulness"), f"{100 * mean(q_values(label, c, 'abstained').values() or [0]):.1f} %"]
                 for c in conds]
        md += [f"## Tabla 4. Métricas de fidelidad por condición (mediana [RIC], n = {n_q} preguntas)", "",
               table(["Condición", "Precisión de citas", "Precisión de citas (borrador)", "Respaldo de afirmaciones",
                      "Completitud", "Citas fundamentadas", "Faithfulness (RAGAS)", "Abstención"], rows4), "",
               "Precisión de citas: marcadores [D#] de la respuesta final que resuelven a una fuente recuperada. "
               "Borrador: la misma medida sobre el primer borrador, antes de la cascada de verificación. Respaldo: "
               "afirmaciones sustantivas con cita inmediata y sin cifras ausentes de la evidencia citada"
               + (" y con implicación NLI ≥ 0,65" if args.nli else "") +
               ". Completitud: grupos de evidencia gold citados / requeridos. Citas fundamentadas: citas que apuntan "
               "a evidencia gold. Las abstenciones no tienen citas ni afirmaciones y quedan fuera de esas medianas.", ""]

        # Table 5
        flags = [("v_invented_citation", "Rechazo por cita inexistente"),
                 ("v_no_citations", "Rechazo por ausencia de citas"),
                 ("v_unsupported_number", "Rechazo por número no respaldado"),
                 ("v_uncited_claims", "Afirmaciones sin cita inmediata detectadas"),
                 ("v_repair_success", "Reparación exitosa en un ciclo"),
                 ("v_prune_synthesis", "Poda de afirmaciones en la síntesis"),
                 ("v_prune_after_repair", "Poda de afirmaciones tras reparación fallida"),
                 ("v_residual_issues", "Entregada con observaciones residuales"),
                 ("v_any", "Al menos una intervención")]
        cols5 = [c for c in ("C5", "C4") if c in conds]
        rows5 = []
        counts5 = {}
        for key, name in flags:
            row = [name]
            for c in cols5:
                runs_c = [r for r in per_run if r["label"] == label and r["condition"] == c]
                hits = sum(1 for r in runs_c if r.get(key))
                counts5[(c, key)] = (hits, len(runs_c))
                row += [hits, f"{100 * hits / len(runs_c):.1f} %" if runs_c else "—"]
            rows5.append(row)
        if cols5:
            n5 = counts5.get(("C5", "v_any"), (0, 0))[1]
            headers = ["Mecanismo"] + [h for c in cols5 for h in (f"{c}: respuestas", "%")]
            md += [f"## Tabla 5. Intervenciones de la cascada de verificación ({n5} ejecuciones de research_rag)", "",
                   table(headers, rows5), "",
                   "C4 (simple_rag) se incluye como referencia: también valida citas y números, pero sin el bucle de "
                   "auditoría ni la reparación localizada. En C5 la poda puede ocurrir ya en la síntesis (dos intentos "
                   "fallidos) y no solo tras la reparación.", ""]

        # Table 7
        c5_runs = [r for r in per_run if r["label"] == label and r["condition"] == "C5"]
        if c5_runs:
            total = sum(r["latency_seconds"] for r in c5_runs)
            share = {s: sum(st["seconds"] for r in c5_runs for st in r["stages"] if st["name"] == s) for s in STAGES}
            other = total - sum(share.values())
            md += ["## Tabla 7. Distribución del tiempo por etapa del bucle de investigación (%)", "",
                   table(["Etapa", *STAGES, "otros"],
                         [["% del total", *[f"{100 * share[s] / total:.1f}" for s in STAGES], f"{100 * other / total:.1f}"],
                          ["mediana s/ejecución", *[_common.fmt(_common.percentile(
                              [sum(st["seconds"] for st in r["stages"] if st["name"] == s) for r in c5_runs], 50), 1)
                              for s in STAGES], "—"]]), "",
                   f"Agregado sobre {len(c5_runs)} ejecuciones de C5; cada nodo del grafo de estados se cronometra al "
                   "registrarse en LangGraph (igual en ambas ramas, sin Langfuse). 'otros' = creación del registro "
                   "de ejecución y transiciones del grafo.", ""]

        # Table 8
        if q_values(label, "C3", "faithfulness") or q_values(label, "C4", "faithfulness"):
            rows8 = []
            for metric, name in (("faithfulness", "Faithfulness"), ("factual_correctness", "Factual correctness"),
                                 ("context_recall", "LLM context recall")):
                a, b = q_values(label, "C3", metric), q_values(label, "C4", metric)
                common = sorted(set(a) & set(b))
                test = wilcoxon([a[q] for q in common], [b[q] for q in common]) if common else None
                ma = mean(a.values()) if a else None
                mb = mean(b.values()) if b else None
                rows8.append([f"*{name}*", _common.fmt(ma), _common.fmt(mb),
                              _common.fmt(mb - ma) if ma is not None and mb is not None else "—",
                              _common.fmt(test["p"], 3) if test else "—", len(common)])
            ragas_meta = _common.RESULTS_DIR / label / "ragas_meta.json"
            judge = json.loads(ragas_meta.read_text())["judge_model"] if ragas_meta.exists() else "?"
            md += ["## Tabla 8. Comparación por RAGAS: recuperación vectorial (C3) frente a GraphRAG (C4)", "",
                   table(["Métrica", "Vectorial (C3)", "GraphRAG (C4)", "Δ", "p (Wilcoxon)", "n"], rows8), "",
                   f"Media sobre preguntas. Juez: `{judge}` vía Ollama. C3 y C4 solo difieren en la inyección del "
                   "contexto del grafo, de modo que Δ aísla su efecto dentro del flujo real.", ""]

        # Values for the prose
        def med(c, m):
            return _common.percentile(list(q_values(label, c, m).values()), 50)

        values[label] = {
            "citation_precision_C5": med("C5", "citation_precision"),
            "citation_precision_draft_C5": med("C5", "citation_precision_draft"),
            "median_latency_C5_s": med("C5", "latency_seconds"),
            "median_latency_C1_s": med("C1", "latency_seconds"),
            "median_latency_C4_simple_rag_s": med("C4", "latency_seconds"),
            "recall20_delta_C1_to_C2": (med("C2", "recall_at_20") - med("C1", "recall_at_20"))
            if med("C2", "recall_at_20") is not None and med("C1", "recall_at_20") is not None else None,
            "ndcg10_delta_C2_to_C3": (med("C3", "ndcg_at_10") - med("C2", "ndcg_at_10"))
            if med("C3", "ndcg_at_10") is not None and med("C2", "ndcg_at_10") is not None else None,
            "verification_intervention_rate_C5_pct": (100 * counts5[("C5", "v_any")][0] / counts5[("C5", "v_any")][1])
            if counts5.get(("C5", "v_any"), (0, 0))[1] else None,
            "research_runs_C5": counts5.get(("C5", "v_any"), (0, 0))[1],
        }

    # Table 6 (both machines)
    a_label = args.config_a or labels[0]
    b_label = args.config_b or (labels[1] if len(labels) > 1 else None)
    latency_cols = [lab for lab in (a_label, b_label) if lab]
    latency_cols += [lab for lab in labels if lab not in latency_cols]  # p. ej. B-t4-4b (B′)

    def latency_text(rows: List[Dict[str, Any]]) -> str:
        lat = [r["latency_seconds"] for r in rows]
        if not lat:
            return "—"
        return (f"{_common.fmt(_common.percentile(lat, 50), 1)} / {_common.fmt(_common.percentile(lat, 95), 1)} / "
                f"{_common.fmt(max(lat), 1)} (n={len(lat)})")

    groups6 = [("`simple_rag` (C4)", lambda r: r["condition"] == "C4")]
    observed = sorted({r["corrective_rounds"] for r in per_run if r["condition"] == "C5"} | {0, 1})
    groups6 += [(f"`research_rag`, {k} ronda(s) correctiva(s)",
                 (lambda k: lambda r: r["condition"] == "C5" and r["corrective_rounds"] == k)(k)) for k in observed]
    groups6 += [("`research_rag`, todas", lambda r: r["condition"] == "C5")]
    rows6 = [[name, *[latency_text([r for r in per_run if r["label"] == lab and pick(r)]) for lab in latency_cols]]
             for name, pick in groups6]
    max_rounds = {lab: metas[lab]["settings"].get("RESEARCH_MAX_ROUNDS") for lab in latency_cols}
    md += ["---", "", "# Comparación entre configuraciones", "",
           "## Tabla 6. Latencia extremo a extremo en segundos (mediana / p95 / máximo)", "",
           table(["Ruta", *[f"{lab} (`{metas[lab]['settings'].get('LLM_MODEL')}`)" for lab in latency_cols]], rows6), "",
           "Ronda correctiva = recuperación adicional tras la auditoría (rondas − 1). "
           f"RESEARCH_MAX_ROUNDS por configuración: {max_rounds} (2 ⇒ como máximo una ronda correctiva). "
           "En lan_client la latencia incluye la red entre el cliente y el servidor de Ollama (ver RTT en la Tabla 2).", ""]
    rows6b = []
    for lab in latency_cols:
        runs_l = [r for r in per_run if r["label"] == lab]
        gen_tokens = sum(r["completion_tokens"] for r in runs_l)
        gen_seconds = sum(r["eval_seconds"] for r in runs_l)
        prompt_tokens = sum(r["prompt_tokens"] for r in runs_l)
        prompt_seconds = sum(r["prompt_eval_seconds"] for r in runs_l)
        rows6b.append([lab, f"{gen_tokens / gen_seconds:.1f}" if gen_seconds else "—",
                       f"{prompt_tokens / prompt_seconds:.1f}" if prompt_seconds else "—",
                       _common.fmt(_common.percentile([r["llm_calls"] for r in runs_l if r["condition"] == "C5"], 50), 0)])
    md += ["### Tabla 6b. Rendimiento del generador (Ollama)", "",
           table(["Config.", "Generación (tokens/s)", "Procesamiento del prompt (tokens/s)",
                  "Llamadas al LLM por research_rag (mediana)"], rows6b), "",
           "Cifras de Ollama (eval_count/eval_duration); no incluyen la llamada del planificador si Ollama no "
           "devuelve metadatos para ella.", ""]

    # Discussion helper: architecture vs model size
    rows9 = []
    for lab in labels:
        for c in ("C3", "C5"):
            if c in metas[lab]["conditions"]:
                rows9.append([lab, metas[lab]["settings"].get("LLM_MODEL"), CONDITION_NAMES[c],
                              cell(lab, c, "citation_precision"), cell(lab, c, "claim_support"),
                              cell(lab, c, "completeness"), cell(lab, c, "faithfulness"),
                              cell(lab, c, "latency_seconds", 1)])
    md += ["## Apoyo a §4.5: salto C3 → C5 frente a salto de modelo", "",
           table(["Config.", "LLM", "Condición", "Precisión de citas", "Respaldo", "Completitud",
                  "Faithfulness", "Latencia (s)"], rows9), "",
           "Atención: A y B no difieren solo en el LLM (almacén, TOP_K, preguntas de evidencia, motor de "
           "embeddings y red; ver Tabla 2). Para aislar el tamaño del modelo usa la etiqueta B-t4-4b (README).", ""]
    (out_dir / "tables.md").write_text("\n".join(md), encoding="utf-8")

    # Statistics
    stats = ["# Pruebas estadísticas", "",
             "Wilcoxon de rangos con signo pareado por pregunta (media de repeticiones), corrección de Holm dentro "
             "de cada métrica y configuración, y tamaño del efecto r (rango-biserial pareado; ±1 = todas las "
             "diferencias en la misma dirección). Δ = mediana de (segunda − primera).", ""]
    for label in labels:
        stats += [f"## {label}", ""]
        for metric in TEST_METRICS:
            results = []
            adjusted = []
            for family in (PAPER_COMPARISONS, EXTENSION_COMPARISONS):
                family_results = []
                for first, second in family:
                    a, b = q_values(label, first, metric), q_values(label, second, metric)
                    common = sorted(set(a) & set(b))
                    if len(common) >= 2:
                        family_results.append(
                            (first, second, wilcoxon([a[q] for q in common], [b[q] for q in common]))
                        )
                results += family_results
                adjusted += holm([res["p"] for _, _, res in family_results]) if family_results else []
            if not results:
                continue
            stats += [f"### {metric}", "", table(
                ["Comparación", "n", "Δ mediana", "p", "p (Holm)", "r"],
                [[f"{f} → {s}", res["n"], _common.fmt(res["median_diff"]), _common.fmt(res["p"], 4),
                  _common.fmt(p_adj, 4), _common.fmt(res["r"], 2)]
                 for (f, s, res), p_adj in zip(results, adjusted)]), ""]
    (out_dir / "stats.md").write_text("\n".join(stats), encoding="utf-8")
    _common.write_json(out_dir / "values.json", values)

    if per_run:
        fields = [k for k in per_run[0] if k != "stages"]
        with (out_dir / "per_run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for r in per_run for k in r if k != "stages"},
                                                               key=lambda k: (k not in fields, k)),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_run)
    metric_names = sorted({m for values_ in perq.values() for m in values_})
    with (out_dir / "per_question_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "condition", "qid", *metric_names])
        for (label, condition, qid), values_ in sorted(perq.items()):
            writer.writerow([label, condition, qid,
                             *[_common.fmt(mean(values_[m]), 4) if values_.get(m) else "" for m in metric_names]])
    print(f"Tablas en {out_dir / 'tables.md'}")
    print(f"Estadística en {out_dir / 'stats.md'}")
    print(json.dumps(values, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
