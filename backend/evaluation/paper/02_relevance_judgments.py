"""Juicios de relevancia (qrels) para Recall@20, nDCG@10 y completitud.

Recall y nDCG necesitan saber qué evidencia es relevante para cada pregunta.
Hay dos formas de obtenerla:

1) Escribirla a mano en questions.jsonl, campo ``gold`` (ver README).
2) Pooling estilo TREC con este script (ojo: el pool solo contiene lo que
   recuperan los canales de ATLAS; decláralo como limitación):

   pool  — para cada pregunta une los top-N de los canales denso, léxico y
           reordenado y genera un CSV para anotar (columna ``relevance``:
           0 = no relevante, 1 = parcialmente, 2 = responde la pregunta).
   apply — lee el CSV anotado y escribe el campo ``gold`` en questions.jsonl
           (guarda antes una copia questions.sin_gold.jsonl).

    python evaluation/paper/02_relevance_judgments.py pool  --user-email tu@correo
    # ... anotar evaluation/paper/judgments_pool.csv en Excel/LibreOffice ...
    python evaluation/paper/02_relevance_judgments.py apply

Anota en UNA sola máquina: las unidades de evidencia quedan como
(archivo, página), que es comparable entre máquinas aunque la extracción
genere fragmentos distintos.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import _common

POOL_CSV = _common.HERE / "judgments_pool.csv"
FIELDS = ["qid", "question", "reference", "filename", "page", "channels", "relevance", "group", "text"]


def pool(args) -> None:
    _common.bootstrap()
    import _instrument

    user_id = _common.resolve_user_id(args.user_email, args.user_id)
    questions = _common.load_questions(args.dataset, args.limit)
    rows: List[Dict[str, Any]] = []
    for index, question in enumerate(questions, 1):
        ranked = _instrument.rankings(question["question"], user_id, args.depth)
        merged: Dict[tuple, Dict[str, Any]] = {}
        for channel in ("rerank", "dense", "lexical"):
            for rank, chunk in enumerate(ranked[channel], 1):
                # One judgment per (file, page): that is the evidence unit.
                key = (_common.normalize_text(chunk.get("filename")), int(chunk.get("page") or 0))
                entry = merged.setdefault(key, {
                    "qid": question["id"],
                    "question": question["question"],
                    "reference": question["reference"],
                    "filename": chunk.get("filename"),
                    "page": int(chunk.get("page") or 0),
                    "channels": [],
                    "relevance": "",
                    "group": "",
                    "text": "",
                })
                entry["channels"].append(f"{channel}#{rank}")
                if len(entry["text"]) < 2500:
                    entry["text"] = (entry["text"] + "\n---\n" + " ".join(str(chunk.get("text", "")).split())).strip("\n-")
        for entry in merged.values():
            entry["channels"] = " ".join(entry["channels"])
            rows.append(entry)
        print(f"  {index}/{len(questions)} {question['id']}: {len(merged)} páginas candidatas")

    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} juicios pendientes en {args.output}. Rellena 'relevance' (0/1/2) y opcionalmente "
          "'group' (misma etiqueta = evidencia intercambiable para completitud).")


def apply(args) -> None:
    questions = _common.load_questions(args.dataset)
    gold: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
    unjudged = 0
    with args.judgments.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get("relevance", "")).strip()
            if value == "":
                unjudged += 1
                continue
            grade = int(float(value))
            if grade <= 0:
                continue
            key = (_common.normalize_text(row["filename"]), int(row["page"]))
            unit = {"filename": row["filename"], "page": int(row["page"]), "grade": grade}
            if row.get("group", "").strip():
                unit["group"] = row["group"].strip()
            current = gold.setdefault(row["qid"], {}).get(key)
            if current is None or grade > current["grade"]:
                gold[row["qid"]][key] = unit
    if unjudged:
        print(f"AVISO: {unjudged} filas sin juicio (se tratan como no relevantes).")

    output = args.output or args.dataset
    if output.resolve() == args.dataset.resolve():
        backup = args.dataset.with_suffix(".sin_gold.jsonl")
        backup.write_text(args.dataset.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copia de seguridad del conjunto original en {backup}")
    with output.open("w", encoding="utf-8") as handle:
        import json

        without = []
        for question in questions:
            units = list(gold.get(question["id"], {}).values())
            if args.merge:
                existing = {(_common.normalize_text(u["filename"]), int(u.get("page") or 0)) for u in units}
                units += [u for u in question.get("gold", [])
                          if (_common.normalize_text(u["filename"]), int(u.get("page") or 0)) not in existing]
            question = {**question, "gold": units}
            if not units:
                without.append(question["id"])
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")
    print(f"Escrito {output}")
    if without:
        print(f"AVISO: {len(without)} preguntas sin evidencia relevante: {', '.join(without)}. "
              "Quedan fuera de Recall/nDCG/completitud.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pool = sub.add_parser("pool", help="Genera el CSV de candidatos a anotar.")
    _common.add_user_args(p_pool)
    p_pool.add_argument("--dataset", type=Path, default=_common.DEFAULT_DATASET)
    p_pool.add_argument("--depth", type=int, default=10, help="Top-N por canal (10 → ~15-25 páginas/pregunta).")
    p_pool.add_argument("--limit", type=int)
    p_pool.add_argument("--output", type=Path, default=POOL_CSV)

    p_apply = sub.add_parser("apply", help="Convierte el CSV anotado en el campo gold.")
    p_apply.add_argument("--dataset", type=Path, default=_common.DEFAULT_DATASET)
    p_apply.add_argument("--judgments", type=Path, default=POOL_CSV)
    p_apply.add_argument("--output", type=Path, help="Por defecto sobrescribe --dataset (con copia de seguridad).")
    p_apply.add_argument("--merge", action="store_true", help="Conserva el gold escrito a mano.")

    args = parser.parse_args()
    pool(args) if args.command == "pool" else apply(args)


if __name__ == "__main__":
    main()
