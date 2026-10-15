#!/usr/bin/env bash
# Ejecuta, para UNA máquina, todo lo que no depende de la otra:
# entorno (Tabla 2), corpus (§3.10) y ablación C1–C5 (Tablas 3–7).
#
#   bash evaluation/paper/run_all.sh <label> <email> [repeticiones]
#   MANIFEST=evaluation/paper/corpus_manifest.csv bash evaluation/paper/run_all.sh B-t4 tu@correo 3
set -euo pipefail
cd "$(dirname "$0")/../.."          # backend/
PY="${PYTHON:-python}"
LABEL="${1:?falta la etiqueta, p. ej. A-mac}"
EMAIL="${2:?falta el email del usuario dueño del corpus}"
REPEATS="${3:-1}"

"$PY" evaluation/paper/00_environment.py --label "$LABEL"
"$PY" evaluation/paper/01_corpus_stats.py --label "$LABEL" --user-email "$EMAIL" ${MANIFEST:+--manifest "$MANIFEST"}
"$PY" evaluation/paper/03_run_ablation.py --label "$LABEL" --user-email "$EMAIL" --repeats "$REPEATS"
echo "Listo: evaluation/paper/results/$LABEL"
