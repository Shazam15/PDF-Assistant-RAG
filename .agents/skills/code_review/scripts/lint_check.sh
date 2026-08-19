#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

echo "🔍 [1/3] Verificando Backend con Flake8..."
flake8 "${REPO_ROOT}/backend/app" --max-line-length=120 --select=E9,F63,F7,F82

echo "🧪 [2/3] Ejecutando pruebas unitarias de backend..."
cd "${REPO_ROOT}/backend" && pytest tests/test_rag_tools.py tests/test_tools.py -q

echo "⚛️ [3/3] Verificando tipos en Frontend..."
cd "${REPO_ROOT}/frontend" && npx tsc --noEmit

echo "✅ Todas las comprobaciones de código pasaron exitosamente."
