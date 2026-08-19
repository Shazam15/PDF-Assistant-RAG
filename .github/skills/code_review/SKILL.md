---
name: code-review
description: >-
  Utiliza este skill cuando el usuario solicite revisar código, auditar la seguridad, depurar errores, ejecutar pruebas o refactorizar archivos en el repositorio.
---

# Code Review & Refactoring Workflow

Sigue este procedimiento estricto para realizar revisiones de código reproducibles y seguras:

## 1. Inspección
1. Inspecciona los archivos relevantes usando herramientas de lectura (`view_file` o servidor MCP de Filesystem).
2. Para revisar cambios recientes o ramas, consulta el estado de Git (`git status`, `git diff`).

## 2. Análisis Estático y Pruebas
1. Ejecuta el análisis de linters del repositorio:
   `make lint` (o `flake8 backend/app` y `cd frontend && npm run lint`)
2. Ejecuta la suite de pruebas automatizadas:
   `make test` (o `cd backend && pytest -v`)
3. Revisa la presencia de secretos o credenciales expuestas (`detect-secrets`).

## 3. Modificación y Edición
1. Aplica cambios mínimos y precisos utilizando herramientas de edición (`replace_file_content` o `multi_replace_file_content`).
2. Conserva la arquitectura existente, tipado estricto y comentarios relevantes.

## 4. Verificación
1. Vuelve a ejecutar las pruebas y linters para certificar que no hay regresiones.
2. Presenta un reporte con la clasificación de hallazgos (Crítico, Medio, Estilo) y los diffs aplicados.
