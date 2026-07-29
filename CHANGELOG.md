# Historial de versiones de ATLAS

Este documento registra las versiones públicas y los hitos arquitectónicos relevantes de ATLAS. El backend expone actualmente la versión `2.0.0`; el valor `0.1.0` de `frontend/package.json` identifica el paquete frontend y no la versión del producto.

El repositorio todavía no contiene tags de release. Por ello, las entradas anteriores a `2.0.0` se presentan como hitos de desarrollo fechados y vinculados a commits, sin asignarles números de versión retroactivos.

## [Unreleased]

### Añadido

- Perfil `ubuntu_t4` para Xeon/Tesla T4 con Ollama local y recuperación en CPU.
- Diagnóstico `make doctor-ubuntu` para GPU, Ollama, modelo, PostgreSQL, extensiones y Redis.
- Arranque de desarrollo `make dev-ubuntu` e instalación CPU `install-backend-ubuntu`.
- Plantillas `systemd` para FastAPI y Celery, y Caddy para frontend, SSE y WebSocket.

### Cambiado

- Esta rama usa `ubuntu_t4` como perfil predeterminado y `127.0.0.1:11434` como Ollama.
- PostgreSQL y Redis se publican solo en localhost; `make docker-up` inicia ambos.
- La caché obtiene `REDIS_URL` y sus límites desde la configuración Pydantic cargada desde `.env`.

### Documentación

- Arquitectura sincronizada con la ingesta adaptable, disponibilidad temprana, timeouts vigentes y despliegues Ubuntu/T4 y Windows/WSL2/T4.
- Incorporado este historial como fuente única para futuras notas de versión.

## [2.0.0] - 2026-07-22

### Añadido

- Perfil `wsl_t4` para backend y frontend nativos en WSL2, PostgreSQL en Docker y Ollama en Windows con una Tesla T4 dedicada.
- Cliente compartido de Ollama con host, timeout y `keep_alive` consistentes en planificación, resumen, memoria, herramientas, auditoría y redacción.
- Comandos `make doctor-wsl`, `make dev-wsl` e `install-backend-wsl`.
- Ingesta con progreso medible, heartbeat, barra de progreso y disponibilidad para consultas desde el 85%.
- Extracción PDF `auto|fast|quality`, con selección adaptativa entre PyMuPDF, Docling y OCR.

### Cambiado

- Presupuesto de investigación ampliado a 1800 segundos, timeout por llamada al LLM de 900 segundos y reserva de síntesis de 600 segundos.
- Qwen3-Embedding-0.6B y Qwen3-Reranker-0.6B se mantienen para índices de investigación de 1024 dimensiones.
- PostgreSQL/pgvector queda como único servicio Docker obligatorio para el despliegue WSL/T4 inicial.

### Corregido

- Renovación anticipada del token antes de abrir WebSocket y fallback silencioso a SSE ante autenticación recuperable.
- La ingesta conserva documentos consultables cuando falla únicamente el enriquecimiento.
- Consultas FTS y síntesis final protegidas frente a puntuación inválida, contexto excesivo y límites prematuros.

Commits principales: `64deee4`, `ba0ee40`.

## Hitos anteriores

### 2026-07-15 — Agente de investigación basado en evidencia

- Incorporado el `StateGraph` de investigación con planificación, recuperación iterativa, claim ledger, auditoría, esquema argumentativo, redacción, verificación y reparación.
- Añadidos perfiles documentales, secciones, chunks jerárquicos, evidencias verificables y ejecuciones de investigación.
- Adoptados PostgreSQL/pgvector, búsqueda léxica global, RRF y modelos Qwen3 para recuperación multilingüe.
- Eliminadas reglas de recuperación dependientes de dominios concretos.

Commit principal: `b9dc562`.

### 2026-07-14 — Enrutamiento adaptativo

- Añadidos los modos `Auto`, `Rápido` e `Investigación`.
- Separadas las rutas `scoped_rag`, `simple_rag`, `research_rag` y `tool_agent`.
- Restringido ReAct a tareas que requieren herramientas; los requisitos de estilo dejaron de activar el agente por sí solos.
- Añadida cancelación de respuestas desde el frontend.

Commit principal: `9d406bb`.

### 2026-07-09 a 2026-07-13 — Alcance documental y citas

- Corregida la conversación con un documento seleccionado y su liberación al cerrar el panel.
- Mejoradas la diversidad documental, las citas estructuradas y la síntesis cuando el agente alcanza sus límites.
- Reforzada la recuperación web para solicitudes que necesitan información externa actual.

Commits principales: `484cef7`, `d291e80`.

### 2026-06-18 a 2026-06-19 — Estabilización del RAG con Ollama

- Migrada la generación principal a Ollama.
- Corregidas salidas JSON inválidas, mensajes duplicados o desaparecidos y persistencia del historial.
- Reparadas la ingesta, la generación de resúmenes y las primeras rutas de razonamiento con documentos.

Commits principales: `2aa7f00`, `449b3ca`, `4cd124e`, `dfcb46c`, `079407b`.

### 2026-02-22 a 2026-06-07 — Base y evolución inicial

- Creada la aplicación RAG inicial con autenticación y soporte para PDF, TXT, DOCX y Markdown.
- Añadidos despliegues en la nube, integraciones de modelos remotos y una nueva interfaz web durante las primeras iteraciones.
- Consolidada la interfaz de documentos y chat, metadatos, visor PDF, controles de accesibilidad y protecciones iniciales de memoria.
- Establecida la base FastAPI, React/Next.js y persistencia documental sobre la que evolucionó ATLAS.

Commits de referencia: `e06f93a`, `c880343`, `a797a11`, `964dddf`, `0552469`, `48838ad`.

## Convención para próximas versiones

- Usar versionado semántico `MAJOR.MINOR.PATCH` para releases del producto.
- Mantener cambios pendientes bajo `Unreleased` y moverlos a una sección fechada al publicar.
- Crear un tag Git `vMAJOR.MINOR.PATCH` para cada release.
- Actualizar conjuntamente la versión de FastAPI en `backend/app/main.py`, este archivo y las notas de despliegue cuando corresponda.
