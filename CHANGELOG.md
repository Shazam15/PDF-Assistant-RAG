# Historial de versiones de ATLAS

Este documento registra las versiones públicas y los hitos arquitectónicos relevantes de ATLAS. El backend expone actualmente la versión `2.0.0`; el valor `0.1.0` de `frontend/package.json` identifica el paquete frontend y no la versión del producto.

El repositorio todavía no contiene tags de release. Por ello, las entradas anteriores a `2.0.0` se presentan como hitos de desarrollo fechados y vinculados a commits, sin asignarles números de versión retroactivos.

> Esta rama (`experimental`) diverge de `frontend/tailwind-v3-no-avx2-fallback` justo después del hito `2.0.0`: ambas comparten la base RAG (fragmentación, embeddings, recuperación híbrida, grafo de investigación), pero esta rama no incluye los perfiles `wsl_t4`/`ubuntu_t4` para un host Xeon/Tesla T4 separado —usa el perfil `local` por defecto y `research_gpu` para producción NVIDIA co-localizada— y en cambio incorpora herramientas MCP y el panel de administración para el grafo de conocimiento, que la otra rama no tiene.

## [Unreleased]

### Añadido

- Herramientas [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) para el agente ReAct: descubrimiento y caché de servidores declarados en `MCP_SERVERS_JSON`, allowlist de solo lectura por defecto, denylist como defensa en profundidad, y adaptador de entrada única para el ejecutor ReAct. Ver [`docs/MCP_TOOLS.md`](docs/MCP_TOOLS.md) para el detalle completo (`12bc7ea`, `44c71f8`, `4dabe2e`).
- Portabilidad de rutas (`~`, variables de entorno) en `MCP_SERVERS_JSON` para que la restricción de directorios del servidor MCP externo resuelva correctamente en cualquier máquina/usuario (`55e9e72`).
- `AGENT_MAX_ITERATIONS` sube de 4 a 6 pasos para dar margen a flujos de herramientas MCP de varios saltos (`44c71f8`).
- `GroundedReActOutputParser` reconoce dinámicamente los nombres de herramientas MCP descubiertas en tiempo de ejecución, recupera una `Action` sin `Action Input` explícito (frecuente en herramientas MCP sin argumentos) y limpia saltos de línea de cola en la entrada parseada (`44c71f8`, `4dabe2e`).
- Panel de administración con visualización del grafo de conocimiento persistido por documento: entidades, relaciones, páginas y conteos, sin exponer texto de chunks ni rutas de almacenamiento (`ba3cb51`).
- Renderizado de fórmulas LaTeX (KaTeX) en las respuestas Markdown del chat y en las conversaciones compartidas (`0fb39a5`).
- Ajuste automático de `RERANK_MAX_LENGTH` al `max_position_embeddings` real del modelo de reranking cargado, para evitar un `IndexError` críptico en vez de un aviso claro al cambiar de modelo.

### Cambiado

- Botón de cancelación: antes solo se comprobaba el evento de cancelación en la rama de saludo del backend; ahora se comprueba en cada punto de control del bucle del agente (router, bucle ReAct, síntesis, validación de citas), y una tarea en segundo plano marca el evento en cuanto el socket se cierra en vez de esperar a que termine el paso en curso (`55e9e72`).
- Borrado de documentos: el botón eliminar pasa de un borrado lógico (`is_deleted=True`) a una rutina de borrado físico compartida (`backend/app/services/cleanup.py`) que limpia el índice vectorial y BM25, el archivo del grafo de conocimiento, las filas de chunk/sección/perfil/evidencia, desvincula (sin borrar) las ejecuciones de investigación y mensajes de chat asociados, y elimina el archivo físico y la fila del documento (`55e9e72`).

### Documentación

- Corregido el presupuesto documentado del grafo de investigación (`RESEARCH_TIMEOUT_SECONDS=1800`, `LLM_REQUEST_TIMEOUT_SECONDS=900`; `docs/ARCHITECTURE.md` describía valores desactualizados de 180/90 segundos de una configuración anterior).
- Añadidas guías dedicadas a los mecanismos internos: [`docs/RETRIEVAL_MATH.md`](docs/RETRIEVAL_MATH.md) (fórmulas de fragmentación, embeddings, fusión híbrida, reranking, verificación NLI y construcción del grafo de conocimiento), [`docs/AGENT_LOOPS.md`](docs/AGENT_LOOPS.md) (enrutador, bucle ReAct y grafo de investigación) y [`docs/MCP_TOOLS.md`](docs/MCP_TOOLS.md) (herramientas MCP en detalle, exclusivo de esta rama).
- Incorporado este historial (`CHANGELOG.md` no existía en esta rama hasta ahora).

## [2.0.0] - 2026-07-22

### Añadido

- Perfil `research_gpu` para producción NVIDIA: GPU dedicada a Ollama y al reranker, embeddings en CPU con lotes de 64 para aprovechar la RAM del servidor.
- Ingesta con progreso medible, heartbeat y disponibilidad para consultas antes de que termine el enriquecimiento (perfil, resúmenes y grafo).
- Extracción PDF `auto|fast|quality`, con selección adaptativa entre PyMuPDF, Docling y OCR.
- Perfil `local_balanced` para Mac con Apple Silicon (MPS), con lotes reducidos para equipos de 8 GB de RAM.

### Cambiado

- Qwen3-Embedding-0.6B y Qwen3-Reranker-0.6B para índices de investigación de 1024 dimensiones en el perfil `research_gpu`.
- `LLM_DISABLE_THINKING` disponible como interruptor explícito para modelos Qwen3.

Commits principales: `909bb1f`, `64deee4`.

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
