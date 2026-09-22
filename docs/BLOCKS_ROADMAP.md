# ATLAS — Reestructuración en bloques + capacidades HERMES/DashAI

## Contexto

Hoy ATLAS lleva al usuario directamente a un chat estilo ChatGPT (`/dashboard`), donde toda la funcionalidad convive en una sola superficie y los modos (`auto`/`quick`/`research`/`code_review`) se eligen en un desplegable dentro del compositor. El objetivo es convertir la entrada en una serie de **bloques**: el primero y principal es el agente de investigación/escritura actual, y a su lado un asistente de programación y un área de gestión de datasets.

Sobre esa base se añaden capacidades inspiradas en **HERMES** (skills que el agente escribe y actualiza solo, memoria persistente entre sesiones) y en **DashAI** (gestión de datasets).

Decisiones ya tomadas que acotan el alcance:

- **El artículo se publica primero** con lo ya hecho (tracing, DOI, ADR). Esto es un proyecto por fases, sin presión de fecha.
- **La Fase 1 es el shell de bloques**, sin funcionalidad nueva.
- **DashAI se limita a datasets**: subir, perfilar, previsualizar y consultar. Sin entrenamiento de modelos ni AutoML.
- **El asistente de programación es de solo lectura** sobre archivos reales, vía el servidor MCP ya configurado con allowlist. Nunca escribe.

### Dos hallazgos que condicionan el diseño

1. **"Workspace" ya está ocupado y significa otra cosa.** `backend/app/routes/workspaces.py` implementa invitaciones por email y `frontend/src/store/workspace-store.ts` define `WorkspaceId = "personal" | "company"`: es **tenencia**, no superficie de trabajo. Los bloques se llaman **block** en todo el código nuevo (`block-store.ts`, `BlockId`, `BlockRail`). No reutilizar `workspace-store.ts`.
2. **El precedente del bloque de código ya existe.** En `backend/app/rag/agent.py:442-446`, `route_query` corta directo a `code_review_agent` cuando el modo es manual, con un comentario que fija la regla "el modo manual siempre gana". Un bloque de programación es una promoción de esto, no una arquitectura nueva.

---

## Fase 1 — Shell de bloques (entregable por sí sola, cero cambio de comportamiento)

**Objetivo:** extraer el shell y el guard de autenticación a un layout, añadir el selector de bloques, y mover el chat actual dentro del bloque de investigación. Al terminar, la app debe comportarse exactamente igual que hoy.

**Ruta:** se mantiene `/dashboard` mediante un **grupo sin ruta** `src/app/(blocks)/`. La URL no cambia (está marcada como favorito y es el aterrizaje tras login, referenciada desde `login/`, `admin/`, `settings/`). Se descarta `/workspace/[block]`: un segmento dinámico no aporta nada para tres superficies escritas a mano y arrastra la colisión de nombre.

**Selector:** un **rail lateral persistente de iconos** (`BlockRail`), no una pantalla lanzadora ni el desplegable del header. Una pantalla lanzadora mete un clic extra entre el login y la única tarea real de hoy; el desplegable del header ya lo ocupa la tenencia y sería ambiguo. El rail ocupa ~56px, siempre visible, y hace que las fases 2 y 3 sean puramente aditivas.

### Crear

| Archivo | Contenido |
|---|---|
| `frontend/src/app/(blocks)/layout.tsx` | Client component. Único lugar del guard de auth para bloques. Renderiza `<BlockRail /> + <Header /> + {children}` |
| `frontend/src/components/layout/BlockRail.tsx` | Rail de iconos (Investigación / Código / Datos); item activo derivado de `usePathname()` |
| `frontend/src/store/block-store.ts` | `BlockId = "research" \| "code" \| "data"`; último bloque visitado en localStorage, solo para "volver al último" — **la URL es la fuente de verdad** |
| `frontend/src/types/document.ts` | `DocInfo` movido literalmente desde `dashboard/page.tsx` (hoy un archivo de página exporta un tipo que otros bloques necesitarían) |
| `frontend/src/hooks/useDocuments.ts` | Extrae `documents`, `activeDoc`, `pdfPage` y el bucle de polling de 3 s desde `dashboard/page.tsx:77-253` |

### Modificar

- **`frontend/src/components/layout/Header.tsx`** — sustituir `viewerOpen` / `onToggleViewer` / `mobileSheetContent` por un único slot opcional `actions?: React.ReactNode` más `title`. El bloque de investigación pasa su toggle de visor y su hoja móvil como `actions`. El Header deja de saber qué es un PDF. La petición decorativa de `Header.tsx:76-79` (hace un GET y descarta el resultado) se deja como está en esta fase: se anota, no se arregla aquí.
- **`frontend/src/app/(blocks)/dashboard/page.tsx`** — queda solo componiendo `DocumentSidebar | ChatSessionSidebar | ChatPanel | PDFViewer` con `useDocuments()`. Se elimina el `useEffect` de auth de `dashboard/page.tsx:157-158`, que pasa al layout.
- **`frontend/src/lib/i18n.ts`** — etiquetas del rail en **los cuatro locales** (en/es/fr/hi); una clave ausente rompe la app.

**Orden:** tipos → hook → layout + rail → generalizar props del Header → adelgazar la página.

**`chat-store`: no se toca en la Fase 1.** Hay una sola superficie de chat, así que des-singletonizar ahora sería churn no verificable. Se difiere a la Fase 2.

**Riesgos:**
- *Doble redirección* si queda el `useEffect` de auth en la página además del layout. Es el bug clásico de este refactor.
- *Remontaje del `PDFViewer`*: está cargado con `dynamic(ssr:false)`; hay que conservar la forma del subárbol de React para que no se reinicialice.

**Verificación:** recorrido Playwright idéntico antes y después (login → `/dashboard` → subir → polling de ingesta → preguntar → citar → abrir PDF en la página N). `git diff --stat` debe mostrar sobre todo movimientos. Comprobar a mano el deep-link `/dashboard?doc=…` y que recargar deslogueado redirige **una sola vez**.

---

## Fase 2 — Bloque de programación (solo lectura)

**Backend (mínimo):** añadir `"code"` al conjunto literal de modos válidos (`backend/app/routes/chat.py:191-195`) y una rama en `route_query` que replique el patrón de `agent.py:442-446`.

Dar a `build_agent_tools` (`backend/app/rag/tools.py:724`) un argumento `profile: "research" | "code" | "data"` en lugar de crear un segundo constructor. El perfil `code` descarta `PDFSearchTool` y conserva `CodeReviewTool`, `SkillTool` y las herramientas MCP.

La garantía de solo lectura ya la dan el allowlist/denylist existentes; hay que **fijarla con un test** que afirme que ningún nombre de herramienta MCP con capacidad de escritura llega al perfil `code`, de modo que lo garantice CI y no la configuración.

**Frontend:** `(blocks)/code/page.tsx` reutiliza `ChatPanel` con una prop `variant`. Aquí sí toca **des-singletonizar `chat-store`**, con **slices por clave**, no factoría + contexto: `messages: Msg[]` pasa a `byBlock: Record<BlockId, {messages, activeSessionId, routingMode}>`. Una factoría con provider obligaría a migrar todos los consumidores fuera del import `useChatStore()` y rompería el modelo de clave única del middleware `persist`.

**Descomposición de `ChatPanel.tsx` (1112 líneas): exactamente tres extracciones, sin reescritura.**

1. `src/hooks/useChatTransport.ts` — el WebSocket construido en línea en `ChatPanel.tsx:331` más el fallback SSE.
2. `src/components/chat/MessageList.tsx`
3. `src/components/chat/Composer.tsx` — input, dictado y el desplegable de modos, cuyas opciones pasan a ser una prop para que el bloque de código muestre modos distintos.

Exportación, estado de herramientas y progreso de investigación se quedan dentro de `ChatPanel`: son específicos de investigación. Objetivo ~400 líneas, no cero.

---

## Fase 3 — Bloque de datasets

**Modelo:** `Dataset` en `backend/app/models.py` espejando `Document` (`models.py:259`): mismo enum de estado, mismo bloque de progreso (`processing_progress/stage/current/total`), mismo borrado lógico, y el mismo `user_id` redundante en las tablas hijas. Más `DatasetProfile` (una fila por dataset, `profile_json` Text). Una tabla nueva solo necesita la clase del modelo: `init_db()` (`backend/app/database.py:214`) la crea. **No añadir scripts `migrate_add_*.py`** — están obsoletos.

**Reutilizar del pipeline de subida:** `validate_upload` (`backend/app/routes/documents.py:137`) añadiendo una rama CSV/parquet paralela a la de PyMuPDF de `documents.py:199`; el esquema `UPLOAD_DIR/<user_id>/<uuid>.<ext>`; el fallback de disponibilidad de workers de `_queue_or_run_ingestion` (`documents.py:79`); y el contrato de polling `GET /{id}/status` para que el frontend reutilice la misma UI de progreso.

**Nuevo:** `backend/app/routes/datasets.py` (registrar en `main.py` junto a las líneas 283-288, **antes** del catch-all de `main.py:354`), una tarea Celery `profile_dataset` en `backend/app/tasks.py`, y `pandas` + `pyarrow` **declarados explícitamente** en `backend/requirements.txt` (hoy solo entran de forma transitiva). El perfilado corre en Celery, nunca en línea, y con tope: muestreo de N filas y abandono por encima de un límite de filas/MB, o un parquet de 2 GB bloquea un worker.

**Herramienta del agente — `DatasetQueryTool`:** **no acepta SQL ni cadenas `df.query` del LLM.** Recibe entrada estructurada (`dataset_id`, `columns`, `filters: [{col, op, value}]`, `agg`, `limit`), construye la llamada pandas en código, acota las filas devueltas y filtra por `user_id`. Se añade al perfil `data` de `build_agent_tools` y se nombra en `AGENT_SYSTEM_PROMPT`.

**Frontend:** `npx shadcn add table select alert-dialog` (no existen hoy).

---

## Fase 4 — Skills auto-escritas (HERMES)

`backend/app/rag/skills.py` (262 líneas) hoy es **estrictamente de solo lectura**. Añadir `write_skill(name, description, body)` que:

1. Genere un slug y escriba **solo** bajo `<root>/<slug>/SKILL.md`, rechazando cualquier ruta resuelta fuera de la raíz.
2. Ejecute el `_RESERVED_REACT_TOKEN_RE` **ya existente** (`skills.py:102`) **antes** de escribir. Ese guard rechaza al cargar cualquier SKILL.md cuyo cuerpo tenga `Thought:`/`Action:`/`Observation:`/`Final Answer:` a principio de línea — y un agente que escribe sobre código emitirá esos tokens de forma natural en ejemplos. Devolver un rechazo estructurado para que el agente reescriba, con escapado como alternativa, nunca descarte silencioso.
3. Llame a `clear_skills_cache()`.

**Descripción obsoleta de `SkillTool`:** hoy el catálogo se hornea en `description` en tiempo de import (`tools.py:802-823`), así que una skill recién escrita no se ve hasta reiniciar. Se arregla construyendo `SkillTool(catalog=skills_catalog_text())` dentro de `build_agent_tools`, que ya se invoca por petición.

**Disparador:** **no engancharlo al bucle ReAct.** Se dispara *después* de terminar la ejecución, en `generate_answer` / `generate_answer_stream` (`backend/app/routes/chat.py:566/588`), como una llamada LLM barata condicionada por una heurística (número de pasos sobre un umbral, o una herramienta usada ≥3 veces).

**Las skills propuestas entran como borradores que requieren aprobación del usuario en la UI.** Que un agente escriba sin supervisión contenido que después se carga siempre en su propio prompt es el borde más afilado de todo este plan.

---

## Fase 5 — Memoria del agente (HERMES)

**Basada en archivos, por usuario: `UPLOAD_DIR/<user_id>/memory/MEMORY.md` y `USER.md`.** El precedente ya existe: `_load_global_style_reference()` (`backend/app/rag/agent.py:208-234`) lee un archivo, lo trunca, lo envuelve en una sección markdown y lo inyecta con `.partial()`. El directorio por usuario ya existe, y un archivo es diffeable y borrable a mano. Una tabla aportaría consultabilidad que no se necesita con un documento por usuario.

Añadir `{agent_memory}` a `AGENT_SYSTEM_PROMPT` (`backend/app/rag/prompts.py:250`) e inyectarlo con `.partial()`, con tope duro (~1500 caracteres) y **acotado**: envuelto en un bloque `<user_memory>` con una línea previa que declare que su contenido es dato suministrado por el usuario y **nunca instrucciones**, y saneado con el mismo regex de tokens ReAct.

Ese regex pasa a ser el guard de inyección compartido por las fases 4 y 5: promoverlo a un helper `sanitize_prompt_block()`.

El "nudge periódico" se implementa igual que el disparador de la Fase 4: post-ejecución, fuera del bucle, nunca a mitad de un ReAct. **No extender `backend/app/scheduler.py`**: hoy tiene tres jobs fijos, jobstore en memoria, sin `CronTrigger` ni jobs por usuario; convertirlo en un cron por usuario es un proyecto mayor que la propia memoria, y esta no lo necesita.

---

## Verificación por fase

| Fase | Cómo se comprueba |
|---|---|
| 1 | Playwright del recorrido completo idéntico antes/después; una sola redirección al recargar deslogueado; el visor PDF no se remonta |
| 2 | Test de backend que afirma que ninguna herramienta MCP de escritura llega al perfil `code`; los dos bloques de chat mantienen historiales independientes |
| 3 | Subida de CSV y parquet con perfilado en Celery; tope de filas respetado; `DatasetQueryTool` rechaza entrada no estructurada; aislamiento por `user_id` |
| 4 | Una skill con `Action:` en el cuerpo se rechaza con mensaje accionable; una skill recién aprobada aparece en el catálogo sin reiniciar |
| 5 | Memoria con texto tipo instrucción no altera el comportamiento del agente; se respeta el tope de caracteres |

Base de referencia: `cd backend && python3 -m pytest tests/ -q` — hoy 3 fallos preexistentes ajenos a este trabajo (`test_router_ignores_filesystem_phrasing_without_mcp_configured`, `test_delete_document_soft_deletes_and_hides_document`, `test_generate_answer_stream_appends_graph_context`). Cualquier fallo adicional es una regresión.
