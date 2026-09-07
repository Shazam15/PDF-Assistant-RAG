# Bucles del agente y uso de herramientas

Este documento complementa [`ARCHITECTURE.md`](ARCHITECTURE.md) con el mecanismo exacto —no solo el diagrama de alto nivel— del enrutador, el agente de herramientas (`tool_agent`) y el grafo de investigación (`research_rag`). Para las fórmulas de recuperación (RRF, reranking, NLI, grafo de conocimiento) ver [`RETRIEVAL_MATH.md`](RETRIEVAL_MATH.md). Los números de línea citados corresponden al código en el momento de escribir este documento.

## 1. Enrutador (`route_query`, `backend/app/rag/agent.py`)

El enrutador nunca llama a un LLM: es una cascada de reglas evaluadas en orden, y la primera que aplica decide la ruta.

1. **Saludo**: si la pregunta normalizada (minúsculas, sin `!?.` finales) coincide exactamente con uno de ~25 saludos fijos en español/inglés → `greeting`.
2. **Modo Rápido explícito**: si `routing_mode == "quick"` → `scoped_rag` (documento seleccionado) o `simple_rag` (corpus completo), sin más evaluación.
3. **Herramienta requerida explícitamente**: `_required_tool` busca frases fijas en tres (o cuatro, ver §5) categorías —web, cálculo, código— sobre la pregunta normalizada; ante una coincidencia → `tool_agent` con `required_tool` fijado.
4. **Documento seleccionado**: si hay `document_id` → `scoped_rag` (con planificador si `routing_mode == "research"`).
5. **Modo Investigación explícito**: `routing_mode == "research"` → `research_rag`.
6. **Puntaje multidocumento ≥ 2**: `_multidocument_score` suma marcadores léxicos en 4 categorías (amplitud +2, comparación +2, síntesis +1, dimensión +1, sin cotas superiores); si el total es ≥2 → `research_rag`.
7. **Puntaje == 1 (ambiguo)**: si ya se conoce cuántos documentos trajo una recuperación previa (`retrieved_document_count`), ≥3 documentos relevantes promueve a `research_rag`; si no, `simple_rag`. Si aún no se conoce ese conteo, la decisión es *provisional*: se recupera primero y se vuelve a invocar `route_query` con el conteo real para resolverla (dos fases, sin exponer la ambigüedad al usuario).
8. **Por defecto**: `simple_rag`.

Ningún requisito de estilo ("actúa como investigador", pide un abstract, secciones o citas) activa por sí solo `research_rag`: esas frases entran en `deliverables`/`constraints` del `ResearchBrief`, nunca en las facetas de evidencia.

## 2. `tool_agent`: bucle ReAct

`get_agent_executor`, `backend/app/rag/agent.py`

### 2.1 Herramientas vinculadas

`PDFSearchTool` (`pdf_search`), `CodeReviewTool` (`code_review`), `MathTool` (`calculator`), `WebSearchTool` (`web_search`) — ver §5 para el detalle de cada una y sus límites.

### 2.2 Tope de iteraciones

```python
research_iterations = max(1, min(3, settings.AGENT_MAX_ITERATIONS - 1))
```

Con `AGENT_MAX_ITERATIONS = 4` (comentario en `config.py`: "tres pasos de investigación más una síntesis final obligatoria"), `research_iterations = 3`. Este valor se pasa a `AgentExecutor(max_iterations=3, early_stopping_method="force")`: LangChain corta el bucle Pensamiento→Acción→Observación tras 3 llamadas a herramientas como máximo, forzando una parada en vez de lanzar una excepción.

### 2.3 Parser de salida tolerante (`GroundedReActOutputParser`)

Si el parseo ReAct estándar falla (`OutputParserException`) pero el texto crudo tiene ≥35 tokens de palabra, se trata como una respuesta final legítima en vez de reintentar el formato — evita agotar el presupuesto de iteraciones por errores de formato cuando el modelo ya redactó una respuesta sustantiva. Lo mismo ocurre si la acción propuesta no corresponde a ninguna herramienta vinculada.

### 2.4 Búsqueda inicial obligatoria

Antes de que el ejecutor ReAct arranque, el sistema invoca `pdf_search` una vez de forma incondicional (`_run_initial_document_search`) para garantizar al menos un intento de evidencia documental, e informa al modelo en la pregunta reformulada que esa búsqueda ya se ejecutó.

### 2.5 La respuesta del ReAct casi nunca se usa tal cual

Tras `executor.invoke(...)`:

1. Si el ejecutor falla pero ya hay fuentes recolectadas de las herramientas, se sintetiza una respuesta parcial en vez de fallar.
2. Si la respuesta cruda contiene frases de parada por límite de iteraciones/tiempo, se descarta y se regenera desde las fuentes acumuladas.
3. **Si la búsqueda inicial obligatoria produjo fuentes, la respuesta `Final Answer:` del LLM se descarta siempre** y se regenera con una pasada de síntesis dedicada (`_generate_partial_answer_from_agent_sources`), que tiene su propio ciclo de reparación de citas (ver §4).
4. Solo si no hubo búsqueda inicial con resultados, se valida/regenera la respuesta cruda por citas.

En la práctica, el texto que el LLM produce dentro del bucle ReAct actúa como una guía de qué buscar, no como la respuesta final entregada al usuario.

## 3. `research_rag`: grafo de investigación (`backend/app/rag/research_agent.py`)

### 3.1 Nodos y transiciones

```
START → understand → retrieve → ledger → audit_start → audit
audit --[after_audit]--> retrieve | outline
outline → draft_start → draft → verify_start → verify
verify --[after_verify]--> repair_start | finalize
repair_start → repair → verify_start   (vuelve a verificar)
finalize → END
```

| Nodo | Función |
| --- | --- |
| `understand` | Construye el `ResearchBrief` (pregunta principal + hasta 6 facetas atómicas) |
| `retrieve` | Recupera evidencia para las facetas pendientes y la fusiona con la existente |
| `ledger` | Construye el *claim ledger* (faceta → evidencia candidata) |
| `audit` | Evalúa cobertura, vacíos y contradicciones; puede filtrar evidencia irrelevante |
| `outline` | Construye el esquema afirmación→fuente a partir del ledger auditado |
| `draft` | Asigna `source_id` (`D1`, `D2`, …) y redacta la síntesis multifuente |
| `verify` | Detecta afirmaciones sin respaldo (ver §4) |
| `repair` | Reescribe una vez las afirmaciones problemáticas |
| `finalize` | Garantiza una respuesta no vacía y cierra la ejecución |

### 3.2 Condición para repetir recuperación (`after_audit`)

```python
synthesis_cutoff = deadline - RESEARCH_SYNTHESIS_RESERVE_SECONDS
puede_reintentar = (
    hay_facetas_pendientes
    and ronda_actual < RESEARCH_MAX_ROUNDS       # 2 por defecto
    and evidencia_nueva_en_la_ultima_ronda > 0     # evita bucles cuando la recuperación se satura
    and time.monotonic() < synthesis_cutoff
)
```

Las cuatro condiciones deben cumplirse a la vez. La tercera es la que impide un ciclo infinito: si una ronda de recuperación no trajo nada nuevo, no vale la pena repetirla aunque falten facetas.

### 3.3 Condición de reparación (`after_verify`)

```python
if hay_problemas and reparaciones_hechas < 1 and time.monotonic() < deadline:
    ir_a("repair")
else:
    ir_a("finalize")
```

Como máximo **una** reparación por ejecución; la respuesta reparada se vuelve a verificar (vuelve al nodo `verify`), pero si sigue teniendo problemas ya no hay una segunda reparación — pasa a `finalize` tal cual.

### 3.4 Presupuesto de tiempo

No existe un presupuesto que se va descontando: se calcula un único `deadline = time.monotonic() + RESEARCH_TIMEOUT_SECONDS` al iniciar la ejecución (1800 s por defecto), y **cada nodo** (excepto `outline` y `finalize`) comprueba al entrar si ya se superó ese `deadline` o si la ejecución fue cancelada, y si es así aborta con `TimeoutError`/`ResearchCancelled`. Adicionalmente, `after_audit` usa un corte derivado —`deadline - RESEARCH_SYNTHESIS_RESERVE_SECONDS` (600 s por defecto)— para dejar de recuperar con suficiente margen para redactar, verificar y reparar.

Si el `TimeoutError` llega a propagarse, `stream_research_agent` lo captura y produce la **mejor respuesta verificable disponible** a partir del estado acumulado (evidencia recuperada hasta ese punto, sin una segunda vuelta de verificación/reparación) en vez de fallar o devolver una lista cruda de chunks. El estado termina como `"timed_out"`, no como error.

`LLM_REQUEST_TIMEOUT_SECONDS` (900 s por defecto) es un límite **independiente**, por llamada HTTP a Ollama; puede agotarse dentro de un nodo antes o después de que quede margen en el `deadline` general de la investigación — no están acoplados.

### 3.5 Fusión de evidencia entre rondas

Cada `document_id`/`chunk_id` (o, si falta, `documento|archivo|página|primeros 200 caracteres`) identifica un candidato de forma única. Al fusionar una nueva ronda con la evidencia existente, un candidato repetido solo sobrescribe sus campos de puntuación si la nueva ocurrencia tiene mayor `relevance_score`; el conteo de "evidencia nueva" que alimenta `after_audit` es la cantidad de candidatos que **no** estaban ya presentes.

### 3.6 Punto de control

En cada cambio de `stage` se persiste un `ResearchRun` (ronda actual, cantidad de evidencia, facetas cubiertas/pendientes, contradicciones, ledger y esquema argumentativo) — esto es lo que permite reanudar o auditar una ejecución después de una interrupción, y es la fuente de los eventos `progress` que recibe el frontend.

Si el paquete `langgraph` no está disponible, existe una máquina de estados equivalente escrita a mano (`_fallback_state_stream`) que reproduce exactamente las mismas condiciones — no es una versión simplificada.

## 4. Verificación y reparación de citas

Ver [`RETRIEVAL_MATH.md` §7](RETRIEVAL_MATH.md#7-verificación-de-afirmaciones) para el detalle matemático completo (chequeo numérico, umbral NLI, regex de citas). En resumen, dentro del bucle del agente:

- Toda oración sustantiva (≥8 tokens de palabra) sin cita `[D#]`/`[W#]` cuenta como problema.
- Toda oración citada con un número/unidad ausente de su fuente citada cuenta como problema, sin necesidad de NLI.
- Si `sources` está vacío o no hay ninguna cita estructurada, la respuesta se reemplaza directamente por el mensaje fijo de evidencia insuficiente.
- Existen tres mecanismos de reparación distintos según el punto del pipeline: (1) regenerar con retroalimentación explícita de los problemas detectados (hasta 2 intentos, se conserva el mejor), (2) poda oración por oración de afirmaciones sin respaldo (`_prune_unsupported_claims`) cuando ningún intento queda limpio, y (3) el nodo `repair` del grafo de investigación, que pide una corrección localizada y cae a la poda si la reparación también falla la validación.

## 5. Herramientas (`backend/app/rag/tools.py`)

| Herramienta | Nombre | Entrada | Aislamiento |
| --- | --- | --- | --- |
| `PDFSearchTool` | `pdf_search` | consulta libre | Llama a `retriever.retrieve`; añade contexto del grafo de conocimiento; envuelve cada resultado entre marcadores `UNTRUSTED DOCUMENT EXCERPT ... END` con una advertencia explícita de no seguir instrucciones dentro del texto (mitigación de inyección de prompt vía documentos) |
| `MathTool` | `calculator` | expresión aritmética | Evalúa con `ast.parse(expr, mode="eval")` restringido a operadores `+ - * / // % **` sobre literales numéricos; cualquier otro nodo del AST (nombres, llamadas, atributos) lanza `ValueError` — no hay `eval()`/`exec()` ni resolución de nombres, así que no hay superficie de ejecución de código |
| `WebSearchTool` | `web_search` | consulta libre | Usa `DDGS().text(...)` (DuckDuckGo, sin clave de API); resultados envueltos entre `UNTRUSTED WEB RESULT ... END`; deduplicados por URL |
| `CodeReviewTool` | `code_review` | código + lenguaje + foco | Nunca ejecuta el código recibido; solo pide al LLM una revisión textual |

No existe una herramienta de ejecución de código ni un intérprete en sandbox: `MathTool` es la única superficie de expresión evaluable, y está restringida a un whitelist de AST numérico.

Todas las herramientas comparten un mecanismo de deduplicación de fuentes por clave (`id`/`chunk_id`/`documento`/`archivo`/`página`/primeros 200 caracteres de texto), conservando la ocurrencia de mayor puntaje.

Esta rama no incluye herramientas MCP (Model Context Protocol); esa capacidad existe únicamente en la rama `experimental` — ver su copia de este documento y `docs/MCP_TOOLS.md` allí.
