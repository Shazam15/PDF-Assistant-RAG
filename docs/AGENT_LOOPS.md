# Bucles del agente y uso de herramientas

Este documento complementa [`ARCHITECTURE.md`](ARCHITECTURE.md) con el mecanismo exacto —no solo el diagrama de alto nivel— del enrutador, el agente de herramientas (`tool_agent`), el grafo de investigación (`research_rag`) y el bucle dedicado de revisión de código (`code_review_agent`). Para las fórmulas de recuperación (RRF, reranking, NLI, grafo de conocimiento) ver [`RETRIEVAL_MATH.md`](RETRIEVAL_MATH.md). Los números de línea citados corresponden al código en el momento de escribir este documento.

## 1. Enrutador (`route_query`, `backend/app/rag/agent.py`)

El enrutador nunca llama a un LLM: es una cascada de reglas evaluadas en orden, y la primera que aplica decide la ruta.

1. **Saludo**: si la pregunta normalizada (minúsculas, sin `!?.` finales) coincide exactamente con uno de ~25 saludos fijos en español/inglés → `greeting`. Se evalúa antes que cualquier modo manual, incluido `code_review` (§4): un simple "Hola" no dispara el bucle dedicado.
2. **Modo Rápido explícito**: si `routing_mode == "quick"` → `scoped_rag` (documento seleccionado) o `simple_rag` (corpus completo), sin más evaluación.
3. **Modo Revisión de Código explícito**: si `routing_mode == "code_review"` → `code_review_agent` (§4) de forma incondicional, sin importar el contenido de la pregunta — mismo precedente que el paso 2.
4. **Herramienta requerida explícitamente**: `_required_tool` busca frases fijas en categorías —web, cálculo, código y, solo si hay al menos un servidor MCP configurado, "archivos" (p. ej. "lista los archivos", "list the files")— sobre la pregunta normalizada; ante una coincidencia → `tool_agent` con `required_tool` fijado. La categoría "código" ya **no** dispara la carga determinista de la skill `code-review` (ver §4.3): en el bucle genérico, esa decisión ahora es del modelo.
5. **Documento seleccionado**: si hay `document_id` → `scoped_rag` (con planificador si `routing_mode == "research"`).
6. **Modo Investigación explícito**: `routing_mode == "research"` → `research_rag`.
7. **Puntaje multidocumento ≥ 2**: `_multidocument_score` suma marcadores léxicos en 4 categorías (amplitud +2, comparación +2, síntesis +1, dimensión +1, sin cotas superiores); si el total es ≥2 → `research_rag`.
8. **Puntaje == 1 (ambiguo)**: si ya se conoce cuántos documentos trajo una recuperación previa (`retrieved_document_count`), ≥3 documentos relevantes promueve a `research_rag`; si no, `simple_rag`. Si aún no se conoce ese conteo, la decisión es *provisional*: se recupera primero y se vuelve a invocar `route_query` con el conteo real para resolverla (dos fases, sin exponer la ambigüedad al usuario).
9. **Por defecto**: `simple_rag`.

Ningún requisito de estilo ("actúa como investigador", pide un abstract, secciones o citas) activa por sí solo `research_rag`: esas frases entran en `deliverables`/`constraints` del `ResearchBrief`, nunca en las facetas de evidencia.

### 1.1 Escalamiento tardío a herramientas MCP (`_fallback_tool_agent_answer`)

El paso 3 solo detecta frases explícitas ("lista los archivos"...); una pregunta como "qué documentos hay en mi carpeta de Desktop" no las contiene y nunca se enruta a `tool_agent` de entrada. Por eso existe un segundo mecanismo, no basado en palabras clave: cuando `simple_rag`/`scoped_rag` terminan **sin evidencia recuperada** (contexto vacío, o la validación de citas colapsa al mensaje fijo de evidencia insuficiente), y hay al menos un servidor MCP configurado, el sistema reintenta la pregunta como si fuera `tool_agent` con `required_tool="files"` antes de rendirse. Cualquier fallo en este intento se registra como advertencia y el flujo cae al mensaje normal de "sin evidencia", nunca a un error visible.

## 2. `tool_agent`: bucle ReAct

`get_agent_executor`, `backend/app/rag/agent.py`

### 2.1 Herramientas vinculadas

`build_agent_tools(user_id, document_id, top_k)` construye la lista completa: los cinco internos —`PDFSearchTool` (`pdf_search`), `CodeReviewTool` (`code_review`), `MathTool` (`calculator`), `WebSearchTool` (`web_search`), `StatisticsTool` (`statistics`)— más `load_mcp_tools()` (herramientas MCP descubiertas en tiempo de ejecución, ver §7). Es la misma lista, y el mismo `AgentExecutor`, el que despacha herramientas internas y MCP: no hay una ruta de despacho separada para MCP dentro del bucle del agente.

### 2.2 Tope de iteraciones

```python
research_iterations = max(1, min(5, settings.AGENT_MAX_ITERATIONS - 1))
```

Con `AGENT_MAX_ITERATIONS = 6` (comentario en `config.py`: "cinco pasos de uso de herramientas más una síntesis final obligatoria"), `research_iterations = 5`. Es más alto que en la rama sin MCP (`min(3, ...)` allí) porque un flujo de herramientas MCP con varios saltos —por ejemplo, listar directorios permitidas y luego leer un archivo— necesita más de 3 pasos, sobre todo con un modelo local pequeño propenso a copiar mal una ruta larga que acaba de leer. Este valor se pasa a `AgentExecutor(max_iterations=research_iterations, early_stopping_method="force")`.

### 2.3 Parser de salida tolerante (`GroundedReActOutputParser`)

Si el parseo ReAct estándar falla (`OutputParserException`) pero el texto crudo tiene ≥35 tokens de palabra, se trata como una respuesta final legítima en vez de reintentar el formato — evita agotar el presupuesto de iteraciones por errores de formato cuando el modelo ya redactó una respuesta sustantiva. Lo mismo ocurre si la acción propuesta no corresponde a ninguna herramienta vinculada.

Dos mejoras adicionales existen en esta rama, motivadas directamente por herramientas MCP sin argumentos (p. ej. `list_allowed_directories`):

- `valid_tool_names` ya no es un conjunto fijo de 4 nombres: `get_agent_executor` lo calcula como `{tool.name for tool in tools} | _AGENT_TOOL_NAMES`, así que una herramienta MCP descubierta en tiempo de ejecución se reconoce como una acción válida en vez de descartarse como "nombre de herramienta inválido".
- `_recover_missing_action_input`: cuando el modelo escribe `Action: list_allowed_directories` sin ninguna línea `Action Input:` (frecuente en modelos locales pequeños ante herramientas sin argumentos, y distinto de una entrada mal formada), el parser sintetiza la llamada con una entrada vacía en vez de fallar el formato y quemar una iteración completa reintentando.
- El `tool_input` parseado se recorta con `.strip()` completo (no solo espacios): el parser base de LangChain solo quita espacios literales, así que una línea en blanco final tras `Action Input: <valor>` sobrevivía como un `\n` de cola — para una herramienta MCP que espera una ruta de archivo, eso convertía `uploads` en `uploads\n`, y la llamada fallaba con "no existe" aunque el modelo hubiera escrito el valor correcto.

### 2.4 Búsqueda inicial obligatoria

Antes de que el ejecutor ReAct arranque, el sistema invoca `pdf_search` una vez de forma incondicional (`_run_initial_document_search`) para garantizar al menos un intento de evidencia documental, e informa al modelo en la pregunta reformulada que esa búsqueda ya se ejecutó.

### 2.5 La respuesta del ReAct casi nunca se usa tal cual

Tras `executor.invoke(...)`:

1. Si el ejecutor falla pero ya hay fuentes recolectadas de las herramientas, se sintetiza una respuesta parcial en vez de fallar.
2. Si la respuesta cruda contiene frases de parada por límite de iteraciones/tiempo, se descarta y se regenera desde las fuentes acumuladas.
3. **Si la búsqueda inicial obligatoria produjo fuentes, la respuesta `Final Answer:` del LLM se descarta siempre** y se regenera con una pasada de síntesis dedicada (`_generate_partial_answer_from_agent_sources`), que tiene su propio ciclo de reparación de citas (ver §5).
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
| `verify` | Detecta afirmaciones sin respaldo (ver §5) |
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

## 4. `code_review_agent`: bucle Percibe-Razona-Actúa (`backend/app/rag/code_review_agent.py`)

Un cuarto modo explícito, `routing_mode == "code_review"`, fuerza incondicionalmente la ruta `code_review_agent` en `route_query` (mismo precedente que `quick`/`research`: el modo manual gana sobre cualquier heurística). Es un módulo aislado del `tool_agent` (§2) y del grafo de investigación (§3) — ninguno de los dos cambia por su existencia — que envuelve su propio `AgentExecutor` ReAct (construido en `get_code_review_executor`, con su propio prompt `CODE_REVIEW_AGENT_PROMPT` y su propio tope de iteraciones `CODE_REVIEW_MAX_ITERATIONS_PER_ROUND`, deliberadamente desacoplado de `AGENT_MAX_ITERATIONS`) en un bucle exterior de rondas, en Python puro.

"Percibe/Razona/Actúa" es una relectura narrativa del mismo ciclo mecánico Thought/Action/Observation/Final Answer — el parser (`GroundedReActOutputParser`, reutilizado sin bifurcar) sigue reconociendo únicamente esos tokens literales.

### 4.1 Condición de término (dos partes, ambas obligatorias)

1. El ejecutor interno llegó a una `Final Answer` por sí mismo, no por agotar `max_iterations` — "el agente decidió que no necesita más herramientas".
2. Una verificación determinista de sintaxis Python (`ast.parse`, solo biblioteca estándar) sobre cualquier código Python que la ronda haya tocado o producido no encontró errores.

Si la ronda no cumple ambas condiciones, los errores de sintaxis se agregan a la siguiente ronda como una nota de "Percepción" y el ejecutor reintenta, acotado por `CODE_REVIEW_MAX_ROUNDS` (rondas) y `CODE_REVIEW_TIMEOUT_SECONDS` (reloj de pared, mismo patrón que `RESEARCH_TIMEOUT_SECONDS` en §3.4). Si se agota el presupuesto sin una ronda limpia, el bucle falla de forma abierta: devuelve la mejor respuesta disponible más un aviso explícito de que la verificación automática no pudo confirmarla, en vez de fallar o simular éxito — la misma convención que `_synthesize_best_available` en el grafo de investigación (§3.4).

Los archivos en lenguajes distintos a Python nunca se verifican automáticamente; el prompt le indica al modelo que lo aclare cuando sea relevante.

### 4.2 Solo lectura/asesoría

Por decisión explícita de producto, este modo nunca escribe ni modifica archivos: no se habilita ninguna herramienta MCP de escritura, y el prompt prohíbe afirmar haber guardado un archivo. El código revisado o generado siempre se entrega como texto en la `Final Answer`, para que el usuario lo aplique manualmente.

### 4.3 Selección de skill: determinista aquí, no en el bucle genérico

`REQUIRED_TOOL_SKILL_MAP` (agent.py) ya **no** incluye `"code"`: en el `tool_agent` genérico (modo `auto`), el modelo decide por sí mismo si llama a `use_skill('code-review')` antes de `code_review`, en vez de que se le fuerce por coincidencia de palabras clave (`statistics` conserva la inyección determinista, sin cambios). En el modo `code_review` dedicado, en cambio, la skill sí se carga de forma determinista al inicio de cada ejecución (`_load_code_review_skill_body`) — la determinación aquí es intencional: el usuario ya eligió explícitamente este modo.

### 4.4 Conocimiento de documentos cargados

El bucle reutiliza `pdf_search` sin cambios: el prompt indica al modelo que lo use antes de generar código cuya lógica deba seguir una especificación, ecuación o comportamiento descrito en un documento cargado por el usuario, y que mencione la fuente de forma informal (no con el contrato estricto `[D#]`-o-rechazo de §5 más abajo, que está calibrado para respuestas en prosa y rechazaría código válido sin citas en línea). El bucle no llama a ese mecanismo de validación de citas en absoluto.

## 5. Verificación y reparación de citas

Ver [`RETRIEVAL_MATH.md` §7](RETRIEVAL_MATH.md#7-verificación-de-afirmaciones) para el detalle matemático completo (chequeo numérico, umbral NLI, regex de citas). En resumen, dentro del bucle del agente:

- Toda oración sustantiva (≥8 tokens de palabra) sin cita `[D#]`/`[W#]` cuenta como problema.
- Toda oración citada con un número/unidad ausente de su fuente citada cuenta como problema, sin necesidad de NLI.
- Si `sources` está vacío o no hay ninguna cita estructurada, la respuesta se reemplaza directamente por el mensaje fijo de evidencia insuficiente.
- Existen tres mecanismos de reparación distintos según el punto del pipeline: (1) regenerar con retroalimentación explícita de los problemas detectados (hasta 2 intentos, se conserva el mejor), (2) poda oración por oración de afirmaciones sin respaldo (`_prune_unsupported_claims`) cuando ningún intento queda limpio, y (3) el nodo `repair` del grafo de investigación, que pide una corrección localizada y cae a la poda si la reparación también falla la validación.

## 6. Herramientas (`backend/app/rag/tools.py`)

| Herramienta | Nombre | Entrada | Aislamiento |
| --- | --- | --- | --- |
| `PDFSearchTool` | `pdf_search` | consulta libre | Llama a `retriever.retrieve`; añade contexto del grafo de conocimiento; envuelve cada resultado entre marcadores `UNTRUSTED DOCUMENT EXCERPT ... END` con una advertencia explícita de no seguir instrucciones dentro del texto (mitigación de inyección de prompt vía documentos) |
| `MathTool` | `calculator` | expresión aritmética | Evalúa con `ast.parse(expr, mode="eval")` restringido a operadores `+ - * / // % **` sobre literales numéricos; cualquier otro nodo del AST (nombres, llamadas, atributos) lanza `ValueError` — no hay `eval()`/`exec()` ni resolución de nombres, así que no hay superficie de ejecución de código |
| `WebSearchTool` | `web_search` | consulta libre | Usa `DDGS().text(...)` (DuckDuckGo, sin clave de API); resultados envueltos entre `UNTRUSTED WEB RESULT ... END`; deduplicados por URL |
| `CodeReviewTool` | `code_review` | código + lenguaje + foco | Nunca ejecuta el código recibido; solo pide al LLM una revisión textual |
| `StatisticsTool` | `statistics` | datos numéricos + operación | Cálculos estadísticos sobre los datos que el propio LLM extrae y pasa como argumento, sin acceso a archivos ni al corpus |

No existe una herramienta de ejecución de código ni un intérprete en sandbox entre las herramientas internas: `MathTool` es la única superficie de expresión evaluable, y está restringida a un whitelist de AST numérico.

Todas las herramientas comparten un mecanismo de deduplicación de fuentes por clave (`id`/`chunk_id`/`documento`/`archivo`/`página`/primeros 200 caracteres de texto), conservando la ocurrencia de mayor puntaje.

## 7. Herramientas MCP (Model Context Protocol)

Además de las herramientas internas, el `tool_agent` puede usar herramientas [MCP](https://modelcontextprotocol.io/) declaradas en `MCP_SERVERS_JSON`, descubiertas y filtradas por `load_mcp_tools()` (`backend/app/rag/tools.py`) y añadidas a la misma lista de herramientas del `AgentExecutor` (§2.1). El mecanismo de aislamiento (solo lectura por defecto, allowlist/denylist, aislamiento de directorios delegado al proceso MCP, caché de descubrimiento y el límite de cancelación cooperativa) se documenta en detalle en [`MCP_TOOLS.md`](MCP_TOOLS.md) — no se duplica aquí para evitar que las dos copias diverjan.

## 8. Cancelación y llamadas a herramientas en curso

El botón de cancelar es cooperativo, no preventivo: una tarea en segundo plano (`backend/app/routes/chat.py`) marca un `cancellation_event` compartido en cuanto el socket se cierra, y `agent.py` comprueba ese evento en cada punto de control del bucle —entre pasos del `AgentExecutor`, antes de cada llamada de síntesis, antes de la validación final de citas—, cortando la respuesta SSE inmediatamente al detectarlo. Lo que **no** hace es interrumpir una llamada a herramienta ya en curso: si el modelo está a mitad de una búsqueda web o de una llamada MCP bloqueada en su propio subproceso, esa llamada corre hasta terminar; la cancelación solo impide que arranque el siguiente paso.
