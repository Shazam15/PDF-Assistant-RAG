# Herramientas MCP (Model Context Protocol)

Esta funcionalidad existe únicamente en la rama `experimental`. Complementa [`AGENT_LOOPS.md` §6](AGENT_LOOPS.md#6-herramientas-mcp-model-context-protocol) con el mecanismo exacto de descubrimiento, aislamiento y despacho de herramientas [MCP](https://modelcontextprotocol.io/). Los números de línea corresponden al código en el momento de escribir este documento.

## 1. Activación y configuración (`backend/app/config.py`)

| Campo | Tipo | Por defecto | Notas |
| --- | --- | --- | --- |
| `MCP_ENABLED` | `bool` | `True` | Interruptor global; sin servidores configurados, el sistema queda inerte de todas formas |
| `MCP_SERVERS_JSON` | `str` | `"{}"` | JSON con la configuración de servidores MCP (comando, args, transporte) |
| `MCP_TOOL_ALLOWLIST` | `list[str]` | `read_file, list_directory, list_allowed_directories, file_exists, get_file_info, grep_file, count_pattern, extract_log_lines` | Coincidencia exacta de nombre; nunca incluye una herramienta mutante por defecto |
| `MCP_TOOL_DENYLIST` | `list[str]` | `[]` en código (el `.env.example` distribuido fija `write_file,edit_file,delete_file,move_file,create_directory` como defensa en profundidad) | Se comprueba **antes** que el allowlist y siempre gana |
| `MCP_TOOL_TIMEOUT_SECONDS` | `int` | `30` | Solo acota el descubrimiento de herramientas (§3), no cada llamada individual |
| `MCP_MAX_RESULT_CHARS` | `int` | `6000` | **Declarado pero no aplicado en ningún lugar del código** — ver nota en §6 |
| `MCP_SERVERS` | `dict` | `{}` | Calculado a partir de `MCP_SERVERS_JSON`, nunca se establece directamente por variable de entorno |

`MCP_TOOL_ALLOWLIST`/`MCP_TOOL_DENYLIST` usan `Annotated[list[str], NoDecode]`: sin `NoDecode`, pydantic-settings intentaría decodificar como JSON cualquier valor de entorno para un campo `list[str]` antes de que el `model_validator` corra, y fallaría con una cadena separada por comas como `MCP_TOOL_ALLOWLIST=read_file,list_directory`. `NoDecode` deja pasar la cadena cruda para que `_parse_mcp_settings` la separe manualmente.

### 1.1 Parseo (`_parse_mcp_settings`, `model_validator(mode="before")`)

1. Si `MCP_SERVERS_JSON` es una cadena vacía → `{}`; si es JSON inválido → `ValueError("MCP_SERVERS_JSON must be valid JSON.")`; si no decodifica a un objeto → `ValueError("MCP_SERVERS_JSON must decode to a JSON object.")`.
2. El resultado pasa por `_expand_mcp_server_paths` (§1.2) antes de guardarse en `MCP_SERVERS`.
3. Para `MCP_TOOL_ALLOWLIST`/`MCP_TOOL_DENYLIST`: **si la variable de entorno no está presente, no se toca** — forzarla a `[]` cuando simplemente está ausente descartaría en silencio el allowlist por defecto de la clase para quien no configuró nada. Si está presente como cadena, se separa por comas y se limpia; si es `None`, se vuelve `[]`.

### 1.2 Portabilidad de rutas (`_expand_mcp_server_paths`)

Introducida en el commit `55e9e72` ("Fixed allowed paths"). Para cada servidor configurado, aplica `os.path.expanduser(os.path.expandvars(arg))` a cada elemento de `args` que sea una cadena (los que no lo son —flags, nombres de paquete— pasan sin tocar). Así, `"~/Documents"` en `MCP_SERVERS_JSON` resuelve al directorio *home* de quien realmente ejecuta el proceso, en cualquier sistema operativo o usuario, en vez de quedar fijado al valor absoluto de quien escribió el `.env`. Antes de este commit, `MCP_SERVERS` se guardaba crudo y una ruta como `/Users/<dev>/Documents` se rompía en cualquier otra máquina.

## 2. Modelo de seguridad: allowlist y denylist (`_is_tool_allowed`, `backend/app/rag/tools.py`)

Denegado por defecto, con permiso explícito:

```
1. si nombre ∈ denylist (en minúsculas): DENEGAR   # el denylist siempre gana, incluso sobre el allowlist
2. si el allowlist está vacío:            DENEGAR   # fail-closed: sin allowlist, nada se permite
3. si nombre ∈ allowlist (en minúsculas): PERMITIR
4. en cualquier otro caso:                DENEGAR
```

`_DANGEROUS_MCP_TOOL_PATTERNS = ("filesystem", "shell", "bash", "exec", "browser", "delete", "write", "edit")` actúa como una segunda comprobación por subcadena, redundante con el paso 4 salvo que la lógica cambie en el futuro — el allowlist por defecto (8 nombres de solo lectura) ya excluye cualquier herramienta mutante. Una prueba de regresión dedicada (`test_default_mcp_allowlist_has_no_write_capable_tools`) hace valer ese contrato.

## 3. Transporte, descubrimiento y caché (`load_mcp_tools`, `backend/app/rag/tools.py`)

- Cliente: `langchain_mcp_adapters.client.MultiServerMCPClient`, importado de forma perezosa (si el paquete no está instalado, MCP se desactiva en silencio con una advertencia, sin afectar al resto de la aplicación).
- El transporte lo decide cada entrada de `MCP_SERVERS` (típicamente `"transport": "stdio"` con `"command": "npx"` y `"args": [...]`): el servidor MCP se lanza como **subproceso hijo** que habla por stdio, no HTTP.
- `load_mcp_tools()`:
  1. Si `MCP_ENABLED` es falso, o `MCP_SERVERS` está vacío → `[]`.
  2. Calcula una clave de caché (`servers` + `allow` + `deny`, JSON con claves ordenadas). Si coincide con la caché en memoria (protegida por un `threading.Lock`), devuelve la lista cacheada sin volver a descubrir — evita relanzar subprocesos y rehacer el *handshake* MCP en cada turno de chat, ya que `build_agent_tools` se invoca por solicitud.
  3. Si no hay caché válida: crea el cliente, y ejecuta `client.get_tools()` en un event loop de `asyncio` nuevo, envuelto en `asyncio.wait_for(timeout=MCP_TOOL_TIMEOUT_SECONDS)`. Un `TimeoutError` o cualquier otra excepción se registra y devuelve `[]` (falla abierto hacia "sin herramientas MCP", nunca hacia un error visible).
  4. Filtra las herramientas descubiertas con `_is_tool_allowed`; las bloqueadas se registran en una sola línea de advertencia.
  5. Cada herramienta permitida pasa por el adaptador de entrada única (§4).
- **La caché es solo de la lista de herramientas, no de las llamadas**: cada invocación de una herramienta abre su propia sesión MCP nueva (comportamiento sin estado de `langchain-mcp-adapters`), así que cachear el descubrimiento no crea conexiones persistentes ni afecta la vigencia de una llamada individual.
- `MCP_TOOL_TIMEOUT_SECONDS` acota exclusivamente esta llamada de descubrimiento. **No existe** un `asyncio.wait_for` equivalente alrededor de una llamada individual a una herramienta — el puente síncrono (§4) ejecuta `asyncio.run(coroutine(**kwargs))` sin timeout propio.

## 4. Adaptador de entrada única para el ejecutor ReAct

El ejecutor ReAct de esta app espera herramientas de una sola entrada de texto. Las herramientas MCP, tal como las construye `langchain-mcp-adapters`, no cumplen ese contrato de dos formas:

1. `args_schema` es un `dict` de JSON-schema crudo, no un modelo Pydantic — el auto-coerción de LangChain de una cadena suelta a una llamada estructurada solo funciona con un modelo Pydantic de un único campo.
2. Solo exponen una `coroutine` async; no hay `func` síncrono, y el camino de ejecución de esta app es síncrono.

`_adapt_mcp_tool_for_single_input_agent` corrige ambas, **mutando la herramienta in-place**:

- Si el esquema requiere más de un campo obligatorio, se registra una advertencia y la herramienta queda **sin adaptar** (no se rompe: simplemente no queda invocable desde el chat, en vez de fallar silenciosamente con datos incompletos).
- Si no requiere ningún campo (p. ej. `list_allowed_directories`), se genera un modelo Pydantic de un campo `str` sin usar, y el puente síncrono lo descarta antes de llamar a la corrutina.
- El puente síncrono (`_sync_bridge`) enruta el argumento posicional único al campo esperado y ejecuta `asyncio.run(coroutine(**kwargs))`.

## 5. Aislamiento de directorios: quién lo aplica realmente

**El código Python de esta aplicación no aísla el sistema de archivos por sí mismo.** La restricción vive enteramente en el proceso externo del servidor MCP: `@modelcontextprotocol/server-filesystem` recibe sus directorios raíz permitidos como argumentos posicionales (`args: ["-y", "@modelcontextprotocol/server-filesystem", "./data/uploads"]`) y es ese binario —no este repositorio— quien rechaza cualquier ruta fuera de esos raíces y expone `list_allowed_directories`. La contribución de esta app al aislamiento son cuatro capas independientes del binario externo:

1. Portabilizar esos `args` por máquina/usuario (`_expand_mcp_server_paths`, §1.2).
2. Restringir qué **nombres** de herramienta son invocables (§2) — incluso dentro de los directorios permitidos, solo operaciones de lectura.
3. Disciplina de *prompting*: el prompt del sistema del agente obliga a llamar primero a `list_allowed_directories` y copiar la ruta devuelta literalmente, advirtiendo explícitamente contra inventar una ruta o usar `.` como atajo (que resolvería a un directorio de trabajo fijo no relacionado).
4. `_uploads_dir_in_mcp_scope()`: antes de usar un atajo que asume que `UPLOAD_DIR` está cubierto por el servidor MCP, compara `os.path.abspath(UPLOAD_DIR)` contra cada `arg` configurado — solo si hay coincidencia exacta se confía en el atajo. Esto importa porque una herramienta MCP que no encuentra una ruta no lanza una excepción Python: devuelve su propio texto `"ENOENT: no such file or directory"` como una observación normal, que terminaría interpretándose como evidencia real de que no existen archivos si no se validara antes el alcance.

El commit `55e9e72` ("Fixed allowed paths") corrigió el punto 1 (portabilidad), no la lógica de aislamiento en sí — esa siempre ha sido responsabilidad del binario externo.

## 6. Reingreso de resultados al contexto del LLM

Una observación de una herramienta MCP reingresa al bucle exactamente igual que cualquier herramienta interna: LangChain serializa cada `AgentStep.observation` de vuelta al *scratchpad* `Thought/Action/Action Input/Observation` que se envía en la siguiente llamada al LLM. A diferencia de `PDFSearchTool`/`WebSearchTool`, que envuelven su salida entre marcadores `UNTRUSTED ... EXCERPT`, el texto de una herramienta MCP pasa **sin envolver**.

Para la interfaz de chat (SSE/WebSocket), cada paso emite eventos `tool_start`/`tool_result` con el texto recortado a **180 caracteres fijos** (`_truncate_tool_text`, exclusivamente para la línea de estado "Usando *tool*…" en la UI). Esto es independiente de `MCP_MAX_RESULT_CHARS`: ese ajuste de configuración existe en `Settings` y se documenta en `.env.example`, pero ningún punto del código lo lee — no hay recorte por tamaño de resultado antes de que el texto completo de una herramienta MCP llegue al LLM. Es un ajuste declarado pero inerte; si se necesita realmente acotar el tamaño de una respuesta MCP antes de que entre al contexto del modelo, ese enforcement todavía no existe y debe añadirse explícitamente.

## 7. Cancelación

Ver [`AGENT_LOOPS.md` §7](AGENT_LOOPS.md#7-cancelación-y-llamadas-a-herramientas-en-curso). En resumen: el evento de cancelación es cooperativo y se comprueba entre pasos del `AgentExecutor`, nunca dentro de una llamada a herramienta ya en curso — una llamada MCP bloqueada en su propio subproceso corre hasta terminar.

## 8. Resumen del modelo de seguridad

| Capa | Por defecto | Cómo se amplía |
| --- | --- | --- |
| Interruptor de la función | `MCP_ENABLED=True`, pero inerte sin servidores | `MCP_ENABLED=False` la desactiva por completo |
| Registro de servidores | `MCP_SERVERS_JSON="{}"` (ninguno) | El administrador debe declarar explícitamente cada servidor |
| Allowlist de nombres | 8 nombres de solo lectura | El administrador debe añadir un nombre explícitamente; nunca se distribuye una herramienta mutante por defecto |
| Denylist de nombres | `[]` en código / `write_file,edit_file,delete_file,move_file,create_directory` en el `.env.example` distribuido | Se comprueba antes que el allowlist; siempre gana |
| Aislamiento de directorios | Lo que el servidor MCP externo reciba como `args` | El administrador lo controla vía `MCP_SERVERS_JSON`; esta app solo portabiliza `~`/variables de entorno, no impone límites propios |
| Timeout de llamada | 30 s, aplicado solo al descubrimiento | `MCP_TOOL_TIMEOUT_SECONDS` |
| Límite de tamaño de resultado | Declarado (`MCP_MAX_RESULT_CHARS=6000`) pero **sin aplicar** en código | Ninguna — ajuste inerte actualmente |
| Cancelación | Cooperativa, entre pasos del ReAct; no interrumpe una llamada en curso | N/A |

## 9. Historial

- `12bc7ea` — primer andamiaje MCP ("not working"); el allowlist por defecto incluía `write_file` en ese momento, y `MCP_SERVERS` se guardaba sin expandir rutas.
- `44c71f8` — hizo las herramientas MCP realmente invocables desde el ejecutor de una sola entrada: quitó `write_file` del allowlist por defecto, añadió `list_allowed_directories`, introdujo `NoDecode` y la guarda de "no tocar allow/denylist si la variable está ausente", subió `AGENT_MAX_ITERATIONS` de 4 a 6, y añadió la caché de descubrimiento (`_mcp_tools_cache`), `_adapt_mcp_tool_for_single_input_agent` y `load_mcp_tools`.
- `4dabe2e` — refinamientos del lado del agente (sin cambios en `tools.py`/`config.py`).
- `55e9e72` — corrigió la portabilidad de rutas (`_expand_mcp_server_paths`), el botón de cancelación (antes solo se comprobaba en la rama de saludo del backend) y, en el mismo commit, un borrado físico incompleto de documentos (sin relación con MCP).
