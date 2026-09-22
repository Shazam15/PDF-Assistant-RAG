# Observabilidad

ATLAS produce dos señales distintas, con herramientas distintas y preguntas distintas:

| Señal | Herramienta | Responde a |
| --- | --- | --- |
| Métricas de sistema | `prometheus-fastapi-instrumentator` + Grafana (`grafana_dashboard.json`) | ¿Cuántas peticiones, con qué latencia y cuántos errores? |
| Traza del razonamiento | Langfuse autoalojado (este documento) | ¿Por qué *esta* respuesta concreta salió así, y en qué etapa se fue el tiempo? |

Las métricas agregan; la traza explica un caso individual. Este documento cubre la segunda.

## 1. Qué se traza

Cada ejecución de `research_rag` produce **una traza** con una observación por etapa sustantiva del grafo documentado en [`AGENT_LOOPS.md`](AGENT_LOOPS.md) §3:

```
understand → retrieve → ledger → audit → outline → draft → verify → repair → finalize
```

Los nodos `*_start` (`audit_start`, `draft_start`, `verify_start`, `repair_start`) **no** se trazan: solo publican una transición de etapa para el stream de progreso y no ejecutan trabajo, así que un span por cada uno solo añadiría ruido.

Toda llamada al LLM hecha con LangChain dentro de una etapa queda anidada bajo el span de esa etapa, porque el `CallbackHandler` de Langfuse se cuelga del contexto OpenTelemetry activo. Eso incluye el planificador, la síntesis, la auditoría, la verificación y la reparación, además del bucle ReAct de `tool_agent`.

## 2. Qué NO sale del proceso

Una decisión de diseño explícita, alineada con el invariante de [`ARCHITECTURE.md`](ARCHITECTURE.md) §Observabilidad ("los logs no deben contener cadena de pensamiento ni el contenido completo de documentos privados"):

`_trace_output` (`backend/app/rag/research_agent.py`) exporta **solo** conteos, etiquetas de etapa y nombres de facetas. Quedan deliberadamente fuera:

- el texto de la evidencia recuperada,
- el texto de las fuentes,
- la respuesta redactada (solo se exporta su longitud en caracteres).

La pregunta del usuario sí se registra, únicamente como `input` del nodo `understand`.

Hay un test de regresión que hace valer este contrato: `test_trace_output_excludes_document_text_and_drafted_answer` en `backend/tests/test_tracing.py`.

## 3. Correlación entre una traza y una ejecución guardada

El `trace_id` se genera **antes** de que corra el primer nodo y viaja por dos caminos a la vez:

- hacia Langfuse, como identificador de la traza;
- hacia la base de datos, dentro de `ResearchRun.state_json` (campo `trace_id`).

Y en sentido inverso, cada span lleva el `run_id` en sus metadatos y como `session_id`. Así se puede saltar de una ejecución almacenada a su traza completa y viceversa, sin necesidad de una columna nueva ni de una migración de esquema.

## 4. Por qué el `trace_id` se pasa explícitamente

`stream_research_agent` es un **generador**: sus etapas se reanudan en el hilo que el consumidor (SSE o WebSocket) tenga a mano, que no es necesariamente el mismo entre un `yield` y el siguiente. Un `with` abierto a lo largo de toda la ejecución dependería de que ese contexto (`contextvar`) se preserve entre suspensiones, cosa que no está garantizada.

Por eso cada etapa abre y cierra su propio span **dentro de una única llamada al nodo** y se engancha a la traza por `trace_id` explícito, en vez de por contexto ambiental heredado. El anidamiento de las llamadas LLM sí usa el contexto ambiental, pero solo dentro de esa misma llamada sincrónica, donde sí es seguro.

## 5. Levantar el stack autoalojado

```bash
docker compose -f docker-compose.langfuse.yml up -d
```

La primera vez descarga ~5.3 GB de imágenes. Cuando termine, `curl -s http://localhost:3001/api/public/health` debe responder `{"status":"OK",...}`.

Nota sobre MinIO: se descarga desde `quay.io`, no desde Docker Hub. MinIO retiró el repositorio `docker.io/minio/minio` (hoy devuelve 404), y usar ese nombre falla con un mensaje engañoso — `pull access denied ... may require 'docker login'`— que parece un problema de credenciales cuando en realidad la imagen ya no está ahí.

Luego abre <http://localhost:3001>, crea una cuenta y un proyecto, copia las claves y ponlas en tu `.env`:

```bash
LANGFUSE_ENABLED=True
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3001
```

El puerto es **3001**, no el 3000 por defecto de Langfuse, porque el frontend Next.js ya ocupa el 3000 en desarrollo local.

Al arrancar, el backend comprueba la conexión una sola vez y registra:

```
Langfuse tracing active host=http://localhost:3001
```

Esa comprobación vive en el arranque y no en el import, para que no ralentice la carga del módulo.

**Si el host no responde en esa comprobación, el tracing se desactiva para todo el proceso**, con dos advertencias explícitas y nada más. No se deja "activado a medias": si se dejara, cada etapa seguiría construyendo spans que el exportador reintenta y registra, enterrando los logs reales de la aplicación bajo errores de conexión de una función que de todos modos no está funcionando.

La consecuencia práctica: **levanta el stack antes que el backend**. Si arrancas Langfuse después, hay que reiniciar el backend para que vuelva a trazar.

## 6. Presupuesto de memoria: dónde conviene correrlo

El stack son seis contenedores. Medido en reposo en una MacBook Pro M1 (arm64 nativo, sin emulación), recién levantado y sin trazas todavía:

| Contenedor | RAM en reposo |
| --- | --- |
| `langfuse-web` | ~1.1 GiB |
| `langfuse-worker` | ~430 MiB |
| `clickhouse` | ~325 MiB |
| `minio` | ~90 MiB |
| `langfuse-postgres` | ~70 MiB |
| `langfuse-redis` | ~16 MiB |
| **Total** | **~2 GiB** |

Las imágenes ocupan además ~5.3 GB en disco.

En la MacBook Pro M1 de 8 GB, esos 2 GB **compiten directamente con Ollama**, que ya es el proceso más hambriento del sistema (con `qwen3:4b` cuantizado quedan unos ~4 GB para macOS, Ollama y el backend: alcanza, pero va justo). Recomendaciones, en orden de preferencia:

1. Correr Langfuse en la máquina NVIDIA/servidor y apuntar `LANGFUSE_HOST` a ella desde el portátil.
2. Levantarlo solo mientras se está instrumentando activamente, y bajarlo (`docker compose -f docker-compose.langfuse.yml down`) al terminar.
3. Si aun así aprieta, bajar `LANGFUSE_SAMPLE_RATE` reduce el volumen enviado, pero **no** reduce la RAM del stack: el coste está en los contenedores, no en el tráfico.

Nótese además que Docker Desktop en macOS no expone Metal a los contenedores, así que ejecutar la aplicación dentro de Docker en esa máquina degradaría el stack de ML a CPU. El stack de Langfuse sí puede vivir en Docker sin ese problema, porque no usa GPU.

## 7. Comportamiento ante fallos (fail-open)

El tracing nunca puede romper una respuesta del chat. Concretamente:

| Situación | Comportamiento |
| --- | --- |
| `langfuse` no instalado | Advertencia al arrancar; tracing desactivado |
| `LANGFUSE_ENABLED=False` | Todas las funciones son no-op; cero coste |
| Claves ausentes | Advertencia al arrancar; tracing desactivado |
| Host inalcanzable al arrancar | Dos advertencias y el tracing se desactiva para todo el proceso (requiere reinicio) |
| El host se cae con el proceso ya arrancado | El SDK reintenta y registra el fallo; las trazas de ese periodo se pierden, la app sigue igual |
| El SDK lanza una excepción a mitad de una etapa | Se registra en `debug`, la etapa devuelve un span no-op y **el nodo se ejecuta igual** |

El último caso tiene test propio: `test_traced_node_still_runs_when_the_trace_backend_fails`.

### No aparece ninguna traza

Por orden de probabilidad:

1. **Falta `LANGFUSE_ENABLED=True`.** Tener las claves puestas no basta; el interruptor es independiente. Desde el arranque esto se avisa explícitamente (`credentials are configured but LANGFUSE_ENABLED is not set to True`), porque es el error más fácil de cometer y el más difícil de diagnosticar.
2. **La variable del host se llama `LANGFUSE_HOST`.** También se acepta `LANGFUSE_BASE_URL` —el nombre que emite la página de onboarding de Langfuse— pero cualquier otro nombre se ignora en silencio y se usa el valor por defecto.
3. **El backend arrancó antes que Langfuse**, así que el tracing se autodesactivó (§5). Reinicia el backend.
4. **La ruta usada no pasa por el grafo.** Solo `research_rag` produce los spans por etapa; `simple_rag` y modo Rápido no entran al grafo, aunque sus llamadas al LLM sí se registran por los callbacks.

Los cambios en `.env` requieren reiniciar el backend: la configuración se cachea con `lru_cache` al arrancar.

## 8. Limitación conocida

`_fallback_state_stream` —la máquina de estados equivalente que se usa **solo** si `langgraph` no está instalado— no está instrumentada. Como `langgraph` es una dependencia declarada en `backend/requirements.txt`, esa ruta no se ejerce en una instalación normal. Si alguna vez se convierte en la ruta principal, habría que instrumentarla aparte.

## 9. Mapa del código

| Área | Archivo |
| --- | --- |
| Configuración (`LANGFUSE_*`) | `backend/app/config.py` |
| Integración, fail-open y helpers | `backend/app/rag/tracing.py` |
| Instrumentación del grafo | `backend/app/rag/research_agent.py` (`_traced_node`, `_trace_output`) |
| Callbacks del ReAct y del LLM | `backend/app/rag/agent.py` |
| Comprobación de conexión al arrancar | `backend/app/main.py` |
| Stack autoalojado | `docker-compose.langfuse.yml` |
| Tests | `backend/tests/test_tracing.py` |
