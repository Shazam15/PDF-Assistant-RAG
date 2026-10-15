# Evaluación para el artículo de ATLAS

Scripts que generan los datos de las tablas del artículo en las dos
configuraciones de hardware. No modifican `backend/app`: instrumentan el código
real en tiempo de ejecución y funcionan en las dos ramas del despliegue:

| Config. | Rama | Perfil | Dónde corre cada cosa |
|---|---|---|---|
| **A** | `experimental` | `local_balanced` | Todo en el Mac M1 (8 GB): backend, Chroma/FTS5, embeddings (MPS), reranker y Ollama |
| **B** | `frontend/tailwind-v3-no-avx2-fallback` | `lan_client` | **Cliente ATLAS** sin GPU: backend, base de datos, reranker y extracción. **Servidor Windows** (Xeon Gold 6258R, 400 GB, Tesla T4): Ollama con el LLM y los embeddings, por la LAN |

| Script | Produce | Dónde |
|---|---|---|
| `00_environment.py` | **Tabla 2**: hardware, versiones, modelo, cuantización, % en VRAM y RTT de red a Ollama | máquina con el backend |
| `00_server_info.ps1` | **Tabla 2**: CPU, RAM, GPU, CUDA y variables de Ollama del servidor Windows | servidor Windows (solo B) |
| `01a_ingest_corpus.py` | Ingesta del corpus en un usuario de evaluación aislado (`atlas-eval@eval.test`) + `corpus_files.csv` con SHA-256 | máquina con el backend |
| `01_corpus_stats.py` | **§3.10 Corpus**: documentos, páginas, fragmentos, % OCR, idioma y dominio | máquina con el backend |
| `02_relevance_judgments.py` | Juicios de relevancia (campo `gold`) por *pooling* | una sola máquina |
| `03_run_ablation.py` | Datos de las **Tablas 3, 4, 5, 6 y 7** (condiciones C1–C5) | máquina con el backend |
| `04_ragas_score.py` | *Faithfulness*, *factual correctness* y *context recall*: **Tabla 4 (columna RAGAS) y Tabla 8** | cualquiera con acceso al Ollama del servidor |
| `05_make_tables.py` | Todas las tablas en Markdown, `stats.md` (Wilcoxon + Holm) y `values.json` (los `[VALOR]`) | una sola máquina |
| `run_all.sh` | Ejecuta 00 + 01 + 03 | máquina con el backend |

Todos los `.py` se ejecutan **desde `backend/`**, con el entorno virtual del
backend activo y la misma `.env` que usa la aplicación. Los resultados quedan en
`evaluation/paper/results/<etiqueta>/`.

Etiquetas sugeridas: `A-mac`, `B-t4` y, opcionalmente, `B-t4-4b` (ver §2).

---

## 0. Preparación (una vez)

1. **Lleva esta carpeta a la rama de la PC.** Los scripts están en `experimental`.
   En el cliente ATLAS (rama `frontend/tailwind-v3-no-avx2-fallback`), copia
   `backend/evaluation/paper/` completa a la misma ruta. No hace falta cambiar
   de rama ni tocar `backend/app`.
2. **Revisa el reranker en el cliente (bug de esa rama).** `lan_client` usa
   mMARCO MiniLM (máximo 514 posiciones) con `RERANK_MAX_LENGTH=1024`. Esa rama
   no tiene el límite automático de `experimental`, así que un fragmento largo
   hace fallar la recuperación con `index 514 is out of bounds`. Hasta corregirlo
   en el código, añade a la `.env` del cliente:

   ```
   RERANK_MAX_LENGTH=506
   ```

   Declara el valor en la Tabla 2 (el script lo registra).
3. **Mismo corpus en las dos configuraciones.** Copia la carpeta del corpus (en el Mac:
   `~/Downloads/ATLAS_corpus_evaluacion/`) y ejecútala con
   `python evaluation/paper/01a_ingest_corpus.py --corpus <carpeta> --label <etiqueta>`; compara después los
   `corpus_files.csv` de ambas máquinas (mismo SHA-256). Luego usa `--user-email atlas-eval@eval.test` en los demás
   scripts. Alternativa manual: Sube los mismos PDF con la
   aplicación en ambas y espera a que estén en estado `ready`. Los nombres de
   archivo deben coincidir: el `gold` se identifica por nombre original + página.
4. **Escribe el conjunto de preguntas** en `evaluation/paper/questions.jsonl`
   (parte de `questions.template.jsonl`). Una línea JSON por pregunta:

   ```json
   {"id": "q001", "question": "...", "reference": "...", "tags": ["lexical"], "gold": [{"filename": "Articulo.pdf", "page": 4, "grade": 2}]}
   ```

   - `reference`: respuesta de referencia (la usa RAGAS).
   - `gold`: evidencia relevante. `page` es opcional (sin página = cualquier parte
     del archivo). `grade`: 1 = parcial, 2 = responde (para nDCG). `group` es
     opcional: las unidades con el mismo `group` son intercambiables al medir
     completitud (por defecto, un grupo por archivo).
   - `tags`: libres. Cada etiqueta genera una Tabla 3b, por ejemplo `lexical` para
     las preguntas con terminología técnica literal que menciona §4.1.
   - El `backend/evaluation/ragas_sample_questions.jsonl` existente pregunta
     sobre el propio repositorio, no sobre tu corpus: no sirve para el artículo.
5. **Si no quieres escribir el `gold` a mano**, usa pooling (en una sola máquina):

   ```bash
   python evaluation/paper/02_relevance_judgments.py pool --user-email TU_EMAIL
   ```

   Abre `evaluation/paper/judgments_pool.csv`, rellena la columna `relevance`
   (0/1/2) y luego:

   ```bash
   python evaluation/paper/02_relevance_judgments.py apply --merge
   ```

   Esto escribe el `gold` dentro de `questions.jsonl` y deja una copia en
   `questions.sin_gold.jsonl`. Copia ese `questions.jsonl` a la otra máquina.
6. **Opcional:** `corpus_manifest.csv` (a partir de `corpus_manifest.template.csv`)
   con el dominio y el idioma de cada archivo, para §3.10.
7. **Rondas correctivas:** con `RESEARCH_MAX_ROUNDS=2` (igual en ambas ramas) hay
   como máximo **una** ronda correctiva. No lo cambies; la Tabla 6 muestra las
   filas de 0 y 1 rondas.
8. **Mantén los modelos cargados durante las mediciones.** En B, `lan_client` ya
   pide `keep_alive=30m` en cada llamada. En el Mac, define `OLLAMA_KEEP_ALIVE=-1`
   antes de `ollama serve`.

---

## 1. Mac (Configuración A)

```bash
cd ~/Documents/GitHub/PDF-Assistant-RAG/backend
source ../.venv/bin/activate
ollama serve        # en otra terminal, si no está corriendo
```

Haz primero una prueba corta (2 preguntas, una repetición) para ver que todo
funciona y medir cuánto tarda cada ejecución; el script imprime un ETA:

```bash
python evaluation/paper/03_run_ablation.py --label A-mac --user-email TU_EMAIL --limit 2
```

Si va bien, borra esa prueba y lanza la corrida completa:

```bash
rm -rf evaluation/paper/results/A-mac
bash evaluation/paper/run_all.sh A-mac TU_EMAIL 3
```

Con 8 GB, cierra el resto de aplicaciones mientras corre. Si el tiempo total es
inviable, usa `--repeats 1` en el Mac o limita las condiciones (`--conditions C4,C5`
basta para la Tabla 6), y declara en el artículo qué se midió dónde.

## 2. PC (Configuración B, perfil `lan_client`)

**En el servidor Windows** (el equipo con la T4), en PowerShell, desde la carpeta
donde copiaste `00_server_info.ps1`:

```powershell
powershell -ExecutionPolicy Bypass -File 00_server_info.ps1 -Output server_info.json
```

Copia `server_info.json` al cliente, en `backend/evaluation/paper/`.

**En el cliente ATLAS**, desde `backend/` y con el entorno activo:

```bash
python evaluation/paper/00_environment.py --label B-t4 --server-info evaluation/paper/server_info.json
python evaluation/paper/01_corpus_stats.py --label B-t4 --user-email TU_EMAIL
python evaluation/paper/03_run_ablation.py --label B-t4 --user-email TU_EMAIL --limit 2
```

Si la prueba va bien, borra `results/B-t4/runs.jsonl` y lanza la corrida completa:

```bash
python evaluation/paper/03_run_ablation.py --label B-t4 --user-email TU_EMAIL --repeats 3
```

`00_environment.py` usa `OLLAMA_BASE_URL` de tu `.env`, mide la latencia de red
cliente → Ollama (RTT, que se suma a cada llamada al LLM y a cada embedding) y
registra el modelo de embeddings que sirve Ollama.

**Experimento recomendado para §4.5 (aislar el tamaño del modelo).** A y B no
difieren solo en el LLM: también cambian el almacén, los TOP_K, el número de
preguntas de evidencia, el motor de embeddings (sentence-transformers frente a
Ollama) y la red. Para comparar 4B con 14B sin confusión, repite B cambiando solo
el modelo; las variables de entorno tienen prioridad sobre el perfil:

```bash
LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M python evaluation/paper/00_environment.py --label B-t4-4b --server-info evaluation/paper/server_info.json
LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M python evaluation/paper/03_run_ablation.py --label B-t4-4b --user-email TU_EMAIL --repeats 3
```

Antes, descarga el modelo en el servidor (`ollama pull qwen3:4b-instruct-2507-q4_K_M`).

## 3. RAGAS (mismo juez para todas las configuraciones)

En el cliente de B, que ya llega al Ollama del servidor, copia
`results/A-mac/` del Mac a `backend/evaluation/paper/results/A-mac/` y ejecuta:

```bash
pip install "ragas>=0.3,<0.4"
python evaluation/paper/04_ragas_score.py --label B-t4  --judge-model qwen3:14b-q4_K_M
python evaluation/paper/04_ragas_score.py --label A-mac --judge-model qwen3:14b-q4_K_M
python evaluation/paper/04_ragas_score.py --label B-t4-4b --judge-model qwen3:14b-q4_K_M   # si lo corriste
```

El juez corre en el Ollama del servidor (dentro de tu LAN; nada sale a terceros).
Si la VRAM no alcanza, baja `--judge-ctx` (por defecto 16384) y
`--max-context-chars`. Declara el juez en el artículo: RAGAS mide concordancia
con ese juez, no verdad factual.

## 4. Tablas

Con todas las carpetas de `results/` en una máquina:

```bash
python evaluation/paper/05_make_tables.py --config-a A-mac --config-b B-t4
```

Para medir además el respaldo con el clasificador NLI (opcional):

```bash
python evaluation/paper/05_make_tables.py --config-a A-mac --config-b B-t4 --nli
```

Abre `evaluation/paper/results/tables/tables.md`. Además se generan:

- `stats.md`: Wilcoxon pareado por pregunta, corrección de Holm y tamaño del efecto r.
- `values.json`: cifras para los `[VALOR]` del resumen y las conclusiones.
- `per_run_metrics.csv` y `per_question_metrics.csv`: datos crudos.

---

## Cómo se mide cada cosa

**Condiciones.** C1–C4 pasan por `_execute_document_route` de `simple_rag`; se
desactiva un componente a la vez intercambiando las funciones que el propio
código consulta:

- C1: canal léxico y reranker apagados.
- C2: reranker apagado.
- C3: sin contexto de grafo.
- C4: `simple_rag` tal cual, que sí inyecta el grafo.
- C5: `run_research_agent` con las dependencias de producción; **no usa el
  grafo** en ninguna de las dos ramas.

Todas comparten los mismos topes de evidencia (`TOP_K_RERANK`, máximo 3
fragmentos por documento). En `experimental` se desactiva además el escalado al
agente de herramientas (MCP), que la otra rama no tiene.

**Recall@20 y nDCG@10** (Tabla 3): funciones de `app/rag/benchmark.py` aplicadas
a «unidades de evidencia» (archivo + página del `gold`; cada unidad se acredita
una sola vez). C1–C4 se puntúan sobre rankings de profundidad fija (densa, RRF,
reordenada) construidos con las mismas funciones y tamaños de pool que
`retrieve()`, antes del corte del grupo líder. C5 se puntúa sobre su evidencia
final. También se reporta el recall de lo que efectivamente recibió el generador.

**Precisión de citas:** marcadores `[D#]` de la respuesta final que resuelven a
una fuente recuperada. La misma medida sobre el **primer borrador** (antes de la
verificación) es la que permite afirmar cuánto aporta la verificación.

**Respaldo de afirmaciones:** frases sustantivas (definición de
`_is_substantive_claim`) con cita inmediata y sin cifras ausentes de la evidencia
citada (`_claim_support_issue`; con `--nli`, además implicación). Usa el mismo
criterio que el sistema, así que C5 está favorecida por construcción: repórtalo
junto a *faithfulness*, que usa un juez externo al sistema.

**Completitud:** grupos de `gold` citados / grupos requeridos.

**Tabla 5:** cada llamada a `_validate_answer_citations`, `_answer_evidence_issues`,
`_prune_unsupported_claims`, `_repair_research_answer` y al verificador queda
registrada con la etapa del grafo en que ocurrió.

**Latencia** (Tabla 6): reloj de pared alrededor de la llamada completa, tras una
ejecución de calentamiento que no se registra. El orden de preguntas y
condiciones se aleatoriza en cada repetición (`--seed`) para no favorecer a
ninguna con la caché de prefijos de Ollama. La caché de respuestas de la app no
interviene, porque se llama al pipeline directamente. La temperatura es 0
(fijada en `get_llm_client`), así que las repeticiones miden no determinismo de
ejecución, no muestreo. En B la latencia incluye la red de la LAN.

**Tabla 7:** cada nodo del grafo LangGraph se cronometra al registrarse
(`StateGraph.add_node`). Funciona igual en ambas ramas y no requiere Langfuse,
que la rama de la PC no tiene.

**Reanudar:** `03` y `04` saltan lo ya registrado. Si cambias la configuración
efectiva, usa otra `--label` (el script avisa si detecta el cambio).

**Privacidad:** `results/` y `server_info.json` contienen fragmentos de tus
documentos, respuestas e IPs de la LAN. Están en `.gitignore`; revísalos antes de
publicarlos como «resultados crudos».

---

## Hallazgos del código que afectan al texto del artículo

1. **Dos ramas, mismo núcleo.** Retrieval, reranking, validación de citas,
   comprobación numérica, poda, auditoría, juez de afirmaciones, reparación y el
   grafo de investigación son funcionalmente idénticos en ambas ramas. Difieren
   en la configuración, en las rutas de herramientas (MCP, `code_review_agent` y
   DOI/Crossref solo en `experimental`) y en la observabilidad (Langfuse solo en
   `experimental`).
2. **Tabla 2:**
   - A usa `Qwen/Qwen3-Embedding-0.6B` (MPS), no E5.
   - B (`lan_client`) usa mMARCO MiniLM en el cliente como reranker, extracción
     `fast`, y embeddings `qwen3-embedding:0.6b` servidos por Ollama.
3. **`research_rag` no usa el grafo:** solo `simple_rag` y `scoped_rag` lo inyectan.
4. **Una sola ronda correctiva** con `RESEARCH_MAX_ROUNDS=2`.
5. **Reparación fallida:** tras la reparación se vuelve a verificar y, si siguen
   los problemas, se entrega la respuesta igualmente. La poda ocurre en la
   síntesis (dos intentos fallidos) o si la reparación inventa citas.
6. **Juez LLM en la verificación:** `_semantic_claim_issues` pide al modelo
   afirmaciones no respaldadas, inferencias no marcadas y citas mal asociadas.
7. **Temperatura 0** fijada en `get_llm_client`; `LLM_TEMPERATURE` solo la usa el
   agente de herramientas.
8. **`run_ragas_eval.py` usa la API de Hugging Face** (en ambas ramas). La
   Tabla 8 se genera aquí con juez local y compara C3 frente a C4.
9. **Bug del reranker en la rama de la PC:** ver §0, punto 2.
