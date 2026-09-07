# Matemática de recuperación e indexación

Este documento complementa [`ARCHITECTURE.md`](ARCHITECTURE.md) con el detalle algorítmico exacto de fragmentación, embeddings, fusión híbrida, reranking, verificación y construcción del grafo de conocimiento. Cada sección cita el archivo y, cuando aporta valor, la línea donde vive la constante o la fórmula. Los números de línea corresponden al estado del código en el momento de escribir este documento; ante una discrepancia, el código es la fuente de verdad.

## 1. Fragmentación jerárquica

`backend/app/rag/chunker.py`

### 1.1 Conteo de tokens

No se usa un tokenizador de subpalabras. `_token_length` estima tokens con una expresión regular Unicode:

```python
len(re.findall(r"\w+|[^\w\s]", text))
```

Cada palabra y cada símbolo de puntuación cuenta como un token. Esta métrica alimenta tanto al `RecursiveCharacterTextSplitter` (como `length_function`) como al agrupador de padres.

### 1.2 División recursiva

`RecursiveCharacterTextSplitter` de LangChain, configurado con:

- `chunk_size = CHUNK_SIZE` (420 por defecto),
- `chunk_overlap = CHUNK_OVERLAP` (80 por defecto),
- `separators = ["\n\n", "\n", ". ", " ", ""]` en orden de preferencia,
- `length_function = _token_length`.

El divisor intenta cortar en el separador más "natural" disponible (párrafo → línea → oración → palabra → carácter) sin exceder `chunk_size` tokens estimados, y repite `chunk_overlap` tokens del final de un fragmento al inicio del siguiente para no perder contexto en el límite.

### 1.3 Agrupación en padres y detección de encabezados

`_annotate_hierarchy` recorre los chunks hijos en orden y:

1. Detecta un posible encabezado con `_heading_candidate`: la primera línea no vacía califica como encabezado si mide ≤120 caracteres, ≤14 palabras, y además es numerada (`^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+`), está en mayúsculas, o sigue *Title Case* sin terminar en puntuación.
2. Si el encabezado detectado difiere de la sección actual, cierra el padre acumulado (`flush_parent`) y abre una nueva sección.
3. Acumula chunks hijos en un padre mientras `current_parent_tokens + chunk_tokens <= PARENT_CHUNK_SIZE` (1600 por defecto); al superarlo, cierra el padre y empieza uno nuevo.
4. Al cerrar un padre, concatena el texto de sus hijos con `\n\n`, calcula `page_start`/`page_end` como el mínimo y máximo de página entre sus hijos, y asigna `parent_id`/`parent_text` a cada hijo.

El hijo es la unidad de recuperación (se embebe y se cita); el padre solo aporta contexto adicional al redactor final.

### 1.4 Tablas e imágenes

Las tablas se extraen como chunks independientes (`chunk_type="table"`) con su `bbox` normalizado a fracción de página `[x0/W, y0/H, x1/W, y1/H]`, sin pasar por el splitter. Las imágenes generan una leyenda vía `app.rag.vision.caption_image` y se insertan como un chunk de texto con `is_image=True`.

## 2. Embeddings y similitud vectorial

`backend/app/rag/embeddings.py`, `backend/app/rag/vectorstore.py`

### 2.1 Normalización y prefijos

`HuggingFaceEmbeddings` se construye con `normalize_embeddings=True`: cada vector se divide por su norma L2, de modo que `‖v‖ = 1`. Con vectores normalizados, el producto punto y la similitud coseno coinciden:

```
cos(a, b) = (a · b) / (‖a‖ ‖b‖) = a · b   cuando ‖a‖ = ‖b‖ = 1
```

Los modelos E5 y Qwen3-Embedding se entrenaron con prefijos asimétricos entre consulta y documento:

- E5 (`_is_e5_model`, detecta `"e5"` en `EMBEDDING_MODEL`): documentos → `"passage: {texto}"`, consultas → `"query: {texto}"`.
- Qwen3-Embedding: consultas → una instrucción fija (`"Instruct: Retrieve passages that provide direct evidence for the research question.\nQuery: {texto}"`); los documentos no llevan prefijo.

Omitir estos prefijos degrada la calidad de recuperación aunque el modelo cargue sin errores, porque el espacio de embeddings fue entrenado con esa asimetría.

### 2.2 Distancia coseno en pgvector

Con `CORPUS_STORE_BACKEND=postgres`, la consulta densa usa el operador `<=>` de pgvector (distancia coseno) directamente en SQL:

```sql
SELECT ..., 1 - (embedding <=> :embedding) AS score
FROM document_chunks
ORDER BY embedding <=> :embedding
LIMIT :top_k
```

`embedding <=> query` devuelve `1 - cos(embedding, query)` (distancia, no similitud), por lo que `1 - distancia` reconstruye la similitud coseno en `[-1, 1]`, típicamente `[0, 1]` para embeddings normalizados de texto. La ordenación `ORDER BY ... <=>` permite usar el índice HNSW sin post-procesar toda la tabla.

### 2.3 ChromaDB (perfil `local`/`local_balanced`)

La colección se crea con metadato `"hnsw:space": "cosine"`. Chroma devuelve una `distance`; el código aplica la misma transformación `similarity = 1 - distance`.

### 2.4 Shortlist de perfiles documentales

Antes de recuperar chunks, `query_document_profiles` reduce el corpus a los documentos semánticamente relevantes comparando el embedding de la pregunta contra el embedding del *perfil* de cada documento (resumen + metodología + hallazgos). En PostgreSQL usa el mismo operador `<=>`; en SQLite, `vectorstore.py` calcula el coseno a mano:

```python
dot = sum(a*b for a, b in zip(vector, query_embedding))
norm_a = sqrt(sum(a**2 for a in vector))
norm_b = sqrt(sum(b**2 for b in query_embedding))
cos = dot / (norm_a * norm_b)
```

## 3. Recuperación léxica

`backend/app/rag/vectorstore.py`, `backend/app/rag/bm25.py`

Existen **tres** implementaciones léxicas según el backend activo:

| Backend | Motor | Fórmula de ranking |
| --- | --- | --- |
| PostgreSQL | `tsvector`/`tsquery` | `ts_rank_cd(to_tsvector('simple', search_text), websearch_to_tsquery('simple', query))` — ranking por densidad de cobertura (*cover density*) de Postgres, con `to_tsvector('simple', ...)` para no aplicar stemming dependiente de idioma |
| SQLite (backend relacional) | FTS5 | `-bm25(document_chunks_fts)` — SQLite implementa BM25 internamente; el signo se invierte porque `bm25()` devuelve puntuaciones donde *menor es mejor* |
| SQLite (fallback legado) | `rank_bm25.BM25Okapi` | BM25 clásico de Okapi sobre un corpus tokenizado con `re.findall(r'\w+', text.lower())` |

BM25 Okapi, para completar la referencia matemática (fórmula estándar que implementa `rank_bm25`):

```
score(D, Q) = Σ_{t∈Q} IDF(t) · (f(t,D)·(k1+1)) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))
```

donde `f(t,D)` es la frecuencia del término en el documento, `|D|` su longitud, `avgdl` la longitud promedio del corpus, y `k1`, `b` las constantes de suavizado por defecto de `rank_bm25` (`k1=1.5`, `b=0.75`).

`query_lexical_chunks` es la ruta usada por la recuperación híbrida en producción (Postgres/SQLite-FTS); `bm25.py` sólo actúa como índice legado por documento, reconstruible desde los chunks guardados.

## 4. Fusión híbrida: Reciprocal Rank Fusion

`backend/app/rag/retriever.py:228-243`

Las puntuaciones densas (similitud coseno) y léxicas (`ts_rank_cd`/BM25) no son comparables entre sí — viven en escalas distintas. En vez de normalizarlas, el sistema fusiona **por posición de ranking**:

```
RRF(d) = Σ_i  w_i / (k + rank_i(d))
```

- `rank_i(d)` es la posición de `d` (1-indexada) dentro del ranking `i` (denso o léxico); un documento ausente de un ranking simplemente no aporta ese término.
- `k = RRF_K = 60` (constante estándar de RRF, suaviza el efecto de las primeras posiciones).
- `w = [0.6, 0.4]` — el ranking denso pesa 1.5× más que el léxico (`retrieve()`, línea 410).

La fusión se ejecuta **una vez por faceta** (pregunta atómica), sobre hasta `candidate_top_k` candidatos por canal (≥50 cuando hay facetas, si no `max(top_k, TOP_K_RETRIEVAL)`).

## 5. Reranking

`backend/app/rag/reranker.py`

### 5.1 Logits crudos, no probabilidades

El cross-encoder (`RERANKER_MODEL`, por defecto `Qwen/Qwen3-Reranker-0.6B` o `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) se invoca con `activation_fn=nn.Identity()` para obtener el logit crudo de cada par `(query, passage)`, evitando que `CrossEncoder` aplique su propia sigmoide interna (lo que calibraría el puntaje dos veces). Los candidatos se ordenan por logit descendente **dentro de una misma consulta**; el logit de una consulta no es comparable con el de otra, así que nunca se aplica un umbral global de probabilidad.

### 5.2 De logit a `relevance_score`

Después de ordenar, cada documento recibe:

```
rerank_rank  = posición 1-indexada tras ordenar por logit
relevance_score = 1 / rerank_rank
```

Es decir, la puntuación que alimenta la selección final **no es el logit** sino el recíproco del rango post-rerank (una segunda fusión por posición, análoga a RRF con `k=0`). Esto hace que la escala de `relevance_score` sea comparable entre facetas distintas (siempre en `(0, 1]`), a costa de perder la magnitud relativa que sí tenía el logit.

### 5.3 Selección del clúster líder (`_relative_rerank_selection`)

`backend/app/rag/retriever.py:246-261`

En vez de un umbral fijo de relevancia, el sistema detecta dónde "cae" la calidad dentro del top-10 (`MAX_EVIDENCE_PER_FACET = 10`) reranked, usando los **logits crudos** (antes de convertirlos a `relevance_score`):

```python
gaps = [scores[i] - scores[i+1] for i in range(len(scores) - 1)]
median_gap = median(|gap| for gap in gaps)
significant = max(0.25, median_gap * 3.0)
# corta en el primer índice ≥ 3 cuyo gap ≥ significant
```

Requiere al menos 4 candidatos con `rerank_score`; si no, devuelve la lista completa sin recortar. El umbral `significant` se adapta a la dispersión típica de esa consulta (3× la mediana de los saltos) con un piso absoluto de `0.25` para evitar cortes espurios cuando todos los gaps son minúsculos.

## 6. Selección final: cobertura de facetas

`backend/app/rag/retriever.py:275-326` (`_select_facet_evidence`)

Una vez fusionada la evidencia de todas las facetas, la selección final es una heurística tipo *cobertura de conjuntos* (greedy set cover), no un simple top-k por puntaje:

1. Ordena todos los candidatos por `relevance_score` descendente.
2. Mientras existan facetas sin cubrir y queden cupos (`top_k`): elige, entre los candidatos que cubren al menos una faceta pendiente y no superan `MAX_CHUNKS_PER_DOCUMENT = 3` chunks de su documento, el que (a) provenga de un documento aún no representado, (b) cubra más facetas nuevas, (c) tenga mayor `relevance_score` — en ese orden de prioridad (tupla de comparación).
3. Marca sus facetas como cubiertas y repite.
4. Si sobran cupos tras cubrir todas las facetas, rellena por `relevance_score` puro, respetando el límite de 3 chunks por documento.

No existe una cuota mínima artificial de documentos: si una sola fuente cubre todas las facetas mejor que cualquier alternativa, el sistema no fuerza diversidad adicional.

## 7. Verificación de afirmaciones

`backend/app/rag/agent.py`

### 7.1 Chequeo numérico (siempre activo)

Antes de cualquier verificación semántica, cada afirmación sustantiva se somete a un chequeo determinista: se extraen los "tokens numéricos" (números con unidad/porcentaje opcional) de la afirmación y de la evidencia citada con la misma expresión regular, y se calcula la diferencia de conjuntos:

```
unsupported = numbers(afirmación) − numbers(evidencia_citada)
```

Si `unsupported` no está vacío, la afirmación se marca como no soportada, sin necesidad de invocar ningún modelo.

### 7.2 Verificación NLI (implementada, actualmente inactiva)

Modelo `NLI_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"`, clasificación de entailment de 3 clases (`entailment`, `neutral`, `contradiction`) vía `softmax` sobre los logits. Una afirmación se rechaza si:

```
entailment < NLI_ENTAILMENT_THRESHOLD (0.65)
   O
entailment ≤ max(neutral, contradiction)
```

La segunda condición es más estricta que un simple umbral: exige que "entailment" sea la clase con mayor probabilidad, no solo que supere 0.65. **Nota de implementación**: el parámetro `verify_entailment` que activa esta ruta está fijado a `False` en todos los puntos de llamada actuales — el modelo NLI está cargado y probado, pero no participa en el flujo de verificación en producción; solo el chequeo numérico y el de citas corren siempre. Activarlo requiere pasar `verify_entailment=True` explícitamente en las llamadas de síntesis.

### 7.3 Validación de citas

Toda respuesta debe contener al menos un marcador `[D#]`/`[W#]` (regex `\[((?:D|W)\d+)\]`) que corresponda a un `source_id` real; una cita a un id inexistente, o la ausencia total de citas, hace que la respuesta se rechace y se reintente o se recorte (`_prune_unsupported_claims`), nunca que se devuelva sin citas verificables.

### 7.4 Reparación

Hasta un ciclo de reparación por respuesta: se reenvían al LLM los problemas detectados (`issues[:8]`) junto con la respuesta previa (truncada) pidiendo una corrección localizada; si la respuesta reparada aún falla la validación, se recurre a `_prune_unsupported_claims`, que elimina oración por oración cualquier afirmación sustantiva sin cita válida o con `unsupported` numérico, en vez de descartar la respuesta completa.

## 8. Grafo de conocimiento (GraphRAG)

`backend/app/rag/graph_builder.py`, `backend/app/rag/graph_retriever.py`

### 8.1 Extracción de entidades

NER con spaCy (`en_core_web_sm`). Una entidad se conserva solo si su etiqueta está en `GRAPH_ENTITY_LABELS` (`PERSON, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, NORP, FAC`). El identificador de nodo es `f"{label}:{texto_normalizado_en_minusculas}"`, así que "OpenAI" y "openai" colapsan al mismo nodo pero "OpenAI" (ORG) y "OpenAI" (PRODUCT) no.

### 8.2 Relaciones por co-ocurrencia

No hay parseo de dependencias ni ventana de oración: **todas las entidades que aparecen en el mismo chunk quedan conectadas entre sí** (todas las combinaciones por pares, `O(k²)` por chunk). El peso de una arista es el número de chunks en los que ambas entidades co-ocurrieron:

```
weight(A, B) = |{ chunk : A ∈ chunk ∧ B ∈ chunk }|
```

Cada nodo acumula además `mentions` (conteo total), y tanto nodos como aristas acumulan el conjunto de páginas y chunks donde aparecieron.

### 8.3 Persistencia y lectura

El grafo se serializa con `networkx.node_link_data` a `data/graphs/<user>_<document>.json`. Al consultar (`get_entity_context`), el sistema:

1. Ubica los nodos que coinciden con la consulta (NER sobre la pregunta; si no hay coincidencia, sub-cadena literal contra el nombre normalizado del nodo).
2. Recorre los vecinos de esos nodos ordenados por peso de arista descendente.
3. Fusiona relaciones equivalentes entre grafos (clave `(min(a,b), max(a,b))` en minúsculas, orden-independiente).
4. **Aquí, no en la construcción, se aplica `GRAPH_MAX_RELATIONSHIPS` (12 por defecto)**: solo las 12 relaciones de mayor peso se formatean como contexto adicional para el LLM. El grafo persistido en disco no tiene ese límite; el límite gobierna cuánto contexto de grafo se inyecta por consulta, no cuánto se almacena.

Cualquier fallo durante la carga o el emparejamiento se registra como advertencia y devuelve una cadena vacía — el grafo nunca puede tumbar una consulta.

## 9. Memoria documental y evidencia verificable

`backend/app/rag/summarizer.py`

El perfil de un documento (resumen, metodología, hallazgos, limitaciones) se redacta con un LLM sobre hasta 18 000 caracteres de chunks concatenados, con salida estructurada (`with_structured_output(..., method="json_mode")`) limitada a 24 elementos de evidencia. Una evidencia solo se conserva si:

```
evidence.exact_quote in chunk[evidence.chunk_index].text
```

Es una comprobación de subcadena literal (sensible a mayúsculas/minúsculas y espacios), no una comparación difusa. Cualquier cita parafraseada, inventada, o apuntando al chunk equivocado se descarta en silencio antes de persistirse; la memoria documental nunca puede introducir una cita no verificable en el sistema.
