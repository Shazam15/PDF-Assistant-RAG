# Guía: configuraciones B y B′ (cliente Zorin OS + servidor Windows)

- **B** — perfil `lan_client`: ATLAS en la PC con Zorin OS (cliente, sin GPU) y Ollama en la PC Windows
  (Xeon Gold 6258R, 400 GB, Tesla T4). Generador `qwen3:14b-q4_K_M`.
- **B′** — idéntica a B, salvo el generador: `qwen3:4b-instruct-2507-q4_K_M` (el mismo modelo que A). Aísla el
  efecto del tamaño del modelo. Usa el **mismo índice** que B: no se vuelve a ingestar.

Etiquetas: `B-t4` y `B-t4-4b`. Todo se ejecuta **de una en una** (B y B′ comparten la GPU).

> Regla de oro: **Ollama 0.33.0 en Windows durante toda la medición.** Con 0.34.4 `research_rag` descarta toda
> la evidencia y se abstiene siempre (comprobado en el Mac). Los comandos de abajo usan `--expect-ollama 0.33.0`:
> si Ollama cambia de versión, la corrida se detiene sola y lo ya registrado sigue siendo válido.

---

## 1. PC Windows (servidor de inferencia)

1. **Instalar exactamente Ollama 0.33.0.** Cierra Ollama (icono de la bandeja → Quit), desinstálalo desde
   *Configuración → Aplicaciones*, descarga `OllamaSetup.exe` de
   <https://github.com/ollama/ollama/releases/tag/v0.33.0> e instálalo. Comprueba en PowerShell:

   ```powershell
   ollama --version
   ```

   Si la app ofrece *Restart to update*, **no lo aceptes** hasta terminar la evaluación.

2. **Variables de entorno** (*Editar las variables de entorno de esta cuenta*), luego cierra y vuelve a abrir Ollama:

   | Variable | Valor | Motivo |
   |---|---|---|
   | `OLLAMA_HOST` | `0.0.0.0` | Escuchar en la red local, no solo en localhost |
   | `OLLAMA_KEEP_ALIVE` | `-1` | Modelo residente: la latencia no incluye cargas |
   | `OLLAMA_NUM_PARALLEL` | `1` | Una petición a la vez, latencias comparables |
   | `OLLAMA_MAX_LOADED_MODELS` | `2` | Generador + modelo de embeddings a la vez |

3. **Firewall** (PowerShell como administrador):

   ```powershell
   New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow -Profile Private
   ```

4. **Modelos**:

   ```powershell
   ollama pull qwen3:14b-q4_K_M
   ollama pull qwen3:4b-instruct-2507-q4_K_M
   ollama pull qwen3-embedding:0.6b
   ollama pull gemma3:12b
   ```

   `gemma3:12b` es el juez de RAGAS (§5): de otra familia que los generadores, para no autoevaluarse.

5. **Energía**: *Configuración → Sistema → Inicio/apagado* → suspensión **Nunca** mientras esté enchufada.

6. **Datos del servidor para la Tabla 2** (copia `00_server_info.ps1` de la carpeta del repositorio):

   ```powershell
   powershell -ExecutionPolicy Bypass -File 00_server_info.ps1 -Output server_info.json
   ```

   Anota la IP de la red local con `ipconfig` (p. ej. `192.168.1.10`).

---

## 2. Llevar los archivos a la PC con Zorin

Desde el **Mac** (cuando la corrida A haya terminado; si no, vuelve a copiar `results/A-mac` al final).
Sustituye `USUARIO@IP_ZORIN` y `RUTA_REPO` (carpeta del repositorio en Zorin):

```bash
rsync -av --exclude '__pycache__' --exclude '*.bak' ~/Documents/GitHub/PDF-Assistant-RAG/backend/evaluation/paper/ USUARIO@IP_ZORIN:RUTA_REPO/backend/evaluation/paper/
```

```bash
rsync -av ~/Downloads/ATLAS_corpus_evaluacion/ USUARIO@IP_ZORIN:~/ATLAS_corpus_evaluacion/
```

Copia también `server_info.json` (de Windows) a `RUTA_REPO/backend/evaluation/paper/` en Zorin. Sin red entre
equipos, usa una memoria USB con las mismas carpetas.

---

## 3. PC con Zorin OS (cliente ATLAS)

1. **Rama y dependencias** (desde la raíz del repositorio, con el entorno virtual del backend activo):

   ```bash
   git checkout frontend/tailwind-v3-no-avx2-fallback
   pip install scipy psutil "ragas>=0.3,<0.4"
   ```

2. **`backend/.env`** — además de tu configuración de base de datos y secretos:

   ```
   MODEL_PROFILE=lan_client
   OLLAMA_BASE_URL=http://IP_WINDOWS:11434
   CPU_THREADS=NUCLEOS_FISICOS_DE_ZORIN
   RERANK_MAX_LENGTH=506
   ```

   `RERANK_MAX_LENGTH=506` es obligatorio en esta rama: sin él, el reranker mMARCO falla con
   `index 514 is out of bounds` en fragmentos largos. No fijes `LLM_MODEL`: el perfil ya usa el de 14B.

3. **OCR**: el Mac indexó con Tesseract solo en inglés. Comprueba los idiomas instalados:

   ```bash
   tesseract --list-langs
   ```

   Si aparece `spa`, el OCR del artículo escaneado (Montero) será mejor que en A: decláralo en el artículo.

4. **Conexión con el servidor** (debe responder `0.33.0`):

   ```bash
   curl -s http://IP_WINDOWS:11434/api/version
   ```

Todos los comandos siguientes se ejecutan desde `RUTA_REPO/backend`.

5. **Ingesta del corpus** en el usuario aislado `atlas-eval@eval.test` (se crea solo):

   ```bash
   python evaluation/paper/01a_ingest_corpus.py --corpus ~/ATLAS_corpus_evaluacion --label B-t4
   ```

   Comprueba que es exactamente el mismo corpus que en A (no debe imprimir nada):

   ```bash
   diff <(cut -d, -f1,2 evaluation/paper/results/A-mac/corpus_files.csv) <(cut -d, -f1,2 evaluation/paper/results/B-t4/corpus_files.csv)
   ```

6. **Tabla 2 y corpus**:

   ```bash
   python evaluation/paper/00_environment.py --label B-t4 --server-info evaluation/paper/server_info.json
   ```

   ```bash
   python evaluation/paper/01_corpus_stats.py --label B-t4 --user-email atlas-eval@eval.test --manifest evaluation/paper/corpus_manifest.csv
   ```

7. **Prueba corta** (2 preguntas × 5 condiciones; imprime un ETA):

   ```bash
   python evaluation/paper/03_run_ablation.py --label B-t4-prueba --user-email atlas-eval@eval.test --limit 2 --expect-ollama 0.33.0
   ```

   Comprueba que C5 no se abstiene (debe mostrar evidencia > 0 y `False`):

   ```bash
   python -c "import json; [print(r['condition'], len(r['evidence']), r['abstained']) for r in map(json.loads, open('evaluation/paper/results/B-t4-prueba/runs.jsonl'))]"
   ```

   Si todo está bien, bórrala: `rm -rf evaluation/paper/results/B-t4-prueba`.

8. **Corrida completa B** dentro de `tmux` (sobrevive si se cierra la terminal o la sesión SSH;
   `sudo apt install tmux` si no está). `systemd-inhibit` evita que Zorin se suspenda:

   ```bash
   tmux new -s atlas
   ```

   ```bash
   mkdir -p evaluation/paper/results/B-t4 && systemd-inhibit --what=idle:sleep --why="Evaluacion ATLAS" python -u evaluation/paper/03_run_ablation.py --label B-t4 --user-email atlas-eval@eval.test --repeats 1 --expect-ollama 0.33.0 >> evaluation/paper/results/B-t4/run.log 2>&1
   ```

   Sal de tmux sin detenerla con `Ctrl+B` y luego `D`; vuelve con `tmux attach -t atlas`. Progreso:

   ```bash
   tail -f evaluation/paper/results/B-t4/run.log
   ```

   Si se interrumpe, relanza el mismo comando: continúa donde quedó. Si hay tiempo, repite con `--repeats 3`
   para añadir las repeticiones 2 y 3 (A tiene 1; en el artículo se declara cuántas tiene cada configuración).

9. **B′** (cuando B haya terminado; mismo índice, solo cambia el generador):

   ```bash
   LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M python evaluation/paper/00_environment.py --label B-t4-4b --server-info evaluation/paper/server_info.json
   ```

   ```bash
   mkdir -p evaluation/paper/results/B-t4-4b && LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M systemd-inhibit --what=idle:sleep --why="Evaluacion ATLAS" python -u evaluation/paper/03_run_ablation.py --label B-t4-4b --user-email atlas-eval@eval.test --repeats 1 --expect-ollama 0.33.0 >> evaluation/paper/results/B-t4-4b/run.log 2>&1
   ```

   Declara en el artículo que la memoria documental del índice (perfiles por documento) la generó el modelo de 14B
   durante la ingesta, y que B′ la reutiliza.

---

## 4. RAGAS (Faithfulness de la Tabla 4 y Tabla 8)

Con las tres corridas terminadas y `results/A-mac` actualizado desde el Mac. Primero libera la VRAM del generador
(está residente por `OLLAMA_KEEP_ALIVE=-1`), en Windows:

```powershell
ollama stop qwen3:14b-q4_K_M
```

En Zorin, el mismo juez para las tres configuraciones:

```bash
python evaluation/paper/04_ragas_score.py --label A-mac --judge-model gemma3:12b
```

```bash
python evaluation/paper/04_ragas_score.py --label B-t4 --judge-model gemma3:12b
```

```bash
python evaluation/paper/04_ragas_score.py --label B-t4-4b --judge-model gemma3:12b
```

Si la VRAM no alcanza, añade `--judge-ctx 8192`.

## 5. Tablas finales

```bash
python evaluation/paper/05_make_tables.py --config-a A-mac --config-b B-t4
```

Resultado en `evaluation/paper/results/tables/` (`tables.md`, `stats.md`, `values.json`). La tabla de apoyo de
§4.5 incluye `B-t4-4b`: la comparación B′ → B es la que sostiene «verificación frente a tamaño del modelo».
