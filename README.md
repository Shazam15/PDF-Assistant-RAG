<div align="center">

<br/>

```
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄               ▄▄▄▄▄▄▄▄▄▄▄     ▄▄▄▄▄▄▄▄▄▄▄    
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌             ▐░░░░░░░░░░░▌   ▐░░░░░░░░░░░▌   
▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ ▐░▌             ▐░█▀▀▀▀▀▀▀█░▌   ▐░█▀▀▀▀▀▀▀▀▀    
▐░▌       ▐░▌     ▐░▌     ▐░▌             ▐░▌       ▐░▌   ▐░▌             
▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░▌             ▐░█▄▄▄▄▄▄▄█░▌   ▐░█▄▄▄▄▄▄▄▄▄    
▐░░░░░░░░░░░▌     ▐░▌     ▐░▌             ▐░░░░░░░░░░░▌   ▐░░░░░░░░░░░▌   
▐░█▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░▌             ▐░█▀▀▀▀▀▀▀█░▌    ▀▀▀▀▀▀▀▀▀█░▌   
▐░▌       ▐░▌     ▐░▌     ▐░▌             ▐░▌       ▐░▌             ▐░▌   
▐░▌       ▐░▌ ▄   ▐░▌ ▄   ▐░█▄▄▄▄▄▄▄▄▄  ▄ ▐░▌       ▐░▌ ▄  ▄▄▄▄▄▄▄▄▄█░▌ ▄ 
▐░▌       ▐░▌▐░▌  ▐░▌▐░▌  ▐░░░░░░░░░░░▌▐░▌▐░▌       ▐░▌▐░▌▐░░░░░░░░░░░▌▐░▌
 ▀         ▀  ▀    ▀  ▀    ▀▀▀▀▀▀▀▀▀▀▀  ▀  ▀         ▀  ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀ 
                                                                          
```

### AI Text and Language Assistant

<br/>
</div>

ATLAS es un asistente de investigación académica basado en recuperación aumentada por generación (RAG). Permite cargar documentos, procesarlos, conversar con un documento seleccionado o investigar sobre todo el corpus, y producir respuestas trazables con citas de página, sección, tabla o figura.

El sistema está diseñado para dos entornos:

- Desarrollo local ligero con SQLite, Chroma y modelos multilingües pequeños.
- Investigación en servidor con PostgreSQL 16, pgvector, CPU Intel y GPU NVIDIA.

La arquitectura detallada está documentada en [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) y el historial de versiones en [CHANGELOG.md](CHANGELOG.md).

## Capacidades

- Ingesta de PDF, DOCX, TXT, Markdown y archivos de código.
- Extracción estructurada con Docling y fallbacks mediante PyMuPDF, pdfplumber y OCR.
- Fragmentación jerárquica en hijos recuperables y contextos padre para síntesis.
- Resúmenes globales y por sección, perfiles documentales y registros de evidencia.
- Recuperación híbrida densa y léxica mediante Reciprocal Rank Fusion (RRF).
- Reranking semántico multilingüe y expansión de contexto.
- Investigación iterativa basada en vacíos de evidencia.
- Comparación multifuente, detección de contradicciones y declaración de incertidumbre.
- Citas inmediatas con documento, página, sección, tabla o figura cuando están disponibles.
- Historial persistente de conversaciones.
- Streaming mediante SSE y WebSocket.
- Cancelación de respuestas en curso.
- Selección de alcance por documento.
- Modos `Auto`, `Rápido` e `Investigación`.
- Autenticación JWT, aislamiento de datos por usuario y caché sensible al modo de consulta.

ATLAS no muestra cadena de pensamiento. Durante una investigación transmite estados verificables, como planificación, búsqueda de evidencia, revisión de vacíos y número de fuentes examinadas.

## Arquitectura

```mermaid
flowchart LR
    Browser["Next.js 16"] --> API["FastAPI"]
    API --> Router["Router adaptativo"]
    API --> Ingestion["Ingesta documental"]
    Ingestion --> Extractor["Docling / PyMuPDF / OCR"]
    Extractor --> Memory["Memoria documental jerárquica"]
    Memory --> Vector["Chroma o pgvector"]
    Memory --> Lexical["FTS5 o PostgreSQL tsvector"]
    Router --> Simple["RAG simple o acotado"]
    Router --> Research["Grafo de investigación"]
    Router --> Tools["Agente de herramientas"]
    Simple --> Retrieval["Dense + lexical + RRF + reranker"]
    Research --> Retrieval
    Retrieval --> Ollama["LLM mediante Ollama"]
    API --> Database["SQLite o PostgreSQL"]
    API --> Redis["Redis / Celery opcional"]
```

### Rutas de consulta

| Ruta | Uso |
|---|---|
| `scoped_rag` | Preguntas sobre un documento seleccionado; nunca ignora el alcance. |
| `simple_rag` | Resumen, extracción, explicación o redacción directa. |
| `research_rag` | Comparación o síntesis de múltiples documentos con rondas correctivas. |
| `tool_agent` | Web, cálculos, revisión de código o acciones que realmente requieren herramientas. |

El modo `Rápido` evita el agente y usa únicamente los documentos cargados. El modo `Investigación` fuerza recuperación profunda. `Auto` decide mediante alcance, estructura sustantiva de la tarea, herramientas necesarias y evidencia recuperada; los requisitos de estilo no activan por sí solos un agente.

## Stack

| Componente | Tecnología |
|---|---|
| Frontend | Next.js 16, React 19, TypeScript |
| Backend | Python 3.11, FastAPI, SQLAlchemy, Alembic |
| LLM | Ollama y modelos Qwen/Mistral configurables |
| Investigación | LangGraph |
| Extracción | Docling, PyMuPDF, pdfplumber, Tesseract |
| Desarrollo local | SQLite, FTS5, Chroma |
| Producción | PostgreSQL 16, pgvector, tsvector, pg_trgm |
| Procesamiento asíncrono | Redis y Celery |
| Embeddings locales | Multilingual E5 |
| Embeddings de investigación | Qwen3-Embedding-0.6B |
| Reranking local | mMARCO MiniLM multilingüe |
| Reranking de investigación | Qwen3-Reranker-0.6B |

## Estructura

```text
PDF-Assistant-RAG/
├── backend/
│   ├── app/
│   │   ├── routes/              # REST, SSE y WebSocket
│   │   ├── rag/                 # Router, recuperación y agente de investigación
│   │   ├── services/            # Ingesta, migración y mantenimiento
│   │   ├── config.py            # Configuración centralizada
│   │   ├── database.py          # SQLite/PostgreSQL
│   │   └── models.py            # Datos y memoria documental
│   ├── alembic/                 # Migraciones de esquema
│   ├── requirements.txt
│   └── tests/
├── frontend/
│   ├── src/app/
│   ├── src/components/
│   └── package.json
├── scripts/
│   └── init_postgres.sql        # Extensiones de PostgreSQL
├── docs/ARCHITECTURE.md
├── CHANGELOG.md
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── .env.example
```

## Requisitos

- Python 3.11. Python 3.9 no es compatible con varias dependencias actuales.
- Node.js 20 o superior.
- npm.
- Make.
- Ollama.
- Poppler, Tesseract y libmagic.
- Git.

Para `research_gpu` también se requiere:

- GPU NVIDIA y driver compatible.
- CUDA visible desde PyTorch y Ollama.
- Se recomiendan al menos 24 GB de VRAM para el modelo predeterminado.
- PostgreSQL 16 con pgvector para el despliegue de producción.

Para `wsl_t4`, Ollama usa la GPU desde Windows y el backend ejecuta embeddings,
reranking y NLI en CPU dentro de WSL2. No se requiere instalar CUDA ni un driver
NVIDIA Linux dentro de WSL.

## Instalación local

### 1. Dependencias del sistema

Ubuntu, Debian o WSL2:

```bash
sudo apt update
sudo apt install -y git make build-essential libmagic1 poppler-utils \
  tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa \
  python3.11 python3.11-venv curl
```

macOS con Homebrew:

```bash
brew install python@3.11 node libmagic poppler tesseract make
```

En Windows se recomienda WSL2:

```powershell
wsl --install -d Ubuntu-24.04
```

Instale el proyecto dentro del sistema de archivos de WSL, por ejemplo `~/PDF-Assistant-RAG`, y no en `/mnt/c`, para evitar degradar el rendimiento de índices y bases de datos.

### 2. Backend y frontend

```bash
git clone <URL_DEL_REPOSITORIO>
cd PDF-Assistant-RAG

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r backend/requirements.txt

cd frontend
npm install
cd ..
```

### 3. Ollama

Instalación en Linux o WSL2:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
ollama serve
```

En macOS y Windows también puede instalarse desde [ollama.com](https://ollama.com/).

### 4. Variables de entorno

```bash
cp .env.example backend/.env
```

Configuración mínima local en `backend/.env`:

```dotenv
SECRET_KEY=reemplazar-por-una-clave-segura
ENVIRONMENT=development
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:7860

MODEL_PROFILE=local_balanced
DEVICE=mps
EMBEDDING_DEVICE=mps
EMBEDDING_BATCH_SIZE=4
RERANKER_DEVICE=cpu
LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M
PDF_EXTRACTION_MODE=fast

DATABASE_URL=sqlite:///./data/app.db
CORPUS_STORE_BACKEND=local
CELERY_ENABLED=False
```

Genere una clave con:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Cree `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:7860
```

`HF_TOKEN` es opcional para modelos públicos, pero recomendable para evitar límites durante descargas de Hugging Face:

```dotenv
HF_TOKEN=hf_token
```

### 5. Base de datos y ejecución

```bash
make migrate PYTHON=.venv/bin/python
make dev PYTHON=.venv/bin/python
```

Servicios:

- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend: [http://localhost:7860](http://localhost:7860)
- Swagger: [http://localhost:7860/docs](http://localhost:7860/docs)

La primera ingesta o consulta puede tardar mientras se descargan los modelos.

## PostgreSQL y pgvector

PostgreSQL no es obligatorio para desarrollo local. Para producción o corpus grandes se recomienda PostgreSQL 16 con pgvector.

La forma más simple de crear la base, el usuario y las extensiones es Docker:

```bash
docker volume create atlas_postgres_data

docker run -d \
  --name atlas-postgres \
  --restart unless-stopped \
  -e POSTGRES_DB=pdf_rag \
  -e POSTGRES_USER=pdf_rag_user \
  -e POSTGRES_PASSWORD=pdf_rag_pass \
  -p 5432:5432 \
  -v atlas_postgres_data:/var/lib/postgresql/data \
  -v "$(pwd)/scripts/init_postgres.sql:/docker-entrypoint-initdb.d/init.sql:ro" \
  pgvector/pgvector:pg16
```

No es necesario crear la base manualmente. La imagen crea `pdf_rag` y `scripts/init_postgres.sql` activa `vector`, `unaccent`, `pg_trgm` y `uuid-ossp` cuando se crea el volumen por primera vez.

Configure `backend/.env`:

```dotenv
DATABASE_URL=postgresql+psycopg://pdf_rag_user:pdf_rag_pass@localhost:5432/pdf_rag
CORPUS_STORE_BACKEND=postgres
```

Después aplique las migraciones:

```bash
make migrate PYTHON=.venv/bin/python
```

Verificación:

```bash
docker exec -it atlas-postgres \
  psql -U pdf_rag_user -d pdf_rag \
  -c "SELECT extname, extversion FROM pg_extension;"
```

Las migraciones crean las tablas y los índices HNSW/GIN. Configure el perfil de modelos antes de migrar para que la dimensión vectorial coincida con el modelo efectivo.

## Perfil NVIDIA

El perfil `research_gpu` está orientado a una máquina con CPU Intel, RAM abundante y GPU NVIDIA:

- Los embeddings se ejecutan en CPU en lotes de 64.
- El reranker usa CUDA.
- Ollama administra la ejecución del LLM en NVIDIA.
- Si CUDA no está disponible para el reranker, este cae a CPU.
- `CPU_THREADS=0` permite que PyTorch determine el paralelismo.

Descargue el modelo predeterminado:

```bash
ollama pull qwen3:30b-a3b
```

Configuración:

```dotenv
MODEL_PROFILE=research_gpu
DEVICE=cuda
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cuda
EMBEDDING_BATCH_SIZE=64
CPU_THREADS=0

LLM_MODEL=qwen3:30b-a3b
LLM_CONTEXT_WINDOW=32768

DATABASE_URL=postgresql+psycopg://pdf_rag_user:pdf_rag_pass@localhost:5432/pdf_rag
CORPUS_STORE_BACKEND=postgres
```

Compruebe el hardware antes de iniciar:

```bash
nvidia-smi
ollama list
```

### Tesla T4 y Qwen3-14B

El perfil `wsl_t4` está diseñado para Windows con WSL2 en modo NAT, un Xeon Gold con
RAM abundante y una Tesla T4 de 16 GB dedicada exclusivamente a Ollama. El backend y
frontend se ejecutan nativamente en WSL; Docker aloja solamente PostgreSQL/pgvector.

#### 1. PowerShell de Windows

Instale el driver NVIDIA de Windows y Ollama. No instale un driver NVIDIA Linux en WSL.
Después, en PowerShell, configure Ollama para aceptar conexiones desde WSL y mantener
un único modelo cargado:

```powershell
setx OLLAMA_HOST "0.0.0.0:11434"
setx OLLAMA_KEEP_ALIVE "30m"
setx OLLAMA_CONTEXT_LENGTH "8192"
setx OLLAMA_MAX_LOADED_MODELS "1"
setx OLLAMA_NUM_PARALLEL "1"
setx OLLAMA_FLASH_ATTENTION "1"
setx OLLAMA_NO_CLOUD "1"
```

Cierre Ollama completamente desde la bandeja y vuelva a abrirlo para aplicar las
variables. Limite el acceso al adaptador virtual de WSL desde PowerShell como
administrador:

```powershell
$wslInterface = Get-NetAdapter | Where-Object Name -Like "*WSL*" | Select-Object -First 1 -ExpandProperty Name
New-NetFirewallRule -DisplayName "Ollama from WSL2" -Direction Inbound -Action Allow `
  -Protocol TCP -LocalPort 11434 -InterfaceAlias $wslInterface
```

Descargue y compruebe el modelo:

```powershell
ollama pull qwen3:14b-q4_K_M
ollama list
nvidia-smi
```

La lista debe incluir una línea similar a:

```text
qwen3:14b-q4_K_M    ...    9.3 GB
```

#### 2. Terminal de Ubuntu/WSL2

Mantenga el repositorio bajo `~/`, no bajo `/mnt/c`, y use Python 3.11:

```bash
git switch codex/experimental-wsl-t4
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
make install-backend-wsl PYTHON="$PWD/.venv/bin/python"

cd frontend
npm install
cd ..
```

El target WSL instala primero PyTorch CPU para que los modelos de recuperación no
reserven VRAM ni descarguen el runtime CUDA.

#### 3. PostgreSQL en Docker

Prepare los archivos de entorno. La contraseña debe ser idéntica en `.env`,
`backend/.env` y `DATABASE_URL`:

```bash
cp .env.example .env
cp .env.example backend/.env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Edite ambos archivos para establecer `SECRET_KEY` y `POSTGRES_PASSWORD`. Después inicie
únicamente PostgreSQL y aplique las migraciones:

```bash
docker compose up -d postgres
docker compose ps postgres
make migrate PYTHON="$PWD/.venv/bin/python"
```

La salida de Docker debe mostrar el contenedor `pdf_rag_postgres` como `healthy`.

#### 4. Diagnóstico y arranque

`doctor-wsl` obtiene la IP del gateway de Windows desde la ruta NAT; no es necesario
editar `.bashrc` ni configurar el modo mirrored:

```bash
make doctor-wsl PYTHON="$PWD/.venv/bin/python"
make dev-wsl PYTHON="$PWD/.venv/bin/python"
```

Salida esperada del diagnóstico:

```text
ATLAS WSL/T4 runtime check
[OK] profile=wsl_t4, llm=qwen3:14b-q4_K_M, CPU threads=28
[OK] Ollama at http://<gateway-windows>:11434 provides qwen3:14b-q4_K_M
[OK] PostgreSQL and pgvector extensions are available
ATLAS is ready to start in WSL/T4 mode.
```

`LLM_DISABLE_THINKING=True` desactiva el modo interno `thinking` de Ollama en planificación,
auditoría y redacción para que no consuma el límite de salida antes de producir la respuesta.
El ciclo verificable del agente (recuperación, ledger, auditoría, reparación y citas) permanece activo.

Después de la primera consulta, `ollama ps` en PowerShell debe mostrar el modelo con
`PROCESSOR` igual a `100% GPU`. Si `doctor-wsl` no puede alcanzar Ollama, compruebe que
la aplicación fue reiniciada después de definir `OLLAMA_HOST` y que la regla de firewall
está asociada al adaptador WSL correcto.

## Modelos grandes y configuración personalizada

Para cambiar libremente todos los modelos utilice `MODEL_PROFILE=custom`. Esto evita que un perfil predeterminado sustituya la selección explícita.

Ejemplo para una máquina con aproximadamente 200 GB de VRAM y 400 GB de RAM:

```bash
ollama pull qwen3:235b-a22b-instruct-2507-q4_K_M
```

```dotenv
MODEL_PROFILE=custom
DEVICE=cuda

LLM_MODEL=qwen3:235b-a22b-instruct-2507-q4_K_M
LLM_CONTEXT_WINDOW=32768
LLM_MAX_NEW_TOKENS=8192

EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIMENSION=1024
EMBEDDING_INDEX_VERSION=hierarchical-qwen3-1024-v1
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=64

RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
RERANKER_DEVICE=cuda
CPU_THREADS=0
```

La cuantización Q4 ocupa aproximadamente 142 GB. Se recomienda comenzar con contexto de 32K o 64K para conservar memoria para KV cache y buffers. Los modelos Q8 o FP16 requieren mucha más memoria y no son la opción inicial recomendada para ese hardware.

Cambiar de modelo de embeddings exige reindexar los documentos. ATLAS registra modelo, dimensión y versión del índice, y puede migrar índices existentes sin borrar archivos, usuarios o conversaciones.

## Procesamiento asíncrono

Para desarrollo, `CELERY_ENABLED=False` procesa documentos en el backend. Para producción configure Redis:

```dotenv
CELERY_ENABLED=True
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
REDIS_URL=redis://localhost:6379/0
```

Inicie Redis y el worker:

```bash
docker run -d --name atlas-redis --restart unless-stopped -p 6379:6379 redis:7-alpine

cd backend
../.venv/bin/celery -A app.celery_app.celery_app worker --loglevel=info
```

El worker debe recibir la misma configuración de base de datos, modelos, almacenamiento y perfil que el backend.


## Variables principales

| Variable | Valor local | Valor de investigación | Propósito |
|---|---|---|---|
| `MODEL_PROFILE` | `local_balanced` | `wsl_t4` | Selecciona el conjunto de modelos y dispositivos. |
| `LLM_MODEL` | `qwen3:4b-instruct-2507-q4_K_M` | `qwen3:14b-q4_K_M` | Modelo servido por Ollama. |
| `DATABASE_URL` | `sqlite:///./data/app.db` | `postgresql+psycopg://...` | Base de datos SQLAlchemy. |
| `CORPUS_STORE_BACKEND` | `local` | `postgres` | Índices locales o pgvector/tsvector. |
| `EMBEDDING_DEVICE` | `mps` | `cpu` | Dispositivo para embeddings. MPS cae a CPU si no está disponible. |
| `RERANKER_DEVICE` | `cpu` | `cpu` | Mantiene libre la T4 para Ollama. |
| `EMBEDDING_BATCH_SIZE` | `4` | `64` | Tamaño de lote de embeddings. |
| `CPU_THREADS` | `4` | `28` | Paralelismo de PyTorch; puede sobrescribirse según la CPU. |
| `PDF_EXTRACTION_MODE` | `fast` | `auto` | Usa PyMuPDF, selección adaptativa o Docling con `quality`. |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `900` | `900` | Límite de una llamada individual al LLM. |
| `RESEARCH_TIMEOUT_SECONDS` | `1800` | `1800` | Presupuesto total de investigación. |
| `RESEARCH_SYNTHESIS_RESERVE_SECONDS` | `600` | `600` | Tiempo reservado para redactar y verificar la respuesta final. |
| `RESEARCH_MAX_ROUNDS` | `2` | `2` | Rondas correctivas máximas. |
| `CELERY_ENABLED` | `False` | `False` | La primera configuración WSL procesa dentro del backend y no inicia Redis. |
| `OLLAMA_BASE_URL` | vacío | dinámico | `make dev-wsl` lo resuelve desde el gateway NAT de Windows. |
| `OLLAMA_KEEP_ALIVE` | `5m` | `30m` | Evita recargar el modelo entre fases de investigación. |

Consulte [.env.example](.env.example) y [backend/app/config.py](backend/app/config.py) para ver todas las opciones.

## Comandos

| Comando | Acción |
|---|---|
| `make install` | Instala backend y frontend. |
| `make migrate` | Inicializa la base y aplica Alembic. |
| `make dev` | Inicia FastAPI y Next.js. |
| `make dev-backend` | Inicia solo FastAPI en el puerto 7860. |
| `make dev-frontend` | Inicia solo Next.js en el puerto 3000. |
| `make test` | Ejecuta las pruebas del backend. |
| `make lint` | Ejecuta lint de backend y frontend. |
| `make build` | Compila el frontend. |

Para usar el entorno virtual del proyecto:

```bash
make test PYTHON=.venv/bin/python
make lint PYTHON=.venv/bin/python
```

Pruebas del frontend:

```bash
cd frontend
npm test
npm run build
```

## API

La especificación OpenAPI completa está disponible en `/docs` durante la ejecución.

Endpoints principales:

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Registro de usuario. |
| `POST` | `/api/v1/auth/login` | Inicio de sesión y tokens JWT. |
| `GET` | `/api/v1/auth/me` | Usuario autenticado. |
| `POST` | `/api/v1/documents/upload` | Carga e ingesta de documentos. |
| `GET` | `/api/v1/documents/` | Lista de documentos. |
| `DELETE` | `/api/v1/documents/{id}` | Elimina un documento y sus índices. |
| `POST` | `/api/v1/chat/ask/stream` | Respuesta mediante SSE. |
| `WS` | `/api/v1/chat/ws` | Chat mediante WebSocket. |
| `GET` | `/api/v1/chat/sessions` | Sesiones persistentes. |

Las solicitudes de chat aceptan `routing_mode: "auto" | "quick" | "research"` y un identificador de documento opcional para restringir el alcance.

Los streams pueden emitir eventos `progress`, `sources`, `token`, `done` y `error`. La cancelación aborta la investigación, recuperación y generación sin guardar respuestas vacías.

## Verificación de una instalación

```bash
curl http://localhost:7860/api/health
ollama list
```

En NVIDIA:

```bash
nvidia-smi
```

En PostgreSQL:

```bash
docker exec -it atlas-postgres \
  psql -U pdf_rag_user -d pdf_rag \
  -c "SELECT extname FROM pg_extension WHERE extname IN ('vector','unaccent','pg_trgm','uuid-ossp');"
```

## Seguridad y operación

- Use una `SECRET_KEY` persistente y aleatoria en producción.
- Configure `ENVIRONMENT=production` y `ALLOWED_ORIGINS` explícitamente.
- No publique PostgreSQL, Redis, Ollama o pgAdmin directamente en Internet.
- Use contraseñas diferentes a los valores de ejemplo.
- Mantenga copias de seguridad del volumen de PostgreSQL y de los archivos subidos.
- Limite la concurrencia de investigaciones para modelos grandes.
- Los proveedores OAuth y SMTP son opcionales y se configuran en `backend/.env`.

## Licencia

Distribuido bajo la licencia MIT. Consulte [license](license).
