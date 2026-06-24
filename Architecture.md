# PDF Assistant RAG — Documentación de Arquitectura y Funcionamiento

> Versión adaptada para implementación con **Ollama** como motor de inferencia local, en sustitución de la API de HuggingFace.

---

## 1. Visión General

PDF Assistant RAG es un sistema de análisis de documentos basado en Inteligencia Artificial que permite a los usuarios cargar documentos PDF y hacer preguntas sobre su contenido. El sistema genera respuestas precisas y citadas mediante un pipeline de Generación Aumentada por Recuperación (RAG) de múltiples etapas.

El sistema está diseñado para ser desplegado en una máquina propia (servidor universitario, computadora de escritorio o servidor en la nube), sin depender de servicios de pago para la generación de respuestas, gracias al uso de Ollama para correr modelos de lenguaje localmente.

---

## 2. Topología del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUARIO FINAL                           │
│                    (Navegador Web)                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP / WebSocket / SSE
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              FRONTEND  (Next.js 16 — Puerto 3000)               │
│  Dashboard · Panel de Chat · Visor de PDF · Gestión de Sesiones │
└─────────────────────┬───────────────────────────────────────────┘
                      │ REST API + JWT
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKEND  (FastAPI — Puerto 7860)                    │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │   Auth   │  │Documentos│  │   Chat   │  │    Admin      │  │
│  │  /auth   │  │/documents│  │  /chat   │  │   /admin      │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│                                                                 │
│  ┌─────────────────── Pipeline RAG ─────────────────────────┐  │
│  │  Chunker → Embeddings → Retriever → Reranker → Agente    │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────┬──────────────────┬──────────────────┬────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌────────────┐   ┌──────────────┐   ┌───────────────┐
│  SQLite DB │   │   ChromaDB   │   │     Redis     │
│ Usuarios,  │   │  Vectores de │   │  Cola Celery  │
│ Documentos,│   │  documentos  │   │  (procesado   │
│ Historial  │   │  por usuario │   │  en background│
└────────────┘   └──────────────┘   └───────────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │  Celery Worker  │
                                   │ (ingesta PDFs)  │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │     OLLAMA      │
                                   │  (Puerto 11434) │
                                   │  llama3.2 /     │
                                   │  mistral:7b /   │
                                   │  llama3.1:8b    │
                                   └─────────────────┘
```

---

## 3. Componentes Principales

### 3.1 Frontend (Next.js 16)

La interfaz de usuario es una aplicación web desarrollada con Next.js que corre en el puerto 3000. Provee las siguientes vistas:

- **Dashboard**: vista general con lista de documentos subidos y su estado de procesamiento.
- **Panel de Chat**: interfaz de conversación con el agente RAG, con soporte para streaming de respuestas token por token.
- **Visor de PDF**: permite visualizar los documentos directamente en el navegador.
- **Gestión de sesiones**: los usuarios pueden crear, renombrar y eliminar sesiones de chat.

La comunicación con el backend se realiza mediante:

- **REST sobre HTTP** para autenticación, subida de documentos y gestión de sesiones.
- **Server-Sent Events (SSE)** para recibir respuestas del chat en tiempo real.
- **WebSocket** como canal alternativo de streaming para el agente.

### 3.2 Backend (FastAPI)

El backend expone una API REST en el puerto 7860 organizada en los siguientes grupos de rutas:

| Grupo | Prefijo | Función |
|---|---|---|
| Autenticación | `/api/v1/auth` | Registro, login, JWT, perfil de usuario |
| Documentos | `/api/v1/documents` | Subida, validación, ingesta, estado, eliminación |
| Chat | `/api/v1/chat` | Preguntas RAG, streaming SSE, historial, sesiones |
| Admin | `/api/v1/admin` | Estadísticas operativas y gestión de usuarios |
| Salud | `/health` | Verificación del estado del sistema |

La documentación interactiva de la API está disponible en `http://localhost:7860/docs` (Swagger UI).

### 3.3 Base de Datos (SQLite)

Almacena todos los metadatos del sistema:

- **Usuarios**: credenciales, roles, tokens de verificación.
- **Documentos**: nombre, ruta, estado de procesamiento, número de páginas, resumen.
- **Sesiones de chat**: título, usuario propietario.
- **Mensajes**: contenido, rol (usuario/asistente), fuentes citadas, feedback.

### 3.4 Almacén de Vectores (ChromaDB)

Almacena los embeddings (representaciones vectoriales) de los fragmentos de texto de cada documento. Cada usuario tiene su propia colección aislada, identificada por su `user_id`. Esto garantiza que un usuario nunca acceda a los vectores de otro.

### 3.5 Cola de Tareas (Redis + Celery)

El procesamiento de documentos es una operación costosa (extracción de texto, generación de embeddings, construcción de grafos de conocimiento). Para no bloquear la API, este trabajo se delega a un worker de Celery que consume tareas de una cola en Redis.

Cuando el usuario sube un documento, la API responde inmediatamente con `202 Accepted` y encola la tarea. El worker la procesa en segundo plano y actualiza el estado en la base de datos.

### 3.6 Motor LLM (Ollama)

**Este es el componente que fue modificado respecto a la versión original del proyecto**, la cual usaba la API de inferencia de HuggingFace.

Ollama corre localmente en el puerto `11434` y sirve los modelos de lenguaje. Las ventajas de este enfoque son:

- **Sin costo**: no hay tokens que consumir ni APIs de pago.
- **Sin dependencia externa**: el sistema funciona completamente offline una vez descargados los modelos.
- **Privacidad**: los documentos nunca salen del servidor.

Los modelos recomendados para este sistema son:

| Modelo | Tamaño | Calidad RAG | Velocidad (CPU) |
|---|---|---|---|
| `llama3.2` | ~2GB | Básica | Rápida |
| `mistral:7b` | ~4GB | Buena | Media |
| `llama3.1:8b` | ~5GB | Muy buena | Lenta en CPU |

El modelo activo se configura en `backend/.env` con la variable `LLM_MODEL`.

---

## 4. Flujo de Ingesta de Documentos

Cuando un usuario sube un PDF, el sistema ejecuta el siguiente proceso:

```
Usuario sube PDF
      │
      ▼
FastAPI valida el archivo
(extensión, tamaño, tipo MIME, legibilidad)
      │
      ▼
Archivo guardado en disco
(./data/uploads/<user_id>/<archivo>)
      │
      ▼
Registro creado en SQLite
(status: "pending")
      │
      ▼
Tarea enviada a Redis
      │
      ▼
Celery Worker recibe la tarea
      │
      ├─► Extracción de texto (pypdf / fallback texto plano)
      │
      ├─► Chunking: división en fragmentos de ~1000 caracteres
      │   con solapamiento de 200 caracteres
      │
      ├─► Embeddings locales (sentence-transformers/all-MiniLM-L6-v2)
      │   y almacenamiento en ChromaDB
      │
      ├─► Índice BM25 para búsqueda léxica
      │
      ├─► Extracción de grafo de conocimiento (spaCy, si disponible)
      │
      └─► Generación de resumen (Ollama)
            │
            ▼
      SQLite actualizado
      (status: "ready", páginas, chunks, resumen)
```

La API del frontend consulta periódicamente el estado del documento hasta que aparece como `ready`.

---

## 5. Flujo de Chat y Recuperación RAG

Cuando el usuario hace una pregunta, el sistema ejecuta el siguiente pipeline:

```
Usuario escribe una pregunta
      │
      ▼
FastAPI valida la sesión y el documento (si aplica)
      │
      ▼
Mensaje del usuario guardado en SQLite
      │
      ▼
Agente ReAct (LangChain + Ollama)
      │
      ├─► Decide usar herramienta: pdf_search
      │         │
      │         ├─► Búsqueda semántica en ChromaDB
      │         │   (vectores del usuario/documento)
      │         │
      │         ├─► Búsqueda léxica BM25
      │         │
      │         ├─► Fusión y deduplicación de candidatos (RRF)
      │         │
      │         └─► Reranking con cross-encoder
      │             (cross-encoder/ms-marco-MiniLM-L-6-v2)
      │
      ├─► Contexto recuperado enviado a Ollama
      │
      └─► Ollama genera la respuesta con streaming
            │
            ▼
      Tokens enviados al frontend via SSE
            │
            ▼
      Respuesta completa guardada en SQLite
      con fuentes citadas (documento, página, fragmento)
```

### 5.1 Herramientas del Agente

El agente ReAct tiene acceso a tres herramientas:

- **`pdf_search`**: busca en los documentos del usuario usando recuperación híbrida (semántica + léxica) con reranking.
- **`calculator`**: evalúa expresiones matemáticas de forma segura.
- **`web_search`**: busca en internet via DuckDuckGo cuando el contenido de los documentos es insuficiente.

---

## 6. Aislamiento de Datos por Usuario

Cada usuario opera en un espacio completamente aislado:

- Sus documentos se guardan en `./data/uploads/<user_id>/`.
- Sus vectores en ChromaDB se almacenan en una colección `user_<user_id>`.
- Todas las consultas a la base de datos y al almacén de vectores filtran por `user_id`.
- Los administradores pueden ver estadísticas agregadas pero no el contenido de los documentos.

---

## 7. Configuración del Sistema (.env)

Las variables más importantes para configurar el sistema en `backend/.env`:

```bash
# Seguridad
SECRET_KEY=clave-aleatoria-larga

# Modelo LLM (Ollama local — sin costo)
LLM_MODEL=mistral:7b

# Reranker (descarga automática de HuggingFace, uso local)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Embeddings (descarga automática, uso local)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Redis (requerido para Celery)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Almacenamiento
UPLOAD_DIR=./data/uploads
CHROMA_PERSIST_DIR=./data/chroma_db
```

---

## 8. Diferencias con la Versión Original

La versión original del proyecto usaba la API de inferencia de HuggingFace con el modelo `Qwen/Qwen2.5-72B-Instruct`. Esta implementación fue modificada para usar Ollama por las siguientes razones:

| Aspecto | Versión Original (HuggingFace) | Esta Versión (Ollama) |
|---|---|---|
| Costo | Tokens limitados, plan de pago requerido | Gratuito, sin límites |
| Privacidad | Documentos enviados a servidores externos | Procesamiento 100% local |
| Dependencia de internet | Requerida para cada pregunta | Solo para descargar el modelo por primera vez |
| Velocidad | Rápida (GPU en la nube) | Depende del hardware local |
| Modelo | Qwen 72B (muy potente) | Mistral 7B / Llama 3.1 8B (buena calidad) |

Los archivos modificados fueron:

- `backend/app/rag/agent.py` — cliente LLM cambiado de `HuggingFaceEndpoint` a `ChatOllama`.
- `backend/app/rag/summarizer.py` — generación de resúmenes adaptada a `ChatOllama`.
- `backend/app/rag/retriever.py` — función `transform_query` simplificada (sin expansión LLM).
- `backend/app/rag/tools.py` — eliminadas definiciones de herramientas de HuggingFace.
- `backend/.env` — variable `LLM_MODEL` apunta ahora al modelo de Ollama.

---

## 9. Requisitos para Correr el Sistema

### Software

- Python 3.11+
- Node.js 18+
- Redis (broker de tareas)
- Ollama (motor LLM local)

### Hardware Recomendado

| Componente | Mínimo | Recomendado |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Almacenamiento | 10 GB | 50 GB |
| CPU | 4 núcleos | 8 núcleos |
| GPU | No requerida | Opcional (acelera Ollama) |

### Comandos de Inicio

```bash
# 1. Iniciar Redis
redis-server --daemonize yes

# 2. Iniciar Ollama
ollama serve &
ollama pull mistral:7b

# 3. Iniciar Celery Worker (en una terminal separada)
cd backend
celery -A app.celery_app worker --loglevel=info --concurrency=1

# 4. Iniciar el sistema completo
make dev
```

---

## 10. Acceso Externo (Producción)

Para que investigadores externos accedan al sistema sin estar en la red local, se recomienda usar **Cloudflare Tunnel**:

1. Crear una cuenta gratuita en cloudflare.com.
2. Registrar un dominio (ejemplo: `investigacion.universidad.edu`).
3. Crear un tunnel desde el dashboard de Cloudflare Zero Trust.
4. Instalar el conector `cloudflared` en el servidor y apuntarlo al puerto 3000.
5. Opcionalmente, configurar **Cloudflare Access** para restringir el acceso solo a correos con dominio institucional (ejemplo: `@utp.ac.pa`).

Este enfoque no requiere abrir puertos en el router, incluye HTTPS automático y funciona desde cualquier conexión a internet.
