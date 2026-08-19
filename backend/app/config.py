"""
Application configuration via pydantic-settings.
All config is loaded from environment variables with sensible defaults.
"""
import json
import os
import secrets
from functools import lru_cache
from typing import Any

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


    # ── MCP Tools ────────────────────────────────────────
    MCP_ENABLED: bool = True
    MCP_SERVERS_JSON: str = "{}"
    MCP_TOOL_ALLOWLIST: list[str] = [
        "read_file",
        "write_file",
        "list_directory",
        "file_exists",
        "get_file_info",
        "grep_file",
        "count_pattern",
        "extract_log_lines",
    ]
    MCP_TOOL_DENYLIST: list[str] = []
    MCP_TOOL_TIMEOUT_SECONDS: int = 30
    MCP_MAX_RESULT_CHARS: int = 6000
    MCP_SERVERS: dict[str, dict[str, Any]] = {}

    @model_validator(mode="before")
    @classmethod
    def _parse_mcp_settings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_servers = data.get("MCP_SERVERS_JSON")
        if raw_servers is None:
            raw_servers = data.get("MCP_SERVERS")

        if isinstance(raw_servers, str):
            if not raw_servers.strip():
                parsed_servers = {}
            else:
                try:
                    parsed_servers = json.loads(raw_servers)
                except json.JSONDecodeError as exc:
                    raise ValueError("MCP_SERVERS_JSON must be valid JSON.") from exc
            if not isinstance(parsed_servers, dict):
                raise ValueError("MCP_SERVERS_JSON must decode to a JSON object.")
            data["MCP_SERVERS"] = parsed_servers
        elif isinstance(raw_servers, dict):
            data["MCP_SERVERS"] = raw_servers
        else:
            data["MCP_SERVERS"] = {}

        for field_name in ("MCP_TOOL_ALLOWLIST", "MCP_TOOL_DENYLIST"):
            value = data.get(field_name)
            if isinstance(value, str):
                data[field_name] = [item.strip() for item in value.split(",") if item.strip()]
            elif value is None:
                data[field_name] = []

        return data

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "ATLAS"
    SECRET_KEY: str = ""
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:7860"

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./data/app.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 25
    DATABASE_POOL_PRE_PING: bool = True
    CORPUS_STORE_BACKEND: str = "auto"

    # ── Auth ─────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_EXPIRY_MINUTES: int = 15
    JWT_REFRESH_EXPIRY_DAYS: int = 7
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_DRIVE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google-drive/callback"
    HF_CLIENT_ID: str = ""
    HF_CLIENT_SECRET: str = ""
    HF_REDIRECT_URI: str = ""
    FRONTEND_URL: str = "http://localhost:3000"
    EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 24

    # ── Email verification ───────────────────────────────
    MAIL_USERNAME: str = ""
    MAIL_PASSWORD: str = ""
    MAIL_FROM: str = ""
    MAIL_SERVER: str = ""
    MAIL_PORT: int = 587
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False

    # Google Drive background sync
    DRIVE_SYNC_ENABLED: bool = False
    DRIVE_SYNC_INTERVAL_MINUTES: int = 60
    GOOGLE_SERVICE_ACCOUNT_FILE: str = ""

    # Celery / Redis background processing
    CELERY_ENABLED: bool = False
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_TRACK_STARTED: bool = True

    # ── Document Processing ──────────────────────────────
    DOC_PROCESSING_TIMEOUT_MINUTES: int = 30
    DOC_PROCESSING_MAX_RETRIES: int = 3
    DOC_PROCESSING_RETRY_DELAY_SECONDS: int = 30
    DOC_CLEANUP_MAX_AGE_DAYS: int = 90

    # ── File Upload ──────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: set = {
        "pdf",
        "docx",
        "txt",
        "md",
        "py",
        "js",
        "ts",
        "tsx",
        "java",
        "cpp",
        "c",
        "cs",
        "go",
        "rs",
        "sql",
        "ipynb",
    }
    ALLOWED_MIME_TYPES: dict = {
        ".pdf": ["application/pdf"],
        ".docx": [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        ],
        ".txt": ["text/plain"],
        ".md": ["text/markdown"],
        ".py": ["text/x-python", "application/x-python-code", "text/plain"],
        ".js": ["application/javascript", "text/javascript", "text/plain"],
        ".ts": ["application/typescript", "text/typescript", "text/plain"],
        ".tsx": ["application/typescript", "text/typescript", "text/plain"],
        ".java": ["text/x-java-source", "text/x-java", "text/plain"],
        ".cpp": ["text/x-c++src", "text/x-c++hdr", "text/plain"],
        ".c": ["text/x-csrc", "text/x-chdr", "text/plain"],
        ".cs": ["text/x-csharp", "text/x-csharp", "text/plain"],
        ".go": ["text/x-go", "text/x-go-source", "text/plain"],
        ".rs": ["text/x-rust", "text/x-rust-source", "text/plain"],
        ".sql": ["text/x-sql", "application/sql", "text/plain"],
        ".ipynb": ["application/json", "text/plain"],
    }

    # ── RAG Pipeline ─────────────────────────────────────
    CHUNK_SIZE: int = 420
    CHUNK_OVERLAP: int = 80
    PARENT_CHUNK_SIZE: int = 1600
    PARENT_CHUNK_OVERLAP: int = 160
    PDF_USE_DOCLING: bool = True
    PDF_EXTRACTION_MODE: str = "auto"
    PDF_USE_UNSTRUCTURED: bool = False
    TOP_K_RETRIEVAL: int = 36 # Fetch a broad candidate pool across documents
    TOP_K_RERANK: int = 16 # Final number of chunks to return after reranking

    # ── Knowledge Graph (GraphRAG) ───────────────────────
    GRAPH_PERSIST_DIR: str = "./data/graphs"
    GRAPH_ENTITY_LABELS: set = {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "NORP",
        "FAC",
    }
    GRAPH_MAX_RELATIONSHIPS: int = 12

    # ── Embeddings (local HuggingFace model) ─────────────
    MODEL_PROFILE: str = "local"
    DEVICE: str = "cpu"
    EMBEDDING_DEVICE: str = "cpu"
    RERANKER_DEVICE: str = "cpu"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_INDEX_VERSION: str = "hierarchical-e5-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    CPU_THREADS: int = 0  # 0 lets PyTorch choose from the available Xeon cores.

    # ── ChromaDB ─────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # ── LLM (HuggingFace Inference API) ──────────────────
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # HuggingFace API token (set in .env)
    LLM_MODEL: str = "qwen3:4b-instruct-2507-q4_K_M"
    LLM_CONTEXT_WINDOW: int = 8192
    LLM_MAX_NEW_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.3
    LLM_REQUEST_TIMEOUT_SECONDS: int = 900
    LLM_DISABLE_THINKING: bool = False
    AGENT_PLANNER_MAX_TOKENS: int = 768
    AGENT_SYNTHESIS_MAX_TOKENS: int = 2048
    AGENT_MAX_ITERATIONS: int = 4  # Three research steps plus one mandatory final synthesis
    RESEARCH_MAX_ROUNDS: int = 2
    RESEARCH_TIMEOUT_SECONDS: int = 1800
    RESEARCH_SYNTHESIS_RESERVE_SECONDS: int = 600
    RESEARCH_MAX_FACETS: int = 6
    RESEARCH_MIN_EVIDENCE_PER_FACET: int = 1
    RESEARCH_PIPELINE_VERSION: str = "evidence-agent-v2"
    SUMMARY_MAX_TOKENS: int = 512

    # ── LangSmith Tracing (optional) ─────────────────────
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "pdf-assistant-rag"

    # ── Reranker ─────────────────────────────────────────
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    RERANK_MAX_LENGTH: int = 2048
    RERANK_RELEVANCE_THRESHOLD: float = 0.5
    RERANK_SCORE_MARGIN: float = 0.15
    RETRIEVAL_PLANNER_MAX_TOKENS: int = 256
    RETRIEVAL_PLANNER_TIMEOUT_SECONDS: int = 30
    RETRIEVAL_PLANNER_VERSION: str = "facets-json-mode-v2"
    NLI_MODEL: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    NLI_ENTAILMENT_THRESHOLD: float = 0.65
    NLI_VERIFIER_VERSION: str = "multilingual-nli-v1"
    # ── Vision / Image captioning ─────────────────────
    VISION_PROVIDER: str | None = None  # e.g. 'openai'
    VISION_MODEL: str | None = None
    OPENAI_API_KEY: str = ""

    # ── Workspace Invitation ─────────────────────────
    APP_URL: str = "http://localhost:3000"
    INVITE_TOKEN_EXPIRY_HOURS: int = 72
    EMAIL_FROM: str = "no-reply@example.com"
    SMTP_HOST: str = ""
    SMTP_PORT: int = 0
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""

    #--- Code Help --------------------------------
    CODE_REVIEW_LLM_MODEL: str = "qwen3-coder"
    CODE_REVIEW_TEMPERATURE: float = 0
    CODE_REVIEW_MAX_CHARS: int = 12000

    @model_validator(mode="after")
    def validate_runtime(self):
        environment = str(self.ENVIRONMENT).lower()
        profile = str(self.MODEL_PROFILE).lower()
        extraction_mode = str(self.PDF_EXTRACTION_MODE).lower()

        # PDF_USE_DOCLING remains a compatibility switch for existing installs.
        # New configurations should use PDF_EXTRACTION_MODE directly.
        if "PDF_EXTRACTION_MODE" not in self.model_fields_set and "PDF_USE_DOCLING" in self.model_fields_set:
            extraction_mode = "quality" if self.PDF_USE_DOCLING else "fast"
            self.PDF_EXTRACTION_MODE = extraction_mode

        if profile == "research_gpu":
            # Keep large, parallel embedding batches on the host CPU/RAM and
            # reserve NVIDIA memory for reranking and Ollama. Explicit env
            # overrides remain available for CPU-only deployments.
            if "DEVICE" not in self.model_fields_set:
                self.DEVICE = "cuda"
            if "EMBEDDING_DEVICE" not in self.model_fields_set:
                self.EMBEDDING_DEVICE = "cpu"
            if "RERANKER_DEVICE" not in self.model_fields_set:
                self.RERANKER_DEVICE = "cuda"
            if "EMBEDDING_BATCH_SIZE" not in self.model_fields_set:
                self.EMBEDDING_BATCH_SIZE = 64
            if "EMBEDDING_MODEL" not in self.model_fields_set:
                self.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
            if "EMBEDDING_DIMENSION" not in self.model_fields_set:
                self.EMBEDDING_DIMENSION = 1024
            if "EMBEDDING_INDEX_VERSION" not in self.model_fields_set:
                self.EMBEDDING_INDEX_VERSION = "hierarchical-qwen3-1024-v1"
            if "RERANKER_MODEL" not in self.model_fields_set:
                self.RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
            if "LLM_MODEL" not in self.model_fields_set:
                self.LLM_MODEL = "qwen3:30b-a3b"
            if "LLM_CONTEXT_WINDOW" not in self.model_fields_set:
                self.LLM_CONTEXT_WINDOW = 32768
            self.RETRIEVAL_PLANNER_VERSION = "research-brief-qwen3-v1"
        elif profile == "local_balanced":
            if "DEVICE" not in self.model_fields_set:
                self.DEVICE = "mps"
            if "EMBEDDING_DEVICE" not in self.model_fields_set:
                self.EMBEDDING_DEVICE = "mps"
            if "RERANKER_DEVICE" not in self.model_fields_set:
                self.RERANKER_DEVICE = "cpu"
            if "EMBEDDING_BATCH_SIZE" not in self.model_fields_set:
                self.EMBEDDING_BATCH_SIZE = 4
            if "CPU_THREADS" not in self.model_fields_set:
                self.CPU_THREADS = 4
            if "EMBEDDING_MODEL" not in self.model_fields_set:
                self.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
            if "EMBEDDING_DIMENSION" not in self.model_fields_set:
                self.EMBEDDING_DIMENSION = 1024
            if "EMBEDDING_INDEX_VERSION" not in self.model_fields_set:
                self.EMBEDDING_INDEX_VERSION = "hierarchical-qwen3-1024-v1"
        elif profile not in {"local", "custom"}:
            raise ValueError("MODEL_PROFILE must be local, local_balanced, custom, or research_gpu")

        if extraction_mode not in {"auto", "fast", "quality"}:
            raise ValueError("PDF_EXTRACTION_MODE must be auto, fast, or quality")

        if self.CORPUS_STORE_BACKEND == "auto":
            self.CORPUS_STORE_BACKEND = (
                "postgres" if self.DATABASE_URL.startswith("postgresql") else "local"
            )
        if self.CORPUS_STORE_BACKEND not in {"local", "postgres"}:
            raise ValueError("CORPUS_STORE_BACKEND must be local, postgres, or auto")
        if self.CORPUS_STORE_BACKEND == "postgres" and not self.DATABASE_URL.startswith("postgresql"):
            raise ValueError("The postgres corpus backend requires a PostgreSQL DATABASE_URL")
        if self.RESEARCH_MAX_ROUNDS < 1 or self.RESEARCH_MAX_ROUNDS > 4:
            raise ValueError("RESEARCH_MAX_ROUNDS must be between 1 and 4")
        if self.EMBEDDING_BATCH_SIZE < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        if self.CPU_THREADS < 0:
            raise ValueError("CPU_THREADS cannot be negative")
        if self.RESEARCH_TIMEOUT_SECONDS < 30 or self.RESEARCH_TIMEOUT_SECONDS > 7200:
            raise ValueError("RESEARCH_TIMEOUT_SECONDS must be between 30 and 7200")
        if self.LLM_REQUEST_TIMEOUT_SECONDS < 10 or self.LLM_REQUEST_TIMEOUT_SECONDS > 3600:
            raise ValueError("LLM_REQUEST_TIMEOUT_SECONDS must be between 10 and 3600")
        if not 10 <= self.RESEARCH_SYNTHESIS_RESERVE_SECONDS < self.RESEARCH_TIMEOUT_SECONDS:
            raise ValueError(
                "RESEARCH_SYNTHESIS_RESERVE_SECONDS must be at least 10 and below RESEARCH_TIMEOUT_SECONDS"
            )

        if not self.SECRET_KEY:
            if environment == "production":
                raise ValueError("SECRET_KEY must be set when ENVIRONMENT=production")
            self.SECRET_KEY = secrets.token_urlsafe(32)
        return self

    @property
    def cors_origins(self) -> list[str]:
        if self.ENVIRONMENT == "production":
            return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]
        return ["*"]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — loaded once on startup."""
    return Settings()
