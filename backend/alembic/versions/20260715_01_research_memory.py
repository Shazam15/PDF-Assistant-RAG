"""Create hierarchical research memory and retrieval indexes."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

from app.config import get_settings


revision = "20260715_01"
down_revision = None
branch_labels = None
depends_on = None


def _guid_type(bind):
    return postgresql.UUID(as_uuid=True) if bind.dialect.name == "postgresql" else sa.String(36)


def _embedding_type(bind):
    if bind.dialect.name == "postgresql":
        from pgvector.sqlalchemy import Vector

        return Vector(get_settings().EMBEDDING_DIMENSION)
    return sa.Text()


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    inspector = inspect(bind)
    guid = _guid_type(bind)
    embedding = _embedding_type(bind)

    if not inspector.has_table("document_profiles"):
        op.create_table(
            "document_profiles",
            sa.Column("id", guid, primary_key=True),
            sa.Column("document_id", guid, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", guid, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("title", sa.String(500), nullable=False),
            sa.Column("summary", sa.Text()),
            sa.Column("methodology", sa.Text()),
            sa.Column("findings", sa.Text()),
            sa.Column("limitations", sa.Text()),
            sa.Column("embedding", embedding),
            sa.Column("section_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("index_version", sa.String(100), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("document_id", name="uq_document_profiles_document_id"),
        )
        op.create_index("ix_document_profiles_user_id", "document_profiles", ["user_id"])

    if not inspector.has_table("document_sections"):
        op.create_table(
            "document_sections",
            sa.Column("id", sa.String(100), primary_key=True),
            sa.Column("document_id", guid, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", guid, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("title", sa.String(500), nullable=False),
            sa.Column("section_path", sa.Text()),
            sa.Column("page_start", sa.Integer(), nullable=False),
            sa.Column("page_end", sa.Integer(), nullable=False),
            sa.Column("summary", sa.Text()),
            sa.Column("text", sa.Text(), nullable=False),
        )
        op.create_index("ix_document_sections_document_id", "document_sections", ["document_id"])
        op.create_index("ix_document_sections_user_id", "document_sections", ["user_id"])

    if not inspector.has_table("document_chunks"):
        op.create_table(
            "document_chunks",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column("document_id", guid, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", guid, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("filename", sa.String(500), nullable=False),
            sa.Column("chunk_index", sa.Integer(), nullable=False),
            sa.Column("parent_id", sa.String(100)),
            sa.Column("section_id", sa.String(100)),
            sa.Column("section_title", sa.String(500)),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("parent_text", sa.Text()),
            sa.Column("page", sa.Integer(), nullable=False),
            sa.Column("page_start", sa.Integer(), nullable=False),
            sa.Column("page_end", sa.Integer(), nullable=False),
            sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("chunk_type", sa.String(40), nullable=False, server_default="text"),
            sa.Column("bbox", sa.Text()),
            sa.Column("table_index", sa.Integer()),
            sa.Column("embedding", embedding),
            sa.Column("search_text", sa.Text()),
        )
        op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
        op.create_index("ix_document_chunks_user_id", "document_chunks", ["user_id"])
        op.create_index("ix_document_chunks_parent_id", "document_chunks", ["parent_id"])
        op.create_index("ix_document_chunks_section_id", "document_chunks", ["section_id"])
        op.create_index("ix_document_chunks_user_document", "document_chunks", ["user_id", "document_id"])

    if not inspector.has_table("document_evidence"):
        op.create_table(
            "document_evidence",
            sa.Column("id", guid, primary_key=True),
            sa.Column("document_id", guid, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", guid, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("chunk_id", sa.String(160), sa.ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False),
            sa.Column("evidence_kind", sa.String(40), nullable=False),
            sa.Column("claim", sa.Text(), nullable=False),
            sa.Column("exact_quote", sa.Text(), nullable=False),
            sa.Column("page", sa.Integer(), nullable=False),
            sa.Column("section_title", sa.String(500)),
            sa.Column("confidence", sa.Float(), nullable=False, server_default="1"),
        )
        op.create_index("ix_document_evidence_document_id", "document_evidence", ["document_id"])
        op.create_index("ix_document_evidence_user_id", "document_evidence", ["user_id"])
        op.create_index("ix_document_evidence_chunk_id", "document_evidence", ["chunk_id"])

    if not inspector.has_table("research_runs"):
        op.create_table(
            "research_runs",
            sa.Column("id", guid, primary_key=True),
            sa.Column("user_id", guid, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("document_id", guid, sa.ForeignKey("documents.id", ondelete="SET NULL")),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("state_json", sa.Text()),
            sa.Column("status", sa.String(30), nullable=False, server_default="running"),
            sa.Column("rounds", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("evidence_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("completed_at", sa.DateTime()),
        )
        op.create_index("ix_research_runs_user_id", "research_runs", ["user_id"])
        op.create_index("ix_research_runs_document_id", "research_runs", ["document_id"])
        op.create_index("ix_research_runs_status", "research_runs", ["status"])

    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_document_profiles_embedding_hnsw "
            "ON document_profiles USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_search_gin "
            "ON document_chunks USING gin (to_tsvector('simple', COALESCE(search_text,text)))"
        )
    else:
        op.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5("
            "chunk_id UNINDEXED, user_id UNINDEXED, document_id UNINDEXED, "
            "filename UNINDEXED, text, tokenize='unicode61')"
        )


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_document_chunks_search_gin")
        op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
        op.execute("DROP INDEX IF EXISTS ix_document_profiles_embedding_hnsw")
    else:
        op.execute("DROP TABLE IF EXISTS document_chunks_fts")
    inspector = inspect(bind)
    for table_name in (
        "document_evidence",
        "research_runs",
        "document_chunks",
        "document_sections",
        "document_profiles",
    ):
        if inspector.has_table(table_name):
            op.drop_table(table_name)
