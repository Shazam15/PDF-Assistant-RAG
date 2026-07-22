"""Add granular document ingestion progress fields."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260720_01"
down_revision = "20260715_01"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    columns = {column["name"] for column in inspect(bind).get_columns("documents")}
    additions = [
        ("processing_current", sa.Integer()),
        ("processing_total", sa.Integer()),
        ("processing_updated_at", sa.DateTime()),
        ("searchable_at", sa.DateTime()),
        ("processing_warning", sa.Text()),
    ]
    for name, column_type in additions:
        if name not in columns:
            op.add_column("documents", sa.Column(name, column_type, nullable=True))

    if bind.dialect.name == "postgresql":
        op.alter_column("documents", "processing_stage", type_=sa.String(40), existing_type=sa.String(20))


def downgrade():
    bind = op.get_bind()
    columns = {column["name"] for column in inspect(bind).get_columns("documents")}
    for name in (
        "processing_warning",
        "searchable_at",
        "processing_updated_at",
        "processing_total",
        "processing_current",
    ):
        if name in columns:
            op.drop_column("documents", name)

    if bind.dialect.name == "postgresql":
        op.alter_column("documents", "processing_stage", type_=sa.String(20), existing_type=sa.String(40))
