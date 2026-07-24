"""
Admin-only operational statistics routes.
"""
import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import get_current_admin
from app.config import get_settings
from app.database import get_db
from app.exceptions import NotFoundException
from app.metrics import get_query_metrics
from app.models import Document, User, ChatMessage
from app.rag.graph_builder import get_graph_path, load_graph
from app.schemas import (
    AdminGraphEdgeResponse,
    AdminGraphListResponse,
    AdminGraphNodeResponse,
    AdminGraphResponse,
    AdminGraphSummaryResponse,
    AdminStatsResponse,
    DiskUsageResponse,
    UserResponse,
)

router = APIRouter(prefix="/admin", tags=["Admin"])
settings = get_settings()
logger = logging.getLogger(__name__)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


@router.get(
    "/stats",
    response_model=AdminStatsResponse,
    summary="Get admin dashboard statistics",
    description=(
        "Returns aggregate user, document, message, query-latency, and disk "
        "usage metrics for authenticated administrators."
    ),
)
def get_admin_stats(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """Return aggregate operational statistics for the admin dashboard.

    The response includes counts for users, uploaded PDFs, all documents, chat
    messages, average RAG query latency, and upload-directory disk usage.
    Access is restricted by the `get_current_admin` dependency.
    """
    upload_dir = Path(settings.UPLOAD_DIR).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    disk_usage = shutil.disk_usage(upload_dir)
    used_percent = (
        round((disk_usage.used / disk_usage.total) * 100, 2)
        if disk_usage.total
        else 0.0
    )
    query_metrics = get_query_metrics()

    total_pdfs_uploaded = (
        db.query(Document)
        .filter(func.lower(Document.original_name).like("%.pdf"))
        .count()
    )

    return AdminStatsResponse(
        total_users=db.query(User).count(),
        total_pdfs_uploaded=total_pdfs_uploaded,
        total_documents=db.query(Document).count(),
        total_messages=db.query(ChatMessage).count(),
        average_query_response_time_ms=float(
            query_metrics["average_query_response_time_ms"]
        ),
        query_count=int(query_metrics["query_count"]),
        disk_space_usage=DiskUsageResponse(
            total_bytes=disk_usage.total,
            used_bytes=disk_usage.used,
            free_bytes=disk_usage.free,
            usage_percent=used_percent,
            upload_dir_bytes=_directory_size(upload_dir),
        ),
        users=db.query(User).all()
    )


@router.get(
    "/users",
    response_model=List[UserResponse],
    summary="List all registered users",
    description="Returns the registered user inventory for authenticated administrators.",
)
def list_all_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """List all registered users.

    Access is restricted to administrators and the response is serialized
    through `UserResponse` so token fields and secrets are not exposed.
    """
    return db.query(User).all()


def _integer_pages(values) -> list[int]:
    pages = []
    for value in values or []:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page >= 0:
            pages.append(page)
    return sorted(set(pages))


@router.get(
    "/graphs",
    response_model=AdminGraphListResponse,
    summary="List persisted document knowledge graphs",
)
def list_knowledge_graphs(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """List graph-enabled documents across users for administrators only."""

    records = (
        db.query(Document, User)
        .join(User, User.id == Document.user_id)
        .filter(Document.is_deleted.is_(False))
        .order_by(Document.uploaded_at.desc())
        .all()
    )
    items = []
    for document, owner in records:
        if not get_graph_path(str(owner.id), str(document.id)).exists():
            continue
        try:
            graph = load_graph(str(owner.id), str(document.id))
        except Exception as exc:
            logger.warning(
                "Could not load graph for admin inventory %s: %s",
                document.id,
                exc,
            )
            continue
        if graph is None:
            continue
        items.append(
            AdminGraphSummaryResponse(
                document_id=str(document.id),
                document_name=document.original_name,
                owner_id=str(owner.id),
                owner_username=owner.username,
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
            )
        )
    return AdminGraphListResponse(items=items, total=len(items))


@router.get(
    "/graphs/{document_id}",
    response_model=AdminGraphResponse,
    summary="Get a document knowledge graph for visualization",
)
def get_knowledge_graph(
    document_id: str,
    max_nodes: int = Query(default=250, ge=10, le=500),
    min_weight: int = Query(default=1, ge=1, le=1000),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """Return a bounded graph DTO without exposing persistence paths or chunks."""

    record = (
        db.query(Document, User)
        .join(User, User.id == Document.user_id)
        .filter(Document.id == document_id, Document.is_deleted.is_(False))
        .first()
    )
    if record is None:
        raise NotFoundException("Document")

    document, owner = record
    try:
        graph = load_graph(str(owner.id), str(document.id))
    except Exception as exc:
        logger.warning("Could not load knowledge graph %s: %s", document.id, exc)
        raise NotFoundException("Knowledge graph") from exc
    if graph is None:
        raise NotFoundException("Knowledge graph")

    ranked_node_ids = sorted(
        graph.nodes,
        key=lambda node_id: (
            int(graph.degree(node_id)),
            int(graph.nodes[node_id].get("mentions", 0)),
            str(graph.nodes[node_id].get("name", node_id)).casefold(),
        ),
        reverse=True,
    )
    selected_node_ids = ranked_node_ids[:max_nodes]
    selected = set(selected_node_ids)

    nodes = [
        AdminGraphNodeResponse(
            id=str(node_id),
            name=str(graph.nodes[node_id].get("name", node_id)),
            label=str(graph.nodes[node_id].get("label", "UNKNOWN")),
            mentions=max(0, int(graph.nodes[node_id].get("mentions", 0))),
            degree=int(graph.degree(node_id)),
            pages=_integer_pages(graph.nodes[node_id].get("pages", [])),
        )
        for node_id in selected_node_ids
    ]

    edges = []
    for source, target, data in graph.edges(data=True):
        weight = max(1, int(data.get("weight", 1)))
        if source not in selected or target not in selected or weight < min_weight:
            continue
        edges.append(
            AdminGraphEdgeResponse(
                id=f"{source}::{target}",
                source=str(source),
                target=str(target),
                weight=weight,
                pages=_integer_pages(data.get("pages", [])),
            )
        )
    edges.sort(key=lambda edge: edge.weight, reverse=True)

    return AdminGraphResponse(
        document_id=str(document.id),
        document_name=document.original_name,
        owner_id=str(owner.id),
        owner_username=owner.username,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        nodes=nodes,
        edges=edges,
        returned_node_count=len(nodes),
        returned_edge_count=len(edges),
        truncated=len(selected_node_ids) < graph.number_of_nodes(),
    )
