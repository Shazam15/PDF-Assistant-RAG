"""Background cleanup jobs for stale documents, and the shared document-purge
routine used both by those jobs and by the immediate hard-delete endpoint."""
import logging
import os
from datetime import datetime, timedelta, timezone

from app.database import get_db_session
from app.config import get_settings
from app.models import Document
from sqlalchemy import and_, or_

logger = logging.getLogger(__name__)
settings = get_settings()


def purge_document(doc: Document, db) -> None:
    """Permanently remove everything for one document: vector/BM25 index
    entries, the knowledge graph, derived relational memory (chunks, sections,
    profile, evidence), the uploaded file on disk, and finally the `Document`
    row itself.

    SQLite (the default backend here) never enforces the `ON DELETE CASCADE`
    declared on DocumentProfile/DocumentSection/DocumentChunk/DocumentEvidence's
    foreign keys — see the missing `PRAGMA foreign_keys` in database.py — so
    those rows are cleaned up explicitly instead of relying on the database to
    cascade them; without this they'd be orphaned forever.

    Each step is best-effort and independently logged: a failure in one
    subsystem (e.g. the vector store) must not stop the others from running,
    since the whole point is to leave nothing behind.

    Does not commit — the caller controls the transaction (either this
    module's own `get_db_session()` context manager, or the request-scoped
    session in the delete endpoint).
    """
    document_id = str(doc.id)
    user_id = str(doc.user_id)

    try:
        from app.rag.vectorstore import delete_document_chunks

        delete_document_chunks(document_id=document_id, user_id=user_id)
    except Exception as e:
        logger.warning("Error cleaning vectors for document %s: %s", document_id, e)

    try:
        from app.rag.graph_builder import delete_graph

        delete_graph(user_id=user_id, document_id=document_id)
    except Exception as e:
        logger.warning("Error deleting knowledge graph for document %s: %s", document_id, e)

    try:
        from app.models import DocumentEvidence, DocumentProfile, DocumentSection

        db.query(DocumentEvidence).filter(DocumentEvidence.document_id == document_id).delete(
            synchronize_session=False
        )
        db.query(DocumentSection).filter(DocumentSection.document_id == document_id).delete(
            synchronize_session=False
        )
        db.query(DocumentProfile).filter(DocumentProfile.document_id == document_id).delete(
            synchronize_session=False
        )
    except Exception as e:
        logger.warning("Error cleaning relational memory for document %s: %s", document_id, e)

    try:
        from app.models import ResearchRun

        db.query(ResearchRun).filter(ResearchRun.document_id == document_id).update(
            {"document_id": None}, synchronize_session=False
        )
    except Exception as e:
        logger.warning("Error detaching research runs for document %s: %s", document_id, e)

    try:
        # Keep the chat transcript (the user's own Q&A record); just drop the
        # now-dangling reference to the document being removed.
        from app.models import ChatMessage

        db.query(ChatMessage).filter(ChatMessage.document_id == document_id).update(
            {"document_id": None}, synchronize_session=False
        )
    except Exception as e:
        logger.warning("Error detaching chat history for document %s: %s", document_id, e)

    try:
        filepath = os.path.join(settings.UPLOAD_DIR, user_id, doc.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.warning("Error deleting file for document %s: %s", document_id, e)

    db.delete(doc)


def cleanup_stale_documents():
    """Mark documents stuck in 'processing' beyond the timeout as failed."""
    timeout_minutes = settings.DOC_PROCESSING_TIMEOUT_MINUTES
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)

    with get_db_session() as db:
        stale = (
            db.query(Document)
            .filter(
                Document.status.in_(("processing", "enriching")),
                Document.processing_started_at.isnot(None),
                or_(
                    Document.processing_updated_at < cutoff,
                    and_(
                        Document.processing_updated_at.is_(None),
                        Document.processing_started_at < cutoff,
                    ),
                ),
                Document.is_deleted.is_(False),
            )
            .all()
        )

        for doc in stale:
            logger.warning(
                "Recovering stale document %s (stuck at '%s' since %s)",
                doc.id,
                doc.processing_stage,
                doc.processing_started_at,
            )
            if doc.status == "enriching" and doc.searchable_at is not None:
                doc.status = "ready"
                doc.processing_progress = 100
                doc.processing_stage = "completed_with_warnings"
                doc.processing_warning = f"Enrichment timed out after {timeout_minutes} minutes"
                doc.completed_at = datetime.now(timezone.utc)
            else:
                doc.status = "failed"
                doc.processing_progress = 0
                doc.error_message = f"Processing timed out after {timeout_minutes} minutes"
                doc.last_error_traceback = "Timed out: no progress update received within the configured timeout window."

        if stale:
            logger.info("Marked %d stale document(s) as failed", len(stale))


def cleanup_old_deleted_documents():
    """Permanently delete documents soft-deleted beyond the max age."""
    max_age_days = settings.DOC_CLEANUP_MAX_AGE_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    with get_db_session() as db:
        old = (
            db.query(Document)
            .filter(
                Document.is_deleted.is_(True),
                Document.deleted_at.isnot(None),
                Document.deleted_at < cutoff,
            )
            .all()
        )

        for doc in old:
            logger.info(
                "Purging old deleted document %s ('%s', deleted %s)",
                doc.id,
                doc.original_name,
                doc.deleted_at,
            )
            purge_document(doc, db)

        if old:
            logger.info("Permanently deleted %d old document(s)", len(old))
