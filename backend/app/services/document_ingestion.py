"""Reusable document ingestion pipeline."""
import traceback
import logging
import time
import threading
from datetime import datetime, timezone

from app.models import Document
from app.rag.chunker import chunk_document, get_page_count
from app.rag.vectorstore import store_chunks
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _load_relational_chunks(db, document_id: str):
    """Reuse persisted neutral chunks before reparsing an uploaded document."""
    from app.models import DocumentChunk

    rows = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id,
    ).order_by(DocumentChunk.chunk_index).all()
    return [
        {
            "text": row.text,
            "page": row.page,
            "page_start": row.page_start,
            "page_end": row.page_end,
            "chunk_index": row.chunk_index,
            "chunk_type": row.chunk_type,
            "parent_id": row.parent_id,
            "parent_text": row.parent_text,
            "section_id": row.section_id,
            "section_title": row.section_title,
            "token_count": row.token_count,
            "bbox": row.bbox,
            "table_index": row.table_index,
        }
        for row in rows
    ]


def _persist_document_memory(
    db,
    document: Document,
    chunks,
    memory,
    original_name: str,
    user_id: str,
) -> None:
    """Replace one document's derived memory atomically after indexing succeeds."""
    from app.models import DocumentEvidence, DocumentProfile, DocumentSection

    document_id = str(document.id)
    db.query(DocumentEvidence).filter(DocumentEvidence.document_id == document_id).delete(synchronize_session=False)
    db.query(DocumentSection).filter(DocumentSection.document_id == document_id).delete(synchronize_session=False)
    db.query(DocumentProfile).filter(DocumentProfile.document_id == document_id).delete(synchronize_session=False)

    profile_text = "\n".join(filter(None, [
        original_name,
        memory.get("summary"),
        memory.get("methodology"),
        memory.get("findings"),
        memory.get("limitations"),
        "; ".join(section.get("title") or "" for section in memory.get("sections") or []),
    ]))
    profile_embedding = None
    if profile_text:
        try:
            from app.rag.embeddings import embed_texts

            profile_embedding = embed_texts([profile_text])[0]
        except Exception as exc:
            logger.warning("Could not embed document profile %s: %s", document_id, exc)

    db.add(DocumentProfile(
        document_id=document_id,
        user_id=user_id,
        title=original_name,
        summary=memory.get("summary"),
        methodology=memory.get("methodology"),
        findings=memory.get("findings"),
        limitations=memory.get("limitations"),
        embedding=profile_embedding,
        section_count=len(memory.get("sections") or []),
        index_version=settings.EMBEDDING_INDEX_VERSION,
    ))
    for section in memory.get("sections") or []:
        db.add(DocumentSection(
            id=f"{document_id}_{section['id']}",
            document_id=document_id,
            user_id=user_id,
            title=section.get("title") or "Document",
            section_path=section.get("title") or "Document",
            page_start=section.get("page_start") or 1,
            page_end=section.get("page_end") or section.get("page_start") or 1,
            summary=section.get("summary"),
            text=section.get("text") or "",
        ))

    chunks_by_index = {int(chunk.get("chunk_index") or 0): chunk for chunk in chunks}
    for evidence in memory.get("evidence") or []:
        chunk_index = int(evidence.get("chunk_index") or 0)
        chunk = chunks_by_index.get(chunk_index)
        quote = str(evidence.get("exact_quote") or "")
        if not chunk or not quote or quote not in str(chunk.get("text") or ""):
            continue
        db.add(DocumentEvidence(
            document_id=document_id,
            user_id=user_id,
            chunk_id=f"{document_id}_{chunk_index}",
            evidence_kind=evidence.get("evidence_kind") or "context",
            claim=evidence.get("claim") or quote,
            exact_quote=quote,
            page=int(chunk.get("page") or 1),
            section_title=chunk.get("section_title") or chunk.get("section"),
            confidence=1.0,
        ))
    document.summary = memory.get("summary") or document.summary
    db.commit()


def migrate_embedding_indexes() -> None:
    """Re-embed existing documents when the configured embedding index changes."""
    import os
    from collections import defaultdict
    from sqlalchemy import or_
    from app.database import SessionLocal
    from app.rag.bm25 import load_bm25_chunks
    from app.rag.embeddings import get_embedding_model
    from app.rag.vectorstore import (
        finish_user_index_migration,
        document_index_is_current,
        start_user_index_migration,
        user_index_needs_migration,
    )

    db = SessionLocal()
    try:
        documents = db.query(Document).filter(
            Document.is_deleted.is_(False),
            or_(Document.status == "ready", Document.processing_stage == "embedding"),
        ).all()
        by_user = defaultdict(list)
        for document in documents:
            by_user[str(document.user_id)].append(document)

        users_to_migrate = {
            user_id: user_documents
            for user_id, user_documents in by_user.items()
            if user_index_needs_migration(user_id)
        }
        if not users_to_migrate:
            return

        # Do not remove a working legacy index until the replacement model is available.
        get_embedding_model()

        for user_id, user_documents in users_to_migrate.items():
            logger.info(
                "Migrating %d documents for user %s to embedding index %s",
                len(user_documents),
                user_id,
                settings.EMBEDDING_INDEX_VERSION,
            )
            start_user_index_migration(user_id)
            user_failed = False
            for document in user_documents:
                filepath = os.path.join(settings.UPLOAD_DIR, user_id, document.filename)
                try:
                    if document.status == "ready" and document_index_is_current(
                        user_id, str(document.id), int(getattr(document, "chunk_count", 0) or 0)
                    ):
                        logger.info("Document %s already uses the current vector index; skipping", document.id)
                        continue
                    document.status = "processing"
                    document.processing_stage = "embedding"
                    document.processing_progress = 70
                    document.error_message = None
                    db.commit()

                    chunks = load_bm25_chunks(user_id, str(document.id))
                    if not chunks:
                        chunks = _load_relational_chunks(db, str(document.id))
                    if not chunks:
                        chunks = chunk_document(
                            filepath,
                            **{
                                key: value
                                for key, value in {
                                    "chunk_size": document.chunk_size,
                                    "chunk_overlap": document.chunk_overlap,
                                }.items()
                                if value is not None
                            },
                        )
                    if not chunks:
                        raise ValueError("No chunks available for embedding migration")

                    document.chunk_count = store_chunks(
                        chunks=chunks,
                        document_id=str(document.id),
                        filename=document.original_name,
                        user_id=user_id,
                    )
                    from app.rag.summarizer import build_document_memory

                    memory = build_document_memory(chunks, use_llm=False)
                    if document.summary:
                        memory["summary"] = document.summary
                    _persist_document_memory(
                        db,
                        document,
                        chunks,
                        memory,
                        document.original_name,
                        user_id,
                    )
                    document.status = "ready"
                    document.processing_stage = "completed"
                    document.processing_progress = 100
                    document.completed_at = datetime.now(timezone.utc)
                    db.commit()
                except Exception as exc:
                    user_failed = True
                    # Keep interrupted migrations discoverable by the next
                    # startup instead of turning a transient error permanent.
                    document.status = "processing"
                    document.processing_stage = "embedding"
                    document.processing_progress = 70
                    document.error_message = f"Embedding migration failed: {exc}"[:1000]
                    db.commit()
                    logger.exception("Embedding migration failed for document %s", document.id)
            if not user_failed:
                finish_user_index_migration(user_id)
                logger.info("Embedding migration completed for user %s", user_id)
    finally:
        db.close()


_STAGE_RANGES = {
    "queued": (0, 0),
    "extracting": (5, 25),
    "extracting_layout": (5, 25),
    "extracting_ocr": (5, 25),
    "chunking": (25, 40),
    "embedding": (40, 80),
    "persisting": (80, 85),
    "searchable": (85, 85),
    "summarizing": (85, 92),
    "building_memory": (92, 96),
    "building_graph": (96, 99),
    "finalizing": (99, 99),
    "completed": (100, 100),
    "completed_with_warnings": (100, 100),
}


def _stage_progress(stage: str, current: int = None, total: int = None) -> int:
    start, end = _STAGE_RANGES.get(stage, (0, 99))
    if current is None or not total or total <= 0:
        return start
    ratio = max(0.0, min(1.0, current / total))
    return round(start + ((end - start) * ratio))


def _update_progress(
    document_id: str,
    progress: int,
    stage: str,
    current: int = None,
    total: int = None,
    error: str = None,
    warning: str = None,
):
    """Persist a monotonic progress update and processing heartbeat."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            existing = int(doc.processing_progress or 0)
            doc.processing_progress = max(existing, min(100, max(0, int(progress))))
            doc.processing_stage = stage
            doc.processing_current = current
            doc.processing_total = total
            doc.processing_updated_at = datetime.now(timezone.utc)
            if error:
                doc.error_message = error
            if warning:
                doc.processing_warning = warning
            db.commit()
    except Exception as e:
        logger.warning("Failed to update progress for %s: %s", document_id, e)
    finally:
        db.close()


class ProgressReporter:
    """Throttle database writes while retaining stage changes and heartbeats."""

    def __init__(
        self,
        document_id: str,
        minimum_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
    ):
        self.document_id = document_id
        self.minimum_interval = minimum_interval
        self.heartbeat_interval = heartbeat_interval
        self.last_stage = None
        self.last_progress = -1
        self.last_update = 0.0
        self.current = None
        self.total = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread = None

    def start(self) -> None:
        if self._heartbeat_thread is not None:
            return

        def heartbeat() -> None:
            while not self._stop_event.wait(self.heartbeat_interval):
                with self._lock:
                    stage = self.last_stage
                    progress = self.last_progress
                    current = self.current
                    total = self.total
                if stage is not None and progress >= 0:
                    _update_progress(self.document_id, progress, stage, current=current, total=total)

        self._heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"ingestion-progress-{self.document_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1)

    def update(
        self,
        stage: str,
        current: int = None,
        total: int = None,
        *,
        force: bool = False,
        warning: str = None,
    ) -> None:
        with self._lock:
            progress = max(self.last_progress, _stage_progress(stage, current, total))
            now = time.monotonic()
            stage_changed = stage != self.last_stage
            if not force and not stage_changed and progress == self.last_progress and now - self.last_update < 5:
                return
            if not force and not stage_changed and now - self.last_update < self.minimum_interval:
                return
            _update_progress(
                self.document_id,
                progress,
                stage,
                current=current,
                total=total,
                warning=warning,
            )
            self.last_stage = stage
            self.last_progress = progress
            self.last_update = now
            self.current = current
            self.total = total


def ingest_document(document_id: str, filepath: str, original_name: str, user_id: str):
    """
    Process a document: chunk it, generate embeddings, store vectors, summarize,
    and update the database record.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    reporter = ProgressReporter(document_id)
    reporter.start()
    became_searchable = False
    try:
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted.is_(False),
        ).first()
        if not doc:
            logger.error("Document %s not found for ingestion", document_id)
            return

        doc.status = "processing"
        doc.processing_stage = "extracting"
        doc.processing_progress = 10
        doc.processing_current = 0
        doc.processing_total = None
        doc.processing_started_at = doc.processing_started_at or datetime.now(timezone.utc)
        doc.processing_updated_at = datetime.now(timezone.utc)
        doc.processing_warning = None
        doc.searchable_at = None
        doc.error_message = None
        doc.last_error_traceback = None
        db.commit()

        page_count = get_page_count(filepath)
        doc.page_count = page_count
        db.commit()
        reporter.update("extracting", 0, page_count, force=True)
        db.expire_all()
        doc = db.query(Document).filter(Document.id == document_id).first()

        def chunk_progress(stage, current, total):
            reporter.update(stage, current, total)

        try:
            chunk_kwargs = {}
            if doc.chunk_size is not None:
                chunk_kwargs["chunk_size"] = doc.chunk_size
            if doc.chunk_overlap is not None:
                chunk_kwargs["chunk_overlap"] = doc.chunk_overlap
            chunks = chunk_document(filepath, progress_callback=chunk_progress, **chunk_kwargs)
        except TypeError:
            chunks = chunk_document(filepath)

        if not chunks:
            doc.status = "failed"
            doc.processing_progress = 0
            doc.error_message = "No text could be extracted from the document"
            db.commit()
            return

        reporter.update("chunking", page_count, page_count, force=True)
        reporter.update("embedding", 0, len(chunks), force=True)

        chunk_count = store_chunks(
            chunks=chunks,
            document_id=document_id,
            filename=original_name,
            user_id=user_id,
            progress_callback=lambda stage, current, total: reporter.update(stage, current, total),
        )

        db.expire_all()
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.chunk_count = chunk_count
        doc.status = "enriching"
        doc.searchable_at = datetime.now(timezone.utc)
        doc.processing_progress = 85
        doc.processing_stage = "searchable"
        doc.processing_current = chunk_count
        doc.processing_total = chunk_count
        doc.processing_updated_at = datetime.now(timezone.utc)
        db.commit()
        became_searchable = True
        reporter.update("searchable", chunk_count, chunk_count, force=True)

        from app.rag import embeddings as embeddings_module

        release_embedding_model = getattr(embeddings_module, "release_embedding_model", lambda: None)
        release_embedding_model()
        warnings = []

        try:
            from app.rag.summarizer import build_document_memory

            reporter.update("summarizing", 0, 1, force=True)
            memory = build_document_memory(
                chunks,
                progress_callback=lambda stage, current, total: reporter.update(stage, current, total),
            )
            _persist_document_memory(db, doc, chunks, memory, original_name, user_id)
        except Exception as e:
            logger.warning("Could not generate structured memory for document %s: %s", document_id, e)
            warnings.append(f"Structured memory: {str(e)[:160]}")
        finally:
            release_embedding_model()

        try:
            reporter.update("building_graph", 0, len(chunks), force=True)
            from app.rag.graph_builder import build_graph, save_graph

            graph = build_graph(chunks)
            save_graph(graph, user_id=user_id, document_id=document_id)
            reporter.update("building_graph", len(chunks), len(chunks), force=True)
        except Exception as e:
            logger.warning("Could not build knowledge graph for document %s: %s", document_id, e)
            warnings.append(f"Knowledge graph: {str(e)[:160]}")

        # ── URL extraction pass (PDF only) ────────────────────────────────
        ext = filepath.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            try:
                reporter.update("finalizing", 0, 1, force=True)
                from app.rag.url_extractor import extract_urls_from_pdf
                import json

                urls = extract_urls_from_pdf(filepath)
                doc.extracted_urls = json.dumps(urls) if urls else None
                db.commit()
                logger.info(
                    "Extracted %s URLs from document %s",
                    len(urls),
                    document_id,
                )
            except Exception as exc:
                logger.warning(
                    "URL extraction failed for document %s: %s",
                    document_id,
                    exc,
                )
                warnings.append(f"URL extraction: {str(exc)[:160]}")
        # ── End URL extraction pass ───────────────────────────────────────

        db.expire_all()
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "ready"
        doc.processing_progress = 100
        doc.processing_stage = "completed_with_warnings" if warnings else "completed"
        doc.processing_current = None
        doc.processing_total = None
        doc.processing_updated_at = datetime.now(timezone.utc)
        doc.processing_warning = "; ".join(warnings) if warnings else None
        doc.completed_at = datetime.now(timezone.utc)
        doc.error_message = None
        db.commit()
        reporter.update(doc.processing_stage, force=True, warning=doc.processing_warning)

        logger.info(
            "Document %s ingested: %s pages, %s chunks",
            document_id,
            page_count,
            chunk_count,
        )

    except Exception as e:
        logger.error("Ingestion error for %s: %s", document_id, e)
        db.rollback()
        try:
            doc = db.query(Document).filter(
                Document.id == document_id,
                Document.is_deleted.is_(False),
            ).first()
            if doc:
                if became_searchable or doc.searchable_at:
                    doc.status = "ready"
                    doc.processing_progress = 100
                    doc.processing_stage = "completed_with_warnings"
                    doc.processing_warning = f"Enrichment interrupted: {str(e)[:300]}"
                    doc.completed_at = datetime.now(timezone.utc)
                else:
                    doc.status = "failed"
                    doc.processing_progress = 0
                    doc.error_message = str(e)[:500]
                doc.last_error_traceback = traceback.format_exc()[:2000]
                doc.processing_updated_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            logger.exception("Failed to mark document %s as failed", document_id)
    finally:
        reporter.stop()
        db.close()
