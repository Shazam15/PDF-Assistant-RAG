"""Reusable document ingestion pipeline."""
import traceback
import logging
from datetime import datetime, timezone

from app.models import Document
from app.rag.chunker import chunk_document, get_page_count
from app.rag.vectorstore import store_chunks
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


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
                    document.status = "processing"
                    document.processing_stage = "embedding"
                    document.processing_progress = 70
                    document.error_message = None
                    db.commit()

                    chunks = load_bm25_chunks(user_id, str(document.id))
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


def _update_progress(document_id: str, progress: int, stage: str, error: str = None):
    """Update document progress fields in the database."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.processing_progress = progress
            doc.processing_stage = stage
            if error:
                doc.error_message = error
            db.commit()
    except Exception as e:
        logger.warning("Failed to update progress for %s: %s", document_id, e)
    finally:
        db.close()


def ingest_document(document_id: str, filepath: str, original_name: str, user_id: str):
    """
    Process a document: chunk it, generate embeddings, store vectors, summarize,
    and update the database record.
    """
    from app.database import SessionLocal

    db = SessionLocal()
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
        doc.error_message = None
        doc.last_error_traceback = None
        db.commit()

        page_count = get_page_count(filepath)
        doc.page_count = page_count
        doc.processing_progress = 20
        db.commit()

        try:
            chunk_kwargs = {}
            if doc.chunk_size is not None:
                chunk_kwargs["chunk_size"] = doc.chunk_size
            if doc.chunk_overlap is not None:
                chunk_kwargs["chunk_overlap"] = doc.chunk_overlap
            doc.processing_stage = "chunking"
            doc.processing_progress = 30
            db.commit()
            chunks = chunk_document(filepath, **chunk_kwargs)
        except TypeError:
            chunks = chunk_document(filepath)

        if not chunks:
            doc.status = "failed"
            doc.processing_progress = 0
            doc.error_message = "No text could be extracted from the document"
            db.commit()
            return

        doc.processing_progress = 50
        doc.processing_stage = "indexing"
        db.commit()

        try:
            from app.rag.graph_builder import build_graph, save_graph

            graph = build_graph(chunks)
            save_graph(graph, user_id=user_id, document_id=document_id)
        except Exception as e:
            logger.warning("Could not build knowledge graph for document %s: %s", document_id, e)

        doc.processing_progress = 70
        doc.processing_stage = "embedding"
        db.commit()

        chunk_count = store_chunks(
            chunks=chunks,
            document_id=document_id,
            filename=original_name,
            user_id=user_id,
        )

        doc.processing_progress = 85
        db.commit()

        try:
            from app.rag.summarizer import build_document_memory

            doc.processing_stage = "memory"
            doc.processing_progress = 88
            db.commit()
            memory = build_document_memory(chunks)
            _persist_document_memory(db, doc, chunks, memory, original_name, user_id)
        except Exception as e:
            logger.warning("Could not generate structured memory for document %s: %s", document_id, e)

        # ── URL extraction pass (PDF only) ────────────────────────────────
        ext = filepath.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            try:
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
        # ── End URL extraction pass ───────────────────────────────────────

        doc.chunk_count = chunk_count
        doc.status = "ready"
        doc.processing_progress = 100
        doc.processing_stage = "completed"
        doc.completed_at = datetime.now(timezone.utc)
        doc.error_message = None
        db.commit()

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
                doc.status = "failed"
                doc.processing_progress = 0
                doc.error_message = str(e)[:500]
                doc.last_error_traceback = traceback.format_exc()[:2000]
                db.commit()
        except Exception:
            logger.exception("Failed to mark document %s as failed", document_id)
    finally:
        db.close()
