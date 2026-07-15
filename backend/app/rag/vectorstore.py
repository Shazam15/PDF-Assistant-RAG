"""
ChromaDB vector store operations.
Per-user collections for data isolation.
"""
import logging
import json
from typing import List, Dict, Any, Optional
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _chunk_payload(chunk: Dict[str, Any], document_id: str, filename: str, user_id: str) -> Dict[str, Any]:
    section = chunk.get("section_title") or chunk.get("section")
    context_labels = [filename]
    if section:
        context_labels.append(str(section))
    if chunk.get("chunk_type") == "table":
        table_index = chunk.get("table_index")
        context_labels.append(f"Table {int(table_index) + 1}" if isinstance(table_index, int) else "Table")
    search_text = "\n".join([*context_labels, chunk["text"]]).strip()
    return {
        "id": f"{document_id}_{chunk['chunk_index']}",
        "text": chunk["text"],
        "search_text": search_text,
        "filename": filename,
        "document_id": str(document_id),
        "user_id": str(user_id),
        "page": int(chunk.get("page") or 1),
        "page_start": int(chunk.get("page_start") or chunk.get("page") or 1),
        "page_end": int(chunk.get("page_end") or chunk.get("page") or 1),
        "chunk_index": int(chunk["chunk_index"]),
        "chunk_type": chunk.get("chunk_type", "text"),
        "parent_id": chunk.get("parent_id"),
        "parent_text": chunk.get("parent_text") or chunk["text"],
        "section_id": chunk.get("section_id"),
        "section": section,
        "token_count": int(chunk.get("token_count") or 0),
        "bbox": chunk.get("bbox", ""),
        "table_index": chunk.get("table_index"),
        "is_image": bool(chunk.get("is_image")),
        "image_caption": chunk.get("image_caption", ""),
    }


def _persist_relational_chunks(payloads: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
    """Persist neutral chunk metadata for PostgreSQL and the SQLite FTS fallback."""
    from app.database import SessionLocal, engine
    from app.models import DocumentChunk
    from sqlalchemy import text

    if not payloads:
        return
    db = SessionLocal()
    try:
        document_id = payloads[0]["document_id"]
        user_id = payloads[0]["user_id"]
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.user_id == user_id,
        ).delete(synchronize_session=False)
        for payload, embedding in zip(payloads, embeddings):
            db.add(DocumentChunk(
                id=payload["id"],
                document_id=document_id,
                user_id=user_id,
                filename=payload["filename"],
                chunk_index=payload["chunk_index"],
                parent_id=payload.get("parent_id"),
                section_id=payload.get("section_id"),
                section_title=payload.get("section"),
                text=payload["text"],
                parent_text=payload.get("parent_text"),
                page=payload["page"],
                page_start=payload["page_start"],
                page_end=payload["page_end"],
                token_count=payload["token_count"],
                chunk_type=payload["chunk_type"],
                bbox=payload.get("bbox"),
                table_index=payload.get("table_index"),
                embedding=embedding if settings.CORPUS_STORE_BACKEND == "postgres" else None,
                search_text=payload["search_text"],
            ))
        db.commit()

        if engine.dialect.name == "sqlite":
            with engine.begin() as connection:
                connection.execute(text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5("
                    "chunk_id UNINDEXED, user_id UNINDEXED, document_id UNINDEXED, filename UNINDEXED, text)"
                ))
                connection.execute(
                    text("DELETE FROM document_chunks_fts WHERE user_id=:user_id AND document_id=:document_id"),
                    {"user_id": user_id, "document_id": document_id},
                )
                connection.execute(
                    text(
                        "INSERT INTO document_chunks_fts(chunk_id,user_id,document_id,filename,text) "
                        "VALUES(:id,:user_id,:document_id,:filename,:search_text)"
                    ),
                    payloads,
                )
    except Exception as exc:
        db.rollback()
        logger.warning("Could not persist relational chunk memory: %s", exc)
    finally:
        db.close()

# ── Singleton ChromaDB client ────────────────────────
_chroma_client = None


def get_chroma_client():
    """Get or create persistent ChromaDB client."""
    global _chroma_client

    if _chroma_client is None:
        import os
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB initialized at {settings.CHROMA_PERSIST_DIR}")

    return _chroma_client


def get_collection_name(user_id: str) -> str:
    """Generate a valid collection name for a user."""
    # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
    clean_id = user_id.replace("-", "_")
    name = f"user_{clean_id}"
    # Truncate if too long
    return name[:63]


def _index_metadata(state: str = "ready") -> Dict[str, str]:
    return {
        "hnsw:space": "cosine",
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_index_version": settings.EMBEDDING_INDEX_VERSION,
        "embedding_index_state": state,
    }


def user_index_needs_migration(user_id: str) -> bool:
    """Return whether a user's vector collection is absent, stale, or incomplete."""
    if settings.CORPUS_STORE_BACKEND == "postgres":
        from app.database import SessionLocal
        from app.models import DocumentProfile

        db = SessionLocal()
        try:
            profiles = db.query(DocumentProfile).filter(DocumentProfile.user_id == user_id).all()
            return not profiles or any(profile.index_version != settings.EMBEDDING_INDEX_VERSION for profile in profiles)
        finally:
            db.close()
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=get_collection_name(user_id))
    except Exception:
        return True
    metadata = collection.metadata or {}
    return (
        metadata.get("embedding_model") != settings.EMBEDDING_MODEL
        or metadata.get("embedding_index_version") != settings.EMBEDDING_INDEX_VERSION
        or metadata.get("embedding_index_state") != "ready"
    )


def start_user_index_migration(user_id: str) -> None:
    """Reset only vectors and mark the replacement index as incomplete."""
    if settings.CORPUS_STORE_BACKEND == "postgres":
        return
    client = get_chroma_client()
    collection_name = get_collection_name(user_id)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    client.get_or_create_collection(name=collection_name, metadata=_index_metadata("migrating"))


def finish_user_index_migration(user_id: str) -> None:
    if settings.CORPUS_STORE_BACKEND == "postgres":
        return
    collection = get_chroma_client().get_collection(name=get_collection_name(user_id))
    collection.modify(metadata=_index_metadata("ready"))


def store_chunks(
    chunks: List[Dict[str, Any]],
    document_id: str,
    filename: str,
    user_id: str,
) -> int:
    """
    Embed and store document chunks in ChromaDB, and build a local BM25 index.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    # Delete existing chunks for this document and user before inserting new ones
    # to avoid stale/orphaned chunks in ChromaDB and BM25.
    delete_document_chunks(document_id, user_id)

    # Generate captions for any extracted images before embedding
    try:
        from app.rag.vision import generate_captions_for_chunks

        generate_captions_for_chunks(chunks)
    except Exception as e:
        logger.warning(f"Could not generate image captions: {e}")

    from app.rag.embeddings import embed_texts

    # ── Prepare batch data ───────────────────────────
    payloads = [_chunk_payload(chunk, document_id, filename, user_id) for chunk in chunks]
    texts = [payload["search_text"] for payload in payloads]
    ids = [payload["id"] for payload in payloads]
    metadatas = [{key: value for key, value in payload.items() if key not in {"id", "user_id", "search_text"} and value is not None}
                 for payload in payloads]

    embeddings: List[List[float]] = []
    for i in range(0, len(texts), 50):
        embeddings.extend(embed_texts(texts[i:i + 50]))
    _persist_relational_chunks(payloads, embeddings)

    if settings.CORPUS_STORE_BACKEND == "postgres":
        logger.info("Stored %d hierarchical chunks in PostgreSQL for document %s", len(payloads), document_id)
        return len(payloads)

    client = get_chroma_client()
    collection_name = get_collection_name(user_id)
    collection = client.get_or_create_collection(name=collection_name, metadata=_index_metadata("ready"))

    # ── Embed and upsert in batches ──────────────────
    batch_size = 50
    total_stored = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        collection.add(
            ids=batch_ids,
            embeddings=embeddings[i:i + len(batch_texts)],
            metadatas=batch_metadatas,
            documents=batch_texts,
        )
        total_stored += len(batch_texts)

    logger.info(f"Stored {total_stored} chunks for document {document_id}")
    return total_stored


def query_chunks(
    query_embedding: List[float],
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Query ChromaDB for relevant chunks.
    Returns list of dicts with text, metadata, and distance.
    """
    if settings.CORPUS_STORE_BACKEND == "postgres":
        return _query_postgres_vectors(query_embedding, user_id, document_id, document_ids, top_k)

    client = get_chroma_client()
    collection_name = get_collection_name(user_id)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        logger.warning(f"Collection {collection_name} not found")
        return []

    # ── Build filter ─────────────────────────────────
    where_filter = None
    if document_id:
        where_filter = {"document_id": {"$eq": document_id}}
    elif document_ids:
        where_filter = {"document_id": {"$in": document_ids}}

    # ── Query ────────────────────────────────────────
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # ── Format results ───────────────────────────────
    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0

            # Convert cosine distance to similarity score (0-1)
            similarity = 1 - distance

            chunks.append({
                "text": metadata.get("text", doc),
                "filename": metadata.get("filename", ""),
                "document_id": metadata.get("document_id", ""),
                "page": metadata.get("page", 1),
                "chunk_index": metadata.get("chunk_index"),
                "chunk_type": metadata.get("chunk_type", "text"),
                "bbox": metadata.get("bbox", ""),
                "parent_id": metadata.get("parent_id"),
                "parent_text": metadata.get("parent_text", metadata.get("text", doc)),
                "section_id": metadata.get("section_id"),
                "section": metadata.get("section"),
                "page_start": metadata.get("page_start", metadata.get("page", 1)),
                "page_end": metadata.get("page_end", metadata.get("page", 1)),
                "token_count": metadata.get("token_count", 0),
                "score": round(similarity, 4),
            })

    return chunks


def _row_to_chunk(row: Any) -> Dict[str, Any]:
    data = dict(row._mapping if hasattr(row, "_mapping") else row)
    return {
        "id": str(data.get("id") or data.get("chunk_id") or ""),
        "text": data.get("text", ""),
        "filename": data.get("filename", ""),
        "document_id": str(data.get("document_id") or ""),
        "page": int(data.get("page") or 1),
        "page_start": int(data.get("page_start") or data.get("page") or 1),
        "page_end": int(data.get("page_end") or data.get("page") or 1),
        "chunk_index": data.get("chunk_index"),
        "chunk_type": data.get("chunk_type") or "text",
        "parent_id": data.get("parent_id"),
        "parent_text": data.get("parent_text") or data.get("text", ""),
        "section_id": data.get("section_id"),
        "section": data.get("section_title"),
        "bbox": data.get("bbox") or "",
        "table_index": data.get("table_index"),
        "score": float(data.get("score") or 0.0),
    }


def _query_postgres_vectors(
    query_embedding: List[float],
    user_id: str,
    document_id: Optional[str],
    document_ids: Optional[List[str]],
    top_k: int,
) -> List[Dict[str, Any]]:
    from app.database import engine
    from sqlalchemy import text

    filters = ["user_id = CAST(:user_id AS uuid)"]
    params: Dict[str, Any] = {"user_id": str(user_id), "embedding": str(query_embedding), "top_k": top_k}
    if document_id:
        filters.append("document_id = CAST(:document_id AS uuid)")
        params["document_id"] = str(document_id)
    elif document_ids:
        filters.append("document_id = ANY(CAST(:document_ids AS uuid[]))")
        params["document_ids"] = "{" + ",".join(map(str, document_ids)) + "}"
    statement = text(
        "SELECT id,text,filename,document_id,page,page_start,page_end,chunk_index,chunk_type,parent_id,"
        "parent_text,section_id,section_title,bbox,table_index,1-(embedding <=> CAST(:embedding AS vector)) AS score "
        f"FROM document_chunks WHERE {' AND '.join(filters)} AND embedding IS NOT NULL "
        "ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT :top_k"
    )
    with engine.connect() as connection:
        return [_row_to_chunk(row) for row in connection.execute(statement, params)]


def query_lexical_chunks(
    query: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """Run one global lexical ranking over the user's corpus."""
    from app.database import engine
    from sqlalchemy import text

    params = {
        "query": query,
        "user_id": str(user_id),
        "document_id": str(document_id or ""),
        "document_ids": "{" + ",".join(map(str, document_ids or [])) + "}",
        "top_k": top_k,
    }
    if engine.dialect.name == "postgresql":
        scope = "AND document_id=CAST(:document_id AS uuid)" if document_id else (
            "AND document_id=ANY(CAST(:document_ids AS uuid[]))" if document_ids else ""
        )
        statement = text(
            "WITH q AS (SELECT websearch_to_tsquery('simple', :query) AS value) "
            "SELECT id,text,filename,document_id,page,page_start,page_end,chunk_index,chunk_type,parent_id,"
            "parent_text,section_id,section_title,bbox,table_index,"
            "ts_rank_cd(to_tsvector('simple', COALESCE(search_text,text)), q.value) AS score "
            "FROM document_chunks,q WHERE user_id=CAST(:user_id AS uuid) " + scope +
            " AND to_tsvector('simple', COALESCE(search_text,text)) @@ q.value "
            "ORDER BY score DESC LIMIT :top_k"
        )
        with engine.connect() as connection:
            return [_row_to_chunk(row) for row in connection.execute(statement, params)]

    if document_id:
        scope = "AND f.document_id=:document_id"
    elif document_ids:
        placeholders = []
        for index, candidate_id in enumerate(document_ids):
            key = f"scope_id_{index}"
            params[key] = str(candidate_id)
            placeholders.append(f":{key}")
        scope = f"AND f.document_id IN ({','.join(placeholders)})"
    else:
        scope = ""
    try:
        statement = text(
            "SELECT c.*, -bm25(document_chunks_fts) AS score FROM document_chunks_fts f "
            "JOIN document_chunks c ON c.id=f.chunk_id WHERE document_chunks_fts MATCH :query "
            "AND f.user_id=:user_id " + scope + " ORDER BY bm25(document_chunks_fts) LIMIT :top_k"
        )
        with engine.connect() as connection:
            return [_row_to_chunk(row) for row in connection.execute(statement, params)]
    except Exception as exc:
        logger.warning("SQLite FTS query failed: %s", exc)
        return []


def query_document_profiles(
    query_embedding: List[float],
    user_id: str,
    document_id: Optional[str] = None,
    top_k: int = 20,
) -> List[str]:
    """Select semantically relevant documents from their learned corpus profiles."""
    from app.database import SessionLocal, engine
    from app.models import DocumentProfile

    db = SessionLocal()
    try:
        query = db.query(DocumentProfile).filter(DocumentProfile.user_id == str(user_id))
        if document_id:
            query = query.filter(DocumentProfile.document_id == str(document_id))
        profiles = query.all()
    finally:
        db.close()
    if not profiles:
        return []

    if engine.dialect.name == "postgresql":
        from sqlalchemy import text

        scope = "AND document_id=CAST(:document_id AS uuid)" if document_id else ""
        statement = text(
            "SELECT document_id FROM document_profiles WHERE user_id=CAST(:user_id AS uuid) " + scope +
            " AND embedding IS NOT NULL ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT :top_k"
        )
        params = {
            "user_id": str(user_id),
            "document_id": str(document_id or ""),
            "embedding": str(query_embedding),
            "top_k": top_k,
        }
        with engine.connect() as connection:
            return [str(row[0]) for row in connection.execute(statement, params)]

    import math

    def cosine(profile) -> float:
        vector = profile.embedding or []
        if not vector or len(vector) != len(query_embedding):
            return float("-inf")
        dot = sum(float(a) * float(b) for a, b in zip(vector, query_embedding))
        norm_a = math.sqrt(sum(float(value) ** 2 for value in vector))
        norm_b = math.sqrt(sum(float(value) ** 2 for value in query_embedding))
        return dot / (norm_a * norm_b) if norm_a and norm_b else float("-inf")

    ranked = sorted(profiles, key=cosine, reverse=True)
    return [str(profile.document_id) for profile in ranked[:top_k] if cosine(profile) != float("-inf")]


def delete_document_chunks(document_id: str, user_id: str):
    """Delete all chunks for a specific document."""
    from app.database import SessionLocal, engine
    from app.models import DocumentChunk
    from sqlalchemy import text

    db = SessionLocal()
    try:
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == str(document_id),
            DocumentChunk.user_id == str(user_id),
        ).delete(synchronize_session=False)
        db.commit()
        if engine.dialect.name == "sqlite":
            with engine.begin() as connection:
                connection.execute(
                    text("DELETE FROM document_chunks_fts WHERE user_id=:user_id AND document_id=:document_id"),
                    {"user_id": str(user_id), "document_id": str(document_id)},
                )
    except Exception as exc:
        db.rollback()
        logger.debug("Relational chunk cleanup skipped: %s", exc)
    finally:
        db.close()

    if settings.CORPUS_STORE_BACKEND == "postgres":
        return
    client = get_chroma_client()
    collection_name = get_collection_name(user_id)

    try:
        from app.rag.bm25 import delete_bm25_index
        delete_bm25_index(document_id, user_id)
    except Exception as e:
        logger.warning(f"Error deleting BM25 index: {e}")

    try:
        collection = client.get_collection(name=collection_name)
        # Get all IDs for this document
        results = collection.get(
            where={"document_id": {"$eq": document_id}},
            include=[],
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
    except Exception as e:
        logger.warning(f"Error deleting chunks: {e}")


def delete_user_collection(user_id: str):
    """Delete entire collection for a user."""
    from app.database import SessionLocal, engine
    from app.models import DocumentChunk
    from sqlalchemy import text

    db = SessionLocal()
    try:
        db.query(DocumentChunk).filter(DocumentChunk.user_id == str(user_id)).delete(synchronize_session=False)
        db.commit()
        if engine.dialect.name == "sqlite":
            with engine.begin() as connection:
                connection.execute(text("DELETE FROM document_chunks_fts WHERE user_id=:user_id"), {"user_id": str(user_id)})
    except Exception as exc:
        db.rollback()
        logger.debug("Relational user index cleanup skipped: %s", exc)
    finally:
        db.close()

    if settings.CORPUS_STORE_BACKEND == "postgres":
        return
    client = get_chroma_client()
    collection_name = get_collection_name(user_id)

    try:
        from app.rag.bm25 import delete_user_bm25_indexes
        delete_user_bm25_indexes(user_id)
    except Exception as e:
        logger.warning(f"Error deleting user BM25 indexes: {e}")

    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted collection {collection_name}")
    except Exception as e:
        logger.warning(f"Error deleting collection: {e}")
