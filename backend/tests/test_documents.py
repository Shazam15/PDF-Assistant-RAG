import importlib
import types
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.models import Document
from app.services.document_ingestion import ingest_document
from app.tasks import process_document
from app.services import document_ingestion


def test_progress_reporter_emits_heartbeat_during_long_stage(monkeypatch):
    updates = []
    monkeypatch.setattr(
        document_ingestion,
        "_update_progress",
        lambda document_id, progress, stage, **kwargs: updates.append((document_id, progress, stage)),
    )
    reporter = document_ingestion.ProgressReporter(
        "doc-1",
        minimum_interval=0,
        heartbeat_interval=0.01,
    )

    reporter.start()
    reporter.update("embedding", 1, 10, force=True)
    time.sleep(0.035)
    reporter.stop()

    assert updates[0] == ("doc-1", 44, "embedding")
    assert len(updates) >= 2


def test_api_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["version"] == "2.0.0"


def test_process_document_does_not_requeue_lost_workers_forever():
    assert process_document.reject_on_worker_lost is False


def test_embedding_migration_reuses_bm25_chunks_and_preserves_document_data(monkeypatch):
    document = SimpleNamespace(
        id="doc-1",
        user_id="user-1",
        filename="stored.pdf",
        original_name="paper.pdf",
        summary="Existing summary",
        chunk_size=None,
        chunk_overlap=None,
        status="ready",
        processing_stage="complete",
        processing_progress=100,
        error_message=None,
        chunk_count=1,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [document]
    monkeypatch.setattr("app.database.SessionLocal", MagicMock(return_value=db))
    embeddings_module = importlib.import_module("app.rag.embeddings")
    monkeypatch.setattr(embeddings_module, "get_embedding_model", MagicMock(return_value=object()))
    vectorstore_module = importlib.import_module("app.rag.vectorstore")
    monkeypatch.setattr(
        vectorstore_module, "user_index_needs_migration", MagicMock(return_value=True), raising=False
    )
    start = MagicMock()
    finish = MagicMock()
    monkeypatch.setattr(vectorstore_module, "start_user_index_migration", start, raising=False)
    monkeypatch.setattr(vectorstore_module, "finish_user_index_migration", finish, raising=False)
    monkeypatch.setattr(vectorstore_module, "document_index_is_current", MagicMock(return_value=False), raising=False)
    chunks = [{"text": "Stored chunk", "page": 1, "chunk_index": 0}]
    bm25_module = importlib.import_module("app.rag.bm25")
    monkeypatch.setattr(bm25_module, "load_bm25_chunks", MagicMock(return_value=chunks))
    chunk_document = MagicMock(side_effect=AssertionError("PDF should not be reparsed"))
    monkeypatch.setattr(document_ingestion, "chunk_document", chunk_document)
    store = MagicMock(return_value=1)
    monkeypatch.setattr(document_ingestion, "store_chunks", store)

    document_ingestion.migrate_embedding_indexes()

    start.assert_called_once_with("user-1")
    finish.assert_called_once_with("user-1")
    store.assert_called_once()
    chunk_document.assert_not_called()
    assert document.summary == "Existing summary"
    assert document.status == "ready"
    assert document.processing_progress == 100


def test_embedding_migration_is_idempotent_for_current_index(monkeypatch):
    document = SimpleNamespace(user_id="user-1", status="ready")
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [document]
    monkeypatch.setattr("app.database.SessionLocal", MagicMock(return_value=db))
    vectorstore_module = importlib.import_module("app.rag.vectorstore")
    monkeypatch.setattr(
        vectorstore_module, "user_index_needs_migration", MagicMock(return_value=False), raising=False
    )
    monkeypatch.setattr(vectorstore_module, "start_user_index_migration", MagicMock(), raising=False)
    monkeypatch.setattr(vectorstore_module, "finish_user_index_migration", MagicMock(), raising=False)
    load_model = MagicMock()
    embeddings_module = importlib.import_module("app.rag.embeddings")
    monkeypatch.setattr(embeddings_module, "get_embedding_model", load_model)

    document_ingestion.migrate_embedding_indexes()

    load_model.assert_not_called()


def test_embedding_migration_failure_remains_retryable(monkeypatch):
    document = SimpleNamespace(
        id="doc-1",
        user_id="user-1",
        filename="stored.pdf",
        original_name="paper.pdf",
        chunk_size=None,
        chunk_overlap=None,
        status="ready",
        processing_stage="completed",
        processing_progress=100,
        error_message=None,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [document]
    monkeypatch.setattr("app.database.SessionLocal", MagicMock(return_value=db))
    embeddings_module = importlib.import_module("app.rag.embeddings")
    monkeypatch.setattr(embeddings_module, "get_embedding_model", MagicMock(return_value=object()))
    vectorstore_module = importlib.import_module("app.rag.vectorstore")
    monkeypatch.setattr(
        vectorstore_module, "user_index_needs_migration", MagicMock(return_value=True), raising=False
    )
    monkeypatch.setattr(vectorstore_module, "start_user_index_migration", MagicMock(), raising=False)
    finish = MagicMock()
    monkeypatch.setattr(vectorstore_module, "finish_user_index_migration", finish, raising=False)
    monkeypatch.setattr(vectorstore_module, "document_index_is_current", MagicMock(return_value=False), raising=False)
    bm25_module = importlib.import_module("app.rag.bm25")
    monkeypatch.setattr(
        bm25_module,
        "load_bm25_chunks",
        MagicMock(return_value=[{"text": "Stored chunk", "page": 1, "chunk_index": 0}]),
    )
    monkeypatch.setattr(document_ingestion, "store_chunks", MagicMock(side_effect=RuntimeError("interrupted")))

    document_ingestion.migrate_embedding_indexes()

    assert document.status == "processing"
    assert document.processing_stage == "embedding"
    assert document.processing_progress == 70
    assert "interrupted" in document.error_message
    finish.assert_not_called()


def test_embedding_migration_skips_documents_already_resumed(monkeypatch):
    document = SimpleNamespace(
        id="doc-1",
        user_id="user-1",
        filename="stored.pdf",
        original_name="paper.pdf",
        chunk_count=12,
        status="ready",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [document]
    monkeypatch.setattr("app.database.SessionLocal", MagicMock(return_value=db))
    embeddings_module = importlib.import_module("app.rag.embeddings")
    monkeypatch.setattr(embeddings_module, "get_embedding_model", MagicMock(return_value=object()))
    vectorstore_module = importlib.import_module("app.rag.vectorstore")
    monkeypatch.setattr(vectorstore_module, "user_index_needs_migration", MagicMock(return_value=True))
    monkeypatch.setattr(vectorstore_module, "start_user_index_migration", MagicMock())
    monkeypatch.setattr(vectorstore_module, "finish_user_index_migration", MagicMock())
    current = MagicMock(return_value=True)
    monkeypatch.setattr(vectorstore_module, "document_index_is_current", current)
    store = MagicMock()
    monkeypatch.setattr(document_ingestion, "store_chunks", store)

    document_ingestion.migrate_embedding_indexes()

    current.assert_called_once_with("user-1", "doc-1", 12)
    store.assert_not_called()
    vectorstore_module.finish_user_index_migration.assert_called_once_with("user-1")


def test_protected_documents_list_requires_auth(client):
    response = client.get("/api/v1/documents/")

    assert response.status_code in (401, 403)


def test_documents_list_authenticated(client, auth_headers, ready_document):
    response = client.get("/api/v1/documents/", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["id"] == ready_document.id
    assert payload["items"][0]["original_name"] == "ready.txt"


def test_documents_list_exposes_ingestion_progress(client, auth_headers, ready_document, db_session):
    ready_document.status = "enriching"
    ready_document.processing_progress = 88
    ready_document.processing_stage = "summarizing"
    ready_document.processing_current = 1
    ready_document.processing_total = 3
    ready_document.searchable_at = datetime.now(timezone.utc)
    db_session.commit()

    response = client.get("/api/v1/documents/", headers=auth_headers)

    assert response.status_code == 200
    document = response.json()["items"][0]
    assert document["status"] == "enriching"
    assert document["processing_progress"] == 88
    assert document["processing_stage"] == "summarizing"
    assert document["processing_current"] == 1
    assert document["processing_total"] == 3
    assert document["searchable_at"] is not None


def test_upload_rejects_unsupported_extension_before_deep_validation(client, auth_headers):
    response = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers,
        files={"file": ("payload.exe", b"binary-data", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "not supported" in response.json()["error"]["message"]


def test_rename_document_updates_original_name(client, auth_headers, ready_document, db_session):
    response = client.patch(
        f"/api/v1/documents/{ready_document.id}",
        headers=auth_headers,
        json={"name": " renamed-report.pdf "},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == ready_document.id
    assert payload["original_name"] == "renamed-report.pdf"

    db_session.refresh(ready_document)
    assert ready_document.original_name == "renamed-report.pdf"
    assert ready_document.filename == "ready.txt"


def test_rename_document_rejects_empty_name(client, auth_headers, ready_document):
    response = client.patch(
        f"/api/v1/documents/{ready_document.id}",
        headers=auth_headers,
        json={"name": "   "},
    )

    assert response.status_code == 422


def test_rename_document_returns_404_for_missing_document(client, auth_headers):
    response = client.patch(
        "/api/v1/documents/00000000-0000-0000-0000-000000000000",
        headers=auth_headers,
        json={"name": "missing.pdf"},
    )

    assert response.status_code == 404


def test_rename_document_returns_403_for_other_users_document(client, auth_headers, db_session, other_user):
    other_document = Document(
        user_id=other_user.id,
        filename="other.txt",
        original_name="other.txt",
        file_size=64,
        status="ready",
    )
    db_session.add(other_document)
    db_session.commit()
    db_session.refresh(other_document)

    response = client.patch(
        f"/api/v1/documents/{other_document.id}",
        headers=auth_headers,
        json={"name": "renamed.txt"},
    )

    assert response.status_code == 403
    db_session.refresh(other_document)
    assert other_document.original_name == "other.txt"


def test_ingest_document_builds_and_saves_graph(db_session, monkeypatch, tmp_path, user):
    document = Document(
        user_id=user.id,
        filename="graph.txt",
        original_name="graph.txt",
        file_size=128,
        status="pending",
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    user_id = user.id
    document_id = document.id
    chunks = [{"text": "OpenAI works with Microsoft.", "page": 1, "chunk_index": 0}]
    saved = {}

    monkeypatch.setattr("app.services.document_ingestion.get_page_count", lambda filepath: 1)
    monkeypatch.setattr("app.services.document_ingestion.chunk_document", lambda filepath: chunks)
    monkeypatch.setattr("app.services.document_ingestion.store_chunks", lambda **kwargs: len(chunks))
    monkeypatch.setattr("app.database.SessionLocal", lambda: db_session)

    fake_summary = types.ModuleType("app.rag.summarizer")
    fake_summary.generate_document_summary = lambda filepath, max_sentences=2: "Summary"
    monkeypatch.setitem(__import__("sys").modules, "app.rag.summarizer", fake_summary)

    monkeypatch.setattr(
        "app.rag.graph_builder.build_graph",
        lambda received_chunks: {"chunks": received_chunks},
    )
    monkeypatch.setattr(
        "app.rag.graph_builder.save_graph",
        lambda graph, user_id, document_id: saved.update(
            {"graph": graph, "user_id": user_id, "document_id": document_id}
        ),
    )

    ingest_document(
        document_id=document_id,
        filepath=str(tmp_path / "graph.txt"),
        original_name=document.original_name,
        user_id=user_id,
    )

    assert saved == {
        "graph": {"chunks": chunks},
        "user_id": user_id,
        "document_id": document_id,
    }
    refreshed = db_session.get(Document, document_id)
    assert refreshed.status == "ready"
    assert refreshed.chunk_count == 1


def test_enrichment_failure_keeps_indexed_document_searchable(db_session, monkeypatch, tmp_path, user):
    document = Document(
        user_id=user.id,
        filename="searchable.txt",
        original_name="searchable.txt",
        file_size=128,
        status="pending",
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    document_id = document.id
    chunks = [{"text": "Verified evidence.", "page": 1, "chunk_index": 0}]
    observed = {}

    monkeypatch.setattr("app.services.document_ingestion.get_page_count", lambda filepath: 1)
    monkeypatch.setattr("app.services.document_ingestion.chunk_document", lambda filepath: chunks)
    monkeypatch.setattr("app.services.document_ingestion.store_chunks", lambda **kwargs: len(chunks))
    monkeypatch.setattr("app.database.SessionLocal", lambda: db_session)

    fake_summary = types.ModuleType("app.rag.summarizer")

    def fail_memory(chunks, progress_callback=None):
        current = db_session.get(Document, document_id)
        observed["status"] = current.status
        observed["searchable_at"] = current.searchable_at
        raise RuntimeError("local Ollama unavailable")

    fake_summary.build_document_memory = fail_memory
    monkeypatch.setitem(__import__("sys").modules, "app.rag.summarizer", fake_summary)
    monkeypatch.setattr("app.rag.graph_builder.build_graph", lambda received_chunks: {})
    monkeypatch.setattr("app.rag.graph_builder.save_graph", lambda *args, **kwargs: None)

    ingest_document(
        document_id=document_id,
        filepath=str(tmp_path / "searchable.txt"),
        original_name=document.original_name,
        user_id=user.id,
    )

    refreshed = db_session.get(Document, document_id)
    assert observed["status"] == "enriching"
    assert observed["searchable_at"] is not None
    assert refreshed.status == "ready"
    assert refreshed.chunk_count == 1
    assert refreshed.searchable_at is not None
    assert "Structured memory" in refreshed.processing_warning


def test_delete_document_soft_deletes_and_hides_document(client, auth_headers, ready_document, db_session, monkeypatch):
    deletion_calls = []
    doc_id = ready_document.id

    monkeypatch.setattr(
        "app.rag.graph_builder.delete_graph",
        lambda user_id, document_id: deletion_calls.append(
            {"user_id": user_id, "document_id": document_id}
        ),
    )

    response = client.delete(
        f"/api/v1/documents/{doc_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert deletion_calls == []

    db_session.refresh(ready_document)
    assert ready_document.is_deleted is True
    assert ready_document.deleted_at is not None

    list_response = client.get("/api/v1/documents/", headers=auth_headers)
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 0

    get_response = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert get_response.status_code == 404
