import importlib
import sys
from unittest.mock import MagicMock


def _load_real_vectorstore():
    missing = object()
    rag_package = sys.modules.get("app.rag")
    previous_attribute = getattr(rag_package, "vectorstore", missing) if rag_package else missing
    fake_module = sys.modules.pop("app.rag.vectorstore", None)
    try:
        return importlib.import_module("app.rag.vectorstore")
    finally:
        if fake_module is not None:
            sys.modules["app.rag.vectorstore"] = fake_module
        if rag_package is not None:
            if previous_attribute is missing:
                try:
                    delattr(rag_package, "vectorstore")
                except AttributeError:
                    pass
            else:
                setattr(rag_package, "vectorstore", previous_attribute)


def test_sqlite_fts_query_escapes_natural_language_punctuation():
    _sqlite_fts_query = _load_real_vectorstore()._sqlite_fts_query

    result = _sqlite_fts_query(
        "¿Qué problema experimental aborda la turbina y cómo afecta el rendimiento?"
    )

    assert "?" not in result
    assert result.startswith('"Qué" OR "problema"')
    assert '"turbina"' in result
    assert '"rendimiento"' in result


def test_sqlite_fts_query_deduplicates_and_limits_terms():
    _sqlite_fts_query = _load_real_vectorstore()._sqlite_fts_query

    assert _sqlite_fts_query("Motor motor MOTOR", max_terms=2) == '"Motor"'
    assert _sqlite_fts_query("uno dos tres", max_terms=2) == '"uno" OR "dos"'
    assert _sqlite_fts_query("¿?!") == ""


def test_store_chunks_deletes_old_chunks(monkeypatch):
    """
    Test that store_chunks cleans up old chunks for the specific document and user
    before embedding and saving the new chunks.
    """
    # Keep track of fake module from sys.modules
    fake_module = sys.modules.get("app.rag.vectorstore")

    # Temporarily remove fake_module to import the real module
    if fake_module:
        del sys.modules["app.rag.vectorstore"]

    try:
        # Import the real module
        import app.rag.vectorstore as real_vectorstore

        # Mock the ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = mock_collection

        # Simulate existing chunks in ChromaDB
        mock_collection.get.return_value = {"ids": ["doc_123_0", "doc_123_1"]}

        monkeypatch.setattr(real_vectorstore, "get_chroma_client", lambda: mock_client)

        # Mock embedding model on the app.rag.embeddings module directly
        import app.rag.embeddings as embeddings_module
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 384]
        monkeypatch.setattr(
            embeddings_module,
            "get_embedding_model",
            lambda: mock_embeddings,
        )

        # Mock BM25 actions to avoid I/O or extra imports
        monkeypatch.setattr("app.rag.bm25.store_bm25_index", lambda *args, **kwargs: None)
        monkeypatch.setattr("app.rag.bm25.delete_bm25_index", lambda *args, **kwargs: None)

        # Execute store_chunks
        chunks = [{"text": "Hello world", "page": 1, "chunk_index": 0}]
        progress_events = []
        real_vectorstore.store_chunks(
            chunks=chunks,
            document_id="doc_123",
            filename="test.pdf",
            user_id="user_123",
            progress_callback=lambda stage, current, total: progress_events.append((stage, current, total)),
        )

        # Assertions
        # 1. It should check for existing chunks using correct metadata filters
        mock_collection.get.assert_called_once_with(
            where={"document_id": {"$eq": "doc_123"}},
            include=[],
        )
        # 2. It should delete those chunks by ID
        mock_collection.delete.assert_called_once_with(ids=["doc_123_0", "doc_123_1"])
        # 3. It should add the new chunks to the collection
        mock_collection.add.assert_called_once()
        add_kwargs = mock_collection.add.call_args.kwargs
        assert add_kwargs["documents"] == ["test.pdf\nHello world"]
        assert add_kwargs["metadatas"][0]["text"] == "Hello world"
        assert ("embedding", 1, 1) in progress_events
        assert progress_events[-1] == ("persisting", 1, 1)

    finally:
        # Restore the fake module
        if fake_module:
            sys.modules["app.rag.vectorstore"] = fake_module
