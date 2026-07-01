import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from app.rag import agent as agent_module
from app.rag.agent import generate_answer, generate_answer_stream

@pytest.fixture
def mock_llm_client():
    with patch("app.rag.agent.get_llm_client") as mock_get:
        client = MagicMock()
        mock_get.return_value = client
        yield client

@pytest.fixture
def mock_retriever():
    with patch("app.rag.agent.retrieve") as mock_retrieve, patch("app.rag.agent.get_entity_context", return_value=""):
        yield mock_retrieve

@pytest.fixture
def mock_agent_executor():
    with patch("app.rag.agent.get_agent_executor") as mock_get:
        executor = MagicMock()
        pdf_tool = MagicMock()
        mock_get.return_value = (executor, pdf_tool, "")
        yield executor, pdf_tool

def test_generate_answer_success(mock_llm_client, mock_retriever):
    mock_retriever.return_value = [
        {
            "text": "This is a test chunk.",
            "filename": "test.pdf",
            "page": 1,
            "score": 0.9,
            "confidence": 90
        }
    ]
    mock_response = MagicMock()
    mock_response.content = "Test answer"
    mock_llm_client.invoke.return_value = mock_response

    result = generate_answer("test question", "user123", "doc123")

    assert result["answer"] == "Test answer\n\nFuentes consultadas: [Fuente: test.pdf, Página 1]"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["filename"] == "test.pdf"
    assert result["sources"][0]["text"] == "This is a test chunk."
    mock_retriever.assert_called_once()

def test_generate_answer_empty_retrieval(mock_llm_client, mock_retriever):
    mock_retriever.return_value = []

    result = generate_answer("test question", "user123", "doc123")

    assert result["answer"] == "No encontré información suficiente en los documentos cargados para responder esta pregunta."
    assert len(result["sources"]) == 0
    mock_llm_client.invoke.assert_not_called()


def test_load_global_style_reference_from_named_file(tmp_path, monkeypatch):
    style_dir = tmp_path / "uploads"
    style_dir.mkdir()
    style_file = style_dir / "PDF_DE_PRUEBA"
    style_file.write_text("Tono solemne y elegante", encoding="utf-8")

    monkeypatch.setattr(agent_module, "settings", SimpleNamespace(UPLOAD_DIR=str(style_dir)))

    reference = agent_module._load_global_style_reference()

    assert "Tono solemne y elegante" in reference
    assert "Referencia de estilo global" in reference


def test_generate_answer_uses_document_style_reference(mock_llm_client, mock_retriever):
    mock_retriever.return_value = [
        {
            "text": "El cielo estaba oscuro y el silencio pesaba sobre la calle.",
            "filename": "book.pdf",
            "page": 1,
            "score": 0.95,
            "confidence": 95,
        }
    ]
    mock_response = MagicMock()
    mock_response.content = "Respuesta estilizada"
    mock_llm_client.invoke.return_value = mock_response

    generate_answer("¿Qué sensación transmite este pasaje?", "user123", "doc123")

    prompt = mock_llm_client.invoke.call_args.args[0][0].content
    assert "Referencia de estilo" in prompt
    assert "book.pdf" in prompt
    assert "silencio pesaba" in prompt


def test_generate_answer_stream_success(mock_llm_client, mock_retriever):
    mock_retriever.return_value = [
        {
            "text": "Chunk text.",
            "filename": "test.pdf",
            "page": 1,
            "score": 0.8,
            "confidence": 80
        }
    ]

    chunk1 = MagicMock()
    chunk1.content = "Hello "
    chunk2 = MagicMock()
    chunk2.content = "world"
    mock_llm_client.stream.return_value = [chunk1, chunk2]

    stream = generate_answer_stream("test question", "user123", "doc123")
    events = list(stream)

    # First event: sources
    sources_event = json.loads(events[0].replace("data: ", "").strip())
    assert sources_event["type"] == "sources"
    assert len(sources_event["data"]) == 1
    assert sources_event["data"][0]["filename"] == "test.pdf"

    token_events = [json.loads(event.replace("data: ", "").strip()) for event in events if '"type": "token"' in event]
    assert "".join(event["data"] for event in token_events) == "Hello world\n\nFuentes consultadas: [Fuente: test.pdf, Página 1]"

    # Last event: done
    done_event = json.loads(events[-1].replace("data: ", "").strip())
    assert done_event["type"] == "done"

def test_generate_answer_greeting(mock_llm_client, mock_retriever):
    # "hi" is a greeting, should skip RAG
    mock_response = MagicMock()
    mock_response.content = "Hello there!"
    mock_llm_client.invoke.return_value = mock_response

    result = generate_answer("hi", "user123")

    assert result["answer"] == "Hello there!"
    assert len(result["sources"]) == 0
    mock_retriever.assert_not_called()

def test_generate_answer_stream_empty_retrieval(mock_llm_client, mock_retriever):
    mock_retriever.return_value = []

    stream = generate_answer_stream("test question", "user123", "doc123")
    events = list(stream)

    sources_event = json.loads(events[0].replace("data: ", "").strip())
    assert sources_event["type"] == "sources"
    assert sources_event["data"] == []

    token_event = json.loads(events[1].replace("data: ", "").strip())
    assert token_event["type"] == "token"
    assert token_event["data"] == "No encontré información suficiente en los documentos cargados para responder esta pregunta."

    # Last event: done
    done_event = json.loads(events[-1].replace("data: ", "").strip())
    assert done_event["type"] == "done"

def test_generate_answer_stream_error(mock_llm_client, mock_retriever):
    mock_retriever.return_value = [
        {"text": "Chunk text.", "filename": "test.pdf", "page": 1, "score": 0.8, "confidence": 80}
    ]
    mock_llm_client.stream.side_effect = Exception("LLM Down")

    stream = generate_answer_stream("test question", "user123", "doc123")
    events = list(stream)

    error_event = [json.loads(e.replace("data: ", "").strip()) for e in events if "error" in e]
    assert len(error_event) > 0
    assert error_event[0]["data"] == "LLM Down"

def test_generate_answer_error(mock_agent_executor, mock_retriever):
    from app.exceptions import ExternalServiceException
    with patch("app.rag.agent.get_llm_client") as mock_get:
        client = MagicMock()
        client.invoke.side_effect = Exception("LLM Down")
        mock_get.return_value = client
        mock_retriever.return_value = [
            {"text": "Chunk text.", "filename": "test.pdf", "page": 1, "score": 0.8, "confidence": 80}
        ]

        with pytest.raises(ExternalServiceException) as exc_info:
            generate_answer("test question", "user123", "doc123")
    assert "LLM Down" in str(exc_info.value)
