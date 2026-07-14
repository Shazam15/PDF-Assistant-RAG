from unittest.mock import MagicMock, patch
from app.rag import agent


def test_generate_answer_appends_graph_context_without_changing_sources(monkeypatch):
    # Mock chunks
    chunks = [
        {
            "text": "Vector context",
            "filename": "doc.pdf",
            "page": 1,
            "score": 0.9,
            "confidence": 100.0,
            "source_type": "document",
            "source_id": "D1",
        }
    ]

    # Mock the executor and the tool
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {"output": "Agent answer [D1]"}
    
    mock_pdf_tool = MagicMock()
    mock_pdf_tool.last_sources = chunks
    mock_pdf_tool.all_sources = []
    mock_web_tool = MagicMock()
    mock_web_tool.last_sources = []
    mock_web_tool.all_sources = []

    monkeypatch.setattr(agent, "get_agent_executor", lambda *args, **kwargs: (mock_executor, mock_pdf_tool, mock_web_tool, ""))

    result = agent.generate_answer(
        "Search the web and explain how OpenAI and Microsoft are related?", "user-1", "doc-1"
    )

    assert result["answer"] == "Agent answer [D1]"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["text"] == "Vector context"
    assert result["sources"][0]["filename"] == "doc.pdf"
    assert result["sources"][0]["source_id"] == "D1"
    mock_executor.invoke.assert_called_once_with(
        {
            "input": "Search the web and explain how OpenAI and Microsoft are related?",
            "chat_history": "",
        }
    )


def test_generate_answer_stream_appends_graph_context(monkeypatch):
    # Mock chunks
    chunks = [
        {
            "text": "Vector stream context",
            "filename": "doc.pdf",
            "page": 1,
            "score": 0.9,
            "confidence": 100.0,
            "source_type": "document",
            "source_id": "D1",
        }
    ]

    # Mock the executor and the tool
    mock_executor = MagicMock()
    # Mock the stream method to yield chunks
    import json
    mock_executor.stream.return_value = iter([
        {"actions": [MagicMock(log="Thought: I should search. Action: pdf_search")]},
        {"intermediate_steps": []}, # This triggers source yielding in my implementation if last_sources is set
        {"output": "Final Answer: Streamed answer [D1]"}
    ])
    
    mock_pdf_tool = MagicMock()
    mock_pdf_tool.last_sources = chunks
    mock_pdf_tool.all_sources = []
    mock_web_tool = MagicMock()
    mock_web_tool.last_sources = []
    mock_web_tool.all_sources = []

    monkeypatch.setattr(agent, "get_agent_executor", lambda *args, **kwargs: (mock_executor, mock_pdf_tool, mock_web_tool, ""))

    events = list(agent.generate_answer_stream("Search the web for OpenAI Microsoft", "user-1", "doc-1"))

    # Verify event types and data
    assert not any("Thinking" in e for e in events)
    assert any("Streamed answer" in e for e in events)
    assert any("Vector stream context" in e for e in events)
    assert events[-1] == f"data: {json.dumps({'type': 'done'})}\n\n"
