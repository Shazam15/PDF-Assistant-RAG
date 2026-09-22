"""Tracing wiring for the research graph.

These tests never talk to a Langfuse server: they inject a fake client at the
module seam so the assertions cover this repository's behaviour (span naming,
trace correlation, what is allowed to leave the process, fail-open) rather than
the SDK's internals.
"""
from contextlib import contextmanager

import pytest

from app.rag import tracing
from app.rag.research_agent import (
    ResearchDependencies,
    _trace_output,
    _traced_node,
    stream_research_agent,
)
from app.rag.retriever import ResearchBrief


class _FakeSpan:
    def __init__(self, name, record):
        self.name = name
        self._record = record

    def update(self, **kwargs):
        self._record.setdefault("updates", []).append(kwargs)


class _FakeClient:
    """Minimal stand-in for the langfuse client surface tracing.py relies on."""

    def __init__(self, raise_on_observation=False):
        self.observations = []
        self.flushed = 0
        self._raise_on_observation = raise_on_observation

    def create_trace_id(self, seed=None):
        return "trace-abc"

    @contextmanager
    def start_as_current_observation(self, *, name, as_type="span", trace_context=None, input=None, metadata=None):
        if self._raise_on_observation:
            raise RuntimeError("langfuse exploded")
        record = {
            "name": name,
            "as_type": as_type,
            "trace_id": getattr(trace_context, "trace_id", None) if trace_context else None,
            "input": input,
            "metadata": metadata,
        }
        self.observations.append(record)
        yield _FakeSpan(name, record)

    def flush(self):
        self.flushed += 1


@pytest.fixture
def fake_langfuse(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(tracing, "_langfuse_get_client", lambda *a, **k: client)
    monkeypatch.setattr(tracing, "_LangfuseTraceContext", lambda trace_id: type("TC", (), {"trace_id": trace_id})())
    monkeypatch.setattr(tracing, "_langfuse_propagate_attributes", None)
    return client


def _source_payload(chunk):
    return {"source_id": chunk["source_id"], "filename": chunk["filename"], "page": chunk["page"]}


def _dependencies():
    return ResearchDependencies(
        plan=lambda _q: ResearchBrief(main_question="main", facets=["methods"]),
        retrieve=lambda **_kwargs: [{
            "id": "chunk-1",
            "filename": "study.pdf",
            "document_id": "doc-1",
            "page": 1,
            "text": "CONFIDENTIAL EVIDENCE TEXT",
            "score": 1.0,
        }],
        source_payload=_source_payload,
        synthesize=lambda *_a: "DRAFTED ANSWER TEXT [D1].",
        verify=lambda *_a: [],
        repair=lambda *_a: pytest.fail("repair should not run"),
        audit=lambda _brief, evidence: {
            "supported": ["methods"],
            "missing": [],
            "conflicts": [],
            "relevant_indices": list(range(len(evidence))),
        },
    )


def test_tracing_disabled_by_default_is_a_no_op():
    assert tracing.LANGFUSE_ENABLED is False
    assert tracing.langfuse_callbacks() == []
    assert tracing.new_trace_id() is None
    tracing.flush_traces()

    with tracing.trace_stage("understand") as span:
        span.update(output={"anything": True})  # must not raise


def test_trace_output_excludes_document_text_and_drafted_answer():
    """The trace backend must not become a second copy of private content."""
    result = {
        "stage": "drafting",
        "evidence": [{"text": "CONFIDENTIAL EVIDENCE TEXT"}],
        "sources": [{"text": "CONFIDENTIAL SOURCE TEXT"}],
        "answer": "DRAFTED ANSWER TEXT",
        "supported_facets": ["methods"],
    }

    summary = _trace_output(result)

    assert "CONFIDENTIAL" not in str(summary)
    assert "DRAFTED ANSWER TEXT" not in str(summary)
    assert summary["evidence_count"] == 1
    assert summary["sources_count"] == 1
    assert summary["answer_chars"] == len("DRAFTED ANSWER TEXT")
    assert summary["supported_facets"] == ["methods"]


def test_traced_node_returns_the_node_result_unchanged(fake_langfuse):
    node = _traced_node("retrieve", lambda _state: {"stage": "retrieving", "evidence": [{"text": "x"}]})

    result = node({"trace_id": "trace-abc", "user_id": "u1", "run_id": "r1"})

    assert result == {"stage": "retrieving", "evidence": [{"text": "x"}]}
    assert fake_langfuse.observations[0]["name"] == "retrieve"
    assert fake_langfuse.observations[0]["trace_id"] == "trace-abc"


def test_traced_node_still_runs_when_the_trace_backend_fails(monkeypatch):
    """A broken tracing backend must never break a chat request."""
    broken = _FakeClient(raise_on_observation=True)
    monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(tracing, "_langfuse_get_client", lambda *a, **k: broken)

    node = _traced_node("draft", lambda _state: {"stage": "drafting", "answer": "ok"})

    assert node({"trace_id": "trace-abc"}) == {"stage": "drafting", "answer": "ok"}


def test_every_research_stage_shares_one_trace_id(fake_langfuse):
    events = list(stream_research_agent("question", "user-1", None, 12, [], _dependencies()))

    assert any(event["type"] == "result" for event in events)
    traced_stages = [obs["name"] for obs in fake_langfuse.observations]
    assert traced_stages == ["understand", "retrieve", "ledger", "audit", "outline", "draft", "verify", "finalize"]
    assert {obs["trace_id"] for obs in fake_langfuse.observations} == {"trace-abc"}
    assert fake_langfuse.flushed == 1

    exported = str(fake_langfuse.observations)
    assert "CONFIDENTIAL EVIDENCE TEXT" not in exported
    assert "DRAFTED ANSWER TEXT" not in exported


def test_research_run_checkpoint_records_the_trace_id(fake_langfuse, monkeypatch):
    """The stored run and the trace must be reachable from each other."""
    captured = {}

    def fake_checkpoint(state, status="running"):
        captured["trace_id"] = state.get("trace_id")

    monkeypatch.setattr("app.rag.research_agent._checkpoint_research_run", fake_checkpoint)

    list(stream_research_agent("question", "user-1", None, 12, [], _dependencies()))

    assert captured["trace_id"] == "trace-abc"
