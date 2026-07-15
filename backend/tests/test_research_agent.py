from threading import Event

import pytest

from app.rag.research_agent import (
    ResearchCancelled,
    ResearchDependencies,
    stream_research_agent,
)
from app.rag.retriever import ResearchBrief


def _source_payload(chunk):
    return {
        "source_id": chunk["source_id"],
        "filename": chunk["filename"],
        "page": chunk["page"],
        "text": chunk["text"],
        "score": chunk.get("score", 1.0),
        "confidence": 100.0,
    }


def test_research_graph_fills_a_missing_facet_then_synthesizes():
    calls = []

    def retrieve(**kwargs):
        calls.append(list(kwargs["facets"]))
        query = kwargs["facets"][0]
        return [{
            "id": f"chunk-{len(calls)}",
            "filename": f"study-{len(calls)}.pdf",
            "document_id": f"doc-{len(calls)}",
            "page": len(calls),
            "text": f"Direct evidence for {query}",
            "facet_ids": ["F1"],
            "facet_queries": {"F1": query},
            "score": 1.0,
        }]

    def audit(brief, evidence):
        covered = {query for item in evidence for query in item.get("facet_queries", {}).values()}
        return {
            "supported": [facet for facet in brief.facets if facet in covered],
            "missing": [facet for facet in brief.facets if facet not in covered],
            "conflicts": [],
            "relevant_indices": list(range(len(evidence))),
        }

    dependencies = ResearchDependencies(
        plan=lambda _question: ResearchBrief(main_question="main", facets=["methods", "results"]),
        retrieve=retrieve,
        source_payload=_source_payload,
        synthesize=lambda _brief, _evidence, _sources, _outline: "Integrated answer [D1] [D2].",
        verify=lambda _answer, _sources, _evidence: [],
        repair=lambda *_args: pytest.fail("repair should not run"),
        audit=audit,
    )

    events = list(stream_research_agent("question", "user", None, 12, [], dependencies))
    result = next(event["data"] for event in events if event["type"] == "result")

    assert calls == [["methods", "results"], ["results"]]
    assert len(result["sources"]) == 2
    assert result["answer"] == "Integrated answer [D1] [D2]."
    assert any(event.get("data", {}).get("stage") == "auditing" for event in events)


def test_research_graph_filters_tangential_evidence_from_final_sources():
    evidence = [
        {"id": "wrong", "filename": "water.pdf", "document_id": "water", "page": 1, "text": "Water treatment."},
        {"id": "right", "filename": "engine.pdf", "document_id": "engine", "page": 2, "text": "Valve timing result."},
    ]
    dependencies = ResearchDependencies(
        plan=lambda _question: ResearchBrief(main_question="engine", facets=["valve timing"]),
        retrieve=lambda **_kwargs: evidence,
        source_payload=_source_payload,
        synthesize=lambda _brief, _evidence, _sources, _outline: "Supported result [D1].",
        verify=lambda _answer, _sources, _evidence: [],
        repair=lambda *_args: "",
        audit=lambda _brief, _evidence: {
            "supported": ["valve timing"],
            "missing": [],
            "conflicts": [],
            "relevant_indices": [1],
        },
    )

    events = list(stream_research_agent("question", "user", None, 8, [], dependencies))
    result = next(event["data"] for event in events if event["type"] == "result")

    assert [source["filename"] for source in result["sources"]] == ["engine.pdf"]


def test_research_graph_honors_cancellation_before_work():
    cancellation = Event()
    cancellation.set()
    dependencies = ResearchDependencies(
        plan=lambda _question: pytest.fail("planner should not run"),
        retrieve=lambda **_kwargs: [],
        source_payload=_source_payload,
        synthesize=lambda *_args: "",
        verify=lambda *_args: [],
        repair=lambda *_args: "",
        cancellation_event=cancellation,
    )

    with pytest.raises(ResearchCancelled):
        list(stream_research_agent("question", "user", None, 8, [], dependencies))


def test_research_graph_synthesizes_accumulated_evidence_after_timeout():
    evidence = [{
        "id": "engine-1",
        "filename": "engine.pdf",
        "document_id": "engine",
        "page": 7,
        "text": "Measured valve timing evidence.",
        "facet_ids": ["F1"],
        "facet_queries": {"F1": "valve timing"},
    }]
    dependencies = ResearchDependencies(
        plan=lambda _question: ResearchBrief(main_question="engine", facets=["valve timing"]),
        retrieve=lambda **_kwargs: evidence,
        source_payload=_source_payload,
        synthesize=lambda _brief, _evidence, _sources, _outline: "Best available answer [D1].",
        verify=lambda *_args: [],
        repair=lambda *_args: "",
        audit=lambda *_args: (_ for _ in ()).throw(TimeoutError("budget")),
    )

    events = list(stream_research_agent("question", "user", None, 8, [], dependencies))
    result = next(event["data"] for event in events if event["type"] == "result")

    assert result["answer"] == "Best available answer [D1]."
    assert [source["filename"] for source in result["sources"]] == ["engine.pdf"]
