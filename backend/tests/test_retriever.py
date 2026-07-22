from unittest.mock import MagicMock

import pytest

from app.rag import retriever
from app.rag import reranker as reranker_module
from app.rag.reranker import Reranker


def test_transform_query_uses_domain_neutral_sentence_segmentation():
    query = "Compare the first method. Evaluate the second method."
    queries = retriever.transform_query(query)

    assert queries == [query, "Compare the first method.", "Evaluate the second method."]


def test_structured_research_plan_uses_dynamic_facets(monkeypatch):
    structured = MagicMock()
    structured.invoke.return_value = retriever.ResearchPlan(
        main_question="¿Qué explica el fenómeno?",
        facets=["evidencia experimental", "modelos teóricos"],
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(retriever, "create_chat_ollama", MagicMock(return_value=llm))

    plan = retriever.build_research_plan("Solicitud extensa sin vocabulario predefinido")

    assert plan.main_question == "¿Qué explica el fenómeno?"
    assert plan.facets == ["evidencia experimental", "modelos teóricos"]
    llm.with_structured_output.assert_called_once_with(
        retriever.ResearchPlan,
        method="json_mode",
    )


def test_structured_research_plan_falls_back_without_domain_expansion(monkeypatch):
    monkeypatch.setattr(retriever, "create_chat_ollama", MagicMock(side_effect=TimeoutError("slow")))

    plan = retriever.build_research_plan("Primera dimensión. Segunda dimensión.")

    assert plan.facets == ["Primera dimensión. Segunda dimensión."]


def test_structured_research_plan_falls_back_on_invalid_structured_result(monkeypatch):
    structured = MagicMock()
    structured.invoke.return_value = None
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(retriever, "create_chat_ollama", MagicMock(return_value=llm))

    plan = retriever.build_research_plan("Compare two independently reported outcomes.")

    assert plan.main_question == "Compare two independently reported outcomes."
    assert plan.facets == ["Compare two independently reported outcomes."]


def test_research_plan_caps_model_and_fallback_facets_at_six(monkeypatch):
    plan = retriever.ResearchPlan(
        main_question="Main question",
        facets=[f"Facet {index}" for index in range(8)],
    )
    assert len(plan.facets) == 6

    monkeypatch.setattr(retriever, "create_chat_ollama", MagicMock(side_effect=RuntimeError("invalid JSON")))
    query = " ".join(f"Dimension {index}." for index in range(8))

    fallback = retriever.build_research_plan(query)

    assert fallback.facets == [query]


def test_retrieve_fans_out_explicit_facets_and_merges_duplicates(monkeypatch):
    searched_queries = []

    monkeypatch.setattr(retriever, "embed_query", lambda query: f"embedding:{query}")

    class FakeReranker:
        def rerank(self, query, documents, top_k):
            scores = {
                "taxes": {"shared": 0.82, "taxes": 0.91},
                "healthcare": {"shared": 0.84, "healthcare": 0.93},
            }[query]
            for document in documents:
                document["relevance_score"] = scores[document["id"]]
            return sorted(documents, key=lambda item: item["relevance_score"], reverse=True)[:top_k]

    monkeypatch.setattr(retriever, "get_reranker", lambda: FakeReranker())

    def fake_query_chunks(query_embedding, user_id, document_id=None, top_k=10):
        searched_queries.append(query_embedding)
        if query_embedding == "embedding:taxes":
            return [
                {
                    "id": "shared",
                    "text": "Shared chunk",
                    "filename": "policy.pdf",
                    "page": 1,
                    "score": 0.2,
                },
                {
                    "id": "taxes",
                    "text": "Tax chunk",
                    "filename": "policy.pdf",
                    "page": 2,
                    "score": 0.7,
                },
            ]

        return [
            {
                "id": "shared",
                "text": "Shared chunk",
                "filename": "policy.pdf",
                "page": 1,
                "score": 0.9,
            },
            {
                "id": "healthcare",
                "text": "Healthcare chunk",
                "filename": "policy.pdf",
                "page": 3,
                "score": 0.8,
            },
        ]

    monkeypatch.setattr(retriever, "query_chunks", fake_query_chunks)

    chunks = retriever.retrieve(
        "How do taxes and healthcare work?",
        user_id="user-1",
        facets=["taxes", "healthcare"],
    )

    assert searched_queries == ["embedding:taxes", "embedding:healthcare"]
    assert [chunk["id"] for chunk in chunks] == ["shared", "healthcare", "taxes"]
    assert set(chunks[0]["requested_facets"].values()) == {"taxes", "healthcare"}
    assert max(chunk["confidence"] for chunk in chunks) == 93.0


def test_facet_selection_prefers_distinct_documents_with_supported_facets():
    chunks = [
        {"id": "a1", "document_id": "a", "facet_ids": ["F1"], "relevance_score": 0.92},
        {"id": "a2", "document_id": "a", "facet_ids": ["F2"], "relevance_score": 0.91},
        {"id": "b1", "document_id": "b", "facet_ids": ["F2"], "relevance_score": 0.88},
    ]

    selected = retriever._select_facet_evidence(chunks, top_k=2)

    assert {chunk["id"] for chunk in selected} == {"a1", "b1"}


def test_reranker_exposes_query_relative_rank_and_raw_score(monkeypatch):
    model = MagicMock()
    model.predict.return_value = [0.0, 2.0]
    reranker = Reranker(model_name="test")
    monkeypatch.setattr(reranker, "_load_model", lambda: model)
    documents = [{"text": "first"}, {"text": "second"}]

    ranked = reranker.rerank("query", documents, top_k=2)

    assert ranked[0]["text"] == "second"
    assert ranked[0]["rerank_score"] == 2.0
    assert ranked[0]["relevance_score"] == 1.0
    assert ranked[1]["rerank_score"] == 0.0
    assert ranked[1]["relevance_score"] == 0.5


def test_reranker_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch):
    cross_encoder = MagicMock()
    monkeypatch.setattr(reranker_module, "CrossEncoder", cross_encoder)
    monkeypatch.setattr(reranker_module.torch.cuda, "is_available", lambda: False)

    Reranker(model_name="test", device="cuda")._load_model()

    assert cross_encoder.call_args.kwargs["device"] == "cpu"


def test_rrf_fuses_rankings_without_comparing_raw_scores():
    dense = [
        {"id": "a", "text": "A", "score": 0.91},
        {"id": "b", "text": "B", "score": 0.90},
    ]
    lexical = [
        {"id": "b", "text": "B", "score": 25.0},
        {"id": "c", "text": "C", "score": 24.0},
    ]

    fused = retriever.reciprocal_rank_fusion([dense, lexical], weights=[0.6, 0.4])

    assert fused[0]["id"] == "b"
    assert fused[0]["retrieval_channels"] == 2
    assert fused[0]["rrf_score"] < 1.0


def test_research_brief_keeps_output_contract_out_of_facets(monkeypatch):
    structured = MagicMock()
    structured.invoke.return_value = retriever.ResearchBrief(
        main_question="Compare the technologies",
        facets=["reported efficiency", "experimental limitations"],
        deliverables=["technical report", "abstract and keywords"],
        constraints=["cite every substantive claim"],
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(retriever, "create_chat_ollama", MagicMock(return_value=llm))

    brief = retriever.build_research_plan("Write a report and abstract comparing technologies")

    assert brief.facets == ["reported efficiency", "experimental limitations"]
    assert "abstract and keywords" in brief.deliverables


def test_deep_retrieval_uses_semantic_profile_shortlist(monkeypatch):
    dense_calls = []
    lexical_calls = []
    candidate_limits = []
    monkeypatch.setattr(retriever, "embed_query", lambda query: [float(len(query))])
    monkeypatch.setattr(
        retriever,
        "query_document_profiles",
        lambda **_kwargs: ["relevant-document"],
    )
    monkeypatch.setattr(retriever, "get_reranker", lambda: None)

    def dense(**kwargs):
        dense_calls.append(kwargs.get("document_ids"))
        candidate_limits.append(kwargs.get("top_k"))
        return [{
            "id": "chunk-1",
            "document_id": "relevant-document",
            "filename": "relevant.pdf",
            "page": 2,
            "text": "Direct evidence",
            "score": 0.8,
        }]

    def lexical(**kwargs):
        lexical_calls.append(kwargs.get("document_ids"))
        candidate_limits.append(kwargs.get("top_k"))
        return []

    monkeypatch.setattr(retriever, "query_chunks", dense)
    monkeypatch.setattr(retriever, "query_lexical_chunks", lexical)

    chunks = retriever.retrieve(
        "Compare reported outcomes",
        user_id="user-1",
        facets=["reported outcome"],
    )

    assert dense_calls == [["relevant-document"]]
    assert lexical_calls == [["relevant-document"]]
    assert candidate_limits == [50, 50]
    assert [chunk["document_id"] for chunk in chunks] == ["relevant-document"]
