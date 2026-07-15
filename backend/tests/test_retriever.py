from app.rag import retriever


def test_transform_query_includes_original_and_dedupes(monkeypatch):
    queries = retriever.transform_query("How do taxes and healthcare work?")

    assert queries[0] == "How do taxes and healthcare work?"
    assert "taxes healthcare work" in queries


def test_transform_query_extracts_core_question_and_adds_bilingual_hints():
    queries = retriever.transform_query(
        'Actúa como investigador. Responde: "¿Qué combinación de tecnologías y estrategias '
        'mejora la sostenibilidad ambiental?". Primero identifica documentos relevantes.'
    )

    assert queries[0] == "¿Qué combinación de tecnologías y estrategias mejora la sostenibilidad ambiental?"
    assert "environmental sustainability technologies strategies" in queries


def test_transform_summary_query_searches_document_structure():
    queries = retriever.transform_query("Haz un resumen de este documento.")

    assert "abstract research objective methodology principal results conclusions" in queries
    assert "resumen objetivo metodologia resultados principales conclusiones" in queries


def test_retrieve_fans_out_transformed_queries_and_merges_duplicates(monkeypatch):
    searched_queries = []

    monkeypatch.setattr(retriever, "transform_query", lambda _query: ["taxes", "healthcare"])
    monkeypatch.setattr(retriever, "embed_query", lambda query: f"embedding:{query}")
    monkeypatch.setattr(retriever, "get_reranker", lambda: None)

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

    chunks = retriever.retrieve("How do taxes and healthcare work?", user_id="user-1")

    assert searched_queries == ["embedding:taxes", "embedding:healthcare"]
    assert [chunk["id"] for chunk in chunks] == ["shared", "taxes", "healthcare"]
    assert chunks[0]["score"] == 0.9
    assert chunks[0]["retrieval_score"] == 1.0
    assert chunks[0]["confidence"] == 100.0
