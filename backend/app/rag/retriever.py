"""
Two-stage retrieval: Hybrid Ensemble (ChromaDB + BM25) + cross-encoder reranking.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional

try:
    # In LangChain 1.3.2+, EnsembleRetriever moved to langchain_classic.
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    class EnsembleRetriever:
        """Small fallback used when optional LangChain classic deps are absent."""

        def __init__(self, retrievers, weights=None):
            self.retrievers = retrievers
            self.weights = weights or [1.0] * len(retrievers)

        def invoke(self, query):
            docs = []
            for retriever in self.retrievers:
                docs.extend(retriever.invoke(query))
            return docs
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangchainDocument
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from app.config import get_settings
from app.rag.embeddings import embed_query
from app.rag.tracing import trace_function
from app.rag.vectorstore import query_chunks
from app.rag.reranker import get_reranker

logger = logging.getLogger(__name__)
settings = get_settings()
MAX_QUERY_VARIANTS = 6
RERANK_POOL_MULTIPLIER = 3
MAX_CHUNKS_PER_DOCUMENT = 3
_GENERIC_RETRIEVAL_TERMS = {
    "academic",
    "academica",
    "academico",
    "actua",
    "afirmaciones",
    "analisis",
    "answer",
    "available",
    "briefly",
    "citas",
    "citation",
    "citations",
    "compare",
    "comparando",
    "conclusion",
    "conclusiones",
    "contained",
    "critical",
    "critico",
    "discusion",
    "document",
    "documento",
    "documentos",
    "documents",
    "evidence",
    "evidencia",
    "explica",
    "finalmente",
    "finally",
    "fortalezas",
    "identify",
    "identifica",
    "incluyendo",
    "informacion",
    "introduccion",
    "investigador",
    "limitations",
    "limitaciones",
    "methodologies",
    "metodologias",
    "organizada",
    "organizado",
    "pagina",
    "paginas",
    "pregunta",
    "primero",
    "propose",
    "propuesta",
    "propuestas",
    "redacta",
    "reference",
    "referencia",
    "relevant",
    "relevantes",
    "responde",
    "respuesta",
    "results",
    "resultados",
    "section",
    "seleccionaste",
    "sources",
    "style",
    "suposiciones",
    "unicamente",
    "using",
    "utilizando",
}


class CustomVectorRetriever(BaseRetriever):
    user_id: str = Field(description="User ID")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    top_k: int = Field(default=10, description="Top K results")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[LangchainDocument]:
        query_vector = embed_query(query)
        candidates = query_chunks(
            query_embedding=query_vector,
            user_id=self.user_id,
            document_id=self.document_id,
            top_k=self.top_k,
        )
        return [LangchainDocument(page_content=c["text"], metadata=c) for c in candidates]


class CustomBM25Retriever(BaseRetriever):
    user_id: str = Field(description="User ID")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    top_k: int = Field(default=10, description="Top K results")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[LangchainDocument]:
        from app.rag.bm25 import query_bm25
        candidates = query_bm25(
            query=query,
            user_id=self.user_id,
            document_id=self.document_id,
            top_k=self.top_k,
        )
        return [LangchainDocument(page_content=c["text"], metadata=c) for c in candidates]


def _generate_query_variants(query: str) -> List[str]:
    """Use Ollama to split/rewrite a user query for semantic search."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.2)
        prompt = (
            "Rewrite the user question into concise semantic search queries for document retrieval. "
            "Split independent topics into separate queries. Return a JSON array of strings only. "
            f"User question: {query}"
        )
        response = llm.invoke([
            SystemMessage(content="You create optimized search queries for a RAG retriever."),
            HumanMessage(content=prompt),
        ])
        import json as _json
        return _json.loads(response.content.strip())
    except Exception:
        return []


def _parse_query_variants(content: str) -> List[str]:
    """Parse LLM output into a list even when it adds light prose around JSON."""
    content = content.strip()
    if not content:
        return []

    parsed = _try_parse_query_json(content)
    if parsed is not None:
        return parsed

    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        parsed = _try_parse_query_json(match.group(0))
        if parsed is not None:
            return parsed

    queries = []
    for line in content.splitlines():
        cleaned = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip().strip('"')
        if cleaned:
            queries.append(cleaned)
    return queries


def _try_parse_query_json(content: str) -> Optional[List[str]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        parsed = parsed.get("queries", [])

    if not isinstance(parsed, list):
        return []

    return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]


def _dedupe_queries(queries: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for query in queries:
        normalized = " ".join(query.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _normalize_query_text(query: str) -> str:
    replacements = str.maketrans(
        "áéíóúüñÁÉÍÓÚÜÑ¿¡",
        "aeiouunAEIOUUN  ",
    )
    return query.translate(replacements).lower()


def _extract_primary_question(query: str) -> str:
    """Prefer the research question over surrounding instructions."""
    candidates = []
    for pattern in (r'"([^"]{12,})"', r"“([^”]{12,})”", r"‘([^’]{12,})’", r"'([^']{12,})'"):
        candidates.extend(match.strip() for match in re.findall(pattern, query))

    question_candidates = [candidate for candidate in candidates if "?" in candidate or "¿" in candidate]
    if question_candidates:
        return max(question_candidates, key=len)
    if candidates:
        return max(candidates, key=len)

    cleaned = query.strip()
    instruction_markers = [
        "Primero,",
        "Luego,",
        "Finalmente,",
        "Redacta ",
        "Todas las afirmaciones",
        "Si la evidencia",
    ]
    for marker in instruction_markers:
        marker_index = cleaned.find(marker)
        if marker_index > 40:
            cleaned = cleaned[:marker_index].strip()
            break

    prefix_patterns = [
        r"^.*?pregunta\s*:?\s*",
        r"^.*?question\s*:?\s*",
    ]
    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    return cleaned or query.strip()


def _keyword_query(query: str) -> str:
    normalized = _normalize_query_text(query)
    keywords: List[str] = []
    seen = set()
    for term in re.findall(r"[a-z0-9]{4,}", normalized):
        if term in _GENERIC_RETRIEVAL_TERMS or term in seen:
            continue
        seen.add(term)
        keywords.append(term)
        if len(keywords) >= 14:
            break
    return " ".join(keywords)


def _bilingual_query_hints(query: str) -> List[str]:
    normalized = _normalize_query_text(query)
    hints: List[str] = []

    if re.search(r"sostenib|ambiental|environmental|sustain", normalized):
        hints.append("environmental sustainability technologies strategies")

    if re.search(r"sostenib|ambiental|environmental|sustain", normalized) and re.search(
        r"tecnolog|technology|technologies|estrateg|strategy|strategies", normalized
    ):
        hints.append("sostenibilidad ambiental tecnologias estrategias agua energia residuos urbano")
        hints.append("sustainable technologies environmental strategies water energy waste urban")

    if re.search(r"energ|energy|renovable|renewable|hydro|hidro", normalized):
        hints.append("renewable energy decentralized energy micro hydro")

    if re.search(r"agua|water|desinfe|potable", normalized):
        hints.append("water treatment disinfection potable water")

    if re.search(r"resum|summary|summar|sintetiz|synthesi", normalized):
        hints.append("abstract research objective methodology principal results conclusions")
        hints.append("resumen objetivo metodologia resultados principales conclusiones")

    if re.search(r"aire|air|atmosfer|atmospher|contamina|pollut|emision|emission", normalized):
        hints.append("urban air quality pollution emissions mitigation technology")

    if (
        re.search(r"aire|air|atmosfer|atmospher", normalized)
        and re.search(r"agua|water", normalized)
        and re.search(r"energ|energy", normalized)
    ):
        hints.append("integrated urban air water energy technologies environmental performance")

    if re.search(r"residu|waste|basura|circular", normalized):
        hints.append("waste management circular economy zero waste")

    if re.search(r"urban|ciudad|city|building|edificio", normalized):
        hints.append("urban sustainability green infrastructure buildings")

    return hints


def _candidate_key(chunk: Dict[str, Any]) -> str:
    for key in ("id", "chunk_id"):
        if chunk.get(key):
            return str(chunk[key])

    text = str(chunk.get("text", ""))
    return "|".join(
        str(part)
        for part in (
            chunk.get("document_id", ""),
            chunk.get("filename", ""),
            chunk.get("page", ""),
            text[:200],
        )
    )


def _merge_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for candidate in candidates:
        candidate_copy = dict(candidate)
        key = _candidate_key(candidate_copy)
        existing = merged.get(key)

        if existing is None or candidate_copy.get("score", 0) > existing.get("score", 0):
            merged[key] = candidate_copy

    return list(merged.values())


def _relevance_score(chunk: Dict[str, Any]) -> float:
    for key in ("rerank_score", "retrieval_score", "score"):
        if key not in chunk:
            continue
        try:
            return float(chunk[key])
        except (TypeError, ValueError):
            continue
    return 0.0


def _diversify_by_document(chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Keep strong chunks while avoiding a final context dominated by one document."""
    if len(chunks) <= top_k:
        return chunks

    ranked = sorted(chunks, key=_relevance_score, reverse=True)
    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    per_document_counts: Dict[str, int] = {}

    def add_chunk(chunk: Dict[str, Any]) -> bool:
        key = _candidate_key(chunk)
        if key in selected_keys:
            return False
        document_key = str(chunk.get("document_id") or chunk.get("filename") or "unknown")
        selected.append(chunk)
        selected_keys.add(key)
        per_document_counts[document_key] = per_document_counts.get(document_key, 0) + 1
        return True

    # First pass: include the best available evidence from each document.
    seen_documents = set()
    for chunk in ranked:
        document_key = str(chunk.get("document_id") or chunk.get("filename") or "unknown")
        if document_key in seen_documents:
            continue
        add_chunk(chunk)
        seen_documents.add(document_key)
        if len(selected) >= top_k:
            return selected

    # Second pass: fill with the strongest remaining chunks, capped per document.
    for chunk in ranked:
        document_key = str(chunk.get("document_id") or chunk.get("filename") or "unknown")
        if per_document_counts.get(document_key, 0) >= MAX_CHUNKS_PER_DOCUMENT:
            continue
        add_chunk(chunk)
        if len(selected) >= top_k:
            return selected

    # Final pass: if there were too few documents, finish by relevance.
    for chunk in ranked:
        add_chunk(chunk)
        if len(selected) >= top_k:
            break

    return selected


def transform_query(query: str) -> List[str]:
    """Rewrite a user question into multiple retrieval-friendly search queries."""
    original = query.strip()
    if not original:
        return []

    primary = _extract_primary_question(original)
    keyword_query = _keyword_query(primary)
    variants = [primary]
    if keyword_query and keyword_query.lower() != primary.lower():
        variants.append(keyword_query)
    variants.extend(_bilingual_query_hints(primary))

    if len(original) <= 220 and original != primary:
        variants.append(original)

    return _dedupe_queries(variants)[:MAX_QUERY_VARIANTS]

@trace_function(
    "retrieve",
    metadata_factory=lambda query, user_id, document_id=None, top_k=None: {
        "user_id": user_id,
        "document_id": document_id,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "top_k_retrieval": settings.TOP_K_RETRIEVAL,
        "top_k_rerank": settings.TOP_K_RERANK,
    },
)

def retrieve(
    query: str,
    user_id: str,
    document_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval pipeline:
    1. Hybrid Search (Vector + BM25 via EnsembleRetriever with RRF) with Query Transformation
    2. Cross-encoder reranking (top-K refined)

    Returns chunks with confidence scores.
    """
    # ── Stage 1: Hybrid Search with Query Transformation ─────────────
    effective_top_k = top_k if top_k is not None else settings.TOP_K_RETRIEVAL
    vector_retriever = CustomVectorRetriever(
        user_id=user_id,
        document_id=document_id,
        top_k=effective_top_k,
    )

    bm25_retriever = CustomBM25Retriever(
        user_id=user_id,
        document_id=document_id,
        top_k=effective_top_k,
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    all_candidates = []
    search_queries = transform_query(query)
    ranking_query = " ".join(search_queries[:3]) if search_queries else query

    for search_query in search_queries:
        docs = ensemble_retriever.invoke(search_query)
        for i, doc in enumerate(docs):
            chunk = doc.metadata.copy()
            chunk["retrieval_score"] = 1.0 / (i + 1)
            chunk.setdefault("score", chunk["retrieval_score"])
            all_candidates.append(chunk)

    if not all_candidates:
        return []

    candidates = _merge_candidates(all_candidates)
    diverse_seeds = _diversify_by_document(candidates, settings.TOP_K_RERANK)

    # ── Stage 2: Cross-encoder reranking ─────────────
    reranker = get_reranker()
    
    if reranker is not None:
        top_chunks = reranker.rerank(
            query=ranking_query,
            documents=candidates,
            top_k=settings.TOP_K_RERANK * RERANK_POOL_MULTIPLIER
        )
    else:
        # Fall back to hybrid scores (no reranker)
        candidates.sort(key=lambda x: x.get("retrieval_score", x.get("score", 0)), reverse=True)
        top_chunks = candidates[: settings.TOP_K_RERANK * RERANK_POOL_MULTIPLIER]

    top_chunks = _merge_candidates([*top_chunks, *diverse_seeds])
    top_chunks = _diversify_by_document(top_chunks, settings.TOP_K_RERANK)

    # top_chunks is now always defined
    # ── Calculate confidence percentages ─────────────
    if top_chunks:
        raw_scores = [_relevance_score(chunk) for chunk in top_chunks]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max(max_score - min_score, 0.001)

        for chunk in top_chunks:
            raw = _relevance_score(chunk)
            chunk["confidence"] = round(((raw - min_score) / score_range) * 100, 1)
            if "rerank_score" in chunk:
                chunk["score"] = round(chunk["rerank_score"], 4)
                del chunk["rerank_score"]

    return top_chunks
