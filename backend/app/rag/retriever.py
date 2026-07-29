"""Domain-neutral hybrid retrieval with RRF and model-relative reranking."""
import logging
import re
import statistics
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
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.rag.embeddings import embed_query
from app.rag.llm_client import create_chat_ollama
from app.rag.tracing import trace_function
from app.rag.vectorstore import query_chunks
try:
    from app.rag.vectorstore import query_document_profiles, query_lexical_chunks
except ImportError:
    def query_lexical_chunks(
        query: str, user_id: str, document_id=None, document_ids=None, top_k: int = 50
    ):
        from app.rag.bm25 import query_bm25
        results = query_bm25(query, user_id, document_id, top_k)
        if document_ids:
            allowed = set(map(str, document_ids))
            results = [item for item in results if str(item.get("document_id")) in allowed]
        return results

    def query_document_profiles(query_embedding, user_id, document_id=None, top_k=20):
        return []
from app.rag.reranker import get_reranker

logger = logging.getLogger(__name__)
settings = get_settings()
MAX_QUERY_VARIANTS = 8
RERANK_POOL_MULTIPLIER = 3
MAX_CHUNKS_PER_DOCUMENT = 3
RRF_K = 60
MAX_FACET_CANDIDATES = 40
MAX_EVIDENCE_PER_FACET = 10


class ResearchBrief(BaseModel):
    """Separate evidence needs from the user's output contract."""

    main_question: str
    facets: List[str]
    deliverables: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)

    @field_validator("main_question")
    @classmethod
    def normalize_main_question(cls, value: str) -> str:
        normalized = " ".join(str(value or "").split())
        if not normalized:
            raise ValueError("main_question cannot be empty")
        return normalized[:2000]

    @field_validator("facets")
    @classmethod
    def normalize_facets(cls, values: List[str]) -> List[str]:
        normalized = []
        seen = set()
        for value in values or []:
            facet = " ".join(str(value or "").split())
            key = facet.lower()
            if facet and key not in seen:
                seen.add(key)
                normalized.append(facet)
            if len(normalized) >= 6:
                break
        if not normalized:
            raise ValueError("at least one facet is required")
        return normalized

    @field_validator("deliverables", "constraints")
    @classmethod
    def normalize_instructions(cls, values: List[str]) -> List[str]:
        return [" ".join(str(value).split())[:500] for value in (values or []) if str(value).strip()][:8]


# Backwards-compatible import name used by routes and existing integrations.
ResearchPlan = ResearchBrief


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


def build_research_plan(query: str) -> ResearchBrief:
    """Create a constrained, domain-independent retrieval plan for research routes."""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = create_chat_ollama(
            temperature=0,
            reasoning=False if settings.LLM_DISABLE_THINKING else None,
            num_predict=settings.RETRIEVAL_PLANNER_MAX_TOKENS,
            timeout_seconds=settings.RETRIEVAL_PLANNER_TIMEOUT_SECONDS,
        )
        # Some Ollama/llama.cpp combinations reject generated JSON-schema
        # grammars. JSON mode still constrains generation to JSON, while the
        # Pydantic parser validates and normalizes the resulting object.
        planner = llm.with_structured_output(ResearchBrief, method="json_mode")
        no_think = "/no_think\n" if settings.LLM_DISABLE_THINKING else ""
        plan = planner.invoke([
            SystemMessage(
                content=(
                    no_think
                    +
                    "Convert the request into a research brief. Separate the substantive question, evidence facets, "
                    "requested deliverables, and evidence constraints. Preserve the user's language. Create no more "
                    "than six atomic facets. A facet must be a question answerable from document content; formatting, "
                    "roles, report sections, citation instructions, abstracts, and keywords belong in deliverables "
                    "or constraints and must never be facets. Do not answer the question. "
                    "Return only one JSON object with this exact shape: "
                    '{"main_question":"...","facets":["..."],"deliverables":["..."],"constraints":["..."]}.'
                )
            ),
            HumanMessage(content=query),
        ])
        facets = _dedupe_queries(plan.facets)[: settings.RESEARCH_MAX_FACETS]
        return ResearchBrief(
            main_question=plan.main_question.strip(),
            facets=facets or [plan.main_question.strip()],
            deliverables=plan.deliverables,
            constraints=plan.constraints,
        )
    except Exception as exc:
        logger.warning("Structured research planning failed; using generic query segmentation: %s", exc)
        original = " ".join(query.split()) or "general research question"
        # A single clean query is safer than treating report instructions as evidence facets.
        return ResearchBrief(main_question=original, facets=[original], deliverables=[], constraints=[])


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


def reciprocal_rank_fusion(
    rankings: List[List[Dict[str, Any]]],
    weights: Optional[List[float]] = None,
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """Fuse heterogeneous rankings without comparing their raw scores."""
    weights = weights or [1.0] * len(rankings)
    fused: Dict[str, Dict[str, Any]] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, candidate in enumerate(ranking, 1):
            key = _candidate_key(candidate)
            current = fused.setdefault(key, dict(candidate))
            current["rrf_score"] = float(current.get("rrf_score", 0.0)) + weight / (k + rank)
            current.setdefault("retrieval_channels", 0)
            current["retrieval_channels"] += 1
    return sorted(fused.values(), key=lambda item: item.get("rrf_score", 0.0), reverse=True)


def _relative_rerank_selection(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the leading cluster using score gaps, never a global probability cutoff."""
    if not ranked:
        return []
    limited = ranked[:MAX_EVIDENCE_PER_FACET]
    raw_scores = [float(item.get("rerank_score", 0.0)) for item in limited]
    if len(raw_scores) < 4 or not any("rerank_score" in item for item in limited):
        return limited

    gaps = [raw_scores[index] - raw_scores[index + 1] for index in range(len(raw_scores) - 1)]
    median_gap = statistics.median(abs(gap) for gap in gaps) if gaps else 0.0
    significant = max(0.25, median_gap * 3.0)
    for index, gap in enumerate(gaps, 1):
        if index >= 3 and gap >= significant:
            return limited[:index]
    return limited


def _relevance_score(chunk: Dict[str, Any]) -> float:
    for key in ("relevance_score", "retrieval_score", "score"):
        if key not in chunk:
            continue
        try:
            return float(chunk[key])
        except (TypeError, ValueError):
            continue
    return 0.0


def _select_facet_evidence(chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Maximize supported facet and document coverage before filling by score."""
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

    all_facets = {facet_id for chunk in ranked for facet_id in chunk.get("facet_ids") or []}
    covered_facets = set()
    while covered_facets != all_facets and len(selected) < top_k:
        candidates = [
            chunk
            for chunk in ranked
            if _candidate_key(chunk) not in selected_keys
            and (set(chunk.get("facet_ids") or []) - covered_facets)
            and per_document_counts.get(
                str(chunk.get("document_id") or chunk.get("filename") or "unknown"), 0
            ) < MAX_CHUNKS_PER_DOCUMENT
        ]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda chunk: (
                str(chunk.get("document_id") or chunk.get("filename") or "unknown")
                not in per_document_counts,
                len(set(chunk.get("facet_ids") or []) - covered_facets),
                _relevance_score(chunk),
            ),
        )
        add_chunk(best)
        covered_facets.update(best.get("facet_ids") or [])

    for chunk in ranked:
        document_key = str(chunk.get("document_id") or chunk.get("filename") or "unknown")
        if per_document_counts.get(document_key, 0) >= MAX_CHUNKS_PER_DOCUMENT:
            continue
        add_chunk(chunk)
        if len(selected) >= top_k:
            return selected

    return selected


def transform_query(query: str) -> List[str]:
    """Provide a domain-neutral fallback when structured planning is unavailable."""
    original = query.strip()
    if not original:
        return []
    segments = [
        segment.strip(" -")
        for segment in re.split(r"(?:\r?\n)+|(?<=[?!.;])\s+", original)
        if len(segment.strip()) >= 8
    ]
    return _dedupe_queries([original, *segments])[:MAX_QUERY_VARIANTS]

@trace_function(
    "retrieve",
    metadata_factory=lambda query, user_id, document_id=None, top_k=None, facets=None: {
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
    facets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval pipeline:
    1. Dense and global lexical rankings fused with reciprocal rank fusion.
    2. Model-relative reranking and facet-aware evidence selection.

    Returns chunks with confidence scores.
    """
    # ── Stage 1: Hybrid Search with Query Transformation ─────────────
    effective_top_k = top_k if top_k is not None else settings.TOP_K_RERANK
    final_top_k = min(effective_top_k, settings.TOP_K_RERANK)
    candidate_top_k = max(50, settings.TOP_K_RETRIEVAL) if facets else max(
        effective_top_k, settings.TOP_K_RETRIEVAL
    )
    merged_evidence: Dict[str, Dict[str, Any]] = {}
    scoped_fallback: Dict[str, Dict[str, Any]] = {}
    search_queries = _dedupe_queries(facets or [query])[:MAX_QUERY_VARIANTS]
    requested_facets = {f"F{index}": facet for index, facet in enumerate(search_queries, 1)}
    reranker = get_reranker()
    document_shortlist = None
    if not document_id and facets:
        document_shortlist = query_document_profiles(
            query_embedding=embed_query(query),
            user_id=user_id,
            top_k=20,
        ) or None
        if document_shortlist:
            logger.info(
                "Document profile shortlist selected %d candidates for deep retrieval.",
                len(document_shortlist),
            )

    for facet_index, search_query in enumerate(search_queries, 1):
        facet_id = f"F{facet_index}"
        vector_kwargs = {
            "query_embedding": embed_query(search_query),
            "user_id": user_id,
            "document_id": document_id,
            "top_k": candidate_top_k,
        }
        lexical_kwargs = {
            "query": search_query,
            "user_id": user_id,
            "document_id": document_id,
            "top_k": candidate_top_k,
        }
        if document_shortlist:
            vector_kwargs["document_ids"] = document_shortlist
            lexical_kwargs["document_ids"] = document_shortlist
        vector_candidates = query_chunks(**vector_kwargs)
        lexical_candidates = query_lexical_chunks(**lexical_kwargs)
        candidates = reciprocal_rank_fusion([vector_candidates, lexical_candidates], weights=[0.6, 0.4])
        if not candidates:
            continue
        if reranker is not None:
            ranked = reranker.rerank(
                query=search_query,
                documents=candidates[:MAX_FACET_CANDIDATES],
                top_k=MAX_FACET_CANDIDATES,
            )
        else:
            ranked = candidates[:MAX_FACET_CANDIDATES]
            for rank, chunk in enumerate(ranked, 1):
                chunk["relevance_score"] = 1.0 / rank
                chunk["rerank_rank"] = rank
        ranked = _relative_rerank_selection(ranked)
        if document_id:
            for chunk in ranked[:final_top_k]:
                key = _candidate_key(chunk)
                current = scoped_fallback.get(key)
                if current is None or _relevance_score(chunk) > _relevance_score(current):
                    scoped_fallback[key] = dict(chunk)
        for chunk in ranked:
            relevance = _relevance_score(chunk)
            key = _candidate_key(chunk)
            current = merged_evidence.get(key)
            if current is None:
                current = dict(chunk)
                current["facet_ids"] = []
                current["facet_scores"] = {}
                current["facet_queries"] = {}
                merged_evidence[key] = current
            current["facet_ids"].append(facet_id)
            current["facet_scores"][facet_id] = round(relevance, 4)
            current["facet_queries"][facet_id] = search_query
            current["context_text"] = current.get("parent_text") or current.get("text", "")
            if relevance >= _relevance_score(current):
                current["relevance_score"] = relevance
                current["score"] = relevance

    if not merged_evidence and document_id:
        for chunk in scoped_fallback.values():
            chunk["facet_ids"] = []
            chunk["facet_scores"] = {}
            chunk["facet_queries"] = {}
        merged_evidence = scoped_fallback

    top_chunks = _select_facet_evidence(list(merged_evidence.values()), final_top_k)
    for chunk in top_chunks:
        relevance = _relevance_score(chunk)
        chunk["score"] = round(relevance, 4)
        chunk["confidence"] = round(relevance * 100.0, 1)
        chunk["requested_facets"] = requested_facets
        chunk.pop("rerank_score", None)
    return top_chunks
