"""Runtime instrumentation and ablation switches for ATLAS.

Nothing in ``backend/app`` is modified. At process start ``install()`` wraps a
handful of module-level functions so each evaluated answer records:

* every citation validation (and why it rejected the draft),
* every deterministic evidence check (uncited claims, unsupported numbers),
* every pruning and repair, tagged with the research stage it ran in,
* the wall time of each research-graph node (wrapped as LangGraph registers it),
* Ollama token counts and durations for every LLM call.

``condition()`` then disables components for the C1–C4 ablations by swapping
the same module attributes the production code looks up at call time.
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

CONDITIONS = {
    "C1": "Recuperación densa únicamente (línea base)",
    "C2": "Híbrida densa + léxica con RRF, sin reordenamiento",
    "C3": "Híbrida + reordenamiento (simple_rag sin grafo)",
    "C4": "C3 + contexto del grafo (simple_rag tal como está en producción)",
    "C5": "research_rag completo (planificación, auditoría, rondas correctivas, verificación, reparación)",
    # Extensions after the paper (feature/claim-mediation); C1–C5 are unchanged.
    "C4M": "C4 + mediación de afirmaciones con cita textual verificada",
    "C5S": "C5 con las auditorías del propio modelo como señal (sin filtrar evidencia ni disparar la reparación)",
    "C5M": "C5S + mediación de afirmaciones en la síntesis y en la reparación",
}
MEDIATION_CONDITIONS = {"C4M", "C5M"}
AUDIT_SIGNAL_CONDITIONS = {"C5S", "C5M"}


def is_research(name: str) -> bool:
    """Conditions that run research_rag instead of simple_rag."""
    return name.startswith("C5")


class Recorder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.stages: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.retrieve_seconds = 0.0
        self.retrieve_calls = 0
        self.graph_seconds = 0.0
        self.graph_context = ""
        self.delivered: Optional[List[Dict[str, Any]]] = None
        self.stage_stack: List[str] = []

    @property
    def stage(self) -> Optional[str]:
        return self.stage_stack[-1] if self.stage_stack else None

    def event(self, kind: str, **data: Any) -> None:
        self.events.append({"kind": kind, "stage": self.stage, "t": time.perf_counter(), **data})


RESEARCH_NODES = ("understand", "retrieve", "ledger", "audit", "outline", "draft", "verify", "repair", "finalize")
REC = Recorder()
_CITATION_RE = re.compile(r"\[((?:D|W)\d+)\]")
_INSTALLED = False


def _classify_citations(agent, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sources:
        return {"category": "no_sources", "cited": 0, "valid": 0}
    normalized = agent._normalize_legacy_citations(answer, sources)
    if re.search(r"\[Fuente:\s*.+?\]", normalized, flags=re.IGNORECASE):
        return {"category": "legacy_label", "cited": 0, "valid": 0}
    valid_ids = {str(s.get("source_id")) for s in sources if s.get("source_id")}
    cited = _CITATION_RE.findall(normalized)
    cited_unique = set(cited)
    if not cited_unique:
        return {"category": "no_citations", "cited": 0, "valid": 0}
    invalid = cited_unique - valid_ids
    return {
        "category": "invented_citation" if invalid else "ok",
        "cited": len(cited),
        "valid": sum(1 for cid in cited if cid in valid_ids),
        "invalid_ids": sorted(invalid),
    }


def install() -> None:
    """Wrap production functions once per process (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    import langchain_ollama
    from langchain_core.callbacks import BaseCallbackHandler

    from app.rag import agent

    # ── Citation validation ──────────────────────────────────────────
    original_validate = agent._validate_answer_citations

    def validate(answer, sources):
        if answer:
            REC.event("validate_citations", **_classify_citations(agent, answer, sources))
        return original_validate(answer, sources)

    agent._validate_answer_citations = validate

    # ── Deterministic evidence checks ────────────────────────────────
    original_issues = agent._answer_evidence_issues

    def evidence_issues(answer, sources, evidence_chunks=None, verify_entailment=False):
        issues = original_issues(answer, sources, evidence_chunks, verify_entailment=verify_entailment)
        REC.event(
            "evidence_check",
            issues=list(issues),
            # The system also checks its own "insufficient evidence" placeholder;
            # those checks say nothing about the model's claims.
            placeholder=(answer or "").strip() == agent.INSUFFICIENT_EVIDENCE_MESSAGE,
            unsupported_number=any("valores no presentes" in issue for issue in issues),
            uncited_claims=any("no tienen una cita inmediata" in issue for issue in issues),
        )
        return issues

    agent._answer_evidence_issues = evidence_issues

    # ── Pruning ──────────────────────────────────────────────────────
    original_prune = agent._prune_unsupported_claims

    def prune(answer, sources, evidence_chunks=None, verify_entailment=False):
        pruned = original_prune(answer, sources, evidence_chunks, verify_entailment=verify_entailment)
        before = [s for s in agent._answer_sentences(answer) if agent._is_substantive_claim(s)]
        after = [s for s in agent._answer_sentences(pruned) if agent._is_substantive_claim(s)]
        REC.event("prune", changed=(pruned or "").strip() != (answer or "").strip(),
                  claims_before=len(before), claims_after=len(after))
        return pruned

    agent._prune_unsupported_claims = prune

    # ── Research verify / repair ─────────────────────────────────────
    original_semantic = agent._semantic_claim_issues

    def semantic_issues(answer, sources, evidence):
        issues = original_semantic(answer, sources, evidence)
        REC.event("verify", issues=list(issues), issue_count=len(issues))
        return issues

    agent._semantic_claim_issues = semantic_issues

    original_repair = agent._repair_research_answer

    def repair(*args, **kwargs):
        REC.event("repair_called")
        return original_repair(*args, **kwargs)

    agent._repair_research_answer = repair

    # ── Claim mediation (C4M, C5M) ───────────────────────────────────
    original_mediate = agent.mediate_claims

    def mediate(*args, **kwargs):
        outcome = original_mediate(*args, **kwargs)
        REC.event("mediate", **outcome.stats)
        return outcome

    agent.mediate_claims = mediate

    # ── Retrieval and graph timing; delivered evidence of simple_rag ─
    original_retrieve = agent.retrieve

    def timed_retrieve(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_retrieve(*args, **kwargs)
        finally:
            REC.retrieve_seconds += time.perf_counter() - start
            REC.retrieve_calls += 1

    agent.retrieve = timed_retrieve

    original_graph = agent.get_entity_context

    def timed_graph(*args, **kwargs):
        start = time.perf_counter()
        try:
            context = original_graph(*args, **kwargs)
            REC.graph_context = context or ""
            return context
        finally:
            REC.graph_seconds += time.perf_counter() - start

    agent.get_entity_context = timed_graph

    original_evidence = agent._retrieve_document_evidence

    def document_evidence(*args, **kwargs):
        context, sources, chunks = original_evidence(*args, **kwargs)
        REC.delivered = chunks
        REC.event("evidence_delivered", chunks=len(chunks), has_context=bool(context))
        return context, sources, chunks

    agent._retrieve_document_evidence = document_evidence

    # ── Research node timing ─────────────────────────────────────────
    # Wrap every node as research_agent registers it on the LangGraph graph.
    # Works on both branches: experimental also wraps nodes in Langfuse spans,
    # the PC branch registers them bare.
    from langgraph.graph import StateGraph

    original_add_node = StateGraph.add_node

    def timed_node(name, action):
        def wrapped(state):
            start = time.perf_counter()
            REC.stage_stack.append(name)
            try:
                return action(state)
            finally:
                REC.stage_stack.pop()
                REC.stages.append({
                    "name": name,
                    "seconds": time.perf_counter() - start,
                    "round": state.get("round") if isinstance(state, dict) else None,
                })

        return wrapped

    def add_node(self, node, action=None, *args, **kwargs):
        if isinstance(node, str) and node in RESEARCH_NODES and callable(action):
            action = timed_node(node, action)
        return original_add_node(self, node, action, *args, **kwargs)

    StateGraph.add_node = add_node

    # ── Ollama token accounting ──────────────────────────────────────
    class TokenHandler(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            for generations in response.generations or []:
                for generation in generations:
                    info = dict(getattr(generation, "generation_info", None) or {})
                    message = getattr(generation, "message", None)
                    info.update(getattr(message, "response_metadata", None) or {})
                    usage = getattr(message, "usage_metadata", None) or {}

                    def seconds(key):
                        value = info.get(key)
                        return round(value / 1e9, 3) if isinstance(value, (int, float)) else None

                    REC.llm_calls.append({
                        "stage": REC.stage,
                        "prompt_tokens": info.get("prompt_eval_count") or usage.get("input_tokens"),
                        "completion_tokens": info.get("eval_count") or usage.get("output_tokens"),
                        "prompt_eval_seconds": seconds("prompt_eval_duration"),
                        "eval_seconds": seconds("eval_duration"),
                        "load_seconds": seconds("load_duration"),
                        "total_seconds": seconds("total_duration"),
                        "done_reason": info.get("done_reason"),
                    })

    handler = TokenHandler()
    base_chat = langchain_ollama.ChatOllama

    class CountingChatOllama(base_chat):
        def __init__(self, **kwargs):
            callbacks = list(kwargs.pop("callbacks", None) or [])
            callbacks.append(handler)
            super().__init__(callbacks=callbacks, **kwargs)

    # experimental: agent.py imports the class, retriever.py imports it lazily.
    # PC branch: every client is built in app.rag.llm_client.
    langchain_ollama.ChatOllama = CountingChatOllama
    if hasattr(agent, "ChatOllama"):
        agent.ChatOllama = CountingChatOllama
    try:
        from app.rag import llm_client

        llm_client.ChatOllama = CountingChatOllama
    except ImportError:
        pass

    _INSTALLED = True


@contextmanager
def _swap(module, name: str, value) -> Iterator[None]:
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


@contextmanager
def condition(name: str) -> Iterator[None]:
    """Disable components for one ablation condition (C4 and C5 run untouched)."""
    from contextlib import ExitStack

    from app.rag import agent, retriever

    with ExitStack() as stack:
        if name in MEDIATION_CONDITIONS:
            stack.enter_context(_swap(agent.settings, "CLAIM_MEDIATION_ENABLED", True))
        if name in AUDIT_SIGNAL_CONDITIONS:
            stack.enter_context(_swap(agent.settings, "RESEARCH_AUDIT_FILTER", False))
            stack.enter_context(_swap(agent.settings, "RESEARCH_SEMANTIC_AUDIT_MODE", "signal"))
        # The evaluation must never escape to the tool agent (web / MCP).
        # Only the experimental branch has this escalation path.
        if hasattr(agent, "_fallback_tool_agent_answer"):
            stack.enter_context(_swap(agent, "_fallback_tool_agent_answer", lambda *a, **k: None))
        if name in {"C1", "C2", "C3"}:
            stack.enter_context(_swap(agent, "get_entity_context", lambda *a, **k: ""))
        if name in {"C1", "C2"}:
            stack.enter_context(_swap(retriever, "get_reranker", lambda *a, **k: None))
        if name == "C1":
            stack.enter_context(_swap(retriever, "query_lexical_chunks", lambda *a, **k: []))
        yield


# ── Fixed-depth rankings for Recall@20 / nDCG@10 (no generation) ─────────


def rankings(query: str, user_id: str, depth: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    """Dense, RRF and reranked rankings built with the same functions and pool
    sizes that ``retriever.retrieve`` uses on the simple_rag path, but without
    the leading-cluster cut and per-document cap, so every condition is scored
    at the same depth."""
    from app.config import get_settings
    from app.rag import retriever
    from app.rag.embeddings import embed_query
    from app.rag.vectorstore import query_chunks

    settings = get_settings()
    pool = max(settings.TOP_K_RERANK, settings.TOP_K_RETRIEVAL, depth)
    dense = query_chunks(query_embedding=embed_query(query), user_id=user_id, top_k=pool)
    lexical = retriever.query_lexical_chunks(query=query, user_id=user_id, top_k=pool)
    fused = retriever.reciprocal_rank_fusion([dense, lexical], weights=[0.6, 0.4])
    reranker = retriever.get_reranker()
    reranked = reranker.rerank(
        query=query,
        documents=[dict(c) for c in fused[: retriever.MAX_FACET_CANDIDATES]],
        top_k=retriever.MAX_FACET_CANDIDATES,
    )
    return {
        "dense": dense[:depth],
        "lexical": lexical[:depth],
        "rrf": fused[:depth],
        "rerank": reranked[:depth],
    }


def chunk_record(chunk: Dict[str, Any], with_context: bool = False) -> Dict[str, Any]:
    record = {
        "source_id": chunk.get("source_id"),
        "document_id": str(chunk.get("document_id") or ""),
        "filename": chunk.get("filename"),
        "page": chunk.get("page"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "section": chunk.get("section"),
        "score": chunk.get("relevance_score", chunk.get("score")),
        "facet_ids": chunk.get("facet_ids") or [],
        "text": chunk.get("text", ""),
    }
    if with_context:
        record["context_text"] = chunk.get("context_text") or chunk.get("parent_text") or chunk.get("text", "")
    return record
