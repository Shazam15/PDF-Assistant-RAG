"""Stateful evidence-seeking research workflow.

The graph exposes progress and evidence decisions, never hidden chain-of-thought.
"""
from __future__ import annotations

import logging
import json
import time
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable, Dict, Iterator, List, Optional, TypedDict

from app.config import get_settings
from app.rag.retriever import ResearchBrief

logger = logging.getLogger(__name__)
settings = get_settings()


class ResearchCancelled(RuntimeError):
    pass


class ResearchState(TypedDict, total=False):
    run_id: Optional[str]
    question: str
    user_id: str
    document_id: Optional[str]
    top_k: Optional[int]
    chat_history: List[Dict[str, str]]
    brief: ResearchBrief
    round: int
    repairs: int
    evidence: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    missing_facets: List[str]
    supported_facets: List[str]
    conflicts: List[str]
    claim_ledger: List[Dict[str, Any]]
    argument_outline: List[Dict[str, Any]]
    new_evidence_count: int
    answer: str
    issues: List[str]
    stage: str
    deadline: float


@dataclass
class ResearchDependencies:
    plan: Callable[[str], ResearchBrief]
    retrieve: Callable[..., List[Dict[str, Any]]]
    source_payload: Callable[[Dict[str, Any]], Dict[str, Any]]
    synthesize: Callable[
        [ResearchBrief, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]],
        str,
    ]
    verify: Callable[[str, List[Dict[str, Any]], List[Dict[str, Any]]], List[str]]
    repair: Callable[[ResearchBrief, List[Dict[str, Any]], List[Dict[str, Any]], str, List[str]], str]
    audit: Optional[Callable[[ResearchBrief, List[Dict[str, Any]]], Dict[str, List[str]]]] = None
    cancellation_event: Optional[Event] = None


def _candidate_key(chunk: Dict[str, Any]) -> str:
    return "|".join(
        str(value) for value in (
            chunk.get("id") or chunk.get("chunk_id") or "",
            chunk.get("document_id") or chunk.get("filename") or "",
            chunk.get("page") or "",
            str(chunk.get("text") or "")[:240],
        )
    )


def _check_budget(state: ResearchState, dependencies: ResearchDependencies) -> None:
    if dependencies.cancellation_event and dependencies.cancellation_event.is_set():
        raise ResearchCancelled("Research request cancelled")
    if time.monotonic() >= state["deadline"]:
        raise TimeoutError("Research time budget exhausted")


def _synthesize_best_available(
    state: ResearchState,
    dependencies: ResearchDependencies,
) -> ResearchState:
    """Produce a grounded answer from accumulated evidence after an early stop."""
    evidence = list(state.get("evidence", []))
    brief = state.get("brief")
    if brief is None:
        question = state.get("question") or "research question"
        brief = ResearchBrief(main_question=question, facets=[question])
        state["brief"] = brief

    for index, chunk in enumerate(evidence, 1):
        chunk["source_id"] = f"D{index}"
        chunk.setdefault("source_type", "document")
    sources = [dependencies.source_payload(chunk) for chunk in evidence]

    answer = ""
    if evidence:
        try:
            ledger = _build_claim_ledger(brief, evidence, state.get("supported_facets", []))
            outline = _build_argument_outline(ledger, evidence)
            answer = dependencies.synthesize(brief, evidence, sources, outline)
        except Exception as exc:
            logger.warning("Best-available research synthesis failed: %s", exc)
    if not answer:
        answer = (
            "No se recuperó evidencia documental suficiente para responder de forma verificable. "
            "No se completaron los vacíos con conocimiento externo."
        )
    state.update({
        "evidence": evidence,
        "sources": sources,
        "answer": answer,
        "stage": "completed",
    })
    return state


def _merge_evidence(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    merged = {_candidate_key(item): dict(item) for item in existing}
    before = len(merged)
    for item in incoming:
        key = _candidate_key(item)
        current = merged.get(key)
        if current is None:
            merged[key] = dict(item)
            continue
        current["facet_ids"] = sorted(set(current.get("facet_ids") or []) | set(item.get("facet_ids") or []))
        current["facet_queries"] = {**current.get("facet_queries", {}), **item.get("facet_queries", {})}
        current["requested_facets"] = {**current.get("requested_facets", {}), **item.get("requested_facets", {})}
        if float(item.get("relevance_score") or 0) > float(current.get("relevance_score") or 0):
            for field in ("relevance_score", "rerank_score", "rerank_rank", "score", "context_text"):
                if field in item:
                    current[field] = item[field]
    values = list(merged.values())
    values.sort(key=lambda item: (float(item.get("rerank_rank") or 9999), -float(item.get("score") or 0)))
    return values, len(merged) - before


def _build_claim_ledger(
    brief: ResearchBrief,
    evidence: List[Dict[str, Any]],
    supported: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Map each atomic evidence question to concrete retrieved excerpts."""
    supported_set = set(supported or [])
    ledger = []
    for claim_index, facet in enumerate(brief.facets, 1):
        evidence_indices = []
        for evidence_index, chunk in enumerate(evidence):
            recovered_queries = set(dict(chunk.get("facet_queries") or {}).values())
            if facet in recovered_queries:
                evidence_indices.append(evidence_index)
        ledger.append({
            "claim_id": f"C{claim_index}",
            "question": facet,
            "evidence_indices": evidence_indices,
            "status": "supported" if facet in supported_set else "candidate" if evidence_indices else "missing",
        })
    return ledger


def _build_argument_outline(
    ledger: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create an internal claim-evidence outline without inventing conclusions."""
    outline = []
    for entry in ledger:
        if entry.get("status") != "supported":
            continue
        source_ids = [
            evidence[index].get("source_id") or f"D{index + 1}"
            for index in entry.get("evidence_indices", [])
            if 0 <= index < len(evidence)
        ]
        outline.append({
            "claim_id": entry["claim_id"],
            "question": entry["question"],
            "source_ids": list(dict.fromkeys(source_ids)),
        })
    return outline


def _fallback_audit(brief: ResearchBrief, evidence: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    supported_queries = {
        query
        for chunk in evidence
        for query in dict(chunk.get("facet_queries") or {}).values()
    }
    supported = [facet for facet in brief.facets if facet in supported_queries]
    missing = [facet for facet in brief.facets if facet not in supported_queries]
    return {"supported": supported, "missing": missing, "conflicts": []}


def _build_graph(dependencies: ResearchDependencies):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None

    def understand(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        brief = dependencies.plan(state["question"])
        return {"brief": brief, "missing_facets": list(brief.facets), "stage": "planning"}

    def retrieve_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        facets = state.get("missing_facets") or state["brief"].facets
        incoming = dependencies.retrieve(
            query=state["brief"].main_question,
            user_id=state["user_id"],
            document_id=state.get("document_id"),
            top_k=state.get("top_k"),
            facets=facets,
        )
        merged, new_count = _merge_evidence(state.get("evidence", []), incoming)
        return {
            "evidence": merged,
            "new_evidence_count": new_count,
            "round": state.get("round", 0) + 1,
            "stage": "retrieving",
        }

    def ledger_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        return {
            "claim_ledger": _build_claim_ledger(state["brief"], state.get("evidence", [])),
            "stage": "building_ledger",
        }

    def audit_start_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        return {"stage": "auditing"}

    def audit_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        audit = (
            dependencies.audit(state["brief"], state.get("evidence", []))
            if dependencies.audit
            else _fallback_audit(state["brief"], state.get("evidence", []))
        )
        evidence = state.get("evidence", [])
        relevant_indices = audit.get("relevant_indices")
        if relevant_indices is not None:
            evidence = [item for index, item in enumerate(evidence) if index in set(relevant_indices)]
        supported = audit.get("supported", [])
        return {
            "evidence": evidence,
            "supported_facets": supported,
            "missing_facets": audit.get("missing", []),
            "conflicts": audit.get("conflicts", []),
            "claim_ledger": _build_claim_ledger(state["brief"], evidence, supported),
            "stage": "auditing",
        }

    def after_audit(state: ResearchState) -> str:
        synthesis_cutoff = state["deadline"] - settings.RESEARCH_SYNTHESIS_RESERVE_SECONDS
        can_retry = (
            bool(state.get("missing_facets"))
            and state.get("round", 0) < settings.RESEARCH_MAX_ROUNDS
            and state.get("new_evidence_count", 0) > 0
            and time.monotonic() < synthesis_cutoff
        )
        return "retrieve" if can_retry else "draft"

    def draft_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        evidence = list(state.get("evidence", []))
        for index, chunk in enumerate(evidence, 1):
            chunk["source_id"] = f"D{index}"
            chunk.setdefault("source_type", "document")
        sources = [dependencies.source_payload(chunk) for chunk in evidence]
        outline = _build_argument_outline(state.get("claim_ledger", []), evidence)
        answer = dependencies.synthesize(state["brief"], evidence, sources, outline)
        return {"evidence": evidence, "sources": sources, "answer": answer, "stage": "drafting"}

    def outline_node(state: ResearchState) -> Dict[str, Any]:
        return {
            "argument_outline": _build_argument_outline(
                state.get("claim_ledger", []), state.get("evidence", [])
            ),
            "stage": "outlining",
        }

    def draft_start_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        return {"stage": "drafting"}

    def verify_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        issues = dependencies.verify(state.get("answer", ""), state.get("sources", []), state.get("evidence", []))
        return {"issues": issues, "stage": "verifying"}

    def verify_start_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        return {"stage": "verifying"}

    def after_verify(state: ResearchState) -> str:
        if state.get("issues") and state.get("repairs", 0) < 1 and time.monotonic() < state["deadline"]:
            return "repair"
        return "finalize"

    def repair_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        answer = dependencies.repair(
            state["brief"],
            state.get("evidence", []),
            state.get("sources", []),
            state.get("answer", ""),
            state.get("issues", []),
        )
        return {"answer": answer, "repairs": state.get("repairs", 0) + 1, "stage": "repairing"}

    def repair_start_node(state: ResearchState) -> Dict[str, Any]:
        _check_budget(state, dependencies)
        return {"stage": "repairing"}

    def finalize_node(state: ResearchState) -> Dict[str, Any]:
        answer = str(state.get("answer") or "").strip()
        if not answer:
            answer = (
                "No se recuperó evidencia documental suficiente para responder de forma verificable. "
                "No se completaron los vacíos con conocimiento externo."
            )
        return {"answer": answer, "stage": "completed"}

    graph = StateGraph(ResearchState)
    graph.add_node("understand", understand)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("ledger", ledger_node)
    graph.add_node("audit_start", audit_start_node)
    graph.add_node("audit", audit_node)
    graph.add_node("outline", outline_node)
    graph.add_node("draft_start", draft_start_node)
    graph.add_node("draft", draft_node)
    graph.add_node("verify_start", verify_start_node)
    graph.add_node("verify", verify_node)
    graph.add_node("repair_start", repair_start_node)
    graph.add_node("repair", repair_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "understand")
    graph.add_edge("understand", "retrieve")
    graph.add_edge("retrieve", "ledger")
    graph.add_edge("ledger", "audit_start")
    graph.add_edge("audit_start", "audit")
    graph.add_conditional_edges("audit", after_audit, {"retrieve": "retrieve", "draft": "outline"})
    graph.add_edge("outline", "draft_start")
    graph.add_edge("draft_start", "draft")
    graph.add_edge("draft", "verify_start")
    graph.add_edge("verify_start", "verify")
    graph.add_conditional_edges("verify", after_verify, {"repair": "repair_start", "finalize": "finalize"})
    graph.add_edge("repair_start", "repair")
    graph.add_edge("repair", "verify_start")
    graph.add_edge("finalize", END)
    return graph.compile()


def _progress_payload(state: ResearchState) -> Dict[str, Any]:
    return {
        "stage": state.get("stage", "planning"),
        "round": state.get("round", 0),
        "max_rounds": settings.RESEARCH_MAX_ROUNDS,
        "facets_completed": len(state.get("supported_facets", [])),
        "facets_total": len(state.get("brief").facets) if state.get("brief") else 0,
        "documents": len({str(item.get("document_id") or item.get("filename")) for item in state.get("evidence", [])}),
        "evidence_count": len(state.get("evidence", [])),
    }


def _fallback_state_stream(
    initial: ResearchState,
    dependencies: ResearchDependencies,
) -> Iterator[ResearchState]:
    """Equivalent local executor used only when LangGraph is not installed."""
    state = dict(initial)
    _check_budget(state, dependencies)
    state["brief"] = dependencies.plan(state["question"])
    state["missing_facets"] = list(state["brief"].facets)
    state["stage"] = "planning"
    yield state

    while True:
        _check_budget(state, dependencies)
        facets = state.get("missing_facets") or state["brief"].facets
        incoming = dependencies.retrieve(
            query=state["brief"].main_question,
            user_id=state["user_id"],
            document_id=state.get("document_id"),
            top_k=state.get("top_k"),
            facets=facets,
        )
        evidence, new_count = _merge_evidence(state.get("evidence", []), incoming)
        state.update({
            "evidence": evidence,
            "new_evidence_count": new_count,
            "round": state.get("round", 0) + 1,
            "stage": "retrieving",
        })
        yield dict(state)
        state["claim_ledger"] = _build_claim_ledger(state["brief"], evidence)
        state["stage"] = "building_ledger"
        yield dict(state)
        state["stage"] = "auditing"
        yield dict(state)
        audit = (
            dependencies.audit(state["brief"], evidence)
            if dependencies.audit
            else _fallback_audit(state["brief"], evidence)
        )
        relevant_indices = audit.get("relevant_indices")
        if relevant_indices is not None:
            state["evidence"] = [
                item for index, item in enumerate(evidence) if index in set(relevant_indices)
            ]
        supported = audit.get("supported", [])
        state.update({
            "supported_facets": supported,
            "missing_facets": audit.get("missing", []),
            "conflicts": audit.get("conflicts", []),
            "claim_ledger": _build_claim_ledger(state["brief"], state["evidence"], supported),
            "stage": "auditing",
        })
        yield dict(state)
        if not (
            state.get("missing_facets")
            and state.get("round", 0) < settings.RESEARCH_MAX_ROUNDS
            and new_count > 0
            and time.monotonic() < state["deadline"] - settings.RESEARCH_SYNTHESIS_RESERVE_SECONDS
        ):
            break

    for index, chunk in enumerate(state.get("evidence", []), 1):
        chunk["source_id"] = f"D{index}"
        chunk.setdefault("source_type", "document")
    state["sources"] = [dependencies.source_payload(chunk) for chunk in state.get("evidence", [])]
    state["argument_outline"] = _build_argument_outline(
        state.get("claim_ledger", []), state.get("evidence", [])
    )
    state["stage"] = "outlining"
    yield dict(state)
    state["stage"] = "drafting"
    yield dict(state)
    state["answer"] = dependencies.synthesize(
        state["brief"], state["evidence"], state["sources"], state["argument_outline"]
    )
    state["stage"] = "drafting"
    yield dict(state)
    state["stage"] = "verifying"
    yield dict(state)
    state["issues"] = dependencies.verify(state["answer"], state["sources"], state["evidence"])
    state["stage"] = "verifying"
    yield dict(state)
    if state["issues"]:
        state["stage"] = "repairing"
        yield dict(state)
        state["answer"] = dependencies.repair(
            state["brief"], state["evidence"], state["sources"], state["answer"], state["issues"]
        )
        state["repairs"] = 1
        state["stage"] = "repairing"
        yield dict(state)
    state["stage"] = "completed"
    yield dict(state)


def _create_research_run(initial: ResearchState) -> Optional[str]:
    try:
        import uuid
        from app.database import SessionLocal
        from app.models import ResearchRun

        uuid.UUID(str(initial["user_id"]))
        db = SessionLocal()
        run = ResearchRun(
            user_id=initial["user_id"],
            document_id=initial.get("document_id"),
            question=initial["question"],
            status="running",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = str(run.id)
        db.close()
        return run_id
    except Exception as exc:
        logger.debug("Research checkpoint creation skipped: %s", exc)
        return None


def _checkpoint_research_run(state: ResearchState, status: str = "running") -> None:
    run_id = state.get("run_id")
    if not run_id:
        return
    try:
        from datetime import datetime, timezone
        from app.database import SessionLocal
        from app.models import ResearchRun

        db = SessionLocal()
        run = db.query(ResearchRun).filter(ResearchRun.id == run_id).first()
        if run:
            run.status = status
            run.rounds = state.get("round", 0)
            run.evidence_count = len(state.get("evidence", []))
            run.state_json = json.dumps({
                "stage": state.get("stage"),
                "supported_facets": state.get("supported_facets", []),
                "missing_facets": state.get("missing_facets", []),
                "conflicts": state.get("conflicts", []),
                "issues": state.get("issues", []),
                "claim_ledger": state.get("claim_ledger", []),
                "argument_outline": state.get("argument_outline", []),
                "evidence_refs": [
                    {
                        "id": item.get("id") or item.get("chunk_id"),
                        "document_id": item.get("document_id"),
                        "page": item.get("page"),
                    }
                    for item in state.get("evidence", [])
                ],
            }, ensure_ascii=False)
            if status in {"completed", "timed_out", "cancelled", "failed"}:
                run.completed_at = datetime.now(timezone.utc)
            db.commit()
        db.close()
    except Exception as exc:
        logger.debug("Research checkpoint update skipped: %s", exc)


def stream_research_agent(
    question: str,
    user_id: str,
    document_id: Optional[str],
    top_k: Optional[int],
    chat_history: Optional[List[Dict[str, str]]],
    dependencies: ResearchDependencies,
) -> Iterator[Dict[str, Any]]:
    initial: ResearchState = {
        "question": question,
        "user_id": user_id,
        "document_id": document_id,
        "top_k": top_k,
        "chat_history": chat_history or [],
        "round": 0,
        "repairs": 0,
        "evidence": [],
        "sources": [],
        "missing_facets": [],
        "supported_facets": [],
        "conflicts": [],
        "claim_ledger": [],
        "argument_outline": [],
        "new_evidence_count": 0,
        "answer": "",
        "issues": [],
        "stage": "planning",
        "deadline": time.monotonic() + settings.RESEARCH_TIMEOUT_SECONDS,
    }
    initial["run_id"] = _create_research_run(initial)
    graph = _build_graph(dependencies)
    state_stream = (
        graph.stream(initial, stream_mode="values", config={"recursion_limit": 24})
        if graph is not None
        else _fallback_state_stream(initial, dependencies)
    )

    last_state = initial
    last_stage = None
    try:
        for state in state_stream:
            last_state = state
            stage = state.get("stage")
            if stage != last_stage:
                last_stage = stage
                _checkpoint_research_run(state)
                yield {"type": "progress", "data": _progress_payload(state)}
    except ResearchCancelled:
        _checkpoint_research_run(last_state, "cancelled")
        raise
    except TimeoutError:
        logger.warning("Research graph reached its %ss budget; finalizing best available state.", settings.RESEARCH_TIMEOUT_SECONDS)
        last_state = _synthesize_best_available(dict(last_state), dependencies)
        terminal_status = "timed_out"
    else:
        terminal_status = "completed"
    _checkpoint_research_run(last_state, terminal_status)
    yield {
        "type": "result",
        "data": {
            "answer": last_state.get("answer") or (
                "La investigación alcanzó su presupuesto temporal sin producir una síntesis verificable. "
                "No se añadieron conclusiones externas a los documentos."
            ),
            "sources": last_state.get("sources", []),
            "evidence": last_state.get("evidence", []),
            "coverage": {
                "supported": last_state.get("supported_facets", []),
                "missing": last_state.get("missing_facets", []),
                "conflicts": last_state.get("conflicts", []),
            },
        },
    }


def run_research_agent(**kwargs) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for event in stream_research_agent(**kwargs):
        if event["type"] == "result":
            result = event["data"]
    return result
