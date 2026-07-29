"""Offline metrics for evaluating evidence-oriented research runs."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Sequence, Set


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    query: str
    relevant_chunk_ids: Set[str]
    graded_relevance: Dict[str, float] = field(default_factory=dict)
    required_evidence_questions: int = 0


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    retrieved_chunk_ids: Sequence[str]
    citation_checks: Sequence[bool]
    claim_support_checks: Sequence[bool]
    supported_evidence_questions: int
    latency_seconds: float


def recall_at_k(case: BenchmarkCase, result: BenchmarkResult, k: int = 20) -> float:
    if not case.relevant_chunk_ids:
        return 1.0
    found = set(result.retrieved_chunk_ids[:k]) & case.relevant_chunk_ids
    return len(found) / len(case.relevant_chunk_ids)


def ndcg_at_k(case: BenchmarkCase, result: BenchmarkResult, k: int = 10) -> float:
    relevance = case.graded_relevance or {
        chunk_id: 1.0 for chunk_id in case.relevant_chunk_ids
    }

    def dcg(values: Sequence[float]) -> float:
        return sum((2**value - 1) / math.log2(rank + 2) for rank, value in enumerate(values))

    actual = [relevance.get(chunk_id, 0.0) for chunk_id in result.retrieved_chunk_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 1.0


def _boolean_precision(checks: Sequence[bool]) -> float:
    return sum(bool(value) for value in checks) / len(checks) if checks else 1.0


def evaluate_case(case: BenchmarkCase, result: BenchmarkResult) -> Dict[str, float]:
    completeness = (
        min(result.supported_evidence_questions, case.required_evidence_questions)
        / case.required_evidence_questions
        if case.required_evidence_questions
        else 1.0
    )
    return {
        "recall_at_20": recall_at_k(case, result, 20),
        "ndcg_at_10": ndcg_at_k(case, result, 10),
        "citation_precision": _boolean_precision(result.citation_checks),
        "claim_support": _boolean_precision(result.claim_support_checks),
        "completeness": completeness,
        "latency_seconds": max(0.0, result.latency_seconds),
    }


def aggregate_benchmark(
    cases: Sequence[BenchmarkCase],
    results: Sequence[BenchmarkResult],
) -> Dict[str, float]:
    results_by_id = {result.case_id: result for result in results}
    rows = [
        evaluate_case(case, results_by_id[case.case_id])
        for case in cases
        if case.case_id in results_by_id
    ]
    if not rows:
        return {}
    return {metric: mean(row[metric] for row in rows) for metric in rows[0]}
