import pytest

from app.rag.benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    aggregate_benchmark,
    evaluate_case,
)


DOMAIN_CASES = [
    ("engines", "Compare reported combustion and boosting strategies"),
    ("medicine", "Compare efficacy and limitations across clinical studies"),
    ("geology", "Synthesize evidence for the reported geological process"),
    ("history", "Contrast primary-source accounts of the event"),
    ("economics", "Compare methods and outcomes in the economic studies"),
    ("law", "Reconcile the legal arguments and cited authorities"),
]


@pytest.mark.parametrize("case_id,query", DOMAIN_CASES)
def test_benchmark_metrics_are_domain_independent(case_id, query):
    case = BenchmarkCase(
        case_id=case_id,
        query=query,
        relevant_chunk_ids={"a", "b"},
        graded_relevance={"a": 2.0, "b": 1.0},
        required_evidence_questions=2,
    )
    result = BenchmarkResult(
        case_id=case_id,
        retrieved_chunk_ids=["a", "irrelevant", "b"],
        citation_checks=[True, False, True],
        claim_support_checks=[True, True],
        supported_evidence_questions=2,
        latency_seconds=12.5,
    )

    metrics = evaluate_case(case, result)

    assert metrics["recall_at_20"] == 1.0
    assert 0.0 < metrics["ndcg_at_10"] < 1.0
    assert metrics["citation_precision"] == pytest.approx(2 / 3)
    assert metrics["claim_support"] == 1.0
    assert metrics["completeness"] == 1.0


def test_benchmark_aggregation_reports_all_acceptance_metrics():
    case = BenchmarkCase("case", "query", {"a"}, required_evidence_questions=1)
    result = BenchmarkResult("case", ["a"], [True], [True], 1, 2.0)

    metrics = aggregate_benchmark([case], [result])

    assert set(metrics) == {
        "recall_at_20",
        "ndcg_at_10",
        "citation_precision",
        "claim_support",
        "completeness",
        "latency_seconds",
    }
