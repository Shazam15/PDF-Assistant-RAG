from unittest.mock import MagicMock

import pytest

from app.rag import code_review_agent


def _fresh_tools():
    pdf_tool = MagicMock(all_sources=[], last_sources=[])
    web_tool = MagicMock(all_sources=[], last_sources=[])
    return pdf_tool, web_tool


def _executor(output: str, intermediate_steps=None):
    executor = MagicMock()
    executor.invoke.return_value = {"output": output, "intermediate_steps": intermediate_steps or []}
    return executor


def test_round_loop_stops_on_first_clean_pass(monkeypatch):
    pdf_tool, web_tool = _fresh_tools()
    executor = _executor("Revisión completa, sin bloques de código adjuntos.")
    get_executor = MagicMock(return_value=(executor, pdf_tool, web_tool, ""))
    monkeypatch.setattr(code_review_agent, "get_code_review_executor", get_executor)

    result = code_review_agent.run_code_review_agent(
        question="Revisa este fragmento por bugs.", user_id="user123"
    )

    assert result["answer"] == "Revisión completa, sin bloques de código adjuntos."
    get_executor.assert_called_once()


def test_round_loop_retries_and_succeeds_after_syntax_fix(monkeypatch):
    pdf_tool, web_tool = _fresh_tools()
    bad_executor = _executor("Aquí está el código:\n```python\ndef f(:\n    pass\n```")
    good_executor = _executor("Aquí está el código corregido:\n```python\ndef f():\n    pass\n```")
    get_executor = MagicMock(
        side_effect=[
            (bad_executor, pdf_tool, web_tool, ""),
            (good_executor, pdf_tool, web_tool, ""),
        ]
    )
    monkeypatch.setattr(code_review_agent, "get_code_review_executor", get_executor)

    result = code_review_agent.run_code_review_agent(
        question="Genera una función de ejemplo.", user_id="user123"
    )

    assert "def f():" in result["answer"]
    assert get_executor.call_count == 2
    second_round_input = good_executor.invoke.call_args.args[0]["input"]
    assert "Nota de Percepción" in second_round_input
    assert "línea" in second_round_input  # the ast.parse error message was fed back


def test_round_loop_fails_open_after_max_rounds(monkeypatch):
    pdf_tool, web_tool = _fresh_tools()
    always_broken = _executor("Sigo intentando:\n```python\ndef f(:\n    pass\n```")
    get_executor = MagicMock(return_value=(always_broken, pdf_tool, web_tool, ""))
    monkeypatch.setattr(code_review_agent, "get_code_review_executor", get_executor)
    monkeypatch.setattr(code_review_agent.settings, "CODE_REVIEW_MAX_ROUNDS", 3)

    result = code_review_agent.run_code_review_agent(question="Genera código.", user_id="user123")

    assert "Nota de verificación automática" in result["answer"]
    assert get_executor.call_count == 3


def test_round_loop_fails_open_on_deadline(monkeypatch):
    pdf_tool, web_tool = _fresh_tools()
    always_broken = _executor("Sigo intentando:\n```python\ndef f(:\n    pass\n```")
    get_executor = MagicMock(return_value=(always_broken, pdf_tool, web_tool, ""))
    monkeypatch.setattr(code_review_agent, "get_code_review_executor", get_executor)
    monkeypatch.setattr(code_review_agent.settings, "CODE_REVIEW_MAX_ROUNDS", 5)

    # time.monotonic() is called once to compute the deadline, then once at the top of
    # each round: round 1's check must pass (proceed), round 2's must fail (break) —
    # stopping after a single round without exhausting all 5.
    times = iter([0.0, 0.0, 10_000.0])
    monkeypatch.setattr(code_review_agent.time, "monotonic", lambda: next(times, 10_000.0))

    result = code_review_agent.run_code_review_agent(question="Genera código.", user_id="user123")

    assert "Nota de verificación automática" in result["answer"]
    assert get_executor.call_count == 1


def test_round_loop_raises_on_cancellation():
    from threading import Event

    cancellation = Event()
    cancellation.set()

    with pytest.raises(code_review_agent.CodeReviewCancelled):
        code_review_agent.run_code_review_agent(
            question="Genera código.", user_id="user123", cancellation_event=cancellation
        )


def test_code_review_agent_skips_citation_validation(monkeypatch):
    """A code-only Final Answer with no [D#] tags must be returned as-is, not rejected the
    way app.rag.agent._validate_answer_citations would reject a citation-less prose answer."""
    pdf_tool, web_tool = _fresh_tools()
    answer_text = "Aquí tienes la función solicitada:\n```python\ndef add(a, b):\n    return a + b\n```"
    executor = _executor(answer_text)
    monkeypatch.setattr(
        code_review_agent, "get_code_review_executor", MagicMock(return_value=(executor, pdf_tool, web_tool, ""))
    )

    result = code_review_agent.run_code_review_agent(question="Escribe una función que sume.", user_id="user123")

    assert result["answer"] == answer_text


def test_check_python_syntax_errors_flags_broken_snippet_and_passes_valid_one():
    final_answer = (
        "Primero, el ejemplo roto:\n```python\ndef broken(:\n    pass\n```\n"
        "Y uno válido:\n```python\ndef ok():\n    return 1\n```"
    )

    errors = code_review_agent._check_python_syntax_errors([], final_answer)

    assert len(errors) == 1
    assert "línea" in errors[0]


def test_check_python_syntax_errors_returns_empty_for_clean_code():
    final_answer = "Aquí está el código:\n```python\ndef ok():\n    return 1\n```"

    assert code_review_agent._check_python_syntax_errors([], final_answer) == []


def test_check_python_syntax_errors_skips_non_python_fences_with_a_language_hint():
    final_answer = "```javascript\nfunction broken( {\n```"

    assert code_review_agent._check_python_syntax_errors([], final_answer) == []
