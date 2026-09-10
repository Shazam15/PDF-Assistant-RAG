"""
Code Review Agent — a dedicated Perceive-Reason-Act loop for the "Code Review" chat mode.

Perceive = the Observation the ReAct executor receives each step (or the initial
question); Reason = its Thought; Act = its Action/tool call — the same mechanical ReAct
cycle used elsewhere in this app (see app.rag.agent.GroundedReActOutputParser), reused
unforked here via import rather than duplicated. "Perceive/Reason/Act" is a narrative
re-framing of that cycle (see CODE_REVIEW_AGENT_PROMPT in app.rag.prompts), not a
different wire protocol — the parser's regexes are still hardcoded to the literal
Thought/Action/Action Input/Observation/Final Answer tokens.

This loop wraps that inner ReAct executor in an OUTER round loop with a two-part
termination condition, checked after each round:
  1. The executor reached a real Final Answer on its own (not an iteration-cap cutoff) —
     "the agent decided it doesn't need more tools."
  2. A deterministic Python-syntax check (ast.parse, stdlib only) over any Python code
     the round touched or produced found zero errors.
Both must hold to finish; otherwise the syntax errors are fed back into the next round's
question as a "Perceive" note and the loop retries, bounded by CODE_REVIEW_MAX_ROUNDS and
CODE_REVIEW_TIMEOUT_SECONDS. If the budget runs out without a clean pass, the loop fails
open (returns the best available answer plus an explicit caveat) rather than crashing or
silently claiming success — the same convention app.rag.research_agent uses.

This module is isolated from the generic ReAct "tool_agent" loop and from
app.rag.research_agent's LangGraph pipeline: reached only via routing_mode="code_review"
(see route_query() in app.rag.agent). app.rag.agent only imports this module lazily,
inside generate_answer()/generate_answer_stream() — never at its own module top level —
so this module can safely import agent.py's internals directly without a circular import.

Per an explicit product decision, this loop's "Act" step is read-only/advisory: no
file-writing MCP tool is enabled, and it must never claim to have written or saved a
file — it always presents reviewed/generated code as text in its Final Answer for the
user to apply manually.
"""
import ast
import json
import logging
import re
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.rag.agent import (
    AGENT_INCOMPLETE_MESSAGE,
    GroundedReActOutputParser,
    _AGENT_TOOL_NAMES,
    _collect_agent_sources,
    _format_chat_history,
    _is_agent_stop_answer,
    _load_global_style_reference,
    _source_payload,
    _truncate_tool_text,
)
from app.rag.prompts import CODE_REVIEW_AGENT_PROMPT
from app.rag.security import MALFORMED_OUTPUT_MESSAGE, OutputParserError, parse_agent_output
from app.rag.skills import load_skill
from app.rag.tools import build_agent_tools

logger = logging.getLogger(__name__)
settings = get_settings()

_CODE_REVIEW_SKILL_NAME = "code-review"
_CODE_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

_VERIFICATION_CAVEAT = (
    "\n\n---\n**Nota de verificación automática:** no fue posible confirmar que el código "
    "propuesto está libre de errores dentro del presupuesto de rondas/tiempo disponible de "
    "este modo. Los archivos en lenguajes distintos a Python no se verifican automáticamente."
)


class CodeReviewCancelled(RuntimeError):
    """Raised when the client cancels mid-loop; callers catch it and stop, never propagates."""


def _looks_python(language: str, file_path: str) -> bool:
    """Err on the side of checking when there's no hint either way: a false-positive
    syntax check on non-Python text just costs a spurious retry round (bounded by the
    round cap), never a wrong claim of success.
    """
    if language and language.strip().lower() in ("python", "py"):
        return True
    if file_path and file_path.strip().lower().endswith(".py"):
        return True
    if not language and not file_path:
        return True
    return False


def _iter_step_actions_and_observations(intermediate_steps: Optional[List[Any]]):
    """Unwrap invoke-style (AgentAction, observation) tuples and streaming AgentStep
    objects alike, mirroring app.rag.agent._step_observations' defensive shape-handling.
    """
    for step in intermediate_steps or []:
        if isinstance(step, (tuple, list)) and len(step) >= 2:
            yield step[0], step[1]
        elif isinstance(step, dict):
            yield step.get("action"), step.get("observation")
        else:
            yield getattr(step, "action", None), getattr(step, "observation", None)


def _extract_code_review_tool_fields(tool_input: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """tool_input may be a structured dict call or a bare ReAct single-input string;
    handle both defensively rather than assuming one shape."""
    if isinstance(tool_input, dict):
        return tool_input.get("code"), tool_input.get("file_path"), tool_input.get("language")
    if isinstance(tool_input, str):
        try:
            parsed = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed.get("code"), parsed.get("file_path"), parsed.get("language")
        return tool_input, None, None
    return None, None, None


def _extract_read_file_path(tool_input: Any) -> str:
    if isinstance(tool_input, str):
        return tool_input.strip()
    if isinstance(tool_input, dict):
        return str(tool_input.get("path") or "").strip()
    return ""


def _check_python_syntax_errors(intermediate_steps: Optional[List[Any]], final_answer_text: str) -> List[str]:
    """Deterministic 'no errors' check — condition 2 of the loop's two-part termination rule.

    Python-only, via ast.parse: pure and stdlib-only, so no temp files are needed for
    in-memory code (fenced Final Answer blocks, code_review tool-call arguments) the way
    py_compile.compile would require (it operates on a file path). A SyntaxError from
    ast.parse is exactly what compile()/py_compile would also raise at the parse stage,
    so this is a faithful syntax check even though it stops short of byte-compiling.

    Non-Python files touched during the loop are skipped, never falsely marked as clean.
    """
    candidates: List[Tuple[str, str]] = []

    for action, observation in _iter_step_actions_and_observations(intermediate_steps):
        if action is None:
            continue
        tool_name = getattr(action, "tool", "")
        tool_input = getattr(action, "tool_input", None)

        if tool_name == "code_review":
            code, file_path, language = _extract_code_review_tool_fields(tool_input)
            if code and _looks_python(language or "", file_path or ""):
                candidates.append((file_path or "code_review: fragmento revisado", code))
        elif tool_name == "read_file":
            path = _extract_read_file_path(tool_input)
            if path.lower().endswith(".py") and observation:
                candidates.append((path, str(observation)))

    for language, body in _CODE_FENCE_RE.findall(final_answer_text or ""):
        if _looks_python(language, ""):
            candidates.append(("bloque de código en la Final Answer", body))

    errors: List[str] = []
    for label, source in candidates:
        try:
            ast.parse(source)
        except SyntaxError as exc:
            errors.append(f"{label}: línea {exc.lineno}: {exc.msg}")
    return errors


def _load_code_review_skill_body() -> str:
    """Deterministically front-load the (Percibe/Razona/Actúa) code-review skill for this
    explicit, deliberate mode — unlike the generic tool_agent loop, where the model now
    decides on its own whether to call use_skill('code-review') (see REQUIRED_TOOL_SKILL_MAP
    in app.rag.agent), determinism here is intentional: the user explicitly chose this mode.
    """
    try:
        body = load_skill(_CODE_REVIEW_SKILL_NAME)
        if not body or body.startswith(f"Skill '{_CODE_REVIEW_SKILL_NAME}' no encontrada"):
            logger.warning("Code review mode: mandatory skill load returned no usable body.")
            return ""
        return body
    except Exception as exc:
        logger.warning("Code review mode: mandatory skill load failed: %s", exc)
        return ""


def _seed_question(question: str, skill_body: str, prior_errors: Optional[List[str]]) -> str:
    parts = [question]
    if skill_body:
        parts.append(f"Procedimiento obligatorio de la skill 'code-review' ya cargado:\n{skill_body}")
    if prior_errors:
        error_lines = "\n".join(f"- {error}" for error in prior_errors)
        parts.append(
            "## Nota de Percepción: errores de sintaxis detectados en la ronda anterior\n"
            f"{error_lines}\n"
            "Corrige estos errores en esta ronda antes de responder con Final Answer."
        )
    return "\n\n".join(parts)


def get_code_review_executor(
    user_id: str,
    document_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
):
    """Build the inner ReAct executor for one round. Mirrors app.rag.agent.get_agent_executor's
    construction closely, with its own prompt/persona and its own, decoupled iteration
    budget (CODE_REVIEW_MAX_ITERATIONS_PER_ROUND) — deliberately not shared with the
    generic tool_agent loop's AGENT_MAX_ITERATIONS, so tuning one can never regress the
    other.
    """
    tools = build_agent_tools(user_id=user_id or "", document_id=document_id, top_k=top_k)
    pdf_tool = next((tool for tool in tools if getattr(tool, "name", "") == "pdf_search"), None)
    web_tool = next((tool for tool in tools if getattr(tool, "name", "") == "web_search"), None)

    chat_llm = ChatOllama(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        reasoning=False if settings.LLM_DISABLE_THINKING else None,
        num_ctx=settings.LLM_CONTEXT_WINDOW,
        num_predict=min(settings.LLM_MAX_NEW_TOKENS, settings.AGENT_PLANNER_MAX_TOKENS),
        client_kwargs={"timeout": settings.LLM_REQUEST_TIMEOUT_SECONDS},
    )
    style_reference = _load_global_style_reference()
    prompt = PromptTemplate.from_template(CODE_REVIEW_AGENT_PROMPT).partial(style_reference=style_reference)
    valid_tool_names = {getattr(tool, "name", "") for tool in tools} | _AGENT_TOOL_NAMES
    agent = create_react_agent(
        chat_llm,
        tools,
        prompt,
        output_parser=GroundedReActOutputParser(valid_tool_names=valid_tool_names),
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=(
            "Formato inválido. Usa Action/Action Input para una herramienta o Final Answer para terminar."
        ),
        max_iterations=settings.CODE_REVIEW_MAX_ITERATIONS_PER_ROUND,
        early_stopping_method="force",
        return_intermediate_steps=True,
    )
    formatted_history = _format_chat_history(chat_history) if chat_history else ""
    return executor, pdf_tool, web_tool, formatted_history


def run_code_review_agent(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    cancellation_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """Non-streaming entry point: run the outer Percibe-Razona-Actúa round loop to completion.

    Never raises for an ordinary failed/incomplete round — fails open with the best
    available answer plus a caveat, mirroring app.rag.research_agent's convention. Only
    an explicit client cancellation propagates, as CodeReviewCancelled.
    """
    deadline = time.monotonic() + settings.CODE_REVIEW_TIMEOUT_SECONDS
    skill_body = _load_code_review_skill_body()
    prior_errors: Optional[List[str]] = None
    best_answer, best_sources = "", []

    for round_number in range(1, settings.CODE_REVIEW_MAX_ROUNDS + 1):
        if cancellation_event is not None and cancellation_event.is_set():
            raise CodeReviewCancelled("Code review cancelled by client.")
        if time.monotonic() >= deadline:
            logger.warning("Code review deadline reached before a clean pass (round %d).", round_number)
            break

        executor, pdf_tool, web_tool, formatted_history = get_code_review_executor(
            user_id, document_id, hf_token, top_k, chat_history
        )
        seeded_question = _seed_question(question, skill_body if round_number == 1 else "", prior_errors)
        try:
            result = executor.invoke({"input": seeded_question, "chat_history": formatted_history})
        except Exception as exc:
            logger.warning("Code review round %d failed: %s", round_number, exc)
            break

        raw_answer = result.get("output", "")
        try:
            answer = parse_agent_output(raw_answer)
        except OutputParserError as exc:
            logger.warning("Rejected malformed code review output: %s", exc)
            answer = MALFORMED_OUTPUT_MESSAGE

        intermediate_steps = result.get("intermediate_steps", [])
        raw_sources = _collect_agent_sources(pdf_tool, web_tool, intermediate_steps)
        sources = [_source_payload(chunk) for chunk in raw_sources]

        finished_cleanly = not _is_agent_stop_answer(answer)
        errors = _check_python_syntax_errors(intermediate_steps, answer) if finished_cleanly else []

        best_answer, best_sources = answer, sources

        if finished_cleanly and not errors:
            logger.info("Code review loop finished cleanly after %d round(s).", round_number)
            return {"answer": answer, "sources": sources}

        if not finished_cleanly:
            logger.warning("Code review round %d hit its iteration cap without a Final Answer.", round_number)
            prior_errors = None
            continue

        logger.info("Code review round %d found %d syntax error(s); retrying.", round_number, len(errors))
        prior_errors = errors

    logger.warning("Code review loop exhausted its round/time budget without a verified-clean pass.")
    return {"answer": (best_answer or AGENT_INCOMPLETE_MESSAGE) + _VERIFICATION_CAVEAT, "sources": best_sources}


def stream_code_review_agent(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    cancellation_event: Optional[Any] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Streaming entry point: yields structured dict events (not raw SSE strings) — the
    caller (app.rag.agent.generate_answer_stream) formats these into SSE, exactly as it
    already does for app.rag.research_agent.stream_research_agent's events, keeping this
    module transport-agnostic.

    Events: {"type": "tool_event", "data": {...}} for tool_start/tool_result within a
    round, {"type": "progress", "data": {"round": n, "stage": ...}} between rounds, and
    exactly one final {"type": "result", "data": {"answer": ..., "sources": [...]}}.
    """
    deadline = time.monotonic() + settings.CODE_REVIEW_TIMEOUT_SECONDS
    skill_body = _load_code_review_skill_body()
    prior_errors: Optional[List[str]] = None
    best_answer, best_sources = "", []

    for round_number in range(1, settings.CODE_REVIEW_MAX_ROUNDS + 1):
        if cancellation_event is not None and cancellation_event.is_set():
            logger.info("Code review stream cancelled by client before round %d.", round_number)
            return
        if time.monotonic() >= deadline:
            logger.warning("Code review deadline reached before a clean pass (round %d).", round_number)
            break

        yield {"type": "progress", "data": {"round": round_number, "stage": "reasoning"}}

        executor, pdf_tool, web_tool, formatted_history = get_code_review_executor(
            user_id, document_id, hf_token, top_k, chat_history
        )
        seeded_question = _seed_question(question, skill_body if round_number == 1 else "", prior_errors)
        accumulated_steps: List[Any] = []
        raw_answer = ""
        stream_failed = False
        try:
            for step in executor.stream({"input": seeded_question, "chat_history": formatted_history}):
                if cancellation_event is not None and cancellation_event.is_set():
                    logger.info("Code review stream cancelled by client mid-round %d.", round_number)
                    return
                if "actions" in step:
                    for action in step.get("actions") or []:
                        tool_name = getattr(action, "tool", "unknown")
                        tool_input = _truncate_tool_text(getattr(action, "tool_input", ""))
                        yield {
                            "type": "tool_event",
                            "data": {"type": "tool_start", "name": tool_name, "summary": tool_input},
                        }
                    continue

                new_steps = step.get("steps") or step.get("intermediate_step") or []
                if new_steps:
                    accumulated_steps.extend(new_steps)
                    for agent_step in new_steps:
                        action = getattr(agent_step, "action", None)
                        tool_name = getattr(action, "tool", "unknown")
                        observation = getattr(agent_step, "observation", None)
                        summary = _truncate_tool_text(observation)
                        if summary:
                            yield {
                                "type": "tool_event",
                                "data": {"type": "tool_result", "name": tool_name, "summary": summary},
                            }
                    continue

                if "output" in step:
                    final_steps = step.get("intermediate_steps") or []
                    if final_steps:
                        accumulated_steps.extend(final_steps)
                    raw_answer = step["output"]
        except Exception as exc:
            logger.warning("Code review streaming round %d failed: %s", round_number, exc)
            stream_failed = True

        if stream_failed and not raw_answer:
            break

        try:
            answer = parse_agent_output(raw_answer) if raw_answer else ""
        except OutputParserError as exc:
            logger.warning("Rejected malformed streamed code review output: %s", exc)
            answer = MALFORMED_OUTPUT_MESSAGE

        raw_sources = _collect_agent_sources(pdf_tool, web_tool, accumulated_steps)
        sources = [_source_payload(chunk) for chunk in raw_sources]
        finished_cleanly = bool(raw_answer) and not _is_agent_stop_answer(answer)
        errors = _check_python_syntax_errors(accumulated_steps, answer) if finished_cleanly else []

        best_answer = answer or best_answer
        best_sources = sources or best_sources

        if finished_cleanly and not errors:
            logger.info("Code review stream finished cleanly after %d round(s).", round_number)
            yield {"type": "result", "data": {"answer": answer, "sources": sources}}
            return

        if not finished_cleanly:
            logger.warning("Code review streaming round %d hit its iteration cap without a Final Answer.", round_number)
            prior_errors = None
            continue

        logger.info("Code review streaming round %d found %d syntax error(s); retrying.", round_number, len(errors))
        prior_errors = errors

    logger.warning("Code review stream exhausted its round/time budget without a verified-clean pass.")
    yield {
        "type": "result",
        "data": {"answer": (best_answer or AGENT_INCOMPLETE_MESSAGE) + _VERIFICATION_CAVEAT, "sources": best_sources},
    }
