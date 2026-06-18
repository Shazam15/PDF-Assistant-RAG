"""
Prompt-injection safeguards for user questions and model outputs.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Dict


PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)\b",
    r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)\b",
    r"\bforget\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)\b",
    r"\breveal\s+(the\s+)?(system|developer)\s+(prompt|message|instructions?)\b",
    r"\b(show|print|display|leak|dump)\s+(the\s+)?(system|developer)\s+(prompt|message|instructions?)\b",
    r"\bact\s+as\s+(the\s+)?(system|developer|admin|root)\b",
    r"\byou\s+are\s+now\s+(the\s+)?(system|developer|admin|root)\b",
    r"\bdisable\s+(all\s+)?(rules?|safety|guardrails?|filters?|restrictions?)\b",
    r"\bbypass\s+(all\s+)?(rules?|safety|guardrails?|filters?|restrictions?)\b",
    r"\boverride\s+(all\s+)?(instructions?|rules?|safety|guardrails?)\b",
    r"\bdo\s+not\s+(follow|obey)\s+(the\s+)?(instructions?|rules?|system)\b",
    r"\bpretend\s+(to\s+be|you\s+are)\s+(the\s+)?(system|developer|admin|root)\b",
]

_COMPILED_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS
]

BLOCKED_INPUT_MESSAGE = (
    "Your message appears to contain prompt-injection instructions and was blocked."
)

MALFORMED_OUTPUT_MESSAGE = (
    "No pude procesar la respuesta del modelo correctamente. Por favor, intenta reformular tu pregunta."
)


@dataclass(frozen=True)
class InputClassification:
    label: str
    is_safe: bool
    reason: str | None = None


from app.exceptions import AppException


class UnsafePromptError(AppException):
    """Raised when user input matches prompt-injection patterns."""

    def __init__(self, message: str = BLOCKED_INPUT_MESSAGE):
        super().__init__("UNSAFE_PROMPT", message, 400)


class OutputParserError(ValueError):
    """Raised when the LLM response does not match the required schema."""


def classify_user_input(text: str) -> InputClassification:
    """Classify a user query as safe or prompt_injection."""
    normalized = " ".join((text or "").split())
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(normalized):
            return InputClassification(
                label="prompt_injection",
                is_safe=False,
                reason=pattern.pattern,
            )

    return InputClassification(label="safe", is_safe=True)


def validate_user_input(text: str) -> None:
    """Raise if the supplied user query should not reach retrieval or the LLM."""
    classification = classify_user_input(text)
    if not classification.is_safe:
        raise UnsafePromptError(BLOCKED_INPUT_MESSAGE)


def parse_agent_output(raw_output: str) -> str:
    """
    Parse the agent's final answer.
    Accepts plain text or JSON with 'answer' key.
    """
    if not raw_output or not raw_output.strip():
        raise OutputParserError("Respuesta vacía del agente.")

    text = raw_output.strip()

    # Intentar JSON primero por compatibilidad
    try:
        # Buscar objeto JSON en el output
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, dict) and "answer" in parsed:
                answer = parsed["answer"].strip()
                if answer:
                    return answer
    except Exception:
        pass

    # Aceptar texto plano directamente
    for prefix in ["Final Answer:", "Respuesta Final:", "Answer:", "final answer:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break

    if len(text) > 10:
        return text

    raise OutputParserError("LLM output is not valid JSON.")


def _load_json_object(raw_output: str) -> Dict[str, Any]:
    """Kept for backward compatibility."""
    content = (raw_output or "").strip()
    if content.lower().startswith("final answer:"):
        content = content.split(":", 1)[1].strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise OutputParserError("LLM output is not valid JSON.") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise OutputParserError("LLM output JSON is malformed.") from exc
    if not isinstance(payload, dict):
        raise OutputParserError("LLM output must be a JSON object.")
    return payload
