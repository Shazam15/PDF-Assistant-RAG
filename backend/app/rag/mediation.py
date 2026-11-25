"""Quote-anchored claim mediation: the model proposes, deterministic code disposes.

The deterministic checks in ``agent.py`` flag substantive sentences that have no
immediate citation, cite an unknown source, or contain figures absent from the
cited excerpt. Before this module, a flagged sentence was simply pruned. Here the
generator gets one batched, schema-constrained call in which, for each flagged
claim, it names the excerpt that supports it and copies a verbatim quote from
it. Each proposal is accepted only when code can verify it:

* the source id was really retrieved;
* the quote is a (whitespace/quote/case-normalised) substring of that excerpt;
* every figure in the claim appears in the quote (after number normalisation);
* optionally, a multilingual NLI model entails the claim from the quote.

Accepted claims get their citation fixed (or are restated to what the quote
supports); everything else is pruned exactly as before. The model never judges
its own output, which is the failure mode the ATLAS paper measured for the
same-model audits (§4.2, §4.5).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_CITATION_RE = re.compile(r"\[((?:D|W)\d+)\]")
MIN_QUOTE_CHARS = 20
MAX_QUOTE_CHARS = 400


class _MediationItem(BaseModel):
    claim: int = Field(description="Number of the flagged claim this item answers.")
    source_id: str = Field(description="Id of the excerpt that supports the claim, e.g. D3. Empty if none does.")
    quote: str = Field(
        description="Shortest span copied character by character from that excerpt that supports the claim. "
        "Empty if none does."
    )
    verdict: Literal["supported", "partial", "unsupported"] = Field(
        description="supported: the quote states the whole claim; partial: it states only part of it; "
        "unsupported: no excerpt states it."
    )
    restated_claim: str = Field(
        description="Only for partial: the claim rewritten, in its original language, to say only what the "
        "quote states. Otherwise empty."
    )


class _MediationResult(BaseModel):
    items: List[_MediationItem] = Field(description="One item per flagged claim.")


@dataclass
class MediationOutcome:
    answer: str
    stats: Dict[str, Any] = field(default_factory=dict)


# ── Deterministic helpers ────────────────────────────────────────────────


_QUOTE_CHARS = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    " ": " ", " ": " ", " ": " ", " ": " ",
    "‐": "-", "‑": "-", "–": "-", "—": "-",
})


def normalize_quote_text(text: str) -> str:
    """Normalise typography only: quotes, dashes, spaces, PDF hyphenation and case."""
    text = (text or "").translate(_QUOTE_CHARS)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip("\"'").strip().casefold()


_NUMBER_RE = re.compile(
    r"(?<![\w.,])("
    r"\d{1,3}(?:[    ]\d{3})+(?:[.,]\d+)?"  # 1 234 or 1 234,5
    r"|\d+(?:[.,]\d+)*"                                    # 1234, 1.234, 1,234.5, 12,5
    r")(?!\w)"
)


def _canonical(value: str) -> str:
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value.lstrip("0") or "0"


def _number_readings(token: str) -> set[str]:
    """All plausible canonical values of a number written with either convention."""
    token = re.sub(r"[    ]", "", token)
    readings: set[str] = set()
    separators = [char for char in token if char in ".,"]
    if not separators:
        return {_canonical(token)}
    last = max(token.rfind("."), token.rfind(","))
    integer, fraction = token[:last], token[last + 1:]
    integer_digits = re.sub(r"[.,]", "", integer)
    # Reading 1: the last separator is decimal and the others group thousands.
    readings.add(_canonical(f"{integer_digits}.{fraction}"))
    # Reading 2: every separator groups thousands (only valid with 3-digit groups).
    groups = re.split(r"[.,]", token)
    if len(set(separators)) == 1 and all(len(group) == 3 for group in groups[1:]):
        readings.add(_canonical("".join(groups)))
    return readings


def normalized_numbers(text: str) -> List[set[str]]:
    """Numbers in ``text`` as sets of readings, ignoring citation markers and units."""
    plain = _CITATION_RE.sub("", text or "")
    return [_number_readings(match.group(1)) for match in _NUMBER_RE.finditer(plain)]


def numbers_supported(claim: str, quote: str) -> bool:
    quote_values: set[str] = set().union(*normalized_numbers(quote))
    return all(readings & quote_values for readings in normalized_numbers(claim))


def _strip_citations(sentence: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", _CITATION_RE.sub("", sentence)).strip()


def _with_citation(sentence: str, source_id: str) -> str:
    plain = _strip_citations(sentence)
    match = re.search(r"([.!?;:]+)$", plain)
    if match:
        return f"{plain[:match.start()].rstrip()} [{source_id}]{match.group(1)}"
    return f"{plain} [{source_id}]"


# ── Mediation ────────────────────────────────────────────────────────────


@dataclass
class _Sentence:
    text: str
    flag: Optional[str] = None
    claim_number: Optional[int] = None


@dataclass
class _Line:
    raw: str
    prefix: str = ""
    sentences: List[_Sentence] = field(default_factory=list)
    passthrough: bool = False


def _parse(answer: str, valid_ids: set[str], support_texts: Dict[str, str]) -> List[_Line]:
    from app.rag import agent

    lines: List[_Line] = []
    for raw in (answer or "").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(_Line(raw=raw, passthrough=True))
            continue
        prefix_match = re.match(r"^(\s*(?:[-*]|\d+[.)])\s*)", raw)
        prefix = prefix_match.group(1) if prefix_match else ""
        body = raw[len(prefix):].strip() if prefix else stripped
        line = _Line(raw=raw, prefix=prefix)
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            sentence = sentence.strip()
            if not sentence:
                continue
            item = _Sentence(text=sentence)
            if agent._is_substantive_claim(sentence):
                cited = set(_CITATION_RE.findall(sentence))
                if not cited:
                    item.flag = "uncited"
                elif not cited.issubset(valid_ids):
                    item.flag = "invalid_citation"
                else:
                    premise = "\n".join(support_texts.get(source_id, "") for source_id in sorted(cited))
                    if agent._claim_support_issue(sentence, premise, False):
                        item.flag = "unsupported_numbers"
            line.sentences.append(item)
        lines.append(line)
    return lines


def _excerpts_block(support_texts: Dict[str, str], max_chars: int) -> str:
    items = [(source_id, text) for source_id, text in support_texts.items() if text.strip()]
    if not items:
        return ""
    per_excerpt = max(300, min(2000, max_chars // len(items) - 40))
    parts = []
    used = 0
    for source_id, text in items:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > per_excerpt:
            text = text[: per_excerpt - 3] + "..."
        part = f"[{source_id}] {text}"
        if used + len(part) > max_chars:
            break
        parts.append(part)
        used += len(part) + 2
    return "\n\n".join(parts)


def _propose(claims: List[str], excerpts: str) -> List[_MediationItem]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.rag import agent

    instruction = (
        "You attach evidence to claims. For each numbered CLAIM, find the EXCERPT that states it and copy the "
        "shortest supporting span from that excerpt character by character into `quote`, keeping its original "
        "language. Never translate, paraphrase or join text from different places. If an excerpt states only "
        "part of the claim, use verdict `partial` and write `restated_claim` with only what the quote states. If "
        "no excerpt states the claim, use verdict `unsupported` with empty source_id and quote. Figures in the "
        "claim must appear in the quote.\n"
        "Respond with only a JSON object {\"items\": [...]} where each item has exactly these keys: "
        "claim (integer), source_id (string), quote (string), "
        "verdict (\"supported\" | \"partial\" | \"unsupported\"), restated_claim (string)."
    )
    numbered = "\n".join(f"{index}. {_strip_citations(claim)}" for index, claim in enumerate(claims, 1))
    model = agent.get_llm_client(max_tokens=settings.CLAIM_MEDIATION_MAX_TOKENS).with_structured_output(
        _MediationResult, method="json_schema"
    )
    result = model.invoke([
        SystemMessage(content=agent._structured_system_prompt(instruction)),
        HumanMessage(content=f"EXCERPTS\n{excerpts}\n\nCLAIMS\n{numbered}"),
    ])
    if not isinstance(result, _MediationResult):
        raise ValueError("mediation model returned an invalid schema")
    return list(result.items)


def _accept(
    item: _MediationItem,
    claim: str,
    support_texts: Dict[str, str],
) -> tuple[Optional[str], str]:
    """Return (replacement sentence, reason). The replacement is None when rejected."""
    from app.rag import agent

    if item.verdict == "unsupported":
        return None, "unsupported"
    source_id = (item.source_id or "").strip().strip("[]")
    if source_id not in support_texts:
        return None, "unknown_source"
    quote = normalize_quote_text(item.quote)
    if not MIN_QUOTE_CHARS <= len(quote) <= MAX_QUOTE_CHARS:
        return None, "quote_length"
    if quote not in normalize_quote_text(support_texts[source_id]):
        return None, "quote_not_in_source"

    candidate = claim if item.verdict == "supported" else (item.restated_claim or "").strip()
    if not candidate or len(re.findall(r"\w+", candidate)) < 3:
        return None, "empty_restatement"
    if not numbers_supported(candidate, item.quote):
        return None, "numbers_not_in_quote"
    if settings.CLAIM_MEDIATION_NLI:
        scores = agent._nli_scores(item.quote, _strip_citations(candidate))
        if scores is not None:
            entailment = scores["entailment"]
            if entailment < settings.NLI_ENTAILMENT_THRESHOLD or entailment <= max(
                scores["neutral"], scores["contradiction"]
            ):
                return None, "nli_not_entailed"
    return _with_citation(candidate, source_id), item.verdict


def mediate_claims(
    answer: str,
    sources: List[Dict[str, Any]],
    evidence: Optional[List[Dict[str, Any]]] = None,
    proposer: Optional[Callable[[List[str], str], List[_MediationItem]]] = None,
) -> MediationOutcome:
    """Fix or prune every flagged claim; unflagged text is returned unchanged."""
    from app.rag import agent

    stats: Dict[str, Any] = {
        "flagged": 0, "mediated": 0, "accepted_supported": 0, "accepted_partial": 0,
        "pruned_over_limit": 0, "rejected": {}, "llm_error": None,
    }
    support_texts = agent._source_support_texts(sources, evidence)
    valid_ids = set(support_texts)
    lines = _parse(answer, valid_ids, support_texts)
    flagged = [sentence for line in lines for sentence in line.sentences if sentence.flag]
    stats["flagged"] = len(flagged)
    if not flagged:
        return MediationOutcome(answer=answer, stats=stats)

    to_mediate = flagged[: max(0, settings.CLAIM_MEDIATION_MAX_CLAIMS)]
    stats["pruned_over_limit"] = len(flagged) - len(to_mediate)
    stats["mediated"] = len(to_mediate)
    for number, sentence in enumerate(to_mediate, 1):
        sentence.claim_number = number

    replacements: Dict[int, str] = {}
    if to_mediate:
        budget = agent._llm_prompt_char_budget(settings.CLAIM_MEDIATION_MAX_TOKENS)
        claims = [sentence.text for sentence in to_mediate]
        excerpts = _excerpts_block(support_texts, max(1200, budget - sum(len(c) for c in claims) - 1500))
        try:
            items = (proposer or _propose)(claims, excerpts)
        except Exception as exc:
            logger.warning("Claim mediation unavailable; pruning flagged claims: %s", exc)
            stats["llm_error"] = str(exc)[:200]
            items = []
        for item in items:
            if not 1 <= item.claim <= len(to_mediate) or item.claim in replacements:
                continue
            replacement, reason = _accept(item, to_mediate[item.claim - 1].text, support_texts)
            if replacement is None:
                stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                continue
            replacements[item.claim] = replacement
            stats["accepted_" + reason] += 1

    output: List[str] = []
    for line in lines:
        if line.passthrough:
            output.append(line.raw)
            continue
        kept = []
        for sentence in line.sentences:
            if not sentence.flag:
                kept.append(sentence.text)
            elif sentence.claim_number in replacements:
                kept.append(replacements[sentence.claim_number])
        if kept:
            output.append(line.prefix + " ".join(kept))

    logger.info(
        "Claim mediation: flagged=%d mediated=%d supported=%d partial=%d rejected=%s over_limit=%d",
        stats["flagged"], stats["mediated"], stats["accepted_supported"], stats["accepted_partial"],
        stats["rejected"], stats["pruned_over_limit"],
    )
    return MediationOutcome(answer="\n".join(output).strip(), stats=stats)


def rescuable_draft(answer: str, sources: List[Dict[str, Any]]) -> bool:
    """A draft rejected only for missing citations, never for invented or legacy ones."""
    if not answer or not sources:
        return False
    if re.search(r"\[Fuente:\s*.+?\]", answer, flags=re.IGNORECASE):
        return False
    valid_ids = {str(source.get("source_id")) for source in sources if source.get("source_id")}
    return set(_CITATION_RE.findall(answer)).issubset(valid_ids)
