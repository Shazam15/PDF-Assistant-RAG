"""
Agentic RAG — intelligent routing using ReAct (Reasoning and Acting).
Intelligently chooses between PDF search, Web Search, and Math tools.
"""
import logging
import json
import math
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Generator, Literal

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.rag.retriever import retrieve
from app.rag.graph_retriever import get_entity_context
from app.rag.prompts import AGENT_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, CODE_REVIEW_PROMPT
from app.exceptions import ExternalServiceException
from app.rag.security import MALFORMED_OUTPUT_MESSAGE, OutputParserError, parse_agent_output
from app.rag.tools import PDFSearchTool, MathTool, CodeReviewTool, WebSearchTool
from app.rag.tracing import trace_function

logger = logging.getLogger(__name__)
settings = get_settings()
ROUTER_VERSION = "adaptive-v1"
RoutingMode = Literal["auto", "quick", "research"]
RoutingRoute = Literal["greeting", "scoped_rag", "simple_rag", "research_rag", "tool_agent"]
_AGENT_TOOL_NAMES = {"pdf_search", "code_review", "calculator", "web_search"}
MIN_SOURCE_SCORE = 0.2
MIN_SOURCE_CONFIDENCE = 10.0
MIN_LEXICAL_OVERLAP = 0.14
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "No encontré información suficiente en las fuentes recuperadas para responder esta pregunta con citas verificables."
)
AGENT_INCOMPLETE_MESSAGE = (
    "El agente no pudo completar el razonamiento multi-paso antes del límite de iteraciones. "
    "Tampoco recuperó evidencia suficiente durante sus iteraciones para redactar una respuesta parcial verificable. "
    "Intenta reformular la pregunta en subpreguntas más específicas o aumenta el límite de iteraciones del agente."
)
_GENERIC_DOCUMENT_TERMS = {
    "documento",
    "documentos",
    "paper",
    "papers",
    "articulo",
    "articulos",
    "artículo",
    "artículos",
    "texto",
    "contenido",
    "fuente",
    "fuentes",
    "resumen",
    "resume",
    "resumir",
    "explica",
    "explicar",
    "analiza",
    "analizar",
    "actua",
    "actuar",
    "academico",
    "academica",
    "academicos",
    "academicas",
    "afirmacion",
    "afirmaciones",
    "analisis",
    "citas",
    "citar",
    "conclusion",
    "conclusiones",
    "contenida",
    "contenidas",
    "contenidos",
    "contradiccion",
    "contradicciones",
    "critico",
    "critica",
    "disponible",
    "disponibles",
    "discusion",
    "evidencia",
    "evidencias",
    "explicito",
    "explicitamente",
    "finalmente",
    "fortaleza",
    "fortalezas",
    "identifica",
    "identificar",
    "informacion",
    "introduccion",
    "investigador",
    "investigadora",
    "limitacion",
    "limitaciones",
    "metodologia",
    "metodologias",
    "pagina",
    "paginas",
    "posible",
    "pregunta",
    "redacta",
    "redactar",
    "relacion",
    "relaciones",
    "relevante",
    "relevantes",
    "responde",
    "respuesta",
    "resultados",
    "seccion",
    "seleccion",
    "seleccionaste",
    "seleccionados",
    "solucion",
    "suposicion",
    "suposiciones",
    "unicamente",
    "utilizando",
}
_STOPWORDS = {
    "acerca",
    "algunas",
    "algunos",
    "como",
    "cómo",
    "cual",
    "cuál",
    "cuales",
    "cuáles",
    "cuando",
    "cuándo",
    "donde",
    "dónde",
    "este",
    "esta",
    "estos",
    "estas",
    "para",
    "pero",
    "porque",
    "sobre",
    "tiene",
    "tienen",
    "what",
    "when",
    "where",
    "which",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "the",
    "and",
    "or",
    "for",
    "about",
}


class GroundedReActOutputParser(ReActSingleInputOutputParser):
    """Finish substantive drafts instead of spending every iteration on format retries."""

    @staticmethod
    def _is_substantive(text: str) -> bool:
        return len(re.findall(r"\w+", text or "")) >= 35

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            parsed = super().parse(text)
        except OutputParserException:
            if self._is_substantive(text):
                logger.warning(
                    "Treating substantive plain-text agent output as a draft for grounded final synthesis."
                )
                return AgentFinish({"output": text.strip()}, text)
            raise

        if isinstance(parsed, AgentAction) and parsed.tool not in _AGENT_TOOL_NAMES:
            if self._is_substantive(text):
                logger.warning(
                    "Agent emitted an invalid action name; preserving its substantive text as a draft."
                )
                return AgentFinish({"output": text.strip()}, text)
        return parsed


def get_llm_client(hf_token: Optional[str] = None, max_tokens: Optional[int] = None):
    """Create an Ollama client (hf_token ignored, kept for compatibility)."""
    return ChatOllama(
        model=settings.LLM_MODEL, 
        temperature=0,
        num_ctx=settings.LLM_CONTEXT_WINDOW,
        num_predict=max_tokens or settings.LLM_MAX_NEW_TOKENS)


def _format_chat_history(messages: List[Dict[str, str]]) -> str:
    if not messages:
        return ""
    lines = ["Previous conversation:"]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _load_global_style_reference() -> str:
    """Load a global writing-style reference from a fixed file name in the upload directory."""
    try:
        upload_dir = getattr(settings, "UPLOAD_DIR", "") or ""
        if not upload_dir:
            return ""

        style_path = os.path.join(upload_dir, "PDF_DE_PRUEBA")
        if not os.path.exists(style_path):
            return ""

        with open(style_path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read().strip()
        if not text:
            return ""

        snippet = re.sub(r"\s+", " ", text)
        if len(snippet) > 700:
            snippet = snippet[:697] + "..."
        return (
            "## Referencia de estilo global\n"
            "Usa este tono, ritmo y forma de expresión como referencia por defecto para todas las respuestas. "
            "No copies literalmente las frases; responde de forma original, elegante y coherente con ese estilo.\n"
            f"Texto de referencia: {snippet}"
        )
    except Exception as exc:
        logger.warning("Could not load global style reference: %s", exc)
        return ""

def get_agent_executor(
    user_id: str,
    document_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
):
    """Initialize the LangChain ReAct agent executor."""
    pdf_tool = PDFSearchTool(user_id=user_id, document_id=document_id, top_k=top_k)
    code_review_tool = CodeReviewTool(user_id=user_id, document_id=document_id, top_k=top_k)
    web_tool = WebSearchTool()
    tools = [pdf_tool, code_review_tool, MathTool(), web_tool]

    chat_llm = ChatOllama(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        num_ctx=settings.LLM_CONTEXT_WINDOW,
        num_predict=min(settings.LLM_MAX_NEW_TOKENS, settings.AGENT_PLANNER_MAX_TOKENS),
    )

    global_style_reference = _load_global_style_reference()
    prompt = PromptTemplate.from_template(AGENT_SYSTEM_PROMPT).partial(style_reference=global_style_reference)
    agent = create_react_agent(
        chat_llm,
        tools,
        prompt,
        output_parser=GroundedReActOutputParser(),
    )

    research_iterations = max(1, min(3, settings.AGENT_MAX_ITERATIONS - 1))

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=(
            "Formato inválido. Usa Action/Action Input para una herramienta o Final Answer para terminar."
        ),
        max_iterations=research_iterations,
        early_stopping_method="force",
        return_intermediate_steps=True,
    )

    formatted_history = _format_chat_history(chat_history) if chat_history else ""
    return executor, pdf_tool, web_tool, formatted_history


def is_greeting(question: str) -> bool:
    """Detect if the question is a casual greeting rather than a document query."""
    greetings = {
        "hi", "hello", "hey", "how are you", "what's up", "whats up",
        "good morning", "good evening", "good afternoon", "thanks", "thank you",
        "bye", "goodbye", "help", "what can you do", "who are you", "hola", "buenos días", 
        "buenas tardes", "buenas noches", "gracias", "adiós", "qué puedes hacer", "quién eres",
        "qué tal", "qué hay", "qué onda", "qué pasa", "cómo estás", "cómo te va", "qué haces",
    }
    return question.lower().strip().rstrip("!?.") in greetings


@dataclass(frozen=True)
class RoutingDecision:
    route: RoutingRoute
    reason: str
    score: int
    mode: RoutingMode
    document_scope: Optional[str]
    provisional: bool = False
    required_tool: Optional[str] = None


def _normalize_routing_mode(mode: str) -> RoutingMode:
    return mode if mode in {"auto", "quick", "research"} else "auto"


def _routing_question(question: str, chat_history: Optional[List[Dict[str, str]]]) -> str:
    """Use the previous user turn only for short, explicitly referential follow-ups."""
    normalized = _normalize_term_text(question)
    follow_up_markers = (
        "eso", "estos", "estas", "continua", "amplia", "comparalos", "compare them",
        "those", "that result", "continue", "expand on",
    )
    if len(normalized.split()) > 18 or not any(marker in normalized for marker in follow_up_markers):
        return question
    for message in reversed(chat_history or []):
        if message.get("role") == "user" and message.get("content"):
            return f"{message['content']}\n{question}"
    return question


def _required_tool(question: str) -> Optional[str]:
    normalized = _normalize_term_text(question)
    tool_signals = {
        "web": (
            "busca en la web", "busca en internet", "fuentes externas", "consulta online",
            "informacion actualizada", "ultimas noticias", "latest information", "search the web",
            "search online", "external sources", "current regulations", "today's",
        ),
        "calculation": (
            "calcula", "calculame", "realiza un calculo", "realiza calculos", "resuelve la ecuacion",
            "haz el calculo", "compute", "calculate", "perform a calculation", "solve the equation",
            "evaluate the formula",
        ),
        "code": (
            "revisa el codigo", "audita el codigo", "depura el codigo", "error en el codigo",
            "code review", "audit the code", "debug the code", "inspect the repository",
        ),
    }
    for tool, markers in tool_signals.items():
        if any(marker in normalized for marker in markers):
            return tool
    return None


def _multidocument_score(question: str) -> int:
    """Score evidence breadth, deliberately ignoring academic style requirements."""
    normalized = _normalize_term_text(question)
    score = 0
    breadth_markers = (
        "multiples documentos", "varios documentos", "todos los documentos", "documentos disponibles",
        "multiples estudios", "varios estudios", "estudios relevantes", "diferentes estudios",
        "multiple documents", "several documents", "all documents", "available documents",
        "multiple studies", "several studies", "relevant studies", "different studies",
    )
    comparison_markers = (
        "compara", "comparacion", "contrasta", "contradiccion", "convergencias", "diferencias entre",
        "compare", "comparison", "contrast", "contradiction", "contradictions", "differences between",
    )
    synthesis_markers = (
        "sintetiza", "sintesis", "integra", "integracion", "solucion integrada", "propone una solucion",
        "synthesize", "synthesis", "integrate", "integration", "integrated solution", "propose a solution",
    )
    dimension_markers = (
        "simultaneamente", "varias dimensiones", "multiples dimensiones", "diferentes dimensiones",
        "simultaneously", "several dimensions", "multiple dimensions", "across dimensions",
    )
    if any(marker in normalized for marker in breadth_markers):
        score += 2
    if any(marker in normalized for marker in comparison_markers):
        score += 2
    if any(marker in normalized for marker in synthesis_markers):
        score += 1
    if any(marker in normalized for marker in dimension_markers):
        score += 1
    return score


def route_query(
    question: str,
    document_id: Optional[str] = None,
    routing_mode: str = "auto",
    chat_history: Optional[List[Dict[str, str]]] = None,
    retrieved_document_count: Optional[int] = None,
) -> RoutingDecision:
    """Choose one route without an LLM call; shared by sync and streaming paths."""
    mode = _normalize_routing_mode(routing_mode)
    routing_question = _routing_question(question, chat_history)
    score = _multidocument_score(routing_question)

    if is_greeting(question):
        return RoutingDecision("greeting", "casual_conversation", 0, mode, document_id)

    if mode == "quick":
        route: RoutingRoute = "scoped_rag" if document_id else "simple_rag"
        return RoutingDecision(route, "manual_quick_documents_only", score, mode, document_id)

    required_tool = _required_tool(routing_question)
    if required_tool:
        return RoutingDecision(
            "tool_agent", f"explicit_{required_tool}_tool_required", score, mode, document_id,
            required_tool=required_tool,
        )

    if document_id:
        reason = "selected_document_deep_retrieval" if mode == "research" else "selected_document"
        return RoutingDecision("scoped_rag", reason, score, mode, document_id)

    if mode == "research":
        return RoutingDecision("research_rag", "manual_research", score, mode, None)

    if score >= 2:
        return RoutingDecision("research_rag", "multidocument_intent", score, mode, None)

    if score == 1:
        if retrieved_document_count is not None:
            if retrieved_document_count >= 3:
                return RoutingDecision("research_rag", "evidence_breadth_promotion", score, mode, None)
            return RoutingDecision("simple_rag", "insufficient_breadth_for_promotion", score, mode, None)
        return RoutingDecision("simple_rag", "ambiguous_breadth_probe", score, mode, None, provisional=True)

    return RoutingDecision("simple_rag", "direct_document_task", score, mode, None)


def _log_routing_decision(decision: RoutingDecision) -> None:
    logger.info(
        "Adaptive routing route=%s reason=%s score=%d mode=%s scope=%s version=%s",
        decision.route,
        decision.reason,
        decision.score,
        decision.mode,
        decision.document_scope or "all_documents",
        ROUTER_VERSION,
    )


def _should_use_agentic_reasoning(question: str) -> bool:
    """Compatibility wrapper; only true when deterministic routing requires tools."""
    return route_query(question).route == "tool_agent"


def _parse_highlight_rects(bbox: Any) -> List[Dict[str, Any]]:
    """Convert stored PDF bbox metadata into frontend highlight rectangles."""
    if not bbox:
        return []

    try:
        data = json.loads(bbox) if isinstance(bbox, str) else bbox
    except (TypeError, json.JSONDecodeError):
        return []

    if isinstance(data, list) and len(data) == 4 and all(isinstance(v, (int, float)) for v in data):
        data = [data]

    rects = []
    if not isinstance(data, list):
        return rects

    for rect in data:
        if not (isinstance(rect, list) and len(rect) == 4):
            continue
        x0, y0, x1, y1 = rect
        if not all(isinstance(v, (int, float)) for v in (x0, y0, x1, y1)):
            continue
        if not all(math.isfinite(float(v)) for v in (x0, y0, x1, y1)):
            continue
        rects.append(
            {
                "left": x0,
                "top": y0,
                "width": max(0, x1 - x0),
                "height": max(0, y1 - y0),
                "unit": "percent",
            }
        )

    return rects

#Carga los documentos y genera un payload de fuente para cada fragmento de texto recuperado.
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _normalize_term_text(text: str) -> str:
    replacements = str.maketrans(
        "áéíóúüñÁÉÍÓÚÜÑ",
        "aeiouunAEIOUUN",
    )
    return str(text or "").translate(replacements).lower()


def _meaningful_terms(text: str) -> set[str]:
    normalized = _normalize_term_text(text)
    terms = set()
    for term in re.findall(r"[a-z0-9]{4,}", normalized):
        if term not in _STOPWORDS:
            terms.add(term)
    return terms


def _stem_term(term: str) -> str:
    for suffix in ("ciones", "cion", "idades", "idad", "mente", "ables", "ible", "ibles", "ados", "adas", "icos", "icas", "es", "s"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def _has_term_match(query_terms: set[str], text_terms: set[str]) -> bool:
    text_stems = {_stem_term(term) for term in text_terms}
    return any(term in text_terms or _stem_term(term) in text_stems for term in query_terms)


def _lexical_overlap(query: str, text: str) -> float:
    query_terms = _meaningful_terms(query) - _GENERIC_DOCUMENT_TERMS
    if not query_terms:
        return 1.0
    text_terms = _meaningful_terms(text)
    matches = sum(1 for term in query_terms if _has_term_match({term}, text_terms))
    return matches / len(query_terms)


def _filter_evidence_chunks(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query_terms = _meaningful_terms(question) - _GENERIC_DOCUMENT_TERMS
    if not query_terms:
        return chunks

    filtered = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        overlap = _lexical_overlap(question, text)
        score = _safe_float(chunk.get("score"), _safe_float(chunk.get("retrieval_score")))
        confidence = _safe_float(chunk.get("confidence"))
        if overlap >= MIN_LEXICAL_OVERLAP or (overlap > 0 and score >= MIN_SOURCE_SCORE):
            chunk["lexical_overlap"] = round(overlap, 3)
            filtered.append(chunk)
        elif confidence >= MIN_SOURCE_CONFIDENCE:
            chunk["lexical_overlap"] = round(overlap, 3)
            filtered.append(chunk)

    return filtered or chunks


def _source_payload(chunk: Dict[str, Any], fallback_id: Optional[str] = None) -> Dict[str, Any]:
    if chunk.get("source_type") == "web":
        source_id = chunk.get("source_id", fallback_id or "")
        text = chunk.get("text") or chunk.get("snippet", "")
        return {
            "source_type": "web",
            "source_id": source_id,
            "title": chunk.get("title", "Web source"),
            "url": chunk.get("url", ""),
            "snippet": chunk.get("snippet", text),
            "text": text,
            "filename": chunk.get("title", "Web source"),
            "page": int(chunk.get("page") or 0),
            "score": _safe_float(chunk.get("score"), 1.0),
            "confidence": _safe_float(chunk.get("confidence", 0)),
            "citation": f"[{source_id}] {chunk.get('title', 'Web source')}".strip(),
        }

    source_id = chunk.get("source_id", fallback_id or "")
    source = {
        "source_type": "document",
        "source_id": source_id,
        "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
        "filename": chunk["filename"],
        "page": chunk["page"],
        "score": _safe_float(chunk.get("score")),
        "confidence": _safe_float(chunk.get("confidence", 0)),
        "lexical_overlap": _safe_float(chunk.get("lexical_overlap", 0)),
        "bbox": chunk.get("bbox", ""),
        "citation": f"[{source_id}] {chunk['filename']}, Página {chunk['page']}".strip(),
    }
    highlight_rects = _parse_highlight_rects(chunk.get("bbox"))
    if highlight_rects:
        source["highlightRects"] = highlight_rects
    return source

#Genera la etiqueta de cita para un fragmento de texto recuperado.
def _citation_label(source: Dict[str, Any]) -> str:
    source_id = source.get("source_id")
    if source_id:
        if source.get("source_type") == "web":
            return f"[{source_id}] {source.get('title') or source.get('filename', 'Fuente web')}"
        return f"[{source_id}] {source['filename']}, Página {source['page']}"
    return f"{source['filename']}, Página {source['page']}"

#Asegura que la respuesta generada por el agente incluya citas a las fuentes utilizadas.
def _ensure_answer_has_citations(answer: str, sources: List[Dict[str, Any]]) -> str:
    if not answer or not sources:
        return answer
    if re.search(r"\[((?:D|W)\d+)\]", answer):
        return answer

    seen = []
    for source in sources:
        label = _citation_label(source)
        if label not in seen:
            seen.append(label)
    return f"{answer}\n\nFuentes consultadas: {'; '.join(seen)}"


def _normalize_legacy_citations(answer: str, sources: List[Dict[str, Any]]) -> str:
    """Convert old '[Fuente: archivo, Página X]' labels to verifiable source IDs."""
    if not answer or not sources:
        return answer

    def replace(match: re.Match[str]) -> str:
        filename = match.group("filename").strip()
        page = int(match.group("page"))
        for source in sources:
            if str(source.get("filename", "")).strip() == filename and int(source.get("page") or 0) == page:
                source_id = source.get("source_id")
                if source_id:
                    return f"[{source_id}]"
        return match.group(0)

    return re.sub(
        r"\[Fuente:\s*(?P<filename>.+?),\s*P(?:á|a)gina\s+(?P<page>\d+)\]",
        replace,
        answer,
        flags=re.IGNORECASE,
    )


def _get_pdf_tool_sources(pdf_tool: PDFSearchTool) -> List[Dict[str, Any]]:
    return list(getattr(pdf_tool, "all_sources", None) or getattr(pdf_tool, "last_sources", []))


def _get_web_tool_sources(web_tool: WebSearchTool) -> List[Dict[str, Any]]:
    return list(getattr(web_tool, "all_sources", None) or getattr(web_tool, "last_sources", []))


def _run_initial_document_search(pdf_tool: PDFSearchTool, question: str) -> str:
    """Guarantee that a complex agent run starts with real document evidence."""
    if not isinstance(pdf_tool, PDFSearchTool):
        return ""
    try:
        logger.info("Agent research action: pdf_search (mandatory initial evidence search).")
        observation = pdf_tool.invoke({"query": question})
        source_count = len(_get_pdf_tool_sources(pdf_tool))
        logger.info("Agent initial pdf_search recovered %d sources.", source_count)
        return str(observation or "")
    except Exception as exc:
        logger.warning("Mandatory initial pdf_search failed: %s", exc)
        return ""


def _agent_question_with_search_state(question: str, initial_search: str) -> str:
    if not initial_search:
        return question
    return (
        f"{question}\n\n"
        "Ya se ejecutó una búsqueda documental inicial y sus resultados se conservarán para la síntesis final. "
        "Usa herramientas adicionales solo si necesitas cubrir un aspecto distinto. Cuando termines, responde "
        "con `Final Answer:`. No inventes fuentes ni identificadores."
    )


def _step_observations(intermediate_steps: Optional[List[Any]]) -> List[str]:
    """Read tool observations from invoke tuples and streaming AgentStep objects."""
    observations = []
    for step in intermediate_steps or []:
        observation = None
        if isinstance(step, (tuple, list)) and len(step) >= 2:
            observation = step[1]
        elif isinstance(step, dict):
            observation = step.get("observation")
        else:
            observation = getattr(step, "observation", None)

        if observation is not None:
            observations.append(str(observation))
    return observations


def _parse_sources_from_observation(observation: str) -> List[Dict[str, Any]]:
    """Recover citation metadata from the exact evidence shown to the ReAct agent."""
    recovered = []
    document_pattern = re.compile(
        r"UNTRUSTED DOCUMENT EXCERPT.*?\n"
        r"Source \[(?P<source_id>D\d+)\] "
        r"\((?P<filename>[^\n]+), Page (?P<page>\d+)\):\n"
        r"(?P<text>.*?)\nEND UNTRUSTED DOCUMENT EXCERPT",
        flags=re.DOTALL,
    )
    for match in document_pattern.finditer(observation or ""):
        recovered.append(
            {
                "source_type": "document",
                "source_id": match.group("source_id"),
                "filename": match.group("filename").strip(),
                "page": int(match.group("page")),
                "text": match.group("text").strip(),
                "score": 1.0,
                "confidence": 100.0,
            }
        )

    web_pattern = re.compile(
        r"UNTRUSTED WEB RESULT.*?\n"
        r"Source \[(?P<source_id>W\d+)\]: (?P<title>.*?)\n"
        r"URL: (?P<url>.*?)\n"
        r"Snippet: (?P<snippet>.*?)\nEND UNTRUSTED WEB RESULT",
        flags=re.DOTALL,
    )
    for match in web_pattern.finditer(observation or ""):
        snippet = match.group("snippet").strip()
        recovered.append(
            {
                "source_type": "web",
                "source_id": match.group("source_id"),
                "title": match.group("title").strip(),
                "filename": match.group("title").strip(),
                "url": match.group("url").strip(),
                "snippet": snippet,
                "text": snippet,
                "page": 0,
                "score": 1.0,
                "confidence": 100.0,
            }
        )
    return recovered


def _agent_source_key(source: Dict[str, Any]) -> tuple[str, str, int, str]:
    source_type = str(source.get("source_type") or "document")
    location = str(source.get("url") or source.get("filename") or "").strip()
    page = int(_safe_float(source.get("page"), 0))
    text = re.sub(r"\s+", " ", str(source.get("text") or source.get("snippet") or "")).strip()
    return source_type, location, page, text[:300]


def _collect_agent_sources(
    pdf_tool: PDFSearchTool,
    web_tool: WebSearchTool,
    intermediate_steps: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Collect all evidence even when LangChain does not preserve tool object state."""
    candidates = [*_get_pdf_tool_sources(pdf_tool), *_get_web_tool_sources(web_tool)]
    for observation in _step_observations(intermediate_steps):
        candidates.extend(_parse_sources_from_observation(observation))

    unique_sources = []
    seen = set()
    for source in candidates:
        if not isinstance(source, dict):
            continue
        key = _agent_source_key(source)
        if key in seen or not (source.get("text") or source.get("snippet")):
            continue
        seen.add(key)
        unique_sources.append(source)

    document_index = 0
    web_index = 0
    for source in unique_sources:
        if source.get("source_type") == "web":
            web_index += 1
            source["source_id"] = f"W{web_index}"
        else:
            document_index += 1
            source["source_type"] = "document"
            source["source_id"] = f"D{document_index}"

    return unique_sources


def _has_relevant_sources(sources: List[Dict[str, Any]]) -> bool:
    return any(
        _safe_float(source.get("score")) >= MIN_SOURCE_SCORE
        or _safe_float(source.get("confidence")) >= MIN_SOURCE_CONFIDENCE
        or _safe_float(source.get("lexical_overlap")) >= MIN_LEXICAL_OVERLAP
        for source in sources
    )


def _validate_answer_citations(answer: str, sources: List[Dict[str, Any]]) -> str:
    if not answer:
        return answer
    if not sources:
        logger.warning("Rejected agent answer because no document or web evidence was recovered.")
        return INSUFFICIENT_EVIDENCE_MESSAGE

    answer = _normalize_legacy_citations(answer, sources)

    if re.search(r"\[Fuente:\s*.+?\]", answer, flags=re.IGNORECASE):
        logger.warning("Rejected answer with legacy or unverifiable citation labels.")
        return INSUFFICIENT_EVIDENCE_MESSAGE

    valid_ids = {str(source.get("source_id")) for source in sources if source.get("source_id")}
    cited_ids = set(re.findall(r"\[((?:D|W)\d+)\]", answer))
    if not cited_ids:
        logger.warning("Rejected answer without structured source citations.")
        return INSUFFICIENT_EVIDENCE_MESSAGE

    invalid_ids = cited_ids - valid_ids
    if invalid_ids:
        logger.warning("Rejected answer with invented citations: %s", sorted(invalid_ids))
        return INSUFFICIENT_EVIDENCE_MESSAGE

    return answer


def _document_key(source: Dict[str, Any]) -> str:
    """Identify a document independently from the chunk used to cite it."""
    return str(source.get("document_id") or source.get("filename") or "").strip()


def _required_document_citations(question: str, sources: List[Dict[str, Any]]) -> int:
    """Require broad comparative requests to actually synthesize multiple documents."""
    available_documents = {
        _document_key(source)
        for source in sources
        if source.get("source_type", "document") == "document" and _document_key(source)
    }
    available_count = len(available_documents)
    if available_count <= 1:
        return available_count

    normalized = _normalize_term_text(question)
    broad_markers = (
        "compara",
        "comparando",
        "multiples",
        "documentos disponibles",
        "estudios relevantes",
        "sintetiza",
        "sintesis",
        "tecnologias",
        "solucion integrada",
    )
    if any(marker in normalized for marker in broad_markers):
        return min(5, available_count)
    return min(2, available_count)


def _cited_document_count(answer: str, sources: List[Dict[str, Any]]) -> int:
    sources_by_id = {
        str(source.get("source_id")): source
        for source in sources
        if source.get("source_id")
    }
    cited_ids = set(re.findall(r"\[((?:D|W)\d+)\]", answer or ""))
    return len(
        {
            _document_key(sources_by_id[source_id])
            for source_id in cited_ids
            if source_id in sources_by_id
            and sources_by_id[source_id].get("source_type", "document") == "document"
            and _document_key(sources_by_id[source_id])
        }
    )


def _generate_agentic_document_answer(
    question: str,
    user_id: str,
    document_id: Optional[str],
    hf_token: Optional[str],
    top_k: Optional[int],
    chat_history: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    logger.info("RAG route: invoking ReAct agent first for complex request.")
    executor, pdf_tool, web_tool, formatted_history = get_agent_executor(
        user_id, document_id, hf_token, top_k, chat_history
    )
    initial_search = _run_initial_document_search(pdf_tool, question)
    agent_question = _agent_question_with_search_state(question, initial_search)
    try:
        result = executor.invoke({"input": agent_question, "chat_history": formatted_history})
    except Exception as exc:
        raw_sources = _collect_agent_sources(pdf_tool, web_tool)
        if raw_sources:
            logger.warning(
                "Agent invocation failed after recovering %d sources; synthesizing from preserved evidence: %s",
                len(raw_sources),
                exc,
            )
            sources = [_source_payload(chunk) for chunk in raw_sources]
            partial_answer = _generate_partial_answer_from_agent_sources(
                question=question,
                raw_sources=raw_sources,
                sources=sources,
                hf_token=hf_token,
                chat_history=chat_history,
            )
            return {"answer": partial_answer, "sources": sources}
        raise

    raw_answer = result.get("output", "")
    try:
        answer = parse_agent_output(raw_answer)
    except OutputParserError as e:
        logger.warning(f"Rejected malformed LLM output: {e}")
        answer = MALFORMED_OUTPUT_MESSAGE

    raw_sources = _collect_agent_sources(
        pdf_tool,
        web_tool,
        result.get("intermediate_steps", []),
    )
    logger.info(
        "Agent completed with %d intermediate steps and %d recovered sources.",
        len(result.get("intermediate_steps", [])),
        len(raw_sources),
    )
    sources = [_source_payload(chunk) for chunk in raw_sources]
    if _is_agent_stop_answer(answer):
        partial_answer = _generate_partial_answer_from_agent_sources(
            question=question,
            raw_sources=raw_sources,
            sources=sources,
            hf_token=hf_token,
            chat_history=chat_history,
        )
        return {"answer": partial_answer, "sources": sources}
    if initial_search and raw_sources:
        logger.info(
            "Agent research phase finished; running mandatory grounded final synthesis from %d sources.",
            len(raw_sources),
        )
        answer = _generate_partial_answer_from_agent_sources(
            question=question,
            raw_sources=raw_sources,
            sources=sources,
            hf_token=hf_token,
            chat_history=chat_history,
            notice="",
        )
        return {"answer": answer, "sources": sources}
    answer = _validate_or_regenerate_agent_answer(
        question=question,
        answer=answer,
        raw_sources=raw_sources,
        sources=sources,
        hf_token=hf_token,
        chat_history=chat_history,
    )

    return {"answer": answer, "sources": sources}


def _is_agent_stop_answer(answer: str) -> bool:
    text = (answer or "").lower()
    stop_phrases = [
        "agent stopped due to iteration limit",
        "agent stopped due to iteration",
        "agent stopped due to time limit",
        "iteration limit or time limit",
        "límite de iteraciones",
        "limite de iteraciones",
        "límite de tiempo",
        "limite de tiempo",
        "condición de alto",
        "condicion de alto",
    ]
    return any(phrase in text for phrase in stop_phrases)


def _retrieve_document_evidence(
    question: str,
    user_id: str,
    document_id: Optional[str],
    top_k: Optional[int],
) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    chunks = retrieve(
        query=question,
        user_id=user_id,
        document_id=document_id,
        top_k=top_k,
    )
    chunks = _filter_evidence_chunks(question, chunks)
    for index, chunk in enumerate(chunks, 1):
        chunk.setdefault("source_type", "document")
        chunk["source_id"] = f"D{index}"

    sources = [_source_payload(chunk) for chunk in chunks]

    # A selected document is an explicit scope chosen by the user. Low semantic
    # scores for generic instructions such as "summarize this document" must not
    # erase otherwise valid chunks from that document.
    if not chunks or (not document_id and not _has_relevant_sources(sources)):
        return "", sources, chunks

    context_parts = [
        (
            f"Fuente [{chunk['source_id']}] ({chunk['filename']}, Página {chunk['page']}):\n"
            f"{chunk['text']}"
        )
        for chunk in chunks
    ]

    try:
        graph_context = get_entity_context(
            query=question,
            user_id=user_id,
            document_id=document_id,
        )
    except Exception as exc:
        logger.warning("Graph context retrieval failed: %s", exc)
        graph_context = ""

    if graph_context:
        context_parts.append(f"Relaciones adicionales:\n{graph_context}")

    return "\n\n".join(context_parts), sources, chunks


def _retrieve_document_context(
    question: str,
    user_id: str,
    document_id: Optional[str],
    top_k: Optional[int],
) -> tuple[str, List[Dict[str, Any]]]:
    """Compatibility wrapper for callers that do not need the full evidence chunks."""
    context, sources, _ = _retrieve_document_evidence(question, user_id, document_id, top_k)
    return context, sources


def _build_style_reference(sources: List[Dict[str, Any]]) -> str:
    """Create a brief style instruction from the retrieved document snippet."""
    if not sources:
        return ""

    sample = next((source for source in sources if source.get("text")), None)
    if not sample:
        return ""

    snippet = re.sub(r"\s+", " ", str(sample.get("text", "")).strip())
    if len(snippet) > 500:
        snippet = snippet[:497] + "..."

    filename = sample.get("filename", "documento")
    return (
        "## Referencia de estilo\n"
        f"Imita el tono, el ritmo y la cadencia del siguiente fragmento del documento '{filename}'. "
        "No copies literalmente las frases; responde de forma original, elegante y coherente con ese estilo.\n"
        f"Fragmento de referencia: {snippet}"
    )


def _build_direct_rag_prompt(
    question: str,
    context: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    style_reference: Optional[str] = None,
) -> str:
    history = _format_chat_history(chat_history) if chat_history else ""
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        style_reference=style_reference or "",
    )
    if history:
        prompt = f"{history}\n\n{prompt}"
    return prompt


def _generate_grounded_selected_document_answer(
    question: str,
    context: str,
    sources: List[Dict[str, Any]],
    hf_token: Optional[str],
    chat_history: Optional[List[Dict[str, str]]],
) -> str:
    """Generate and verify a response for an explicitly selected document."""
    from langchain_core.messages import HumanMessage

    prompt = _build_direct_rag_prompt(
        question,
        context,
        chat_history,
        style_reference=_load_global_style_reference(),
    )
    prompt += (
        "\n\nEl usuario seleccionó explícitamente este documento. Responde sobre él aunque la solicitud sea "
        "genérica, como resumir, explicar o redactar. Usa citas [D#] en línea y no confundas una omisión "
        "de formato de cita con falta de evidencia."
    )
    chat_llm = get_llm_client(hf_token, max_tokens=settings.AGENT_SYNTHESIS_MAX_TOKENS)
    last_error: Optional[Exception] = None
    completed_generation = False

    for attempt in range(2):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                "\n\nLa respuesta anterior omitió o inventó identificadores de fuente. Redacta nuevamente la "
                "respuesta completa y usa únicamente las citas [D#] visibles en el contexto. No menciones "
                "esta revisión."
            )
        try:
            response = chat_llm.invoke([HumanMessage(content=attempt_prompt)])
            answer = parse_agent_output(response.content)
            completed_generation = True
        except Exception as exc:
            last_error = exc
            logger.warning("Selected-document synthesis attempt %d failed: %s", attempt + 1, exc)
            continue

        answer = _validate_answer_citations(answer, sources)
        if answer != INSUFFICIENT_EVIDENCE_MESSAGE:
            return answer

    if not completed_generation and last_error is not None:
        raise last_error
    return _build_evidence_only_fallback(sources, "")


def _build_agent_source_context(raw_sources: List[Dict[str, Any]]) -> str:
    """Format sources already recovered by the agent without performing a new retrieval."""
    context_parts = []
    seen_ids = set()
    total_chars = 0
    max_context_chars = 14000

    # Put one excerpt from every document first so repeated chunks from the
    # highest-ranked papers cannot consume the context window by themselves.
    ordered_sources = []
    deferred_sources = []
    seen_documents = set()
    for source in raw_sources:
        document_key = _document_key(source) or str(source.get("url") or "")
        if document_key and document_key not in seen_documents:
            ordered_sources.append(source)
            seen_documents.add(document_key)
        else:
            deferred_sources.append(source)
    ordered_sources.extend(deferred_sources)

    for index, source in enumerate(ordered_sources, 1):
        if total_chars >= max_context_chars:
            break
        source_id = source.get("source_id") or f"D{index}"
        if source_id in seen_ids:
            continue
        seen_ids.add(source_id)

        text = str(source.get("text") or source.get("snippet") or "").strip()
        if not text:
            continue
        remaining_chars = max_context_chars - total_chars
        source_limit = min(1000, remaining_chars)
        if len(text) > source_limit:
            text = text[: max(0, source_limit - 3)] + "..."

        if source.get("source_type") == "web":
            title = source.get("title") or source.get("filename") or "Fuente web"
            url = source.get("url", "")
            context_part = (
                f"Fuente [{source_id}] ({title})\n"
                f"URL: {url}\n"
                f"{text}"
            )
            context_parts.append(context_part)
            total_chars += len(context_part)
            continue

        filename = source.get("filename", "documento")
        page = source.get("page", "?")
        context_part = (
            f"Fuente [{source_id}] ({filename}, Página {page}):\n{text}"
        )
        context_parts.append(context_part)
        total_chars += len(context_part)

    return "\n\n".join(context_parts)


def _build_partial_agent_answer_prompt(
    question: str,
    context: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    draft_answer: Optional[str] = None,
) -> str:
    history = _format_chat_history(chat_history) if chat_history else ""
    response_kind = "final"
    task_description = (
        "La fase de investigación recuperó evidencia documental. Realiza una síntesis nueva e independiente; "
        "no reutilices conclusiones preliminares del planificador."
    )
    prompt = f"""{history}

{task_description}
Redacta ahora la mejor respuesta académica {response_kind} usando ÚNICAMENTE la evidencia recuperada abajo.

## Evidencia recuperada

{context}

## Pregunta original

{question}

## Instrucciones

- Responde en español académico.
- No hagas una nueva búsqueda ni asumas información externa.
- Antes de redactar, construye internamente una matriz de evidencia: tema, metodología, resultado, fortaleza y limitación de cada documento. No muestres esa matriz ni tu razonamiento interno.
- Examina TODAS las fuentes proporcionadas. Para solicitudes comparativas, integra al menos cinco documentos distintos si están disponibles; no centres toda la respuesta en los dos primeros.
- Organiza la respuesta exactamente según los productos y el idioma pedidos por el usuario. No repitas la pregunta ni estas instrucciones.
- Redacta con densidad académica y sin redundancias; completa todos los productos solicitados dentro del espacio disponible.
- Distingue resultados explícitos de inferencias. No afirmes beneficios sobre aire, agua o energía si el fragmento citado no los respalda.
- Cita cada afirmación sustantiva usando solo los identificadores visibles en la evidencia, por ejemplo [D1], [D2] o [W1].
- Distribuye las citas a lo largo del análisis y compara convergencias, contradicciones y limitaciones entre estudios.
- Si una conclusión queda incompleta por falta de evidencia, indícalo explícitamente.
- No menciones límites de iteraciones, borradores, prompts, herramientas, razonamiento interno ni cadena de pensamiento.

## Respuesta {response_kind}
"""
    return prompt.strip()


def _generate_partial_answer_from_agent_sources(
    question: str,
    raw_sources: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    hf_token: Optional[str],
    chat_history: Optional[List[Dict[str, str]]],
    draft_answer: Optional[str] = None,
    notice: str = "",
) -> str:
    """Synthesize an answer from the agent's already-recovered evidence after an iteration stop."""
    if not raw_sources or not sources:
        return AGENT_INCOMPLETE_MESSAGE

    context = _build_agent_source_context(raw_sources)
    if not context:
        return AGENT_INCOMPLETE_MESSAGE

    from langchain_core.messages import HumanMessage

    chat_llm = get_llm_client(hf_token, max_tokens=settings.AGENT_SYNTHESIS_MAX_TOKENS)
    # Do not pass the planner draft to the writer. Early planner answers tend to
    # anchor the synthesis on the first one or two documents it noticed.
    prompt = _build_partial_agent_answer_prompt(question, context, chat_history)
    required_documents = _required_document_citations(question, sources)
    document_source_ids: Dict[str, List[str]] = {}
    for source in sources:
        if source.get("source_type", "document") != "document":
            continue
        document_key = _document_key(source)
        source_id = str(source.get("source_id") or "")
        if document_key and source_id:
            document_source_ids.setdefault(document_key, []).append(source_id)
    coverage_guide = "\n".join(
        f"- {document}: {', '.join(f'[{source_id}]' for source_id in source_ids)}"
        for document, source_ids in document_source_ids.items()
    )
    if required_documents:
        prompt += (
            "\n\n## Cobertura documental obligatoria\n"
            f"La respuesta debe citar evidencia de al menos {required_documents} documentos distintos de esta lista:\n"
            f"{coverage_guide}\n"
            "Los identificadores de varias páginas del mismo archivo cuentan como un solo documento. "
            "No muestres esta lista de control en la respuesta."
        )

    for attempt in range(2):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                "\n\n## Revisión obligatoria\n"
                f"La versión anterior no integró el mínimo de {required_documents} documentos distintos. "
                "Redacta la respuesta completa nuevamente desde la evidencia, ampliando la comparación y "
                "manteniendo citas válidas. No comentes esta revisión en la respuesta."
            )
        try:
            response = chat_llm.invoke([HumanMessage(content=attempt_prompt)])
            answer = parse_agent_output(response.content)
        except Exception as exc:
            logger.warning("Grounded synthesis attempt %d failed: %s", attempt + 1, exc)
            continue

        if _is_agent_stop_answer(answer):
            continue
        answer = _validate_answer_citations(answer, sources)
        cited_documents = _cited_document_count(answer, sources)
        if answer != INSUFFICIENT_EVIDENCE_MESSAGE and cited_documents >= required_documents:
            logger.info(
                "Grounded synthesis accepted with %d cited documents (required=%d).",
                cited_documents,
                required_documents,
            )
            return answer
        logger.warning(
            "Grounded synthesis cited %d distinct documents; %d required. Retrying.",
            cited_documents,
            required_documents,
        )

    return _build_evidence_only_fallback(raw_sources, "")


def _build_evidence_only_fallback(raw_sources: List[Dict[str, Any]], notice: str) -> str:
    """Guarantee a cited, evidence-only response when the final synthesis call fails."""
    evidence_lines = []
    diverse_sources = []
    repeated_sources = []
    seen_documents = set()
    for source in raw_sources:
        document_key = _document_key(source) or str(source.get("url") or "")
        if document_key and document_key not in seen_documents:
            diverse_sources.append(source)
            seen_documents.add(document_key)
        else:
            repeated_sources.append(source)

    for source in [*diverse_sources, *repeated_sources][:12]:
        source_id = source.get("source_id")
        text = re.sub(
            r"\s+",
            " ",
            str(source.get("text") or source.get("snippet") or "").strip(),
        )
        if not source_id or not text:
            continue
        excerpt = text[:420] + ("..." if len(text) > 420 else "")
        if source.get("source_type") == "web":
            label = source.get("title") or source.get("filename") or "Fuente web"
        else:
            label = f"{source.get('filename', 'documento')}, página {source.get('page', '?')}"
        evidence_lines.append(f"- [{source_id}] **{label}**: {excerpt}")

    if not evidence_lines:
        return AGENT_INCOMPLETE_MESSAGE

    prefix = f"{notice}\n\n" if notice else ""
    return prefix + (
        "No fue posible completar la síntesis interpretativa, pero sí conservar la evidencia verificable "
        "recuperada durante la fase de investigación:\n\n"
        + "\n".join(evidence_lines)
        + "\n\nLas relaciones o conclusiones que no se desprendan explícitamente de estos fragmentos "
        "deben considerarse insuficientemente sustentadas."
    )


def _validate_or_regenerate_agent_answer(
    question: str,
    answer: str,
    raw_sources: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    hf_token: Optional[str],
    chat_history: Optional[List[Dict[str, str]]],
) -> str:
    validated_answer = _validate_answer_citations(answer, sources)
    if validated_answer != INSUFFICIENT_EVIDENCE_MESSAGE:
        return validated_answer

    if not raw_sources or not sources or not answer or answer == MALFORMED_OUTPUT_MESSAGE:
        return validated_answer

    logger.warning(
        "Agent answer had invalid or missing citations; regenerating from agent-recovered sources only."
    )
    return _generate_partial_answer_from_agent_sources(
        question=question,
        raw_sources=raw_sources,
        sources=sources,
        hf_token=hf_token,
        chat_history=chat_history,
        notice="",
    )


def _generate_direct_document_answer(
    question: str,
    user_id: str,
    document_id: Optional[str],
    hf_token: Optional[str],
    top_k: Optional[int],
    chat_history: Optional[List[Dict[str, str]]],
    evidence: Optional[tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    logger.info("RAG route: using direct document RAG for simple request.")
    if evidence is None:
        context, sources, _ = _retrieve_document_evidence(question, user_id, document_id, top_k)
    else:
        context, sources, _ = evidence
    if not context:
        return {
            "answer": "No encontré información suficiente en los documentos cargados para responder esta pregunta.",
            "sources": [],
        }

    if document_id:
        answer = _generate_grounded_selected_document_answer(
            question=question,
            context=context,
            sources=sources,
            hf_token=hf_token,
            chat_history=chat_history,
        )
        return {"answer": answer, "sources": sources}

    from langchain_core.messages import HumanMessage

    style_reference = _load_global_style_reference()
    chat_llm = get_llm_client(hf_token)
    prompt = _build_direct_rag_prompt(question, context, chat_history, style_reference=style_reference)
    response = chat_llm.invoke([HumanMessage(content=prompt)])
    answer = parse_agent_output(response.content)
    if _is_agent_stop_answer(answer):
        answer = "No pude generar una respuesta final estable. Reformulo con el contexto recuperado: " + (
            "No encontré información suficiente en los documentos cargados para responder esta pregunta."
        )
    answer = _validate_answer_citations(answer, sources)
    return {"answer": answer, "sources": sources}


def _effective_retrieval_top_k(decision: RoutingDecision, top_k: Optional[int]) -> Optional[int]:
    if decision.mode == "research" or decision.route == "research_rag":
        return max(top_k or 0, settings.TOP_K_RERANK)
    return top_k


def _retrieved_document_count(raw_sources: List[Dict[str, Any]]) -> int:
    return len({_document_key(source) for source in raw_sources if _document_key(source)})


def _execute_document_route(
    decision: RoutingDecision,
    question: str,
    user_id: str,
    document_id: Optional[str],
    hf_token: Optional[str],
    top_k: Optional[int],
    chat_history: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """Retrieve once, then execute either scoped/simple or deep research synthesis."""
    retrieval_top_k = _effective_retrieval_top_k(decision, top_k)
    evidence = _retrieve_document_evidence(question, user_id, document_id, retrieval_top_k)
    context, sources, raw_sources = evidence

    if decision.provisional:
        decision = route_query(
            question=question,
            document_id=document_id,
            routing_mode=decision.mode,
            chat_history=chat_history,
            retrieved_document_count=_retrieved_document_count(raw_sources),
        )
        _log_routing_decision(decision)

    if not context:
        return {
            "answer": "No encontré información suficiente en los documentos cargados para responder esta pregunta.",
            "sources": [],
        }

    if decision.route == "research_rag":
        logger.info("RAG route: deep multi-document retrieval and grounded synthesis without ReAct.")
        answer = _generate_partial_answer_from_agent_sources(
            question=question,
            raw_sources=raw_sources,
            sources=sources,
            hf_token=hf_token,
            chat_history=chat_history,
            notice="",
        )
        return {"answer": answer, "sources": sources}

    return _generate_direct_document_answer(
        question=question,
        user_id=user_id,
        document_id=document_id,
        hf_token=hf_token,
        top_k=retrieval_top_k,
        chat_history=chat_history,
        evidence=evidence,
    )


@trace_function(
    "generate_answer",
    metadata_factory=lambda question, user_id, document_id=None, **kwargs: {
        "user_id": user_id,
        "document_id": document_id,
        "llm_model": settings.LLM_MODEL,
    },
)
def generate_answer(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    routing_mode: RoutingMode = "auto",
) -> Dict[str, Any]:
    """Generate through the deterministic adaptive router."""

    decision = route_query(question, document_id, routing_mode, chat_history)
    _log_routing_decision(decision)

    # ── Handle greetings ─────────────────────────────
    if decision.route == "greeting":
        chat_llm = get_llm_client(hf_token)
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content="Eres Document AI Analyst, un asistente de IA amigable. Responde SIEMPRE en español."),
                HumanMessage(content=question),
            ]
            response = chat_llm.invoke(messages)
            answer = response.content.strip()
        except Exception:
            answer = "¡Hola! Soy ATLAS. ¿En qué puedo ayudarte hoy?"
        return {"answer": answer, "sources": []}

    if decision.route == "tool_agent":
        try:
            return _generate_agentic_document_answer(
                question=question,
                user_id=user_id,
                document_id=document_id,
                hf_token=hf_token,
                top_k=top_k,
                chat_history=chat_history,
            )
        except Exception as e:
            logger.error("Tool agent failed; direct document fallback is not equivalent: %s", e)
            return {"answer": AGENT_INCOMPLETE_MESSAGE, "sources": []}

    try:
        return _execute_document_route(
            decision=decision,
            question=question,
            user_id=user_id,
            document_id=document_id,
            hf_token=hf_token,
            top_k=top_k,
            chat_history=chat_history,
        )
    except Exception as e:
        logger.error("Document RAG generation error on route %s: %s", decision.route, e)
        raise ExternalServiceException("Ollama", str(e)) from e




@trace_function(
    "generate_answer_stream",
    metadata_factory=lambda question, user_id, document_id=None, **kwargs: {
        "user_id": user_id,
        "document_id": document_id,
        "llm_model": settings.LLM_MODEL,
    },
)
def generate_answer_stream(
    question: str,
    user_id: str,
    document_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    routing_mode: RoutingMode = "auto",
) -> Generator[str, None, None]:
    """Stream a response using the same deterministic router as the REST path."""

    decision = route_query(question, document_id, routing_mode, chat_history)
    _log_routing_decision(decision)

    # ── Handle greetings ─────────────────────────────
    if decision.route == "greeting":
        yield f"data: {json.dumps({'type': 'sources', 'data': []})}\n\n"
        chat_llm = get_llm_client(hf_token)
        try:
            from langchain_core.messages import HumanMessage
            for chunk in chat_llm.stream([HumanMessage(content=question)]):
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    use_agentic_first = decision.route == "tool_agent"

    if not use_agentic_first:
        try:
            result = _execute_document_route(
                decision=decision,
                question=question,
                user_id=user_id,
                document_id=document_id,
                hf_token=hf_token,
                top_k=top_k,
                chat_history=chat_history,
            )
            yield f"data: {json.dumps({'type': 'sources', 'data': result['sources']})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': result['answer']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        except Exception as e:
            logger.error("Streaming document route %s failed: %s", decision.route, e)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

    pdf_tool = None
    web_tool = None
    accumulated_steps: List[Any] = []
    sources_sent = False
    answer_sent = False
    sent_source_keys = set()

    try:
        logger.info("Streaming RAG route: invoking ReAct agent first for complex request.")
        executor, pdf_tool, web_tool, formatted_history = get_agent_executor(
            user_id, document_id, hf_token, top_k, chat_history
        )
        initial_search = _run_initial_document_search(pdf_tool, question)
        agent_question = _agent_question_with_search_state(question, initial_search)
        initial_sources = _collect_agent_sources(pdf_tool, web_tool)
        if initial_sources:
            sources = [_source_payload(chunk) for chunk in initial_sources]
            sent_source_keys = {_agent_source_key(source) for source in initial_sources}
            sources_sent = True
            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        for step in executor.stream({"input": agent_question, "chat_history": formatted_history}):
            if "actions" in step:
                for action in step.get("actions") or []:
                    logger.info(
                        "Agent research action: %s input=%s",
                        getattr(action, "tool", "unknown"),
                        str(getattr(action, "tool_input", ""))[:300],
                    )
                continue

            new_steps = step.get("steps") or step.get("intermediate_step") or []
            if new_steps:
                accumulated_steps.extend(new_steps)
                for agent_step in new_steps:
                    action = getattr(agent_step, "action", None)
                    logger.info(
                        "Agent research observation received from %s.",
                        getattr(action, "tool", "unknown"),
                    )
                tool_sources = _collect_agent_sources(pdf_tool, web_tool, accumulated_steps)
                current_source_keys = {_agent_source_key(source) for source in tool_sources}
                if tool_sources and current_source_keys != sent_source_keys:
                    sources = [_source_payload(chunk) for chunk in tool_sources]
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                    sources_sent = True
                    sent_source_keys = current_source_keys
                continue

            if "output" in step:
                final_steps = step.get("intermediate_steps") or []
                if final_steps:
                    accumulated_steps.extend(final_steps)
                tool_sources = _collect_agent_sources(pdf_tool, web_tool, accumulated_steps)
                sources = [_source_payload(chunk) for chunk in tool_sources]
                current_source_keys = {_agent_source_key(source) for source in tool_sources}
                if not sources_sent or current_source_keys != sent_source_keys:
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                    sources_sent = True
                    sent_source_keys = current_source_keys

                try:
                    clean_answer = parse_agent_output(step["output"])
                except OutputParserError as e:
                    logger.warning(f"Rejected malformed streamed LLM output: {e}")
                    clean_answer = MALFORMED_OUTPUT_MESSAGE

                if _is_agent_stop_answer(clean_answer):
                    logger.warning("Streaming agent stopped before a final answer.")
                    break

                if initial_search and tool_sources:
                    logger.info(
                        "Agent research phase finished; running mandatory grounded final synthesis from %d sources.",
                        len(tool_sources),
                    )
                    clean_answer = _generate_partial_answer_from_agent_sources(
                        question=question,
                        raw_sources=tool_sources,
                        sources=sources,
                        hf_token=hf_token,
                        chat_history=chat_history,
                        notice="",
                    )
                else:
                    clean_answer = _validate_or_regenerate_agent_answer(
                        question=question,
                        answer=clean_answer,
                        raw_sources=tool_sources,
                        sources=sources,
                        hf_token=hf_token,
                        chat_history=chat_history,
                    )
                yield f"data: {json.dumps({'type': 'token', 'data': clean_answer})}\n\n"
                answer_sent = True

        if answer_sent:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        if use_agentic_first:
            tool_sources = _collect_agent_sources(pdf_tool, web_tool, accumulated_steps)
            sources = [_source_payload(chunk) for chunk in tool_sources]
            current_source_keys = {_agent_source_key(source) for source in tool_sources}
            if not sources_sent or current_source_keys != sent_source_keys:
                yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
            logger.info(
                "Streaming agent stopped with %d observed steps and %d recovered sources; synthesizing partial answer.",
                len(accumulated_steps),
                len(tool_sources),
            )
            partial_answer = _generate_partial_answer_from_agent_sources(
                question=question,
                raw_sources=tool_sources,
                sources=sources,
                hf_token=hf_token,
                chat_history=chat_history,
            )
            yield f"data: {json.dumps({'type': 'token', 'data': partial_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        context, sources = _retrieve_document_context(question, user_id, document_id, top_k)
        if not context:
            fallback_answer = "No encontré información suficiente en los documentos cargados para responder esta pregunta."
            yield f"data: {json.dumps({'type': 'sources', 'data': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': fallback_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        from langchain_core.messages import HumanMessage

        style_reference = _load_global_style_reference()
        chat_llm = get_llm_client(hf_token)
        prompt = _build_direct_rag_prompt(question, context, chat_history, style_reference=style_reference)

        try:
            if hasattr(chat_llm, "stream"):
                collected_chunks = []
                for chunk in chat_llm.stream([HumanMessage(content=prompt)]):
                    content = getattr(chunk, "content", "")
                    if content:
                        collected_chunks.append(str(content))
                answer = parse_agent_output("".join(collected_chunks))
            else:
                response = chat_llm.invoke([HumanMessage(content=prompt)])
                answer = parse_agent_output(response.content)
        except Exception as stream_error:
            logger.error(f"Direct RAG streaming fallback error: {stream_error}")
            answer = "No pude generar una respuesta final estable."

        answer = _validate_answer_citations(answer, sources)
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        if not answer_sent:
            yield f"data: {json.dumps({'type': 'token', 'data': answer})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    except Exception as e:
        logger.warning("Agentic streaming failed: %s", e)
        if use_agentic_first:
            if pdf_tool is not None and web_tool is not None:
                tool_sources = _collect_agent_sources(pdf_tool, web_tool, accumulated_steps)
                if tool_sources:
                    sources = [_source_payload(chunk) for chunk in tool_sources]
                    logger.warning(
                        "Recovering complex streaming response from %d sources preserved before the error.",
                        len(tool_sources),
                    )
                    partial_answer = _generate_partial_answer_from_agent_sources(
                        question=question,
                        raw_sources=tool_sources,
                        sources=sources,
                        hf_token=hf_token,
                        chat_history=chat_history,
                    )
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                    yield f"data: {json.dumps({'type': 'token', 'data': partial_answer})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
            yield f"data: {json.dumps({'type': 'sources', 'data': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': AGENT_INCOMPLETE_MESSAGE})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        try:
            context, sources = _retrieve_document_context(question, user_id, document_id, top_k)
            if not context:
                yield f"data: {json.dumps({'type': 'sources', 'data': []})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'data': 'No encontré información suficiente en los documentos cargados para responder esta pregunta.'})}\n\n"
            else:
                from langchain_core.messages import HumanMessage

                style_reference = _load_global_style_reference()
                chat_llm = get_llm_client(hf_token)
                prompt = _build_direct_rag_prompt(question, context, chat_history, style_reference=style_reference)
                if hasattr(chat_llm, "stream"):
                    collected_chunks = []
                    for chunk in chat_llm.stream([HumanMessage(content=prompt)]):
                        content = getattr(chunk, "content", "")
                        if content:
                            collected_chunks.append(str(content))
                    answer = parse_agent_output("".join(collected_chunks))
                else:
                    response = chat_llm.invoke([HumanMessage(content=prompt)])
                    answer = parse_agent_output(response.content)

                answer = _validate_answer_citations(answer, sources)
                yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'data': answer})}\n\n"
        except Exception as fallback_error:
            logger.error(f"Direct RAG streaming fallback error: {fallback_error}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(fallback_error)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
