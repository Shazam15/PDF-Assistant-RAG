"""
Agentic RAG — intelligent routing using ReAct (Reasoning and Acting).
Intelligently chooses between PDF search, Web Search, and Math tools.
"""
import logging
import json
import os
import re
from typing import List, Dict, Any, Optional, Generator

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.rag.retriever import retrieve
from app.rag.graph_retriever import get_entity_context
from app.rag.prompts import AGENT_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, CODE_REVIEW_PROMPT
from app.exceptions import ExternalServiceException
from app.rag.security import MALFORMED_OUTPUT_MESSAGE, OutputParserError, parse_agent_output
from app.rag.tools import PDFSearchTool, MathTool, CodeReviewTool
from app.rag.tracing import trace_function

logger = logging.getLogger(__name__)
settings = get_settings()


def get_llm_client(hf_token: Optional[str] = None):
    """Create an Ollama client (hf_token ignored, kept for compatibility)."""
    return ChatOllama(
        model=settings.LLM_MODEL, 
        temperature=0,
        num_predict=settings.LLM_MAX_NEW_TOKENS)


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
    tools = [pdf_tool, code_review_tool, MathTool()]

    chat_llm = ChatOllama(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        num_predict=settings.LLM_MAX_NEW_TOKENS
    )

    global_style_reference = _load_global_style_reference()
    prompt = PromptTemplate.from_template(AGENT_SYSTEM_PROMPT).partial(style_reference=global_style_reference)
    agent = create_react_agent(chat_llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        early_stopping_method="force",
    )

    formatted_history = _format_chat_history(chat_history) if chat_history else ""
    return executor, pdf_tool, formatted_history


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
def _source_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    source = {
        "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
        "filename": chunk["filename"],
        "page": chunk["page"],
        "score": chunk["score"],
        "confidence": chunk.get("confidence", 0),
        "bbox": chunk.get("bbox", ""),
    }
    highlight_rects = _parse_highlight_rects(chunk.get("bbox"))
    if highlight_rects:
        source["highlightRects"] = highlight_rects
    return source

#Genera la etiqueta de cita para un fragmento de texto recuperado.
def _citation_label(source: Dict[str, Any]) -> str:
    return f"[Fuente: {source['filename']}, Página {source['page']}]"

#Asegura que la respuesta generada por el agente incluya citas a las fuentes utilizadas.
def _ensure_answer_has_citations(answer: str, sources: List[Dict[str, Any]]) -> str:
    if not answer or not sources:
        return answer
    if re.search(r"\[Fuente:\s*.+?,\s*P(?:á|a)gina\s+\d+\]", answer, flags=re.IGNORECASE):
        return answer

    seen = []
    for source in sources:
        label = _citation_label(source)
        if label not in seen:
            seen.append(label)
    return f"{answer}\n\nFuentes consultadas: {'; '.join(seen)}"


def _get_pdf_tool_sources(pdf_tool: PDFSearchTool) -> List[Dict[str, Any]]:
    return list(getattr(pdf_tool, "all_sources", None) or getattr(pdf_tool, "last_sources", []))


def _generate_agentic_document_answer(
    question: str,
    user_id: str,
    document_id: Optional[str],
    hf_token: Optional[str],
    top_k: Optional[int],
    chat_history: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    executor, pdf_tool, formatted_history = get_agent_executor(
        user_id, document_id, hf_token, top_k, chat_history
    )
    result = executor.invoke({"input": question, "chat_history": formatted_history})

    raw_answer = result.get("output", "")
    try:
        answer = parse_agent_output(raw_answer)
    except OutputParserError as e:
        logger.warning(f"Rejected malformed LLM output: {e}")
        answer = MALFORMED_OUTPUT_MESSAGE

    sources = [_source_payload(chunk) for chunk in _get_pdf_tool_sources(pdf_tool)]
    answer = _ensure_answer_has_citations(answer, sources)

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


def _retrieve_document_context(
    question: str,
    user_id: str,
    document_id: Optional[str],
    top_k: Optional[int],
) -> tuple[str, List[Dict[str, Any]]]:
    chunks = retrieve(
        query=question,
        user_id=user_id,
        document_id=document_id,
        top_k=top_k,
    )
    sources = [_source_payload(chunk) for chunk in chunks]

    if not chunks:
        return "", sources

    context_parts = [
        (
            f"Fragmento {index} ({chunk['filename']}, Página {chunk['page']}):\n"
            f"{chunk['text']}"
        )
        for index, chunk in enumerate(chunks, 1)
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

    return "\n\n".join(context_parts), sources


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


def _generate_direct_document_answer(
    question: str,
    user_id: str,
    document_id: Optional[str],
    hf_token: Optional[str],
    top_k: Optional[int],
    chat_history: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    context, sources = _retrieve_document_context(question, user_id, document_id, top_k)
    if not context:
        return {
            "answer": "No encontré información suficiente en los documentos cargados para responder esta pregunta.",
            "sources": [],
        }

    from langchain_core.messages import HumanMessage

    style_reference = _build_style_reference(sources)
    chat_llm = get_llm_client(hf_token)
    prompt = _build_direct_rag_prompt(question, context, chat_history, style_reference=style_reference)
    response = chat_llm.invoke([HumanMessage(content=prompt)])
    answer = parse_agent_output(response.content)
    if _is_agent_stop_answer(answer):
        answer = "No pude generar una respuesta final estable. Reformulo con el contexto recuperado: " + (
            "No encontré información suficiente en los documentos cargados para responder esta pregunta."
        )
    answer = _ensure_answer_has_citations(answer, sources)
    return {"answer": answer, "sources": sources}


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
) -> Dict[str, Any]:
    """Agentic generation: retrieve via tools → reason → generate answer."""

    # ── Handle greetings ─────────────────────────────
    if is_greeting(question):
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
            answer = "¡Hola! Soy Document AI Analyst. ¿En qué puedo ayudarte con tus documentos?"
        return {"answer": answer, "sources": []}

    try:
        result = _generate_agentic_document_answer(
            question=question,
            user_id=user_id,
            document_id=document_id,
            hf_token=hf_token,
            top_k=top_k,
            chat_history=chat_history,
        )
        if not _is_agent_stop_answer(result["answer"]):
            return result
        logger.warning("Agent stopped before a final answer; falling back to direct RAG.")
    except Exception as e:
        logger.warning("Agentic RAG failed; falling back to direct RAG: %s", e)

    try:
        return _generate_direct_document_answer(
            question=question,
            user_id=user_id,
            document_id=document_id,
            hf_token=hf_token,
            top_k=top_k,
            chat_history=chat_history,
        )
    except Exception as e:
        logger.error(f"Direct RAG generation error: {e}")
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
) -> Generator[str, None, None]:
    """Streaming Agentic pipeline."""

    # ── Handle greetings ─────────────────────────────
    if is_greeting(question):
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

    try:
        executor, pdf_tool, formatted_history = get_agent_executor(
            user_id, document_id, hf_token, top_k, chat_history
        )

        sources_sent = False
        answer_sent = False

        for step in executor.stream({"input": question, "chat_history": formatted_history}):
            if "actions" in step:
                continue

            if "intermediate_steps" in step:
                tool_sources = _get_pdf_tool_sources(pdf_tool)
                if not sources_sent and tool_sources:
                    sources = [_source_payload(chunk) for chunk in tool_sources]
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                    sources_sent = True
                continue

            if "output" in step:
                tool_sources = _get_pdf_tool_sources(pdf_tool)
                sources = [_source_payload(chunk) for chunk in tool_sources]
                if not sources_sent:
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                    sources_sent = True

                try:
                    clean_answer = parse_agent_output(step["output"])
                except OutputParserError as e:
                    logger.warning(f"Rejected malformed streamed LLM output: {e}")
                    clean_answer = MALFORMED_OUTPUT_MESSAGE

                if _is_agent_stop_answer(clean_answer):
                    logger.warning("Streaming agent stopped before a final answer; falling back to direct RAG.")
                    break

                clean_answer = _ensure_answer_has_citations(clean_answer, sources)
                yield f"data: {json.dumps({'type': 'token', 'data': clean_answer})}\n\n"
                answer_sent = True

        if answer_sent:
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

        style_reference = _build_style_reference(sources)
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

        answer = _ensure_answer_has_citations(answer, sources)
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        if not answer_sent:
            yield f"data: {json.dumps({'type': 'token', 'data': answer})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    except Exception as e:
        logger.warning("Agentic streaming failed; falling back to direct RAG: %s", e)
        try:
            context, sources = _retrieve_document_context(question, user_id, document_id, top_k)
            if not context:
                yield f"data: {json.dumps({'type': 'sources', 'data': []})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'data': 'No encontré información suficiente en los documentos cargados para responder esta pregunta.'})}\n\n"
            else:
                from langchain_core.messages import HumanMessage

                style_reference = _build_style_reference(sources)
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

                answer = _ensure_answer_has_citations(answer, sources)
                yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'data': answer})}\n\n"
        except Exception as fallback_error:
            logger.error(f"Direct RAG streaming fallback error: {fallback_error}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(fallback_error)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return