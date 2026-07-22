import logging
from app.config import get_settings
from app.rag.llm_client import create_chat_ollama
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

#from app.rag.agent import get_llm_client

logger = logging.getLogger(__name__)
settings = get_settings()


class EvidenceDraft(BaseModel):
    evidence_kind: str
    claim: str
    exact_quote: str
    chunk_index: int


class DocumentMemoryDraft(BaseModel):
    summary: str
    methodology: str = ""
    findings: str = ""
    limitations: str = ""
    evidence: List[EvidenceDraft] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def limit_evidence(cls, values: List[EvidenceDraft]) -> List[EvidenceDraft]:
        return values[:24]


def _extractive_summary(text: str, max_sentences: int) -> str | None:
    sentences = []
    for sentence in text.replace("\n", " ").split("."):
        cleaned = " ".join(sentence.split())
        if cleaned:
            sentences.append(cleaned)
        if len(sentences) >= max_sentences:
            break
    if not sentences:
        return None
    return ". ".join(sentences) + "."


def generate_document_summary_from_chunks(
    chunks: List[Dict[str, Any]],
    max_sentences: int = 3,
) -> str | None:
    """Generar un resumen corto del documento usando chunk extraidos"""
    if not chunks:
        return None

    chunk_texts = []
    for chunk in chunks[:10]:
        text = chunk.get("text")
        if isinstance(text, str) and text.strip():
            chunk_texts.append(text.strip())

    text_to_summarise = " ".join(chunk_texts).strip()
    if not text_to_summarise:
        return None

    prompt = f"Resume el siguiente texto en {max_sentences} oraciones:\n\n{text_to_summarise}"

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        chat_llm = create_chat_ollama(
            temperature=0.3,
            reasoning=False if settings.LLM_DISABLE_THINKING else None,
            num_predict=settings.SUMMARY_MAX_TOKENS,
        )
        response = chat_llm.invoke([
            SystemMessage(content="Eres un asistente útil y conciso que resume documentos de manera clara y precisa."),
            HumanMessage(content=prompt),
        ])
        summary = response.content.strip()
        return summary or _extractive_summary(text_to_summarise, max_sentences)
    except Exception as e:
        logger.warning("LLM summary generation failed; using extractive summary: %s", e)
        return _extractive_summary(text_to_summarise, max_sentences)


def build_document_memory(
    chunks: List[Dict[str, Any]],
    use_llm: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """Create a hierarchical profile and only retain evidence with literal provenance."""
    sections: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        section_id = str(chunk.get("section_id") or "S0")
        section = sections.setdefault(section_id, {
            "id": section_id,
            "title": chunk.get("section_title") or chunk.get("section") or "Document",
            "page_start": int(chunk.get("page_start") or chunk.get("page") or 1),
            "page_end": int(chunk.get("page_end") or chunk.get("page") or 1),
            "texts": [],
        })
        section["page_start"] = min(section["page_start"], int(chunk.get("page") or 1))
        section["page_end"] = max(section["page_end"], int(chunk.get("page") or 1))
        section["texts"].append(str(chunk.get("text") or ""))

    section_payloads = []
    for section in sections.values():
        section_text = "\n\n".join(section.pop("texts")).strip()
        section_payloads.append({
            **section,
            "text": section_text,
            "summary": _extractive_summary(section_text, 3) or "",
        })

    representative_chunks = []
    total_chars = 0
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text or total_chars >= 18000:
            continue
        excerpt = text[: min(len(text), 18000 - total_chars)]
        representative_chunks.append(f"[chunk {chunk.get('chunk_index', 0)}]\n{excerpt}")
        total_chars += len(excerpt)
    corpus_text = "\n\n".join(representative_chunks)
    fallback_text = " ".join(str(chunk.get("text") or "") for chunk in chunks[:20])
    fallback_summary = _extractive_summary(fallback_text, 5) or ""
    draft = DocumentMemoryDraft(summary=fallback_summary)

    if corpus_text and use_llm:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = create_chat_ollama(
                temperature=0,
                reasoning=False if settings.LLM_DISABLE_THINKING else None,
                num_predict=min(1200, settings.LLM_MAX_NEW_TOKENS),
            )
            structured = llm.with_structured_output(DocumentMemoryDraft, method="json_mode")
            no_think = "/no_think\n" if settings.LLM_DISABLE_THINKING else ""
            if progress_callback:
                progress_callback("summarizing", 0, 1)
            draft = structured.invoke([
                SystemMessage(content=(
                    no_think
                    +
                    "Build a neutral document memory from the supplied chunks. Summarize scope, methodology, "
                    "findings, and limitations only when visible. Evidence kinds may be method, result, limitation, "
                    "definition, or context. Every exact_quote must be copied verbatim from its referenced chunk. "
                    "Do not add outside knowledge. Return JSON matching the schema."
                )),
                HumanMessage(content=corpus_text),
            ])
            if progress_callback:
                progress_callback("summarizing", 1, 1)
        except Exception as exc:
            logger.warning("Structured document memory generation failed: %s", exc)

    chunks_by_index = {int(chunk.get("chunk_index") or 0): chunk for chunk in chunks}
    verified_evidence = []
    for evidence in draft.evidence:
        chunk = chunks_by_index.get(evidence.chunk_index)
        if not chunk or evidence.exact_quote not in str(chunk.get("text") or ""):
            continue
        verified_evidence.append(evidence.model_dump())

    if progress_callback:
        progress_callback("building_memory", 1, 1)

    return {
        "summary": draft.summary or fallback_summary,
        "methodology": draft.methodology,
        "findings": draft.findings,
        "limitations": draft.limitations,
        "sections": section_payloads,
        "evidence": verified_evidence,
    }


def generate_document_summary(filePath: str, max_sentences: int = 3) -> str | None:
    """
    Extraer el texto de los primeros fragmentos del documento y pedir al LLM que resuma.
    Devuelve un resumen corto como cadena, o None si falla.


    Args:
        filePath (str): Path al archivo del documento.
        max_sentences (int): Maximo de oraciones que puede tener el resumen.
    
    Returns:
        str | None: Texto del resumen o None si falla la generación del resumen.
    
    Nota:
        - Esta función está diseñada para ser llamada después de que un documento se haya subido y procesado.
        - Usa los primeros fragmentos del documento para generar un resumen, que luego se almacena en la base de datos.      
    """
    from app.rag.chunker import chunk_document

    try:
        chunks = chunk_document(filePath)

        if not chunks:
            logger.warning(f"No chunks extracted from {filePath}, cannot summarise.")
            return None

        return generate_document_summary_from_chunks(chunks, max_sentences=max_sentences)

    except Exception as e:
        logger.error(f"Summary generation failed for {filePath}: {e}")
        return None
