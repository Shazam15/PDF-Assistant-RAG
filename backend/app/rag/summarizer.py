import logging
from app.config import get_settings
from typing import Any, Dict, List

#from app.rag.agent import get_llm_client

logger = logging.getLogger(__name__)
settings = get_settings()


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
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        chat_llm = ChatOllama(
            model=settings.LLM_MODEL, 
            temperature=0.3,
            num_predict=settings.SUMMARY_MAX_TOKENS
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
