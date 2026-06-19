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
    """Generate a short document summary from already extracted chunks."""
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

    prompt = f"Summarise the following text in {max_sentences} sentences:\n\n{text_to_summarise}"

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        chat_llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.3)
        response = chat_llm.invoke([
            SystemMessage(content="You are a helpful assistant that summarizes documents."),
            HumanMessage(content=prompt),
        ])
        summary = response.content.strip()
        return summary or _extractive_summary(text_to_summarise, max_sentences)
    except Exception as e:
        logger.warning("LLM summary generation failed; using extractive summary: %s", e)
        return _extractive_summary(text_to_summarise, max_sentences)


def generate_document_summary(filePath: str, max_sentences: int = 3) -> str | None:
    """
    Extract text from the first few chunks of the document and ask LLM to summarise.
    Returns a short summary string, or None on failure.

    Args:
        filePath (str): Path to the document file.
        max_sentences (int): Maximum number of sentences in the summary.
    
    Returns:
        str | None: Summary text or None if summarisation fails.
    
    Note:
        - This function is designed to be called after a document is uploaded and processed.
        - It uses the first few chunks of the document to generate a summary, which is then stored in the database.        
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
