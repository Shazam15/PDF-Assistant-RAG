"""Multilingual reranking with raw, model-relative scores."""

import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder
import torch
from torch import nn

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Reranker Class ─────────────────────────────────────
class Reranker:
    """Reranks documents using a cross-encoder model (BGE reranker)."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the reranker model.

        Args:
            model_name: HuggingFace model ID (defaults to settings.RERANKER_MODEL).
            device: 'cpu', 'cuda', or None (auto-detect).
        """
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device = device
        self._model: Optional[CrossEncoder] = None

    @staticmethod
    def _safe_max_length(model_name: str, configured_max_length: int) -> int:
        """Clamp RERANK_MAX_LENGTH to what the model's position embeddings actually
        support. CrossEncoder's `max_length` only controls tokenizer truncation — it
        is never validated against the model's own `max_position_embeddings`, so a
        too-large value (e.g. the 2048 default, sized for Qwen3-Reranker) silently
        tokenizes sequences the model itself cannot index, crashing deep inside the
        forward pass with `IndexError: index N is out of bounds for dimension 1 with
        size N` instead of a clear error at load time.
        """
        try:
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(model_name)
            max_position_embeddings = getattr(model_config, "max_position_embeddings", None)
            if isinstance(max_position_embeddings, int) and max_position_embeddings > 0:
                # A small safety margin covers architectures (e.g. XLM-RoBERTa) that
                # reserve a couple of leading position ids for padding/offset, so the
                # true usable length is a few tokens below max_position_embeddings.
                safe_limit = max(16, max_position_embeddings - 8)
                if configured_max_length > safe_limit:
                    logger.warning(
                        "RERANK_MAX_LENGTH=%d exceeds what %s supports (max_position_embeddings=%d); "
                        "clamping to %d.",
                        configured_max_length,
                        model_name,
                        max_position_embeddings,
                        safe_limit,
                    )
                    return safe_limit
        except Exception as exc:
            logger.warning("Could not introspect max_position_embeddings for %s: %s", model_name, exc)
        return configured_max_length

    # Lazy-load the model when needed to avoid long startup times
    def _load_model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading reranker: {self.model_name}")
            requested_device = self.device or get_settings().RERANKER_DEVICE
            device = requested_device
            if requested_device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
                logger.warning(
                    "CUDA was requested for the reranker but is unavailable; falling back to CPU."
                )
            kwargs = {
                "max_length": self._safe_max_length(self.model_name, get_settings().RERANK_MAX_LENGTH),
                "device": device,
            }
            if "qwen3-reranker" in self.model_name.lower():
                kwargs.update({
                    "prompts": {
                        "research": (
                            "Given a research question, determine whether the document passage provides direct "
                            "evidence that helps answer the query"
                        )
                    },
                    "default_prompt_name": "research",
                })
            self._model = CrossEncoder(self.model_name, **kwargs)
            logger.info("Reranker loaded successfully")
        return self._model

    # Reranking method that takes a query and a list of documents, and returns them sorted by relevance
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The user query.
            documents: List of document dicts (must contain text_key field).
            top_k: Number of top documents to return after reranking.
            text_key: Key in document dict that holds the text content.

        Returns:
            List of reranked documents (same dicts, but sorted by relevance).
        """
        if not documents:
            return []

        model = self._load_model()

        # Repeat neutral document/section context for the cross-encoder while
        # retaining the literal child text separately for citation checks.
        pairs = []
        for document in documents:
            labels = [document.get("filename"), document.get("section") or document.get("section_title")]
            passage = "\n".join(
                [str(label) for label in labels if label] + [str(document[text_key])]
            )
            pairs.append((query, passage))

        # Request raw logits explicitly. CrossEncoder otherwise applies its own
        # sigmoid for single-label models, which would calibrate the score twice.
        scores = model.predict(pairs, activation_fn=nn.Identity())

        # Raw logits are only comparable inside this query. They are deliberately
        # not converted to probabilities or compared with a global threshold.
        scored = list(zip(scores, documents))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top_k documents
        reranked = [doc for _, doc in scored[:top_k]]

        # Attach rerank_score to each returned document
        for rank, (score, doc) in enumerate(scored, 1):
            if doc in reranked:
                logit = float(score)
                doc["rerank_score"] = logit
                doc["rerank_rank"] = rank
                doc["relevance_score"] = 1.0 / rank

        return reranked


# Singleton instance for global reuse
_reranker_instance: Optional[Reranker] = None

# Function to get the global reranker instance
def get_reranker(model_name: Optional[str] = None) -> Reranker:
    """Get or create the global reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker(model_name=model_name)
    return _reranker_instance


def release_reranker() -> None:
    global _reranker_instance
    if _reranker_instance is not None:
        _reranker_instance._model = None
    _reranker_instance = None
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
