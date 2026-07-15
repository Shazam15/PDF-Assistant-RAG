"""
Response caching utility for PDF-Assistant-RAG.

Supports two backends:
- Redis (preferred, for production)
- LRU in-memory cache (fallback for development or when Redis is unavailable)

Cache key is a SHA-256 hash of (document_id, question) to ensure keys are
short, stable, and unique across all question/document combinations.
"""

import hashlib
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all values come from environment variables
# ---------------------------------------------------------------------------

CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # seconds; default 1 hour
REDIS_URL: Optional[str] = os.getenv("REDIS_URL", None)
LRU_MAX_SIZE: int = int(os.getenv("CACHE_LRU_MAX_SIZE", "128"))
ROUTER_CACHE_VERSION = "adaptive-v1"

# ---------------------------------------------------------------------------
# Redis client (lazy init — only created when REDIS_URL is set)
# ---------------------------------------------------------------------------

_redis_client = None
_redis_available = False


def _get_redis():
    """
    Lazily initialise the Redis client.
    Returns the client if Redis is reachable, otherwise returns None.
    After one failure, stops retrying for the process lifetime.
    """
    global _redis_client, _redis_available

    if _redis_client is not None:
        return _redis_client if _redis_available else None

    if not REDIS_URL:
        logger.info("REDIS_URL not set — using LRU in-memory cache.")
        _redis_available = False
        return None

    try:
        import redis  # soft import — redis package is optional

        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        client.ping()
        _redis_client = client
        _redis_available = True
        logger.info("Redis cache connected at %s", REDIS_URL)
        return _redis_client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis unavailable (%s) — falling back to LRU cache.", exc)
        _redis_available = False
        return None


# ---------------------------------------------------------------------------
# LRU in-memory fallback
# ---------------------------------------------------------------------------

_lru_store: dict = {}
_lru_order: list = []


def _cached_answer_text(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("answer", ""))
    return str(payload or "")


def _is_invalid_cached_answer(answer: Any) -> bool:
    text = _cached_answer_text(answer).lower()
    invalid_phrases = [
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
    return any(phrase in text for phrase in invalid_phrases)


def _lru_get(key: str) -> Optional[str]:
    return _lru_store.get(key)


def _lru_set(key: str, value: str) -> None:
    if key in _lru_store:
        _lru_order.remove(key)
    elif len(_lru_store) >= LRU_MAX_SIZE:
        oldest = _lru_order.pop(0)
        del _lru_store[oldest]
    _lru_store[key] = value
    _lru_order.append(key)


def _lru_delete(key: str) -> None:
    if key in _lru_store:
        del _lru_store[key]
        _lru_order.remove(key)


# ---------------------------------------------------------------------------
# Public API — these are the only functions chat.py needs
# ---------------------------------------------------------------------------


def make_cache_key(
    document_id: str,
    question: str,
    user_id: str = "",
    top_k: Optional[int] = None,
    routing_mode: str = "auto",
) -> str:
    """
    Generate a stable, short cache key from document_id + question.

    SHA-256 gives us a 64-char hex string that is:
    - Always the same length regardless of question length
    - Unique per (document_id, question) pair
    - Safe for Redis keys and dict keys
    """
    raw = (
        f"{ROUTER_CACHE_VERSION}:{routing_mode}:{user_id}:{document_id}:"
        f"{top_k or ''}:{question.strip().lower()}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_response(
    document_id: str,
    question: str,
    user_id: str = "",
    top_k: Optional[int] = None,
    routing_mode: str = "auto",
) -> Optional[dict[str, Any]]:
    """
    Look up a cached answer for a (document_id, question) pair.
    Returns the answer string on hit, None on miss.
    """
    key = make_cache_key(
        document_id, question, user_id=user_id, top_k=top_k, routing_mode=routing_mode
    )
    r = _get_redis()

    if r is not None:
        try:
            value = r.get(key)
            if value:
                logger.debug("Cache HIT (Redis) for key %s", key[:12])
                payload = json.loads(value)
                if _is_invalid_cached_answer(payload):
                    r.delete(key)
                    return None
                if isinstance(payload, dict):
                    return payload
                return {"answer": str(payload), "sources": []}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis GET failed (%s) — checking LRU.", exc)

    value = _lru_get(key)
    if value:
        logger.debug("Cache HIT (LRU) for key %s", key[:12])
        payload = json.loads(value)
        if _is_invalid_cached_answer(payload):
            _lru_delete(key)
            return None
        if isinstance(payload, dict):
            return payload
        return {"answer": str(payload), "sources": []}

    logger.debug("Cache MISS for key %s", key[:12])
    return None


def set_cached_response(
    document_id: str,
    question: str,
    answer: str,
    sources: Optional[list[dict[str, Any]]] = None,
    user_id: str = "",
    top_k: Optional[int] = None,
    routing_mode: str = "auto",
) -> None:
    """
    Store an answer. Tries Redis first; falls back to LRU.
    TTL is controlled by the CACHE_TTL environment variable.
    """
    if _is_invalid_cached_answer(answer):
        logger.debug("Skipping invalid cache answer for question: %s", question[:40])
        return

    key = make_cache_key(
        document_id, question, user_id=user_id, top_k=top_k, routing_mode=routing_mode
    )
    serialised = json.dumps({"answer": answer, "sources": sources or []})
    r = _get_redis()

    if r is not None:
        try:
            r.setex(key, CACHE_TTL, serialised)
            logger.debug("Cache SET (Redis) key %s TTL %ds", key[:12], CACHE_TTL)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis SET failed (%s) — storing in LRU.", exc)

    _lru_set(key, serialised)
    logger.debug("Cache SET (LRU) key %s", key[:12])


def invalidate_cache(
    document_id: str,
    question: str,
    user_id: str = "",
    top_k: Optional[int] = None,
    routing_mode: str = "auto",
) -> None:
    """Remove one cache entry — useful when a document is re-indexed."""
    key = make_cache_key(
        document_id, question, user_id=user_id, top_k=top_k, routing_mode=routing_mode
    )
    r = _get_redis()
    if r is not None:
        try:
            r.delete(key)
        except Exception:  # noqa: BLE001
            pass
    _lru_delete(key)
