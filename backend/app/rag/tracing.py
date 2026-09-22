"""
Optional tracing helpers for the RAG pipeline: LangSmith (hosted) and Langfuse
(self-hosted). Safe to import even when neither backend is installed or configured.

Both integrations are strictly fail-open: a tracing backend that is unreachable,
misconfigured or not installed must never break a chat request, so every call
below degrades to a no-op instead of raising.
"""
import logging
import os
from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    from langsmith import traceable as _langsmith_traceable
except Exception:  # pragma: no cover - optional dependency safety
    _langsmith_traceable = None


def configure_langsmith() -> bool:
    """Configure LangSmith environment variables when tracing is enabled."""
    if not settings.LANGSMITH_TRACING:
        return False

    if not settings.LANGSMITH_API_KEY:
        logger.warning("LangSmith tracing enabled but LANGSMITH_API_KEY is not set; tracing disabled.")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    return _langsmith_traceable is not None


LANGSMITH_ENABLED = configure_langsmith()


def _sanitize_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    return {key: value for key, value in (metadata or {}).items() if value is not None}


def _build_traceable(name: str, run_type: str, metadata: Optional[dict[str, Any]] = None):
    """Build a LangSmith traceable decorator safely across versions."""
    if _langsmith_traceable is None:
        return None

    sanitized = _sanitize_metadata(metadata)
    try:
        return _langsmith_traceable(
            name=name,
            run_type=run_type,
            metadata=sanitized or None,
        )
    except TypeError:
        return _langsmith_traceable(name=name, run_type=run_type)


def trace_call(
    name: str,
    fn: Callable[..., Any],
    *args: Any,
    run_type: str = "chain",
    metadata: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Execute a callable with LangSmith tracing when available."""
    if not LANGSMITH_ENABLED:
        return fn(*args, **kwargs)

    decorator = _build_traceable(name, run_type, metadata)
    if decorator is None:
        return fn(*args, **kwargs)

    traced_fn = decorator(fn)
    return traced_fn(*args, **kwargs)


def trace_function(
    name: str,
    *,
    run_type: str = "chain",
    metadata_factory: Optional[Callable[..., dict[str, Any]]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator wrapper that becomes a no-op when LangSmith is disabled."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            metadata = metadata_factory(*args, **kwargs) if metadata_factory else None
            return trace_call(
                name,
                fn,
                *args,
                run_type=run_type,
                metadata=metadata,
                **kwargs,
            )

        return wrapped

    return decorator


# ── Langfuse (self-hosted) ───────────────────────────────────────────────────

try:
    from langfuse import get_client as _langfuse_get_client
    from langfuse import propagate_attributes as _langfuse_propagate_attributes
    from langfuse.types import TraceContext as _LangfuseTraceContext
except Exception:  # pragma: no cover - optional dependency safety
    _langfuse_get_client = None
    _langfuse_propagate_attributes = None
    _LangfuseTraceContext = None


def configure_langfuse() -> bool:
    """Publish Langfuse credentials to the environment when tracing is enabled.

    The environment is the configuration channel on purpose: langfuse's
    ``get_client()`` (and therefore the LangChain ``CallbackHandler``, which
    resolves its client that way) reads these variables rather than any instance
    this module could construct and hand over. This function deliberately makes
    no network call, so importing the module never slows application startup —
    connectivity is checked separately by ``verify_langfuse_connection()``.
    """
    if not settings.LANGFUSE_ENABLED:
        # Credentials without the switch is almost always an oversight, and the
        # symptom — traces that never arrive, with nothing in the logs — gives no
        # hint at all. Say it out loud instead of starting up silently inert.
        if settings.LANGFUSE_PUBLIC_KEY or settings.LANGFUSE_SECRET_KEY:
            logger.warning(
                "Langfuse credentials are configured but LANGFUSE_ENABLED is not set to True; "
                "no traces will be produced. Set LANGFUSE_ENABLED=True to turn tracing on."
            )
        return False

    if _langfuse_get_client is None:
        logger.warning("Langfuse tracing enabled but the langfuse package is not installed; tracing disabled.")
        return False

    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.warning(
            "Langfuse tracing enabled but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not set; tracing disabled."
        )
        return False

    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = settings.ENVIRONMENT
    os.environ["LANGFUSE_SAMPLE_RATE"] = str(settings.LANGFUSE_SAMPLE_RATE)
    if settings.LANGFUSE_DEBUG:
        os.environ["LANGFUSE_DEBUG"] = "true"
    return True


LANGFUSE_ENABLED = configure_langfuse()

_langfuse_handler: Any = None
_langfuse_handler_failed = False


def _client() -> Any:
    if not LANGFUSE_ENABLED or _langfuse_get_client is None:
        return None
    try:
        return _langfuse_get_client()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Langfuse client unavailable: %s", exc)
        return None


def disable_langfuse(reason: str) -> None:
    """Turn tracing off for the rest of the process.

    Leaving it nominally enabled against a host that is not there is worse than
    off: every stage would still build spans the exporter then retries and logs,
    burying real application logs under connection errors for a feature that is
    not working anyway. Tracing stays off until the process restarts, so start
    the stack before the backend rather than alongside it.
    """
    global LANGFUSE_ENABLED
    if LANGFUSE_ENABLED:
        LANGFUSE_ENABLED = False
        logger.warning("Langfuse tracing disabled for this process: %s", reason)


def verify_langfuse_connection() -> bool:
    """Check credentials against the configured host. Never raises.

    Kept out of ``configure_langfuse()`` so that import time stays network-free;
    call this once from application startup to surface a misconfigured or
    unreachable self-hosted instance as a single explicit warning instead of a
    stream of background exporter errors.
    """
    client = _client()
    if client is None:
        return False
    try:
        return bool(client.auth_check())
    except Exception as exc:
        logger.warning(
            "Langfuse is enabled but %s is unreachable or rejected the credentials (%s).",
            settings.LANGFUSE_HOST,
            exc,
        )
        return False


def langfuse_handler() -> Any:
    """Return a cached LangChain callback handler, or None when disabled."""
    global _langfuse_handler, _langfuse_handler_failed
    if not LANGFUSE_ENABLED or _langfuse_handler_failed:
        return None
    if _langfuse_handler is None:
        try:
            from langfuse.langchain import CallbackHandler

            _langfuse_handler = CallbackHandler()
        except Exception as exc:
            logger.warning("Langfuse LangChain callback unavailable; LLM calls will not be traced: %s", exc)
            _langfuse_handler_failed = True
            return None
    return _langfuse_handler


def langfuse_callbacks() -> list[Any]:
    """Callbacks list for LangChain/LangGraph calls; empty when tracing is off.

    The handler nests whatever it records under whichever span is active in the
    ambient OpenTelemetry context, so a call made inside ``trace_stage`` lands
    under that stage rather than in a trace of its own.
    """
    handler = langfuse_handler()
    return [handler] if handler is not None else []


def new_trace_id(seed: Optional[str] = None) -> Optional[str]:
    """Mint a trace id up front so every stage of a run can attach to it.

    A long-lived ``with`` block cannot carry the root trace here: the research
    workflow is a generator, and its stages resume in whatever thread the
    consumer happens to use, which is not guaranteed to preserve the contextvar
    the ambient context relies on. Passing this id explicitly to each stage keeps
    the trace tree correct regardless of how the generator is consumed.
    """
    client = _client()
    if client is None:
        return None
    try:
        return client.create_trace_id(seed=seed) if seed else client.create_trace_id()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Langfuse trace id creation skipped: %s", exc)
        return None


class _NullSpan:
    """No-op stand-in so callers never branch on whether tracing is active."""

    def update(self, **kwargs: Any) -> None:
        return None

    def score(self, *args: Any, **kwargs: Any) -> None:
        return None


@contextmanager
def trace_stage(
    name: str,
    *,
    trace_id: Optional[str] = None,
    as_type: str = "span",
    input: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    trace_name: Optional[str] = None,
) -> Iterator[Any]:
    """Record one stage of a run, attaching it to ``trace_id`` when given.

    Yields a span-like object exposing ``update()``. When tracing is disabled or
    anything goes wrong, yields a no-op span so the caller's body always runs.
    """
    client = _client()
    if client is None:
        yield _NullSpan()
        return

    trace_attributes = {
        key: value
        for key, value in (
            ("user_id", user_id),
            ("session_id", session_id),
            ("trace_name", trace_name),
            ("tags", tags),
        )
        if value is not None
    }

    with ExitStack() as stack:
        try:
            if trace_attributes and _langfuse_propagate_attributes is not None:
                stack.enter_context(_langfuse_propagate_attributes(**trace_attributes))
            trace_context = (
                _LangfuseTraceContext(trace_id=trace_id)
                if trace_id and _LangfuseTraceContext is not None
                else None
            )
            span = stack.enter_context(
                client.start_as_current_observation(
                    name=name,
                    as_type=as_type,
                    trace_context=trace_context,
                    input=input,
                    metadata=_sanitize_metadata(metadata) or None,
                )
            )
        except Exception as exc:
            logger.debug("Langfuse stage %s not traced: %s", name, exc)
            span = _NullSpan()
        yield span


def flush_traces() -> None:
    """Flush pending traces. Safe to call when tracing is disabled."""
    client = _client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Langfuse flush skipped: %s", exc)
