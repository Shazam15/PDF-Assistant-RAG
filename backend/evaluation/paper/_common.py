"""Shared helpers for the paper evaluation scripts.

Every script calls ``bootstrap()`` before importing ``app.*``: the backend reads
``.env`` and its ``./data`` paths relative to the working directory, so the
scripts must run with ``backend/`` as cwd regardless of where they are invoked.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parents[1]
REPO_DIR = BACKEND_DIR.parent
RESULTS_DIR = HERE / "results"
DEFAULT_DATASET = HERE / "questions.jsonl"


def bootstrap(verbose: bool = False) -> None:
    os.chdir(BACKEND_DIR)
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def results_dir(label: str) -> Path:
    path = RESULTS_DIR / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "backend/app"],
            cwd=REPO_DIR, text=True, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


# ── Users ────────────────────────────────────────────────────────────────


def resolve_user_id(user_email: Optional[str], user_id: Optional[str]) -> str:
    """Return the owner id of the indexed corpus (requires bootstrap())."""
    if user_id:
        return str(user_id)
    from app.database import SessionLocal
    from app.models import User

    db = SessionLocal()
    try:
        if user_email:
            user = db.query(User).filter(User.email == user_email).first()
            if user is None:
                raise SystemExit(f"No existe un usuario con email {user_email!r}")
            return str(user.id)
        users = db.query(User).all()
        if len(users) == 1:
            return str(users[0].id)
        emails = ", ".join(sorted(u.email for u in users))
        raise SystemExit(f"Hay {len(users)} usuarios; indica --user-email ({emails})")
    finally:
        db.close()


def add_user_args(parser) -> None:
    parser.add_argument("--user-email", help="Email del usuario dueño del corpus indexado.")
    parser.add_argument("--user-id", help="UUID del usuario (alternativa a --user-email).")


# ── JSONL ────────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON inválido ({exc})") from exc


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_questions(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    questions = []
    seen = set()
    for row in read_jsonl(path):
        missing = {"id", "question", "reference"} - set(row)
        if missing:
            raise SystemExit(f"{path}: la pregunta {row.get('id')!r} no tiene {sorted(missing)}")
        if row["id"] in seen:
            raise SystemExit(f"{path}: id duplicado {row['id']!r}")
        seen.add(row["id"])
        row.setdefault("gold", [])
        row.setdefault("tags", [])
        questions.append(row)
        if limit and len(questions) >= limit:
            break
    if not questions:
        raise SystemExit(f"{path} no contiene preguntas")
    return questions


# ── Gold evidence matching ───────────────────────────────────────────────
#
# A gold "evidence unit" is {"filename", "page"?, "quote"?, "grade"?, "group"?}.
#   page mode : a chunk matches when it comes from that file and its page span
#               contains the page (robust across machines/extraction modes).
#   quote mode: a chunk matches when its text contains the quote (chunk-precise,
#               only valid when both machines produced identical chunks).
#   auto      : quote when the unit has one, otherwise page.
# A unit with neither page nor quote matches any chunk of the file.


def normalize_text(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _page_span(chunk: Dict[str, Any]) -> tuple[int, int]:
    def as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    page = as_int(chunk.get("page"), 0)
    start = as_int(chunk.get("page_start"), page)
    end = as_int(chunk.get("page_end"), page)
    return min(start, end, page or start), max(start, end, page)


def unit_matches(chunk: Dict[str, Any], unit: Dict[str, Any], mode: str = "page") -> bool:
    if normalize_text(chunk.get("filename")) != normalize_text(unit.get("filename")):
        return False
    quote = unit.get("quote")
    if quote and mode in {"quote", "auto"}:
        return normalize_text(quote) in normalize_text(chunk.get("text"))
    if unit.get("page") not in (None, ""):
        start, end = _page_span(chunk)
        return start <= int(unit["page"]) <= end
    return True


def credited_ids(chunks: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any]], mode: str) -> List[str]:
    """Map a ranked chunk list to unit ids, crediting each unit once (first hit).

    The result can be fed to ``app.rag.benchmark`` so Recall@k / nDCG@k are the
    repository's own implementation, computed over evidence units.
    """
    credited: set[int] = set()
    out: List[str] = []
    for rank, chunk in enumerate(chunks):
        hit = next(
            (index for index, unit in enumerate(units)
             if index not in credited and unit_matches(chunk, unit, mode)),
            None,
        )
        if hit is None:
            out.append(f"__nonrelevant_{rank}")
        else:
            credited.add(hit)
            out.append(f"u{hit}")
    return out


def retrieval_metrics(chunks: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any]], mode: str) -> Dict[str, Optional[float]]:
    if not units:
        return {"recall_at_20": None, "ndcg_at_10": None}
    from app.rag.benchmark import BenchmarkCase, BenchmarkResult, ndcg_at_k, recall_at_k

    case = BenchmarkCase(
        case_id="q",
        query="",
        relevant_chunk_ids={f"u{index}" for index in range(len(units))},
        graded_relevance={f"u{index}": float(unit.get("grade") or 1) for index, unit in enumerate(units)},
    )
    result = BenchmarkResult(
        case_id="q",
        retrieved_chunk_ids=credited_ids(chunks, units, mode),
        citation_checks=[],
        claim_support_checks=[],
        supported_evidence_questions=0,
        latency_seconds=0.0,
    )
    return {"recall_at_20": recall_at_k(case, result, 20), "ndcg_at_10": ndcg_at_k(case, result, 10)}


def evidence_groups(units: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group gold units into required evidence items (default: one per file)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for unit in units:
        key = str(unit.get("group") or normalize_text(unit.get("filename")))
        groups.setdefault(key, []).append(unit)
    return groups


# ── Small stats helpers ──────────────────────────────────────────────────


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)


def median_iqr(values: Sequence[Optional[float]]) -> tuple[Optional[float], Optional[float], Optional[float], int]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None, None, 0
    return percentile(clean, 50), percentile(clean, 25), percentile(clean, 75), len(clean)


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def fmt_median_iqr(values: Sequence[Optional[float]], digits: int = 3) -> str:
    median, q1, q3, n = median_iqr(values)
    if median is None:
        return "—"
    return f"{median:.{digits}f} [{q1:.{digits}f}–{q3:.{digits}f}]"
