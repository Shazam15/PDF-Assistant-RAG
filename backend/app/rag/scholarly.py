"""Resolve scholarly identifiers (DOI) for web search results.

A citation in ATLAS is only worth as much as its auditability, so the rule here
is that a wrong DOI is worse than no DOI: every identifier this module attaches
must either be read literally out of the source, or confirmed against the source
title before it is accepted. When in doubt the source simply carries no DOI.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Crossref's recommended pattern. The suffix is greedy by design, so callers must
# strip the sentence punctuation that follows a DOI quoted inside prose.
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9<>+\[\]]+", re.IGNORECASE)

# Trailing characters that belong to the surrounding sentence, not to the DOI.
_DOI_TRAILING_NOISE = ".,;:)]}'\"<>"

# A DOI suffix may legitimately contain "/", so the pattern above cannot tell
# where the identifier ends and the rest of a URL path begins. These are the
# trailing pieces publishers append to a DOI-based URL — Frontiers' ".../full",
# Springer's ".../<doi>.pdf" — which would otherwise be swallowed into the DOI
# and produce an identifier that resolves nowhere.
_DOI_URL_TRAILING_SEGMENTS = frozenset({
    "full", "fulltext", "full-text", "abstract", "pdf", "epdf", "html", "htm",
    "meta", "figures", "references", "citations", "supplemental", "supplementary",
    "download", "print", "summary", "toc",
})
_DOI_FILE_SUFFIX = re.compile(r"\.(pdf|html?|xml|json|epub|txt)$", re.IGNORECASE)

_ARXIV_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", re.IGNORECASE)

# Hosts worth paying a network round trip for when no DOI is present in the page
# itself. Everything else is left alone rather than guessed at.
_SCHOLARLY_HOSTS = (
    "doi.org",
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "sciencedirect.com",
    "springer.com",
    "link.springer.com",
    "nature.com",
    "onlinelibrary.wiley.com",
    "tandfonline.com",
    "sagepub.com",
    "acs.org",
    "ieee.org",
    "ieeexplore.ieee.org",
    "jstor.org",
    "mdpi.com",
    "frontiersin.org",
    "plos.org",
    "journals.plos.org",
    "biomedcentral.com",
    "cambridge.org",
    "oup.com",
    "academic.oup.com",
    "scielo.org",
    "redalyc.org",
    "dialnet.unirioja.es",
    "researchgate.net",
    "semanticscholar.org",
    "sae.org",
)


def normalize_doi(raw: str) -> Optional[str]:
    """Return a bare, lowercased DOI, or None when the string holds no valid one."""
    if not raw:
        return None
    candidate = unquote(str(raw)).strip()
    match = _DOI_PATTERN.search(candidate)
    if not match:
        return None
    doi = match.group(0).rstrip(_DOI_TRAILING_NOISE)
    doi = _DOI_FILE_SUFFIX.sub("", doi)

    # Drop publisher path segments the greedy suffix pulled in. Done in a loop
    # because a URL can stack them, as in ".../<doi>/full/pdf".
    while True:
        head, separator, tail = doi.rpartition("/")
        # Keep the DOI prefix intact: the first "/" separates 10.xxxx from the
        # suffix, and removing that would leave an unusable identifier.
        if not separator or "/" not in head or tail.lower() not in _DOI_URL_TRAILING_SEGMENTS:
            break
        doi = head

    doi = _DOI_FILE_SUFFIX.sub("", doi.rstrip(_DOI_TRAILING_NOISE))

    # A DOI suffix cannot be empty, and a lone prefix is not a usable identifier.
    if "/" not in doi or doi.split("/", 1)[1] == "":
        return None
    return doi.lower()


def doi_url(doi: str) -> str:
    """Canonical resolver URL for a DOI."""
    return f"https://doi.org/{doi}"


def _is_scholarly_host(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return any(host == known or host.endswith(f".{known}") for known in _SCHOLARLY_HOSTS)


def _doi_from_arxiv(url: str) -> Optional[str]:
    """arXiv mints a DOI for every submission, derivable from the identifier."""
    match = _ARXIV_PATTERN.search(url or "")
    if not match:
        return None
    arxiv_id = match.group(1).split("v")[0]
    return f"10.48550/arxiv.{arxiv_id}"


def _normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title or "").lower()).strip()


def _titles_match(left: str, right: str) -> bool:
    """Compare titles in a way that respects word order.

    A character-similarity ratio is not usable here: "Attention Is All You Need"
    and "Is Attention All You Need?" share nearly every character but are
    different papers asking opposite questions, and a fuzzy ratio scores them as
    a match. Requiring equality — or containment, which absorbs the site suffix
    search engines append, as in "... - arXiv" — keeps word order meaningful.
    """
    left_norm, right_norm = _normalized_title(left), _normalized_title(right)
    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm or left_norm in right_norm or right_norm in left_norm


def lookup_doi_by_title(title: str) -> Optional[str]:
    """Ask Crossref for a DOI, accepting it only if the title really matches.

    Crossref always returns its best guess, so an unchecked result would happily
    attach a confident-looking DOI for a different paper. The returned title is
    compared against the one we searched for and anything below the configured
    similarity threshold is discarded.

    Never raises: a network failure means no DOI, never a failed search.
    """
    if not settings.DOI_CROSSREF_LOOKUP or not str(title or "").strip():
        return None

    try:
        import httpx

        response = httpx.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": title, "rows": 1, "select": "DOI,title"},
            headers={"User-Agent": f"{settings.APP_NAME}/1.0 (academic RAG assistant)"},
            timeout=settings.DOI_CROSSREF_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        response.raise_for_status()
        items = (response.json().get("message") or {}).get("items") or []
    except Exception as exc:
        logger.debug("Crossref DOI lookup skipped for %r: %s", str(title)[:80], exc)
        return None

    if not items:
        return None

    candidate = items[0]
    candidate_titles = candidate.get("title") or []
    candidate_title = candidate_titles[0] if candidate_titles else ""
    if not _titles_match(title, candidate_title):
        logger.debug(
            "Crossref returned %r for %r; below the title-match threshold, discarding.",
            str(candidate_title)[:80],
            str(title)[:80],
        )
        return None

    return normalize_doi(candidate.get("DOI") or "")


def resolve_source_doi(source: Dict[str, Any]) -> Optional[str]:
    """Find the DOI for one web result, cheapest and most reliable path first."""
    url = str(source.get("url") or "")

    # 1. Stated literally in the URL (doi.org links, and publishers that put the
    #    DOI in the path). Nothing to verify: it is the identifier itself.
    doi = normalize_doi(url)
    if doi:
        return doi

    # 2. Derivable from an arXiv identifier without a network call.
    doi = _doi_from_arxiv(url)
    if doi:
        return doi

    # 3. Quoted in the page text the search engine returned.
    for field in ("snippet", "text", "title"):
        doi = normalize_doi(str(source.get(field) or ""))
        if doi:
            return doi

    # 4. Only now pay for a lookup, and only for hosts where one is plausible.
    if _is_scholarly_host(url):
        return lookup_doi_by_title(str(source.get("title") or ""))

    return None


def enrich_sources_with_doi(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach `doi` and a canonical `doi_url` to web sources, in place.

    Sources that already carry a DOI are left untouched, and a source whose DOI
    cannot be established keeps no DOI field at all rather than an empty one.
    """
    for source in sources:
        if source.get("doi"):
            continue
        try:
            doi = resolve_source_doi(source)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("DOI resolution skipped for a source: %s", exc)
            continue
        if doi:
            source["doi"] = doi
            source["doi_url"] = doi_url(doi)
    return sources
