"""DOI resolution for web citations.

The central contract under test is that ATLAS never attaches an identifier it
could not establish: a wrong DOI would look authoritative while pointing at a
different paper, which is worse for an auditable-citations product than no DOI.
"""
import pytest

from app.rag import scholarly
from app.rag.scholarly import (
    enrich_sources_with_doi,
    lookup_doi_by_title,
    normalize_doi,
    resolve_source_doi,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _crossref_payload(doi, title):
    return {"message": {"items": [{"DOI": doi, "title": [title]}]}}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://doi.org/10.1038/s41586-021-03819-3", "10.1038/s41586-021-03819-3"),
        ("10.1016/J.ENCONMAN.2019.01.012", "10.1016/j.enconman.2019.01.012"),
        ("see doi:10.1000/xyz123.", "10.1000/xyz123"),
        ("(10.1234/abc-def_9)", "10.1234/abc-def_9"),
        ("https://example.com/no-doi-here", None),
        ("10.1234", None),
        ("10.1234/", None),
        ("", None),
        # Publisher URL suffixes that the greedy DOI suffix would swallow. Each
        # of these was produced by a real web search and resolves nowhere.
        (
            "https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2025.1549160/full",
            "10.3389/fmicb.2025.1549160",
        ),
        (
            "https://link.springer.com/content/pdf/10.1007/s13369-022-06947-7.pdf",
            "10.1007/s13369-022-06947-7",
        ),
        ("https://example.org/10.1093/nar/gkaa1100/full/pdf", "10.1093/nar/gkaa1100"),
        # The prefix must survive: stripping back past the first "/" would leave
        # "10.3389", which is not an identifier.
        ("10.3389/full", "10.3389/full"),
    ],
)
def test_normalize_doi_reads_identifiers_and_rejects_malformed_ones(raw, expected):
    assert normalize_doi(raw) == expected


def test_doi_is_read_from_a_publisher_url_without_any_network_call(monkeypatch):
    monkeypatch.setattr(
        "httpx.get", lambda *a, **k: pytest.fail("no lookup should happen for a stated DOI")
    )
    source = {"url": "https://www.sciencedirect.com/science/article/pii/10.1016/j.apenergy.2020.114574"}

    assert resolve_source_doi(source) == "10.1016/j.apenergy.2020.114574"


def test_arxiv_doi_is_derived_from_the_identifier(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: pytest.fail("arXiv needs no lookup"))

    assert resolve_source_doi({"url": "https://arxiv.org/abs/1706.03762v5"}) == "10.48550/arxiv.1706.03762"


def test_doi_quoted_in_the_snippet_is_used(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: pytest.fail("snippet DOI needs no lookup"))
    source = {"url": "https://example.org/article", "snippet": "Published as doi:10.1103/PhysRevLett.116.061102."}

    assert resolve_source_doi(source) == "10.1103/physrevlett.116.061102"


def test_a_non_scholarly_host_without_a_doi_is_never_looked_up(monkeypatch):
    """Latency is only worth paying where a DOI is plausible."""
    monkeypatch.setattr("httpx.get", lambda *a, **k: pytest.fail("no lookup for a general web page"))

    assert resolve_source_doi({"url": "https://someblog.example/post", "title": "A blog post"}) is None


def test_crossref_doi_is_accepted_when_the_title_matches(monkeypatch):
    monkeypatch.setattr(
        "httpx.get",
        lambda *a, **k: _FakeResponse(
            _crossref_payload("10.1038/nature12373", "Attention Is All You Need")
        ),
    )

    assert lookup_doi_by_title("Attention is all you need") == "10.1038/nature12373"


def test_crossref_doi_is_discarded_when_it_answers_with_a_different_paper(monkeypatch):
    """Crossref always returns its best guess; an unchecked guess is a wrong citation."""
    monkeypatch.setattr(
        "httpx.get",
        lambda *a, **k: _FakeResponse(
            _crossref_payload("10.9999/unrelated", "Gut microbiota composition in adults")
        ),
    )

    assert lookup_doi_by_title("Charge motion requirements for a class-leading GTDI engine") is None


def test_a_reordered_title_is_not_treated_as_the_same_paper(monkeypatch):
    """Regression: Crossref's top hit for "Attention Is All You Need" is the
    distinct paper "Is Attention All You Need?". The two share almost every
    character, so a fuzzy similarity ratio accepted it and attached a book
    chapter's DOI to the transformer paper."""
    monkeypatch.setattr(
        "httpx.get",
        lambda *a, **k: _FakeResponse(
            _crossref_payload("10.1007/978-3-031-84300-6_13", "Is Attention All You Need?")
        ),
    )

    assert lookup_doi_by_title("Attention Is All You Need") is None


def test_a_search_engine_site_suffix_still_matches(monkeypatch):
    """Search results routinely append the site name to the title."""
    monkeypatch.setattr(
        "httpx.get",
        lambda *a, **k: _FakeResponse(
            _crossref_payload("10.1007/978-3-211-47113-5", "Charging the Internal Combustion Engine")
        ),
    )

    assert (
        lookup_doi_by_title("Charging the Internal Combustion Engine - SpringerLink")
        == "10.1007/978-3-211-47113-5"
    )


def test_crossref_failure_never_breaks_resolution(monkeypatch):
    def explode(*_args, **_kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr("httpx.get", explode)

    assert lookup_doi_by_title("Any title at all") is None


def test_lookup_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(scholarly.settings, "DOI_CROSSREF_LOOKUP", False)
    monkeypatch.setattr("httpx.get", lambda *a, **k: pytest.fail("lookup is disabled"))

    assert lookup_doi_by_title("Attention is all you need") is None


def test_enrichment_adds_doi_and_url_only_where_one_was_established(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: _FakeResponse({"message": {"items": []}}))
    sources = [
        {"url": "https://doi.org/10.1038/s41586-021-03819-3", "title": "A paper"},
        {"url": "https://someblog.example/post", "title": "A blog post"},
    ]

    enriched = enrich_sources_with_doi(sources)

    assert enriched[0]["doi"] == "10.1038/s41586-021-03819-3"
    assert enriched[0]["doi_url"] == "https://doi.org/10.1038/s41586-021-03819-3"
    assert "doi" not in enriched[1]
    assert "doi_url" not in enriched[1]


def test_existing_doi_is_never_overwritten(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: pytest.fail("already resolved"))
    sources = [{"url": "https://doi.org/10.1/other", "doi": "10.1234/already-known"}]

    assert enrich_sources_with_doi(sources)[0]["doi"] == "10.1234/already-known"
