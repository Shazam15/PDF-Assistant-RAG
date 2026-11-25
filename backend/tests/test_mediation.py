from unittest.mock import MagicMock

import pytest

from app.rag import agent as agent_module
from app.rag import mediation
from app.rag.mediation import _MediationItem, mediate_claims, normalized_numbers, numbers_supported

SOURCES = [
    {
        "source_id": "D1",
        "filename": "dengue.pdf",
        "text": "El boletín reporta 12.5% de casos graves en la provincia de Panamá Oeste durante 2023.",
    },
    {
        "source_id": "D2",
        "filename": "canal.pdf",
        "text": "The Panama Canal expansion opened in June 2016 with a third set of locks.",
    },
]


def _proposer(*items):
    calls = []

    def propose(claims, excerpts):
        calls.append((claims, excerpts))
        return [_MediationItem(**item) for item in items]

    propose.calls = calls
    return propose


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(mediation.settings, "CLAIM_MEDIATION_MAX_CLAIMS", 8)
    monkeypatch.setattr(mediation.settings, "CLAIM_MEDIATION_NLI", False)


def test_number_normalization_accepts_reformatted_figures():
    assert numbers_supported("Hubo 12,5 % de casos graves", "reporta 12.5% de casos")
    assert numbers_supported("unos 1 234 casos", "se registraron 1,234 casos")
    assert numbers_supported("unos 1234 casos", "se registraron 1.234 casos")
    assert not numbers_supported("Hubo 13 % de casos graves", "reporta 12.5% de casos")
    assert normalized_numbers("texto [D12] sin cifras") == []


def test_supported_claim_gets_verified_citation():
    answer = "El boletín indica que el 12,5 % de los casos fueron graves en Panamá Oeste durante 2023."
    proposer = _proposer({
        "claim": 1, "source_id": "D1", "verdict": "supported", "restated_claim": "",
        "quote": "reporta 12.5% de casos graves en la provincia de Panamá Oeste durante 2023",
    })

    outcome = mediate_claims(answer, SOURCES, proposer=proposer)

    assert outcome.answer.endswith("durante 2023 [D1].")
    assert outcome.stats["accepted_supported"] == 1
    assert "[D1]" in proposer.calls[0][1]


def test_invented_quote_is_rejected_and_pruned():
    answer = (
        "Introducción breve.\n"
        "El boletín indica que el 12,5 % de los casos fueron graves en Panamá Oeste durante 2023."
    )
    proposer = _proposer({
        "claim": 1, "source_id": "D1", "verdict": "supported", "restated_claim": "",
        "quote": "el 12.5% de los casos fueron graves según el ministerio de salud",
    })

    outcome = mediate_claims(answer, SOURCES, proposer=proposer)

    assert outcome.answer == "Introducción breve."
    assert outcome.stats["rejected"] == {"quote_not_in_source": 1}


def test_figure_missing_from_quote_is_rejected():
    answer = "La ampliación del canal se inauguró en junio de 2016 y costó 5250 millones de dólares [D2]."
    proposer = _proposer({
        "claim": 1, "source_id": "D2", "verdict": "supported", "restated_claim": "",
        "quote": "The Panama Canal expansion opened in June 2016",
    })

    outcome = mediate_claims(answer, SOURCES, proposer=proposer)

    assert outcome.answer == ""
    assert outcome.stats["rejected"] == {"numbers_not_in_quote": 1}


def test_partial_claim_is_restated_to_what_the_quote_supports():
    answer = "La ampliación del canal se inauguró en junio de 2016 y costó 5250 millones de dólares [D2]."
    proposer = _proposer({
        "claim": 1, "source_id": "D2", "verdict": "partial",
        "quote": "The Panama Canal expansion opened in June 2016",
        "restated_claim": "La ampliación del canal se inauguró en junio de 2016.",
    })

    outcome = mediate_claims(answer, SOURCES, proposer=proposer)

    assert outcome.answer == "La ampliación del canal se inauguró en junio de 2016 [D2]."
    assert outcome.stats["accepted_partial"] == 1


def test_unflagged_sentences_are_untouched_and_model_is_not_called():
    answer = "## Resultado\n- La ampliación del canal incluyó un tercer juego de esclusas según el documento [D2]."
    proposer = _proposer()

    outcome = mediate_claims(answer, SOURCES, proposer=proposer)

    assert outcome.answer == answer
    assert proposer.calls == []


def test_lexical_number_false_positive_is_recovered():
    # The paper's numeric check reads "2016 con" as the token "2016con" and
    # flags a correct claim; mediation compares values and keeps it.
    answer = "- La ampliación del canal abrió en junio de 2016 con un tercer juego de esclusas [D2]."
    assert agent_module._answer_evidence_issues(answer, SOURCES)
    proposer = _proposer({
        "claim": 1, "source_id": "D2", "verdict": "supported", "restated_claim": "",
        "quote": "opened in June 2016 with a third set of locks",
    })

    outcome = mediate_claims(answer, SOURCES, proposer=proposer)

    assert outcome.answer == answer
    assert outcome.stats["accepted_supported"] == 1


def test_model_failure_falls_back_to_pruning():
    answer = "Primera parte.\nEsta afirmación sustantiva no tiene ninguna cita que la respalde en absoluto."

    def broken(_claims, _excerpts):
        raise RuntimeError("ollama down")

    outcome = mediate_claims(answer, SOURCES, proposer=broken)

    assert outcome.answer == "Primera parte."
    assert "ollama down" in outcome.stats["llm_error"]


def test_claims_over_the_limit_are_pruned(monkeypatch):
    monkeypatch.setattr(mediation.settings, "CLAIM_MEDIATION_MAX_CLAIMS", 1)
    answer = (
        "El boletín indica que el 12,5 % de los casos fueron graves en Panamá Oeste durante 2023. "
        "Esta segunda afirmación sustantiva tampoco tiene ninguna cita que la respalde."
    )
    proposer = _proposer({
        "claim": 1, "source_id": "D1", "verdict": "supported", "restated_claim": "",
        "quote": "12.5% de casos graves en la provincia de Panamá Oeste durante 2023",
    })

    outcome = mediate_claims(answer, SOURCES, proposer=proposer)

    assert outcome.answer.endswith("durante 2023 [D1].")
    assert "segunda" not in outcome.answer
    assert outcome.stats["pruned_over_limit"] == 1


def test_default_proposer_uses_json_schema_structured_output(monkeypatch):
    structured = MagicMock()
    structured.invoke.return_value = mediation._MediationResult(items=[])
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(agent_module, "get_llm_client", MagicMock(return_value=llm))

    mediate_claims("Esta afirmación sustantiva no tiene ninguna cita que la respalde en absoluto.", SOURCES)

    llm.with_structured_output.assert_called_once_with(mediation._MediationResult, method="json_schema")


def test_uncited_draft_is_rescued_only_when_mediation_is_enabled(monkeypatch):
    draft = "El boletín indica que el 12,5 % de los casos fueron graves en Panamá Oeste durante 2023."
    item = _MediationItem(
        claim=1, source_id="D1", verdict="supported", restated_claim="",
        quote="12.5% de casos graves en la provincia de Panamá Oeste durante 2023",
    )
    structured = MagicMock()
    structured.invoke.return_value = mediation._MediationResult(items=[item])
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(agent_module, "get_llm_client", MagicMock(return_value=llm))

    monkeypatch.setattr(agent_module.settings, "CLAIM_MEDIATION_ENABLED", False)
    assert agent_module._rescue_uncited_draft(draft, SOURCES) == ""

    monkeypatch.setattr(agent_module.settings, "CLAIM_MEDIATION_ENABLED", True)
    assert agent_module._rescue_uncited_draft(draft, SOURCES).endswith("[D1].")
    # Invented citations are never rescued.
    assert agent_module._rescue_uncited_draft(draft + " Otra cosa [D9].", SOURCES) == ""


def test_semantic_audit_signal_mode_does_not_trigger_repair(monkeypatch):
    audit = agent_module._ClaimAudit(
        unsupported_claims=[], unmarked_inferences=["Algo inferido."], wrong_citations=[]
    )
    structured = MagicMock()
    structured.invoke.return_value = audit
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(agent_module, "get_llm_client", MagicMock(return_value=llm))
    answer = "La ampliación del canal incluyó un tercer juego de esclusas según el documento [D2]."

    monkeypatch.setattr(agent_module.settings, "RESEARCH_SEMANTIC_AUDIT_MODE", "gate")
    assert any("inferencias" in issue for issue in agent_module._semantic_claim_issues(answer, SOURCES, SOURCES))

    monkeypatch.setattr(agent_module.settings, "RESEARCH_SEMANTIC_AUDIT_MODE", "signal")
    assert agent_module._semantic_claim_issues(answer, SOURCES, SOURCES) == []

    llm.with_structured_output.reset_mock()
    monkeypatch.setattr(agent_module.settings, "RESEARCH_SEMANTIC_AUDIT_MODE", "off")
    assert agent_module._semantic_claim_issues(answer, SOURCES, SOURCES) == []
    llm.with_structured_output.assert_not_called()
