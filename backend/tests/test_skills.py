import os
import textwrap

import pytest

from app.rag import skills as skills_module


@pytest.fixture(autouse=True)
def _reset_skills_cache():
    """Every test gets a clean cache so SKILLS_DIR overrides actually take effect."""
    skills_module.clear_skills_cache()
    yield
    skills_module.clear_skills_cache()


def _write_skill(tmp_path, slug, name, description, body, resource: tuple[str, str] | None = None):
    skill_dir = tmp_path / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    if resource:
        relative_path, content = resource
        resource_path = skill_dir / relative_path
        resource_path.parent.mkdir(parents=True, exist_ok=True)
        resource_path.write_text(content, encoding="utf-8")
    return skill_dir


def test_discover_skills_finds_the_bundled_skills():
    """The two skills shipped with the app must always be discoverable at their real path."""
    catalog = skills_module.discover_skills(force_reload=True)
    assert "code-review" in catalog
    assert "statistical-analysis" in catalog
    assert catalog["code-review"].description


def test_discover_skills_scans_configured_directory(tmp_path, monkeypatch):
    _write_skill(tmp_path, "demo", "demo-skill", "A demo skill.", "Follow these steps.")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    catalog = skills_module.discover_skills(force_reload=True)

    assert set(catalog) == {"demo-skill"}
    assert catalog["demo-skill"].description == "A demo skill."


def test_discover_skills_skips_files_without_frontmatter(tmp_path, monkeypatch):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("No frontmatter here.", encoding="utf-8")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    assert skills_module.discover_skills(force_reload=True) == {}


def test_discover_skills_skips_frontmatter_missing_required_fields(tmp_path, monkeypatch):
    skill_dir = tmp_path / "incomplete"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: incomplete\n---\nBody.", encoding="utf-8")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    assert skills_module.discover_skills(force_reload=True) == {}


def test_skills_catalog_text_lists_name_and_description(tmp_path, monkeypatch):
    _write_skill(tmp_path, "demo", "demo-skill", "A demo skill.", "Body.")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    catalog_text = skills_module.skills_catalog_text()

    assert "demo-skill" in catalog_text
    assert "A demo skill." in catalog_text


def test_load_skill_with_empty_input_returns_catalog(tmp_path, monkeypatch):
    _write_skill(tmp_path, "demo", "demo-skill", "A demo skill.", "Body.")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    assert "demo-skill" in skills_module.load_skill("")


def test_load_skill_returns_tier2_instructions(tmp_path, monkeypatch):
    _write_skill(tmp_path, "demo", "demo-skill", "A demo skill.", "Step 1. Do the thing.")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    result = skills_module.load_skill("demo-skill")

    assert "Step 1. Do the thing." in result


def test_load_skill_unknown_name_lists_available_skills(tmp_path, monkeypatch):
    _write_skill(tmp_path, "demo", "demo-skill", "A demo skill.", "Body.")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    result = skills_module.load_skill("nonexistent-skill")

    assert "no encontrada" in result.lower()
    assert "demo-skill" in result


def test_load_skill_lists_resources_and_loads_tier3_content(tmp_path, monkeypatch):
    _write_skill(
        tmp_path,
        "demo",
        "demo-skill",
        "A demo skill.",
        "Body.",
        resource=("references/checklist.md", "- item one\n- item two"),
    )
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    tier2 = skills_module.load_skill("demo-skill")
    assert "references/checklist.md" in tier2

    tier3 = skills_module.load_skill("demo-skill/references/checklist.md")
    assert "item one" in tier3


def test_load_skill_resource_rejects_path_traversal(tmp_path, monkeypatch):
    _write_skill(tmp_path, "demo", "demo-skill", "A demo skill.", "Body.")
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    result = skills_module.load_skill("demo-skill/../secret.txt")

    assert "top secret" not in result
    assert "no encontrado" in result.lower()


def test_discover_skills_skips_body_containing_final_answer_token(tmp_path, monkeypatch):
    """A skill body echoed back as a tool Observation must never contain a literal ReAct
    control token, or a model prone to imitating formatting it just saw could copy it into
    its own next Thought/Action and desync the output parser."""
    _write_skill(tmp_path, "bad", "bad-skill", "A bad skill.", "Step 1.\nFinal Answer: done.")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    assert skills_module.discover_skills(force_reload=True) == {}


def test_discover_skills_skips_body_containing_action_input_token(tmp_path, monkeypatch):
    _write_skill(tmp_path, "bad", "bad-skill", "A bad skill.", "Step 1.\nAction Input: rm -rf /")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    assert skills_module.discover_skills(force_reload=True) == {}


def test_load_skill_resolves_case_and_separator_variants(tmp_path, monkeypatch):
    _write_skill(tmp_path, "demo", "demo-skill", "A demo skill.", "Step 1. Do the thing.")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    assert "Step 1. Do the thing." in skills_module.load_skill("Demo_Skill")
    assert "Step 1. Do the thing." in skills_module.load_skill("DEMO SKILL")


def test_load_skill_resolves_code_review_tool_name_alias():
    result = skills_module.load_skill("code_review")
    assert "Revisión y Generación de Código" in result


def test_load_skill_resolves_statistics_tool_name_alias():
    result = skills_module.load_skill("statistics")
    assert "Análisis Estadístico" in result


def test_load_skill_tier2_appends_apply_dont_restate_directive(tmp_path, monkeypatch):
    _write_skill(tmp_path, "demo", "demo-skill", "A demo skill.", "Body.")
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    result = skills_module.load_skill("demo-skill")

    assert "No repitas, resumas ni parafrasees" in result


def test_load_skill_tier3_resource_does_not_append_directive(tmp_path, monkeypatch):
    _write_skill(
        tmp_path,
        "demo",
        "demo-skill",
        "A demo skill.",
        "Body.",
        resource=("references/checklist.md", "- item one"),
    )
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))

    result = skills_module.load_skill("demo-skill/references/checklist.md")

    assert "No repitas, resumas ni parafrasees" not in result


def test_clear_skills_cache_forces_rescan(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_module.settings, "SKILLS_DIR", str(tmp_path))
    assert skills_module.discover_skills() == {}

    _write_skill(tmp_path, "demo", "demo-skill", "A demo skill.", "Body.")
    # Without clearing the cache, the stale (empty) result would still be returned.
    skills_module.clear_skills_cache()

    assert "demo-skill" in skills_module.discover_skills()
