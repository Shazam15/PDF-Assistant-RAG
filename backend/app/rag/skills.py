"""
Agent Skills — modular, filesystem-based procedures the ReAct agent can load on demand.

A skill is a directory under `settings.SKILLS_DIR` containing a `SKILL.md` file with
YAML frontmatter (`name`, `description`) followed by step-by-step instructions, plus
optional `scripts/`, `references/`, and `assets/` subfolders for resources the
instructions may point to.

Skills use progressive disclosure, mirroring how the app already avoids loading MCP
tool schemas or document chunks until they're actually needed:

  Tier 1 (metadata)      - name + description for every skill. Cheap enough to embed
                            directly in the `use_skill` tool description, so the agent
                            always sees the catalog without spending a tool call on it.
  Tier 2 (instructions)  - the full SKILL.md body. Loaded when the agent calls
                            `use_skill` with a bare skill name.
  Tier 3 (resources)     - files under scripts/references/assets. Loaded when the
                            agent calls `use_skill` with "<skill>/<relative-path>".
"""
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SKILL_FILENAME = "SKILL.md"
_RESOURCE_SUBFOLDERS = ("scripts", "references", "assets")
_MAX_RESOURCE_CHARS = 20_000

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)

# A skill's body is echoed back to the ReAct agent verbatim as a tool Observation. If it
# contains one of the agent's own control tokens, a model prone to imitating formatting it
# just saw could copy it into its next Thought/Action and desync GroundedReActOutputParser's
# regex-based parsing (see app/rag/agent.py). Skills are developer-authored, version-controlled
# content, so rejecting a violation at load time (visible in logs, fixable immediately) is far
# cheaper than a live parsing desync at runtime.
_RESERVED_REACT_TOKEN_RE = re.compile(
    r"(?im)^\s*(?:Action(?:\s+\d+)?\s*(?:Input)?\s*:|Thought\s*:|Observation\s*:|Final\s+Answer\s*:)"
)

# A model that gets a "not found" Observation on a slightly-off skill name burns a whole retry
# iteration for nothing. These map each backing tool's own name to the skill that documents how
# to use it, so asking for the tool's name resolves the skill on the first try too.
_TOOL_NAME_SKILL_ALIASES: Dict[str, str] = {
    "code_review": "code-review",
    "statistics": "statistical-analysis",
}


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: str  # absolute path to the skill's own directory


_skills_lock = threading.Lock()
_skills_cache: Optional[Dict[str, SkillMetadata]] = None
_skills_cache_root: Optional[str] = None


def _skills_root() -> str:
    configured = getattr(settings, "SKILLS_DIR", "") or ""
    return os.path.abspath(configured) if configured else ""


def _parse_skill_file(skill_md_path: str) -> Optional[Dict[str, str]]:
    try:
        with open(skill_md_path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError as exc:
        logger.warning("Could not read skill file %s: %s", skill_md_path, exc)
        return None

    match = _FRONTMATTER_RE.match(raw)
    if not match:
        logger.warning("Skill file %s is missing YAML frontmatter; skipping.", skill_md_path)
        return None

    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning("Skill file %s has invalid YAML frontmatter: %s", skill_md_path, exc)
        return None

    name = str(frontmatter.get("name") or "").strip()
    description = str(frontmatter.get("description") or "").strip()
    if not name or not description:
        logger.warning("Skill file %s must declare both 'name' and 'description'; skipping.", skill_md_path)
        return None

    body = match.group(2).strip()
    reserved_match = _RESERVED_REACT_TOKEN_RE.search(body)
    if reserved_match:
        logger.warning(
            "Skill file %s contains a reserved ReAct control token (%r) in its body; skipping it "
            "to avoid desyncing the agent's output parser when it's echoed back as a tool Observation.",
            skill_md_path,
            reserved_match.group(0).strip(),
        )
        return None

    return {"name": name, "description": description, "body": body}


def discover_skills(force_reload: bool = False) -> Dict[str, SkillMetadata]:
    """Scan SKILLS_DIR for `<skill>/SKILL.md` files (Tier 1: metadata only).

    Cached per-process like `load_mcp_tools()`; call `clear_skills_cache()` after
    changing SKILLS_DIR at runtime (e.g. in tests).
    """
    root = _skills_root()
    global _skills_cache, _skills_cache_root
    with _skills_lock:
        if not force_reload and _skills_cache is not None and _skills_cache_root == root:
            return _skills_cache

        discovered: Dict[str, SkillMetadata] = {}
        if root and os.path.isdir(root):
            for entry in sorted(os.listdir(root)):
                skill_dir = os.path.join(root, entry)
                skill_md = os.path.join(skill_dir, SKILL_FILENAME)
                if not os.path.isfile(skill_md):
                    continue
                parsed = _parse_skill_file(skill_md)
                if parsed is None:
                    continue
                discovered[parsed["name"]] = SkillMetadata(
                    name=parsed["name"], description=parsed["description"], path=skill_dir
                )

        _skills_cache = discovered
        _skills_cache_root = root
        return discovered


def clear_skills_cache() -> None:
    """Drop the cached skill catalog so the next discover_skills() call re-scans disk."""
    global _skills_cache, _skills_cache_root
    with _skills_lock:
        _skills_cache = None
        _skills_cache_root = None


def skills_catalog_text() -> str:
    """Render Tier-1 metadata for every skill, for embedding in the tool description."""
    skills = discover_skills()
    if not skills:
        return "No hay skills instaladas."
    return "\n".join(f"- {meta.name}: {meta.description}" for meta in skills.values())


def _list_skill_resources(skill_dir: str) -> List[str]:
    resources: List[str] = []
    for subfolder in _RESOURCE_SUBFOLDERS:
        folder = os.path.join(skill_dir, subfolder)
        if not os.path.isdir(folder):
            continue
        for dirpath, _dirnames, filenames in os.walk(folder):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                resources.append(os.path.relpath(full_path, skill_dir))
    return sorted(resources)


def _normalize_skill_key(value: str) -> str:
    return re.sub(r"[\s_]+", "-", (value or "").strip().lower())


def _lookup_skill(skills: Dict[str, SkillMetadata], query: str) -> Optional[SkillMetadata]:
    """Resolve a skill by exact name, then a case/separator-insensitive match, then a known
    backing-tool alias — so a model that types 'Code_Review' or 'statistics' still resolves on
    the first try instead of burning a retry iteration on a "not found" Observation.
    """
    if query in skills:
        return skills[query]

    normalized_query = _normalize_skill_key(query)
    for name, meta in skills.items():
        if _normalize_skill_key(name) == normalized_query:
            return meta

    aliased_name = _TOOL_NAME_SKILL_ALIASES.get(query) or _TOOL_NAME_SKILL_ALIASES.get(normalized_query)
    if aliased_name:
        normalized_alias = _normalize_skill_key(aliased_name)
        for name, meta in skills.items():
            if _normalize_skill_key(name) == normalized_alias:
                return meta

    return None


def _resolve_resource_path(skill_dir: str, relative_path: str) -> Optional[str]:
    """Resolve a Tier-3 resource path, refusing anything that escapes the skill's own directory."""
    skill_dir_abs = os.path.abspath(skill_dir)
    candidate = os.path.abspath(os.path.join(skill_dir_abs, relative_path))
    if os.path.commonpath([candidate, skill_dir_abs]) != skill_dir_abs:
        return None
    return candidate if os.path.isfile(candidate) else None


def load_skill(query: str) -> str:
    """Load a skill for the agent: Tier 1 catalog, Tier 2 instructions, or a Tier 3 resource.

    - Empty input -> the Tier 1 catalog (name + description of every skill).
    - "<skill>" -> that skill's full SKILL.md body (Tier 2), plus an index of any
      scripts/references/assets it ships (Tier 3 index, not their content).
    - "<skill>/<relative-path>" -> the raw content of that one resource file (Tier 3),
      e.g. "code-review/references/checklist.md".
    """
    skills = discover_skills()
    normalized = (query or "").strip().strip("/")
    if not normalized:
        return skills_catalog_text()

    skill_name, _, resource_path = normalized.partition("/")
    meta = _lookup_skill(skills, skill_name.strip())
    if meta is None:
        available = ", ".join(sorted(skills)) or "ninguna"
        return f"Skill '{skill_name}' no encontrada. Skills disponibles: {available}."

    if not resource_path:
        parsed = _parse_skill_file(os.path.join(meta.path, SKILL_FILENAME))
        body = parsed["body"] if parsed else ""
        resources = _list_skill_resources(meta.path)
        if resources:
            resource_lines = "\n".join(f"- {resource}" for resource in resources)
            body += (
                f"\n\n## Recursos adicionales de esta skill\n"
                f"Usa 'use_skill' con \"{meta.name}/<ruta>\" para leer alguno de estos archivos:\n"
                f"{resource_lines}"
            )
        # Tier-2 only: a verbose Observation invites a long paraphrasing Thought from a small
        # model, which can trip GroundedReActOutputParser's premature-AgentFinish shortcut for
        # substantive (>=35 word) text with no Action line. Nudge it to act, not summarize.
        body += (
            "\n\n---\n"
            "Aplica este procedimiento en tu próxima Action de inmediato. No repitas, resumas ni "
            "parafrasees estos pasos en tu Thought: úsalo solo para decidir la siguiente Action concreta."
        )
        return body

    resolved = _resolve_resource_path(meta.path, resource_path)
    if resolved is None:
        return f"Recurso '{resource_path}' no encontrado dentro de la skill '{meta.name}'."
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError as exc:
        return f"No se pudo leer el recurso '{resource_path}': {exc}"
    if len(content) > _MAX_RESOURCE_CHARS:
        content = content[:_MAX_RESOURCE_CHARS] + "\n...[truncado]"
    return content
