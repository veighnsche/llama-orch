from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

# --- Whitelist parsing from placeholders.md ---

_BULLET = re.compile(r"^\s*\*\s+")
_BACKTICKS_ALL = re.compile(r"`([^`]+)`")
_BACKTICK_DESC = re.compile(r"\*\s+`[a-zA-Z0-9_]+`\s*(?:—|-)\s*(.*)")
_VAR = re.compile(r"{{\s*([#/])?\s*([a-zA-Z0-9_]+)\s*}}")


def parse_whitelist(md_path: Path) -> Tuple[Set[str], Dict[str, str]]:
    """Parse placeholders.md and return (whitelist, descriptions).
    It looks for bullet lines like: * `placeholder` — description
    """
    text = md_path.read_text(encoding="utf-8")
    wl: Set[str] = set()
    desc: Dict[str, str] = {}
    for line in text.splitlines():
        if not _BULLET.search(line):
            continue
        chunks = _BACKTICKS_ALL.findall(line)
        if not chunks:
            continue
        md = _BACKTICK_DESC.search(line)
        desc_val = md.group(1).strip() if md else ""
        for chunk in chunks:
            for key in re.split(r"\s*/\s*", chunk):
                key = key.strip()
                if not key or not re.match(r"^[a-zA-Z0-9_\*]+$", key):
                    continue
                wl.add(key)
                desc.setdefault(key, desc_val)
    # Common section/variable names that may not appear as bullets explicitly
    for section in (
        "leningen",
        "omzetstromen",
        "indicatieve_heffing_jaar",
        "in",
        "mapping",
        "out",
        # widely used simple vars sometimes omitted from the list
        "start_maand",
    ):
        wl.add(section)

    # Expand shorthand/wildcards commonly used in placeholders.md text
    expansions = {
        # price sensitivity
        "omzet_*": ["omzet_min10", "omzet_basis", "omzet_plus10"],
        "marge_*": ["marge_min10", "marge_basis", "marge_plus10"],
        "runway_*": ["runway_min10", "runway_basis", "runway_plus10"],
        "prijs_*": ["prijs_min10", "prijs_basis", "prijs_plus10"],
        "contrib_*": ["contrib_min10", "contrib_basis", "contrib_plus10"],
        "marge_pct_*": ["marge_pct_min10", "marge_pct_basis", "marge_pct_plus10"],
        "ltv_cac_*": ["ltv_cac_pricemin10", "ltv_cac_ratio", "ltv_cac_varplus10"],
    }
    for wildcard, keys in expansions.items():
        if wildcard in text:
            for k in keys:
                wl.add(k)
                desc.setdefault(k, "expanded from wildcard")

    # Numeric pair shorthand like `omzet_pricevol1/2` and `marge_pricevol1/2`
    for prefix in ("omzet_pricevol", "marge_pricevol"):
        if prefix in text:
            for n in ("1", "2"):
                k = f"{prefix}{n}"
                wl.add(k)
                desc.setdefault(k, "expanded from pair shorthand")

    # Explicit multi-token backticks often formatted with slashes
    explicit_tokens = [
        # pricing elasticities and volumes
        "volume_plus15", "volume_min10", "volume_basis",
        # stress shorthands appearing in templates
        "stress_omzet_min30", "stress_dso_plus30", "stress_opex_plus20",
        # misc singletons used in templates
        "var_basis", "marge_pricemin10", "marge_varplus10", "dekking_pct",
        # explicit price sensitivity tokens
        "prijs_min10", "prijs_basis", "prijs_plus10",
        # explicit contrib tokens
        "contrib_min10", "contrib_basis", "contrib_plus10",
        # explicit LTV/CAC stress tokens
        "ltv_cac_pricemin10", "ltv_cac_varplus10",
        # explicit margin pct tokens
        "marge_pct_min10", "marge_pct_basis", "marge_pct_plus10",
    ]
    for k in explicit_tokens:
        wl.add(k)
        desc.setdefault(k, "explicitly included common key")
    return wl, desc


# --- Template scanning ---

def find_placeholders(template_text: str) -> Tuple[Set[str], Set[str]]:
    """Return (variables, sections) referenced by a template.
    Sections use {{#name}}...{{/name}}. Variables are bare {{name}}.
    """
    vars_used: Set[str] = set()
    sections: Set[str] = set()
    for m in _VAR.finditer(template_text):
        sig = m.group(1)
        name = m.group(2)
        if sig == '#':
            sections.add(name)
        elif sig == '/':
            # end section, ignore
            pass
        else:
            vars_used.add(name)
    return vars_used, sections


# --- Minimal Mustache-like renderer (variables + list sections) ---

class RenderError(RuntimeError):
    pass


def _render_section(block: str, items: Any, whitelist: Set[str]) -> str:
    # If items is falsy, render nothing
    if not items:
        return ""
    out = []
    if isinstance(items, list):
        for it in items:
            out.append(render_template(block, it, whitelist))
        return "".join(out)
    # Truthy scalar or dict: render once with same or nested context
    ctx = items if isinstance(items, dict) else {}
    return render_template(block, ctx, whitelist)


def render_template(template_text: str, data: Dict[str, Any], whitelist: Set[str]) -> str:
    # Fail fast on unknown placeholders (not in whitelist)
    vars_used, sections = find_placeholders(template_text)
    unknown = [k for k in vars_used.union(sections) if k not in whitelist]
    if unknown:
        raise RenderError(f"Unknown template key(s): {', '.join(sorted(unknown))}")

    # Process sections first (innermost first). We use a stack-based regex search.
    # Pattern {{#name}}...{{/name}}
    sec_open = re.compile(r"{{\s*#\s*([a-zA-Z0-9_]+)\s*}}")
    while True:
        m = list(sec_open.finditer(template_text))
        if not m:
            break
        # Take the last (innermost) open
        last = m[-1]
        name = last.group(1)
        # Find its closing tag after last.end()
        close = re.search(r"{{\s*/\s*" + re.escape(name) + r"\s*}}", template_text[last.end():])
        if not close:
            raise RenderError(f"Unclosed section: {name}")
        a = last.start()
        b = last.end() + close.end()
        inner = template_text[last.end(): last.end() + close.start()]
        rendered = _render_section(inner, data.get(name), whitelist)
        template_text = template_text[:a] + rendered + template_text[b:]

    # Now variables
    def repl(m: re.Match) -> str:
        if m.group(1):  # was a section marker, already processed
            return ""
        key = m.group(2)
        if key not in whitelist:
            raise RenderError(f"Unknown template key: {key}")
        if key not in data:
            raise RenderError(f"Template missing required placeholder: {key}")
        v = data.get(key)
        return "" if v is None else str(v)

    return _VAR.sub(repl, template_text)


def check_templates(templates_dir: Path, whitelist: Set[str]) -> List[Tuple[str, List[str]]]:
    """Return list of (template_relpath, unknown_keys[])"""
    issues: List[Tuple[str, List[str]]] = []
    for p in sorted(templates_dir.glob("*.md.tpl")):
        txt = p.read_text(encoding="utf-8")
        vars_used, sections = find_placeholders(txt)
        unknown = [k for k in vars_used.union(sections) if k not in whitelist]
        if unknown:
            issues.append((str(p), sorted(unknown)))
    return issues


def generate_template_keys_md(templates_dir: Path, placeholders_md: Path, out_path: Path) -> None:
    wl, desc = parse_whitelist(placeholders_md)
    rows = ["# TEMPLATE_KEYS (generated)\n"]
    for p in sorted(templates_dir.glob("*.md.tpl")):
        txt = p.read_text(encoding="utf-8")
        vars_used, sections = find_placeholders(txt)
        rows.append(f"\n## {p.name}\n\n")
        rows.append("Placeholder | Kind | Required | Description\n")
        rows.append("---|---|---|---\n")
        for k in sorted(vars_used):
            kind = "var"
            required = "yes"
            rows.append(f"{k} | {kind} | {required} | {desc.get(k, '')}\n")
        for s in sorted(sections):
            kind = "section"
            required = "conditional"
            rows.append(f"{s} | {kind} | {required} | {desc.get(s, '')}\n")
    out_path.write_text("".join(rows), encoding="utf-8")
