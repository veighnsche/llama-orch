from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

# --- Whitelist parsing from placeholders.md ---

_BACKTICK_ITEM = re.compile(r"\*\s+`([a-zA-Z0-9_]+)`\s+—\s+(.*)")
_VAR = re.compile(r"{{\s*([#/])?\s*([a-zA-Z0-9_]+)\s*}}")


def parse_whitelist(md_path: Path) -> Tuple[Set[str], Dict[str, str]]:
    """Parse placeholders.md and return (whitelist, descriptions).
    It looks for bullet lines like: * `placeholder` — description
    """
    text = md_path.read_text(encoding="utf-8")
    wl: Set[str] = set()
    desc: Dict[str, str] = {}
    for line in text.splitlines():
        m = _BACKTICK_ITEM.search(line)
        if m:
            key = m.group(1).strip()
            wl.add(key)
            desc.setdefault(key, m.group(2).strip())
    # Common section names that may not appear as bullets explicitly
    for section in ("leningen", "omzetstromen", "indicatieve_heffing_jaar"):
        wl.add(section)
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
