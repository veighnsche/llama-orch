from __future__ import annotations

from pathlib import Path
from typing import Dict


def simple_placeholder_render(template_path: Path, output_path: Path, mapping: Dict[str, str]) -> None:
    text = template_path.read_text(encoding="utf-8")
    out = text
    for k, v in mapping.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out, encoding="utf-8")


def render_template_jinja(template_path: Path, output_path: Path, context: Dict) -> None:
    """Render a markdown template using Jinja2 with the given context.
    This supports filters like join that are present in the template file.
    """
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        trim_blocks=False,
        lstrip_blocks=False,
        undefined=StrictUndefined,  # catch missing keys early
    )
    tpl = env.get_template(template_path.name)
    rendered = tpl.render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
