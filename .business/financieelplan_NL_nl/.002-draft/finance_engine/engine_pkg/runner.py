from __future__ import annotations

from ..io.writer import ensure_dir
from ..config import OUTPUTS
from ..render.template import render_template_jinja  # re-export for tests
from ..validation.validate import validate_inputs  # re-export for tests

from .orchestrator import run_pipeline


def run() -> int:
    ensure_dir(OUTPUTS)
    return run_pipeline()
