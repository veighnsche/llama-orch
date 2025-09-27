from __future__ import annotations

from ..config import OUTPUTS
from ..io.writer import ensure_dir
from .orchestrator import run_pipeline


def run() -> int:
    ensure_dir(OUTPUTS)
    return run_pipeline()
