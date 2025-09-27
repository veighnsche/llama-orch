from __future__ import annotations

from typing import Any, Dict

import pytest

from finance_engine.config import OUTPUTS
from finance_engine.io.writer import ensure_dir
from finance_engine.engine_pkg.orchestrator import run_pipeline


def test_engine_returns_nonzero_on_validation_failure(monkeypatch):
    def fake_validate_inputs(*, config: Dict[str, Any], costs: Dict[str, Any], lending: Dict[str, Any], price_sheet):
        return {"errors": ["boom"], "warnings": []}

    ensure_dir(OUTPUTS)
    rc = run_pipeline(validate_port=fake_validate_inputs)
    assert rc == 1, "Engine should return 1 when validation produces errors"

    # validation_report.json should be written even on failure
    assert (OUTPUTS / "validation_report.json").exists()
