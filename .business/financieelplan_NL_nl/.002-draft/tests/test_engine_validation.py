from __future__ import annotations

from typing import Any, Dict

import pytest

from finance_engine.engine_pkg import runner as engine_runner
from finance_engine.config import OUTPUTS


def test_engine_returns_nonzero_on_validation_failure(monkeypatch):
    def fake_validate_inputs(*args, **kwargs) -> Dict[str, Any]:
        return {"errors": ["boom"], "warnings": []}

    monkeypatch.setattr(engine_runner, "validate_inputs", fake_validate_inputs, raising=True)

    rc = engine_runner.run()
    assert rc == 1, "Engine should return 1 when validation produces errors"

    # validation_report.json should be written even on failure
    assert (OUTPUTS / "validation_report.json").exists()
