from __future__ import annotations

from pathlib import Path
import shutil

from finance_engine.config import INPUTS
from finance_engine.engine_pkg.validation.preflight import run_preflight


def copy_inputs(tmp: Path) -> Path:
    dst = tmp / "inputs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in INPUTS.iterdir():
        if p.is_file():
            shutil.copy(p, dst / p.name)
    return dst


def test_overrides_optional_normal_mode(tmp_path, monkeypatch):
    d = copy_inputs(tmp_path)
    # Write overrides with one expired and one active entry
    (d / "overrides.yaml").write_text(
        """
price_overrides:
  Llama-3.1-8B:
    unit_price_eur_per_1k_tokens: 0.15
    expires_on: 1999-01-01
  Mixtral-8x7B:
    unit_price_eur_per_1k_tokens: 0.39
    expires_on: 2099-12-31
        
""",
        encoding="utf-8",
    )
    # Normal mode: should be OK (expired allowed but shown), not strict
    res = run_preflight(d)
    assert res.ok


def test_overrides_expired_strict_fails(tmp_path, monkeypatch):
    d = copy_inputs(tmp_path)
    (d / "overrides.yaml").write_text(
        """
price_overrides:
  Llama-3.1-8B:
    unit_price_eur_per_1k_tokens: 0.15
    expires_on: 1999-01-01
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENGINE_STRICT_VALIDATION", "1")
    res = run_preflight(d)
    assert not res.ok
    monkeypatch.delenv("ENGINE_STRICT_VALIDATION", raising=False)
