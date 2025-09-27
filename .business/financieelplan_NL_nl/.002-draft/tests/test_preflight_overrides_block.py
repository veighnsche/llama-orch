from __future__ import annotations

from pathlib import Path
import shutil

from finance_engine.config import INPUTS
from finance_engine.engine_pkg.validation.registry import run_preflight
from finance_engine.engine_pkg.validation.shared import build_preflight_markdown


def copy_inputs(tmp: Path) -> Path:
    dst = tmp / "inputs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in INPUTS.iterdir():
        if p.is_file():
            shutil.copy(p, dst / p.name)
    return dst


def test_overrides_applied_block_and_reconcile_suggested(tmp_path: Path):
    d = copy_inputs(tmp_path)
    # Price overrides: one expired, one active
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
    # Capacity overrides: one expired
    (d / "capacity_overrides.yaml").write_text(
        """
capacity_overrides:
  Llama-3.1-8B:
    tps_override_tokens_per_sec: 100.0
    preferred_gpu: L40S
    expires_on: 1999-01-01
""",
        encoding="utf-8",
    )
    res = run_preflight(d)
    assert res.ok
    md = build_preflight_markdown(res)
    assert "Overrides Applied:" in md
    assert "- prices:" in md
    assert "- capacity:" in md
    assert "Reconcile suggested:" in md
