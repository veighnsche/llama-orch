from __future__ import annotations

from pathlib import Path
import shutil

from finance_engine.config import INPUTS
from finance_engine.engine_pkg.validation.validate_scenarios import validate


def copy_inputs(tmp: Path) -> Path:
    dst = tmp / "inputs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in INPUTS.iterdir():
        if p.is_file():
            shutil.copy(p, dst / p.name)
    return dst


def test_scenarios_good(tmp_path: Path):
    d = copy_inputs(tmp_path)
    (d / "scenarios.yaml").write_text("""
monthly:
  worst_m_tokens: 1
  base_m_tokens: 5
  best_m_tokens: 10
""", encoding="utf-8")
    fr = validate(d)
    assert fr.ok


def test_scenarios_bad_negative(tmp_path: Path):
    d = copy_inputs(tmp_path)
    (d / "scenarios.yaml").write_text("""
monthly:
  worst_m_tokens: -1
  base_m_tokens: 5
  best_m_tokens: 10
""", encoding="utf-8")
    fr = validate(d)
    assert not fr.ok
