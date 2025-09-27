from __future__ import annotations

from pathlib import Path
import shutil

from finance_engine.config import INPUTS
from finance_engine.engine_pkg.validation.preflight.validate_gpu_pricing import validate


def copy_inputs(tmp: Path) -> Path:
    dst = tmp / "inputs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in INPUTS.iterdir():
        if p.is_file():
            shutil.copy(p, dst / p.name)
    return dst


def test_gpu_pricing_good(tmp_path: Path):
    d = copy_inputs(tmp_path)
    (d / "gpu_pricing.yaml").write_text(
        """
private_tap_markup_by_gpu:
  L40S: 45
  A100: 60
""",
        encoding="utf-8",
    )
    fr = validate(d)
    assert fr.ok and fr.count == 2


def test_gpu_pricing_bad_percent(tmp_path: Path):
    d = copy_inputs(tmp_path)
    (d / "gpu_pricing.yaml").write_text(
        """
private_tap_markup_by_gpu:
  L40S: -5
""",
        encoding="utf-8",
    )
    fr = validate(d)
    assert not fr.ok
