from pathlib import Path
import json

from finance_engine.engine_pkg import run
from finance_engine.config import OUTPUTS


def test_smoke_run(tmp_path, monkeypatch):
    # Ensure outputs directory is the default; engine writes to ./outputs
    rc = run()
    assert rc in (0, 1)

    # Must produce validation report regardless; success depends on errors
    vr = OUTPUTS / "validation_report.json"
    assert vr.exists(), "validation_report.json missing"

    report = json.loads(vr.read_text(encoding="utf-8"))
    if report.get("errors"):
        # When there are errors, engine exits non-zero and may skip some artifacts
        assert rc == 1
    else:
        # When no errors, we expect summary and placeholder template
        assert rc == 0
        assert (OUTPUTS / "run_summary.json").exists()
        assert (OUTPUTS / "run_summary.md").exists()
        assert (OUTPUTS / "template_filled.md").exists()
