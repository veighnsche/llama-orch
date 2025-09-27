from __future__ import annotations

import types

from finance_engine.config import OUTPUTS
from finance_engine.io.writer import ensure_dir
from finance_engine.engine_pkg.orchestrator import run_pipeline


def test_fallback_render_when_template_raises(monkeypatch):
    # Inject a renderer that raises to trigger fallback
    def boom_renderer(template_path, out_path, ctx):
        raise RuntimeError("boom")

    ensure_dir(OUTPUTS)
    rc = run_pipeline(render_port=boom_renderer)
    assert rc == 0

    plan = OUTPUTS / "financial_plan.md"
    assert plan.exists(), "financial_plan.md should be written even on fallback"
    content = plan.read_text(encoding="utf-8")
    assert "Financial Plan (Fallback Render)" in content
    # Should still reference charts in fallback
    assert "charts/model_margins_per_1m.png" in content
