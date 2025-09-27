from __future__ import annotations

import types

from finance_engine.config import OUTPUTS
from finance_engine.engine_pkg import runner as engine_runner


def test_fallback_render_when_template_raises(monkeypatch):
    # Force the template renderer to raise so fallback kicks in
    monkeypatch.setattr(
        engine_runner,
        "render_template_jinja",
        lambda template_path, out_path, ctx: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=True,
    )

    rc = engine_runner.run()
    assert rc == 0

    plan = OUTPUTS / "financial_plan.md"
    assert plan.exists(), "financial_plan.md should be written even on fallback"
    content = plan.read_text(encoding="utf-8")
    assert "Financial Plan (Fallback Render)" in content
    # Should still reference charts in fallback
    assert "charts/model_margins_per_1m.png" in content
