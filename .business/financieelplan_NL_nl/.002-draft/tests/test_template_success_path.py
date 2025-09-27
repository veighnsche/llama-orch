from __future__ import annotations

from finance_engine.config import OUTPUTS
from finance_engine.engine_pkg import run as engine_run


def test_template_renders_without_fallback():
    assert engine_run() == 0
    plan = OUTPUTS / "financial_plan.md"
    assert plan.exists(), "financial_plan.md should exist"
    content = plan.read_text(encoding="utf-8")
    assert "Financial Plan (Fallback Render)" not in content, "Template should render without fallback"
    # spot-check a few context variables appear in rendered markdown
    assert "Public Tap" in content
    assert "Private Tap" in content
    assert "Model Economics" in content
