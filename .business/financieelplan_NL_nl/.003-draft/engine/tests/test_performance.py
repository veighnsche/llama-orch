from __future__ import annotations
from pathlib import Path
import time
import psutil
from runner import runner
from .util_inputs import make_minimal_inputs


def test_minimal_e2e_perf(tmp_path: Path):
    inputs = make_minimal_inputs(tmp_path)
    out = tmp_path / "out"
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    runner.execute(inputs, out, ["public", "private"], seed=424242, fail_on_warning=True, max_concurrency=2)
    dt = time.perf_counter() - t0
    rss_after = proc.memory_info().rss
    delta_mb = (rss_after - rss_before) / (1024 * 1024)
    # Home-profile defaults (adjust in CI if needed)
    assert dt <= 10.0, f"E2E runtime too slow: {dt:.2f}s"
    assert delta_mb <= 512.0, f"Memory growth too high: {delta_mb:.1f} MB"
