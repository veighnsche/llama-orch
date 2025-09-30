from __future__ import annotations
from pathlib import Path
from runner import runner
from .util_inputs import make_minimal_inputs


def _read_sums(p: Path) -> dict[str, str]:
    m = {}
    for line in p.read_text().strip().splitlines():
        if not line.strip():
            continue
        sha, name = line.strip().split(None, 1)
        m[name.strip()] = sha.strip()
    return m


def test_determinism_sha256sums(tmp_path: Path):
    inputs = make_minimal_inputs(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    runner.execute(inputs, out1, ["public", "private"], seed=424242, fail_on_warning=True, max_concurrency=2)
    runner.execute(inputs, out2, ["public", "private"], seed=424242, fail_on_warning=True, max_concurrency=4)
    sums1 = _read_sums(out1 / "SHA256SUMS")
    sums2 = _read_sums(out2 / "SHA256SUMS")
    assert sums1 == sums2, "SHA256SUMS must match across runs with same seed and inputs"
