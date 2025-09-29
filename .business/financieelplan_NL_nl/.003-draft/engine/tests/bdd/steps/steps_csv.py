from __future__ import annotations
import csv
from pathlib import Path
from pytest_bdd import then, parsers


def _read_csv(path: Path):
    with path.open() as f:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []
        rows = [row for row in rdr]
    return headers, rows


@then(parsers.parse('the outputs directory should contain CSV "{name}" with headers {headers}'))
def then_outputs_contains_csv_with_headers(ctx, name: str, headers: str):
    out = ctx['outputs']
    p = out / name
    assert p.exists(), f"missing CSV: {name}"
    hdr, _ = _read_csv(p)
    expected = [h.strip() for h in headers.split(',')]
    # Allow subset check to support superset headers in files
    for h in expected:
        assert h in hdr, f"missing header {h} in {name}: got {hdr}"


@then(parsers.parse('CSV "{name}" column "{col}" should be monotonic nondecreasing'))
def then_csv_col_monotone(ctx, name: str, col: str):
    out = ctx['outputs']
    _, rows = _read_csv(out / name)
    vals = []
    for r in rows:
        v = r.get(col)
        assert v is not None, f"missing column {col} in row"
        try:
            vals.append(float(v))
        except Exception:
            vals.append(0.0)
    assert all(vals[i] <= vals[i+1] for i in range(len(vals)-1)), f"{col} not monotone nondecreasing: {vals}"


@then(parsers.parse('CSV "{name}" column "{col}" should be all >= {threshold:f}'))
def then_csv_col_all_ge(ctx, name: str, col: str, threshold: float):
    out = ctx['outputs']
    _, rows = _read_csv(out / name)
    for r in rows:
        v = r.get(col)
        assert v is not None, f"missing column {col}"
        try:
            x = float(v)
        except Exception:
            x = 0.0
        assert x >= threshold, f"value {x} < {threshold}"


@then(parsers.parse('CSV "{name}" column "{col}" should be all "{value}"'))
def then_csv_col_all_string(ctx, name: str, col: str, value: str):
    out = ctx['outputs']
    _, rows = _read_csv(out / name)
    for r in rows:
        assert str(r.get(col, '')).strip() == value
