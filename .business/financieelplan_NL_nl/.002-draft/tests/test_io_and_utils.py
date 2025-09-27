from __future__ import annotations

from pathlib import Path
import pandas as pd

from finance_engine.io.loader import load_yaml
from finance_engine.utils.markdown import df_to_markdown_rows


def test_load_yaml_empty(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    data = load_yaml(p)
    assert data == {}


def test_df_to_markdown_rows_float_formatting():
    df = pd.DataFrame({"a": [1.2345, 2.0], "b": ["x", "y"]})
    body = df_to_markdown_rows(df, ["a", "b"])
    lines = body.splitlines()
    assert lines[0] == "| 1.23 | x |"
    assert lines[1] == "| 2.00 | y |"
