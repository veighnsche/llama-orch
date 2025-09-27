from __future__ import annotations

from typing import Iterable, List
import pandas as pd


def df_to_markdown_rows(df: pd.DataFrame, columns: Iterable[str]) -> str:
    """Return markdown table body rows (no header) for the given df and column order."""
    out_lines: List[str] = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                # Format floats nicely
                cells.append(f"{val:.2f}")
            else:
                cells.append(str(val))
        out_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(out_lines)
