from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .shared import FileReport

try:
    import pandera as pa
    from pandera import Column, Check
    schema = pa.DataFrameSchema({
        "sku": Column(str, nullable=False),
        "category": Column(str, nullable=False),
        "unit": Column(str, nullable=False),
    })
except Exception:
    schema = None  # type: ignore


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "price_sheet.csv"
    fr = FileReport(name="price_sheet.csv", ok=True)
    try:
        df = pd.read_csv(p)
        req = ["sku", "category", "unit"]
        for c in req:
            if c not in df.columns:
                fr.ok = False
                fr.errors.append(f"missing column: {c}")
        if not fr.ok:
            return fr
        if schema is not None:
            try:
                _ = schema.validate(df, lazy=True)
            except Exception as e:  # pragma: no cover
                fr.ok = False
                fr.errors.append(str(e))
                return fr
        mask_pub = df["category"].astype(str) == "public_tap"
        units: List[str] = sorted(set(df.loc[mask_pub, "unit"].dropna().astype(str)))
        if any(u not in ("1k_tokens", "1M_tokens") for u in units):
            fr.ok = False
            fr.errors.append("unit must be one of {1k_tokens, 1M_tokens} for public_tap rows")
        # Unit price column is optional; pricing is derived from policy+costs and not read from price_sheet
        fr.count = len(df)
        fr.info["units_public"] = units
        fr.info["skus"] = sorted(set(df["sku"].astype(str)))
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
