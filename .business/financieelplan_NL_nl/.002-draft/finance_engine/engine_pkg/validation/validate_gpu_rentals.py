from __future__ import annotations

from pathlib import Path

import pandas as pd

from .shared import FileReport, is_number
from ...io.loader import read_csv


# Optional pandera schema validation (new shape)
try:
    import pandera as pa
    from pandera import Column, Check

    schema = pa.DataFrameSchema({
        "gpu": Column(str, nullable=False),
        "vram_gb": Column(object, checks=Check(lambda x: x.apply(lambda v: is_number(v) and float(v) >= 0))),
        "provider": Column(str, nullable=False),
        "usd_hr": Column(object, checks=Check(lambda x: x.apply(lambda v: is_number(v) and float(v) > 0))),
    })
except Exception:  # pandera not installed
    schema = None  # type: ignore


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "gpu_rentals.csv"
    fr = FileReport(name="gpu_rentals.csv", ok=True)
    try:
        df = read_csv(p)
        req = ["gpu", "vram_gb", "provider", "usd_hr"]
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
        # manual checks complement
        if df["gpu"].isna().any() or (df["gpu"].astype(str).str.strip() == "").any():
            fr.ok = False
            fr.errors.append("gpu column has empty values")
        if df["provider"].isna().any() or (df["provider"].astype(str).str.strip() == "").any():
            fr.ok = False
            fr.errors.append("provider column has empty values")
        # Ensure usd_hr numeric and > 0
        usd = pd.to_numeric(df["usd_hr"], errors="coerce")
        if usd.isna().any() or (usd <= 0).any():
            fr.ok = False
            fr.errors.append("usd_hr must be numeric and > 0 for all rows")
        fr.count = len(df)
        fr.info["gpus"] = sorted(set(df["gpu"].astype(str)))
        fr.info["providers"] = sorted(set(df["provider"].astype(str)))
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
