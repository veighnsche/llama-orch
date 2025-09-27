from __future__ import annotations

from pathlib import Path

import pandas as pd

from .shared import FileReport, is_number

try:
    import pandera as pa
    from pandera import Column, Check
    schema = pa.DataFrameSchema({
        "model_name": Column(str, nullable=False),
        "gpu": Column(str, nullable=False),
        "throughput_tokens_per_sec": Column(object, checks=Check(lambda x: x.apply(lambda v: is_number(v) and float(v) >= 0))),
    })
except Exception:
    schema = None  # type: ignore


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "tps_model_gpu.csv"
    fr = FileReport(name="tps_model_gpu.csv", ok=True)
    try:
        df = pd.read_csv(p)
        req = ["model_name", "gpu", "throughput_tokens_per_sec"]
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
        if df["model_name"].isna().any() or (df["model_name"].astype(str).str.strip() == "").any():
            fr.ok = False
            fr.errors.append("model_name column has empty values")
        if df["gpu"].isna().any() or (df["gpu"].astype(str).str.strip() == "").any():
            fr.ok = False
            fr.errors.append("gpu column has empty values")
        fr.count = len(df)
        fr.info["gpus"] = sorted(set(df["gpu"].astype(str)))
        fr.info["models"] = sorted(set(df["model_name"].astype(str)))
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
