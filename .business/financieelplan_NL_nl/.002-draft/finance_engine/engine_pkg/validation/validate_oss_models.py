from __future__ import annotations

from pathlib import Path

import pandas as pd

from .shared import FileReport


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "oss_models.csv"
    fr = FileReport(name="oss_models.csv", ok=True)
    try:
        df = pd.read_csv(p)
        if "name" not in df.columns:
            fr.ok = False
            fr.errors.append("missing column: name")
            return fr
        if df["name"].isna().any() or (df["name"].astype(str).str.strip() == "").any():
            fr.ok = False
            fr.errors.append("name column has empty values")
        # Any spec column present?
        if not any(c in df.columns for c in ("variant_size_b", "context_tokens", "license")):
            fr.ok = False
            fr.errors.append("no spec columns present (variant_size_b/context_tokens/license)")
        fr.count = len(df)
        # Generate pseudo SKUs for coverage heuristics
        if "variant_size_b" in df.columns:
            def mk_sku(row: pd.Series) -> str:
                try:
                    size = row.get("variant_size_b")
                    if pd.notna(size):
                        try:
                            f = float(size)
                            if f.is_integer():
                                size_str = f"{int(f)}B"
                            else:
                                size_str = f"{f}B"
                        except Exception:
                            size_str = str(size)
                    else:
                        size_str = ""
                    name = str(row.get("name", "")).strip().replace(" ", "-").replace(".", "-")
                    return f"{name}-{size_str}" if size_str else name
                except Exception:
                    return str(row.get("name")).strip()
            fr.info["pseudo_skus"] = sorted(set(df.apply(mk_sku, axis=1)))
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
