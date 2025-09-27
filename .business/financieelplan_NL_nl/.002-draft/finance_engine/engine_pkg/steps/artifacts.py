from __future__ import annotations

from typing import Any, Dict
import math
import pandas as pd

from ...config import OUTPUTS, ENGINE_VERSION
from ...io.writer import write_csv, write_yaml, write_text
from ...utils.vat import vat_set_aside_rows
from ...utils.coerce import get_float
from ...types.inputs import Config


def write_artifacts(*, config: Config, agg: Dict[str, Any]) -> None:
    # Core CSV artifacts
    write_csv(OUTPUTS / "model_price_per_1m_tokens.csv", agg["model_df"]) 
    write_csv(OUTPUTS / "public_tap_scenarios.csv", agg["public_df"]) 
    write_csv(OUTPUTS / "private_tap_economics.csv", agg["private_df"]) 

    be_df = pd.DataFrame([
        {
            k: (round(v, 2) if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in agg["break_even"].items()
        }
    ])
    write_csv(OUTPUTS / "break_even_targets.csv", be_df)

    write_csv(OUTPUTS / "loan_schedule.csv", agg["loan_df"]) 

    # 24-month timeseries artifacts
    if agg.get("ts_public_df") is not None:
        write_csv(OUTPUTS / "public_timeseries_24m.csv", agg["ts_public_df"]) 
    if agg.get("ts_private_df") is not None:
        write_csv(OUTPUTS / "private_timeseries_24m.csv", agg["ts_private_df"]) 
    if agg.get("ts_total_df") is not None:
        write_csv(OUTPUTS / "total_timeseries_24m.csv", agg["ts_total_df"]) 

    # Curated profitability outputs
    curated = agg.get("pub_curated_df")
    curated = curated if isinstance(curated, type(agg.get("pub_df"))) else agg.get("pub_df")
    if curated is not None:
        write_csv(OUTPUTS / "model_profitability.csv", curated)
        # Simple curated catalog as Markdown
        pol = agg.get("policy_public_tap", {})
        min_gm = pol.get("min_gross_margin_pct")
        lines = [
            "# Curated Public Tap Catalog",
            "",
            f"Minimum gross margin threshold (worst-case): {min_gm}%" if isinstance(min_gm, (int, float)) else "",
            "",
            "| Model | GPU | TPS | Sell €/1M | Gross Margin % (min) | Gross Margin % (med) |",
            "|-------|-----|-----|-----------:|----------------------:|----------------------:|",
        ]
        try:
            df = curated.copy()
            df = df.sort_values(by="gross_margin_pct_med", ascending=False)
            for _, r in df.iterrows():
                lines.append(
                    f"| {r.get('model')} | {r.get('gpu')} | {round(float(r.get('tps', 0.0)),2)} | "
                    f"{round(float(r.get('sell_per_1m_eur', 0.0)),2)} | {round(float(r.get('gross_margin_pct_min', 0.0)),2)}% | {round(float(r.get('gross_margin_pct_med', 0.0)),2)}% |"
                )
        except Exception:
            pass
        write_text(OUTPUTS / "curated_catalog.md", "\n".join([ln for ln in lines if ln is not None]) + "\n")

    # VAT set-aside examples
    vat_rate = get_float(config, ["tax_billing", "vat_standard_rate_pct"], 21.0)
    write_csv(OUTPUTS / "vat_set_aside.csv", pd.DataFrame(vat_set_aside_rows(vat_rate)))

    # Assumptions YAML for tests to sanity-check
    eur_usd = get_float(config, ["fx", "eur_usd_rate"], 1.08)
    default_markup = get_float(config, ["pricing_inputs", "private_tap_default_markup_over_provider_cost_pct"], 50.0)
    assumptions = {
        "engine_version": ENGINE_VERSION,
        "fx": {"eur_usd_rate": eur_usd},
        "private": {"default_markup_over_cost_pct": default_markup},
    }
    write_yaml(OUTPUTS / "assumptions.yaml", assumptions)

    # Optional: acquisition funnel base snapshot (if funnel driver)
    funnel = agg.get("funnel_base")
    if isinstance(funnel, dict) and funnel:
        try:
            write_csv(OUTPUTS / "acquisition_funnel_base.csv", pd.DataFrame([funnel]))
        except Exception:
            pass
    # Optional: unit economics snapshot
    unit = agg.get("unit_economics")
    if isinstance(unit, dict) and unit:
        try:
            write_csv(OUTPUTS / "unit_economics.csv", pd.DataFrame([unit]))
        except Exception:
            pass

    # Optional: competitor caps applied view
    try:
        md = agg.get("model_df")
        if md is not None and "competitor_cap_applied" in md.columns:
            caps = md[["model", "sell_per_1m_eur", "competitor_cap_applied"]].copy()
            write_csv(OUTPUTS / "competitor_caps_applied.csv", caps)
    except Exception:
        pass
