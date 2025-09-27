from __future__ import annotations

from typing import Any, Dict
import math
import pandas as pd

from ...config import OUTPUTS, ENGINE_VERSION
from ...io.writer import write_csv, write_yaml
from ...utils.vat import vat_set_aside_rows


def write_artifacts(*, config: Dict[str, Any], agg: Dict[str, Any]) -> None:
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

    # VAT set-aside examples
    vat_rate = float(config.get("tax_billing", {}).get("vat_standard_rate_pct", 21.0))
    write_csv(OUTPUTS / "vat_set_aside.csv", pd.DataFrame(vat_set_aside_rows(vat_rate)))

    # Assumptions YAML for tests to sanity-check
    eur_usd = float(config.get("fx", {}).get("eur_usd_rate", 1.08))
    raw_markup = config.get("pricing_inputs", {}).get("private_tap_default_markup_over_provider_cost_pct")
    default_markup = float(raw_markup if isinstance(raw_markup, (int, float, str)) else 50.0)
    assumptions = {
        "engine_version": ENGINE_VERSION,
        "fx": {"eur_usd_rate": eur_usd},
        "private": {"default_markup_over_cost_pct": default_markup},
    }
    write_yaml(OUTPUTS / "assumptions.yaml", assumptions)
