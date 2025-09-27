from __future__ import annotations

from typing import Any, Dict, List


def validate_inputs(*, config: Dict[str, Any], costs: Dict[str, Any], lending: Dict[str, Any], price_sheet) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    # Required config keys
    required_paths = [
        ("currency",),
        ("pricing_inputs", "fx_buffer_pct"),
        ("tax_billing", "vat_standard_rate_pct"),
        ("finance", "marketing_allocation_pct_of_inflow"),
    ]
    for path in required_paths:
        node = config
        for k in path:
            if not isinstance(node, dict) or k not in node:
                errors.append(f"config.yaml missing key: {'/'.join(path)}")
                break
            node = node[k]

    # Business fixed cost can be null → warn
    try:
        business = config.get("finance", {}).get("fixed_costs_monthly_eur", {}).get("business")
        if business is None:
            warnings.append("config.finance.fixed_costs_monthly_eur.business is null; will fall back to extra.business_fixed_costs.business or 0.")
    except Exception:
        warnings.append("config.finance.fixed_costs_monthly_eur not fully specified.")

    # price_sheet columns minimal check
    required_cols = [
        "sku",
        "category",
        "unit",
        "unit_price_eur_per_1k_tokens",
    ]
    for col in required_cols:
        if col not in getattr(price_sheet, 'columns', []):
            errors.append(f"price_sheet.csv missing column '{col}'")

    return {"errors": errors, "warnings": warnings, "info": info}
