from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .shared import FileReport, non_empty_string, is_number, require_keys
from ....io.loader import load_yaml


# Optional pydantic-based validation if available
try:
    from pydantic import BaseModel, Field, ValidationError
    from typing import Optional

    class FinanceModel(BaseModel):
        marketing_allocation_pct_of_inflow: Optional[float] = Field(default=None, ge=0, le=100)

    class TaxBillingModel(BaseModel):
        vat_standard_rate_pct: Optional[float] = Field(default=None, ge=0, le=100)

    class RefundsModel(BaseModel):
        allowed: Optional[bool] = None

    class LegalPolicyModel(BaseModel):
        refunds: Optional[RefundsModel] = None

    class ConfigModel(BaseModel):
        currency: str
        finance: Optional[FinanceModel] = None
        tax_billing: Optional[TaxBillingModel] = None
        legal_policy: Optional[LegalPolicyModel] = None

    def _validate_with_pydantic(obj: Dict[str, Any], fr: FileReport) -> None:
        try:
            _ = ConfigModel.model_validate(obj)  # pydantic v2
        except Exception as e:  # pragma: no cover - version differences
            try:
                _ = ConfigModel.parse_obj(obj)  # pydantic v1
            except ValidationError as ve:
                fr.ok = False
                fr.errors.extend([str(err) for err in ve.errors()])
                return
        # No additional errors
        return
except Exception:  # pydantic not installed
    def _validate_with_pydantic(obj: Dict[str, Any], fr: FileReport) -> None:  # type: ignore
        return


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "config.yaml"
    fr = FileReport(name="config.yaml", ok=True)
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        # currency
        if not non_empty_string(obj.get("currency")):
            fr.ok = False
            fr.errors.append("currency: required non-empty string")
        # finance.marketing_allocation_pct_of_inflow (0-100)
        ok, v = require_keys(obj, ["finance", "marketing_allocation_pct_of_inflow"])
        if ok and v is not None:
            if not is_number(v) or not (0 <= float(v) <= 100):
                fr.ok = False
                fr.errors.append("finance.marketing_allocation_pct_of_inflow: must be numeric 0–100 (percent)")
        # tax_billing.vat_standard_rate_pct (0-100)
        ok, v = require_keys(obj, ["tax_billing", "vat_standard_rate_pct"])
        if ok and v is not None:
            if not is_number(v) or not (0 <= float(v) <= 100):
                fr.ok = False
                fr.errors.append("tax_billing.vat_standard_rate_pct: must be numeric 0–100 (percent)")
        # legal_policy.refunds.allowed (bool) if present
        ok, v = require_keys(obj, ["legal_policy", "refunds", "allowed"])
        if ok and v is not None and not isinstance(v, bool):
            fr.ok = False
            fr.errors.append("legal_policy.refunds.allowed: must be boolean if present")
        fr.count = len(obj)
        # Optional schema validation
        _validate_with_pydantic(obj, fr)
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
