from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .shared import FileReport, require_keys, is_number
from ....io.loader import load_yaml

try:
    from pydantic import BaseModel, Field, ValidationError

    class RepaymentModel(BaseModel):
        term_months: int = Field(gt=0)
        interest_rate_pct: float = Field(ge=0)

    class LoanRequestModel(BaseModel):
        amount_eur: float = Field(gt=0)

    class LendingModel(BaseModel):
        loan_request: LoanRequestModel
        repayment_plan: RepaymentModel

    def _validate_with_pydantic(obj: Dict[str, Any], fr: FileReport) -> None:
        try:
            _ = LendingModel.model_validate(obj)  # type: ignore[attr-defined]
        except Exception:
            try:
                _ = LendingModel.parse_obj(obj)  # type: ignore[attr-defined]
            except ValidationError as ve:
                fr.ok = False
                fr.errors.extend([str(err) for err in ve.errors()])
except Exception:  # pydantic not installed
    def _validate_with_pydantic(obj: Dict[str, Any], fr: FileReport) -> None:  # type: ignore
        return


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "lending_plan.yaml"
    fr = FileReport(name="lending_plan.yaml", ok=True)
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        ok_amt, amount = require_keys(obj, ["loan_request", "amount_eur"])
        ok_term, term = require_keys(obj, ["repayment_plan", "term_months"])
        ok_rate, rate = require_keys(obj, ["repayment_plan", "interest_rate_pct"])
        if not ok_amt or not is_number(amount) or float(amount) <= 0:
            fr.ok = False
            fr.errors.append("loan_request.amount_eur: required > 0")
        if not ok_term or not is_number(term) or float(term) <= 0:
            fr.ok = False
            fr.errors.append("repayment_plan.term_months: required > 0")
        if not ok_rate or not is_number(rate) or float(rate) < 0:
            fr.ok = False
            fr.errors.append("repayment_plan.interest_rate_pct: required ≥ 0")
        fr.count = len(obj)
        _validate_with_pydantic(obj, fr)
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
