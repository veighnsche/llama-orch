from __future__ import annotations
from pathlib import Path
from typing import Iterable
from .io_utils import write_rows


def write_pnl(out_dir: Path, pub_rev_m, prv_rev_m, pub_cost_m, prv_cost_m, opex_fixed_m, dep_m, ebitda_m, interest_m, ebt_m, tax_m, net_income_m) -> str:
    header = [
        "month","revenue_public","revenue_private","cogs_public","cogs_private","opex_fixed","depreciation","EBITDA","interest","EBT","tax","NetIncome",
    ]
    rows = (
        [str(i), f"{pub_rev_m[i]}", f"{prv_rev_m[i]}", f"{pub_cost_m[i]}", f"{prv_cost_m[i]}", f"{opex_fixed_m[i]}", f"{dep_m[i]}", f"{ebitda_m[i]}", f"{interest_m[i]}", f"{ebt_m[i]}", f"{tax_m[i]}", f"{net_income_m[i]}"]
        for i in range(len(ebitda_m))
    )
    return write_rows(out_dir / "pnl_by_month.csv", header, rows)


def write_cashflow(out_dir: Path, starting_cash_m, cash_from_ops_m, wc_delta_m, vat_cash_m, capex_m, debt_service_interest_m, debt_service_principal_m, ending_cash_m) -> str:
    header = [
        "month","starting_cash","cash_from_ops","working_capital_delta","vat_cash","capex","debt_service_interest","debt_service_principal","ending_cash"
    ]
    rows = (
        [str(i), f"{starting_cash_m[i]}", f"{cash_from_ops_m[i]}", f"{wc_delta_m[i]}", f"{vat_cash_m[i]}", f"{capex_m[i]}", f"{debt_service_interest_m[i]}", f"{debt_service_principal_m[i]}", f"{ending_cash_m[i]}"]
        for i in range(len(ending_cash_m))
    )
    return write_rows(out_dir / "cashflow_by_month.csv", header, rows)
