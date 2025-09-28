"""Private clients model (scaffold)."""

def expected_new_clients(budget_eur: float, cac_eur: float) -> float:
    if cac_eur <= 0:
        return 0.0
    return budget_eur / cac_eur
