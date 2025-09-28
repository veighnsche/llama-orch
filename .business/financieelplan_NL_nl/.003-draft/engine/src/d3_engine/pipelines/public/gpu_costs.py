"""GPU cost calculations (scaffold)."""

def cost_eur_per_1m(eur_hr: float, tokens_per_hour: float) -> float:
    if tokens_per_hour <= 0:
        return float("inf")
    return eur_hr / (tokens_per_hour / 1_000_000.0)
