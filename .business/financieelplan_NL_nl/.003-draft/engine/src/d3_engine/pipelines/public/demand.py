"""Public demand model (scaffold)."""

def monthly_tokens(new_customers: float, tokens_per_conversion_mean: float) -> float:
    return max(0.0, new_customers) * max(0.0, tokens_per_conversion_mean)
