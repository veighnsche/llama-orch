"""Private pricing (scaffold)."""

def sell_eur_hr(provider_eur_hr_med: float, markup_pct: float) -> float:
    return provider_eur_hr_med * (1.0 + markup_pct / 100.0)
