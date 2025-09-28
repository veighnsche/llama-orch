"""FX helpers (scaffold)."""

def eur_per_usd(usd: float, eur_usd_fx_rate: float, fx_buffer_pct: float) -> float:
    return usd / max(eur_usd_fx_rate, 1e-9) * (1.0 + fx_buffer_pct / 100.0)
