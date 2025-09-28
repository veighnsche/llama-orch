"""TPS heuristics (scaffold)."""

def estimate_tps(model: str, gpu: str, vram_gb: float) -> float:
    # naive placeholder
    base = 1000.0
    if "H100" in gpu:
        base *= 3.0
    elif "A100" in gpu:
        base *= 2.0
    elif "L4" in gpu or "L40" in gpu:
        base *= 1.2
    return base * max(vram_gb, 1.0) / 40.0
