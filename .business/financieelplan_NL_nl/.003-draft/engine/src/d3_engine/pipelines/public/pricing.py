"""Public pricing (scaffold)."""

def round_to_increment(value: float, inc: float) -> float:
    if inc <= 0:
        return value
    return round(value / inc) * inc
