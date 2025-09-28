"""Private provider costs (scaffold)."""

def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    mid = n // 2
    return (s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2)
