from __future__ import annotations

import re


def canonicalize_model_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s).lower()
