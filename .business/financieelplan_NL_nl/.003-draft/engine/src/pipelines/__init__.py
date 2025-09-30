"""Pipelines registry to resolve pipeline names to compute_rows callables."""
from __future__ import annotations
from typing import Callable, Dict

from .public.artifacts import compute_rows as public_compute_rows
from .private.artifacts import compute_rows as private_compute_rows

REGISTRY: Dict[str, Callable[[dict], dict]] = {
    "public": public_compute_rows,
    "private": private_compute_rows,
}
