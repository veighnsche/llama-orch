from __future__ import annotations

# Shim module for legacy import path in tests:
# finance_engine.engine_pkg.validation.preflight.validate_gpu_pricing
# Re-export the validate() function from the canonical module.
from ..validate_gpu_pricing import validate  # noqa: F401
