from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class RenderPort(Protocol):
    def __call__(self, template_file: Path, out_path: Path, context: Dict[str, Any]) -> None: ...


@runtime_checkable
class ValidatePort(Protocol):
    def __call__(
        self,
        *,
        config: Dict[str, Any],
        costs: Dict[str, Any],
        lending: Dict[str, Any],
        price_sheet: pd.DataFrame,
    ) -> Dict[str, Any]: ...


def get_default_renderer() -> RenderPort:
    # Local import to avoid import cycles at module import time
    from ..render.template import render_template_jinja

    return render_template_jinja


def get_default_validator() -> ValidatePort:
    # Local import to avoid import cycles at module import time
    from ..validation.validate import validate_inputs

    return validate_inputs
