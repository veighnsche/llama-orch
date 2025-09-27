from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dict. Returns empty dict if file is empty.
    Raises FileNotFoundError if the file does not exist.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file using pandas.
    Caller is responsible for column validations.
    """
    return pd.read_csv(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
