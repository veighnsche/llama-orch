from __future__ import annotations

import shutil
from pathlib import Path
import pytest

from finance_engine.config import OUTPUTS


@pytest.fixture(autouse=True)
def clean_outputs_before_tests():
    # Clean outputs directory before each test session
    if OUTPUTS.exists():
        for p in OUTPUTS.iterdir():
            if p.is_file():
                p.unlink(missing_ok=True)
            else:
                shutil.rmtree(p, ignore_errors=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    yield
