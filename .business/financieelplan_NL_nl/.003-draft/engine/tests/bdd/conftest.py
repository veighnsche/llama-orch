import os
import sys
from pathlib import Path
import pytest


def _find_draft_dir(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if p.name == '.003-draft':
            return p
    # Fallback: assume tests/ is under engine/
    return start.resolve().parents[2]


@pytest.fixture(scope='session')
def draft_dir() -> Path:
    return _find_draft_dir(Path(__file__))


@pytest.fixture(scope='session')
def engine_src(draft_dir: Path) -> Path:
    return draft_dir / 'engine' / 'src'


@pytest.fixture(scope='session', autouse=True)
def add_engine_src_to_path(engine_src: Path):
    sys.path.insert(0, str(engine_src))
    os.environ['PYTHONPATH'] = f"{engine_src}:{os.environ.get('PYTHONPATH','')}"


@pytest.fixture()
def ctx():
    return {}


# Note: avoid plugin-specific hooks here to allow running without pytest-bdd plugin preloaded
