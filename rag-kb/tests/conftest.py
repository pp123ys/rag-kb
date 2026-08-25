import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from ragkb.config import Settings


@pytest.fixture
def settings():
    return Settings()
