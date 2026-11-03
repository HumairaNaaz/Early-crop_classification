import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def fake_cache(tmp_path_factory):
    """Synthetic CatCrops -> preprocessed cache, shared by all tests."""
    from data import preprocess as P
    from tests.synthetic import make_fake_catcrops

    tmp = tmp_path_factory.mktemp("fake_catcrops")
    make_fake_catcrops(tmp, fields_per_class=10)
    out = tmp / "cache"
    P.main(["--root", str(tmp), "--out", str(out), "--workers", "2"])
    return out
