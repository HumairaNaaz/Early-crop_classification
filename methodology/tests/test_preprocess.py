"""Data pipeline tests (no torch needed)."""
import numpy as np
import pandas as pd
import pytest

from data import preprocess as P
from data.cache import YearCache, compute_stats, make_split, N_BANDS


@pytest.fixture(scope="module")
def cache_dir(fake_cache):  # built once in conftest.py
    return fake_cache


def test_eighteen_classes_and_exclusions(cache_dir):
    c = YearCache.load(cache_dir, 2022)
    assert len(c.meta["class_names"]) == 18
    assert set(np.unique(c.label)) == set(range(18))
    assert c.meta["fields_after_class_filter"] == 180  # 20 classes x 10, minus 2 excluded classes


def test_previous_year_kept_with_negative_days(cache_dir):
    c = YearCache.load(cache_dir, 2022)
    days, vals = c.field_obs(0)
    assert days.min() < 0 and days.max() >= 0, "both years must be present"
    assert days.min() >= -366 and days.max() <= 365
    assert np.all(np.diff(days) > 0), "one observation per date, sorted"
    assert vals.shape[1] == 11


def test_duplicate_tiles_merged(cache_dir, tmp_path):
    df = pd.DataFrame({
        **{b: [1000.0, 3000.0] for b in P.BANDS},
        "CP": [0.0, 80.0], "doa": ["2022-03-01", "2022-03-01"],
    })
    days, vm, vc = P.aggregate_field(df, 2022)
    assert len(days) == 1 and days[0] == 59
    assert np.isclose(vm[0, 0], 2000.0)          # plain mean
    assert np.isclose(vc[0, 0], 1000.0)          # cloudy view dropped when a clear one exists


def test_rows_outside_two_years_dropped():
    df = pd.DataFrame({**{b: [1.0, 2.0, 3.0] for b in P.BANDS}, "CP": [0, 0, 0],
                       "doa": ["2020-06-01", "2021-06-01", "2023-06-01"]})
    days, _, _ = P.aggregate_field(df, 2022)
    assert list(days) == [(pd.Timestamp("2021-06-01") - pd.Timestamp("2022-01-01")).days]


def test_prev_labels_and_rotation_groups(cache_dir):
    c = YearCache.load(cache_dir, 2023)
    g = c.rotation_group()
    assert set(g) <= {"same", "changed", "unknown"}
    assert (g == "same").sum() > 0 and (g == "changed").sum() > 0
    assert np.all(c.prev_label >= -1) and np.all(c.prev_label < 18)


def test_split_and_stats(cache_dir):
    c = YearCache.load(cache_dir, 2022)
    tr, va = make_split(c)
    assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(c)
    assert abs(len(va) - 0.1 * len(c)) <= 1
    s = compute_stats(c, tr, 18)
    assert s["profiles"].shape == (18, 74, N_BANDS)
    assert np.isfinite(s["profiles"]).all()
    assert sum(s["class_counts"]) == len(tr)
    assert 0 not in s["mun_vocab"].values()      # 0 reserved for unknown


def test_reading_from_zip_archive_matches_folder(tmp_path):
    """The downloaded archive can be preprocessed without unpacking it, with identical output."""
    import shutil
    import zipfile

    from tests.synthetic import make_fake_catcrops

    src = tmp_path / "src"
    make_fake_catcrops(src, fields_per_class=3)
    root = P.find_dataset_root(src)
    archive = tmp_path / "1.archive"  # kagglehub's name for the zip download
    shutil.make_archive(str(tmp_path / "ds"), "zip", root_dir=src)
    (tmp_path / "ds.zip").rename(archive)
    assert P.find_dataset_root(archive).startswith(P.ZIP_TAG)
    with zipfile.ZipFile(archive) as z:
        assert any(n.endswith("2022/L2A/lleida.csv") for n in z.namelist())

    P.main(["--root", str(src), "--out", str(tmp_path / "from_dir"), "--workers", "2"])
    P.main(["--root", str(archive), "--out", str(tmp_path / "from_zip"), "--workers", "2"])
    for year in (2022, 2023):
        a = np.load(tmp_path / "from_dir" / f"{year}.npz")
        b = np.load(tmp_path / "from_zip" / f"{year}.npz")
        assert set(a.files) == set(b.files)
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
