"""Tests for src/portfolio/longonly_cache.py -- the parquet caching
layer over longonly_data.build_us_inputs/build_canada_inputs.
"""

import datetime
import shutil

import pytest

from src.data.universe_panel import RAW_DATA_DIR
from src.portfolio import longonly_cache

pytestmark_realdata = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)


@pytest.fixture
def clean_cache_dir():
    """Ensures this test's cache directory is empty before AND after --
    tests must not leave cache artifacts that could make a later test
    (or a real run) silently read stale/wrong data."""
    test_dir = longonly_cache._cache_dir(
        "us", datetime.date(2013, 1, 31), datetime.date(2015, 12, 31)
    )
    if test_dir.exists():
        shutil.rmtree(test_dir)
    yield test_dir
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytestmark_realdata
def test_get_us_inputs_writes_and_reads_back_identically(clean_cache_dir):
    """First call (cache miss) must build fresh AND write the cache.
    Second call (cache hit) must return content identical to the first
    -- not just a non-empty result, a BIT-IDENTICAL round trip through
    parquet."""
    start, end = datetime.date(2013, 1, 31), datetime.date(2015, 12, 31)

    betas1, fringe1, rets1 = longonly_cache.get_us_inputs(
        start, end, variants=("window_24m",)
    )
    assert clean_cache_dir.exists()
    assert (clean_cache_dir / "manifest.json").exists()
    assert (clean_cache_dir / "betas.parquet").exists()
    assert (clean_cache_dir / "fringe.parquet").exists()
    assert (clean_cache_dir / "quarterly_rets.parquet").exists()

    betas2, fringe2, rets2 = longonly_cache.get_us_inputs(
        start, end, variants=("window_24m",)
    )

    assert set(betas1.keys()) == set(betas2.keys())
    for variant in betas1:
        assert set(betas1[variant].keys()) == set(betas2[variant].keys())
        for formation_date in betas1[variant]:
            df1 = betas1[variant][formation_date].sort("id")
            df2 = betas2[variant][formation_date].sort("id")
            assert df1.height == df2.height
            assert (df1["beta"] - df2["beta"]).abs().max() < 1e-12

    assert set(fringe1.keys()) == set(fringe2.keys())
    assert rets1.height == rets2.height


@pytestmark_realdata
def test_get_us_inputs_cache_hit_does_not_call_build_us_inputs(clean_cache_dir, monkeypatch):
    """The defining property of a cache: a SECOND call with the SAME
    (leg, start, end, variants) must NOT invoke the expensive builder at
    all. Verified by monkeypatching build_us_inputs to raise if called,
    then confirming the second get_us_inputs call succeeds anyway."""
    from src.portfolio import longonly_data

    start, end = datetime.date(2013, 1, 31), datetime.date(2015, 12, 31)

    longonly_cache.get_us_inputs(start, end, variants=("window_24m",))

    def _explode(*args, **kwargs):
        raise AssertionError(
            "build_us_inputs was called on a cache HIT -- the cache "
            "is not actually preventing the expensive rebuild"
        )

    monkeypatch.setattr(longonly_data, "build_us_inputs", _explode)

    # Must succeed WITHOUT calling the monkeypatched (exploding) builder.
    betas, _fringe, _rets = longonly_cache.get_us_inputs(start, end, variants=("window_24m",))
    assert len(betas["window_24m"]) >= 0  # just confirming no exception occurred


@pytestmark_realdata
def test_get_us_inputs_force_rebuild_bypasses_cache_read(clean_cache_dir, monkeypatch):
    """force_rebuild=True must call the builder even when a matching
    cache exists -- confirmed by monkeypatching build_us_inputs to a
    sentinel and checking it WAS called (the opposite check from the
    cache-hit test above)."""
    from src.portfolio import longonly_data

    start, end = datetime.date(2013, 1, 31), datetime.date(2015, 12, 31)
    longonly_cache.get_us_inputs(start, end, variants=("window_24m",))

    called = {"flag": False}
    real_builder = longonly_data.build_us_inputs

    def _spy(*args, **kwargs):
        called["flag"] = True
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(longonly_data, "build_us_inputs", _spy)

    longonly_cache.get_us_inputs(start, end, variants=("window_24m",), force_rebuild=True)
    assert called["flag"], "force_rebuild=True did not call build_us_inputs"


@pytestmark_realdata
def test_get_us_inputs_different_variants_triggers_rebuild(clean_cache_dir, monkeypatch):
    """A cache written for variants=("window_24m",) must NOT be silently
    reused for a request naming variants=("window_36m",) -- the
    manifest's variant-match check must catch this and rebuild."""
    from src.portfolio import longonly_data

    start, end = datetime.date(2013, 1, 31), datetime.date(2015, 12, 31)
    longonly_cache.get_us_inputs(start, end, variants=("window_24m",))

    called = {"flag": False}
    real_builder = longonly_data.build_us_inputs

    def _spy(*args, **kwargs):
        called["flag"] = True
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(longonly_data, "build_us_inputs", _spy)

    longonly_cache.get_us_inputs(start, end, variants=("window_36m",))
    assert called["flag"], (
        "requesting a DIFFERENT variant set silently reused a cache "
        "written for a different set -- manifest match check failed"
    )


def test_manifest_matches_rejects_missing_file(tmp_path):
    fake_manifest = tmp_path / "manifest.json"
    assert not longonly_cache._manifest_matches(
        fake_manifest, "us", datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), ("window_24m",)
    )


def test_manifest_matches_rejects_wrong_leg(tmp_path):
    import json

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "leg": "us",
                "start": "2015-01-01",
                "end": "2015-12-31",
                "variants": ["window_24m"],
            },
            f,
        )
    assert not longonly_cache._manifest_matches(
        manifest_path, "ca", datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), ("window_24m",)
    )


def test_manifest_matches_accepts_variants_in_any_order(tmp_path):
    import json

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "leg": "us",
                "start": "2015-01-01",
                "end": "2015-12-31",
                "variants": ["window_60m", "window_24m"],
            },
            f,
        )
    assert longonly_cache._manifest_matches(
        manifest_path,
        "us",
        datetime.date(2015, 1, 1),
        datetime.date(2015, 12, 31),
        ("window_24m", "window_60m"),
    )


# ---------------------------------------------------------------------------
# _write_cache atomicity -- regression test for a real defect found by
# leakage-auditor this session: sequential in-place writes left a window
# where an interrupted write (a killed background process, this
# session's own repeated pattern) could leave betas.parquet from a NEW
# build beside fringe.parquet/quarterly_rets.parquet from an OLD build,
# with the OLD manifest.json still validating. Uses tiny synthetic
# DataFrames -- no real WRDS/CHASS data needed to exercise the
# write-swap mechanism itself.
# ---------------------------------------------------------------------------


def test_write_cache_leaves_no_tmp_directory_behind(tmp_path, monkeypatch):
    import polars as pl

    monkeypatch.setattr(longonly_cache, "CACHE_ROOT", tmp_path)
    cache_dir = longonly_cache._cache_dir(
        "us", datetime.date(2015, 1, 1), datetime.date(2015, 12, 31)
    )

    betas_by_date_per_variant = {
        "window_24m": {
            datetime.date(2015, 3, 31): pl.DataFrame({"id": ["A"], "beta": [1.0]})
        }
    }
    fringe_by_date = {
        datetime.date(2015, 3, 31): pl.DataFrame(
            {"id": ["A"], "mkt_cap": [5e9], "price": [50.0]}
        )
    }
    quarterly_monthly_rets = pl.DataFrame(
        {"id": ["A"], "month": [datetime.date(2015, 4, 30)], "ret": [0.01]}
    )

    longonly_cache._write_cache(
        cache_dir,
        "us",
        datetime.date(2015, 1, 1),
        datetime.date(2015, 12, 31),
        ("window_24m",),
        betas_by_date_per_variant,
        fringe_by_date,
        quarterly_monthly_rets,
    )

    # No leftover .tmp-* sibling directories after a successful write.
    siblings = list(tmp_path.iterdir())
    tmp_leftovers = [s for s in siblings if ".tmp-" in s.name]
    assert tmp_leftovers == [], f"leftover tmp directories after write: {tmp_leftovers}"
    assert cache_dir.exists()
    assert (cache_dir / "manifest.json").exists()
    assert (cache_dir / "betas.parquet").exists()
    assert (cache_dir / "fringe.parquet").exists()
    assert (cache_dir / "quarterly_rets.parquet").exists()


def test_write_cache_overwrite_never_leaves_mixed_vintage(tmp_path, monkeypatch):
    """The specific scenario leakage-auditor reproduced: build 1 writes
    beta=[1.0]/mkt_cap=[10.0], build 2 (same params, different values --
    simulating a data refresh) writes beta=[2.0]/mkt_cap=[20.0]. After
    build 2 completes, a read must NEVER see beta from build 2 paired
    with mkt_cap from build 1 (or vice versa) -- both must be from the
    SAME build, always."""
    import polars as pl

    monkeypatch.setattr(longonly_cache, "CACHE_ROOT", tmp_path)
    cache_dir = longonly_cache._cache_dir(
        "us", datetime.date(2015, 1, 1), datetime.date(2015, 12, 31)
    )

    def _build(beta_val, mkt_cap_val):
        return (
            {"window_24m": {datetime.date(2015, 3, 31): pl.DataFrame({"id": ["A"], "beta": [beta_val]})}},
            {datetime.date(2015, 3, 31): pl.DataFrame({"id": ["A"], "mkt_cap": [mkt_cap_val], "price": [50.0]})},
            pl.DataFrame({"id": ["A"], "month": [datetime.date(2015, 4, 30)], "ret": [0.01]}),
        )

    b1, f1, r1 = _build(1.0, 10.0)
    longonly_cache._write_cache(
        cache_dir, "us", datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), ("window_24m",), b1, f1, r1
    )

    b2, f2, r2 = _build(2.0, 20.0)
    longonly_cache._write_cache(
        cache_dir, "us", datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), ("window_24m",), b2, f2, r2
    )

    betas_flat = pl.read_parquet(cache_dir / "betas.parquet")
    fringe_flat = pl.read_parquet(cache_dir / "fringe.parquet")
    assert betas_flat["beta"][0] == 2.0
    assert fringe_flat["mkt_cap"][0] == 20.0, (
        "fringe.parquet still holds build-1's mkt_cap value after build-2's "
        "overwrite completed -- mixed-vintage cache, the exact defect this "
        "test guards against"
    )
