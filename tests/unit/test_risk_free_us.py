"""Tests for src/data/risk_free_us.py -- US risk-free rate, both paths
specified in config/risk_free.yaml:

- load_us_rf_ken_french(): Ken French monthly RF (percent units, and the
  file's two-table parse trap -- see the module docstring).
- load_us_rf_aqr_gate4(): AQR's own RF sheet, used only for Gate 4.
"""


import datetime

import polars as pl
import pytest

from src.data import risk_free_us

_KEN_FRENCH_FIXTURE = """This file was created using the 202607 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.


,Mkt-RF,SMB,HML,RF
192607,   2.89,  -2.42,  -2.75,   0.22
192608,   2.44,  -1.40,   4.19,   0.25

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
  1927,  29.44,  -3.05,  -3.36,   3.12
  1928,  35.56,   3.73,  -5.26,   3.56
"""


@pytest.fixture
def fixture_csv(tmp_path, monkeypatch):
    path = tmp_path / "ff_factors_fixture.csv"
    path.write_text(_KEN_FRENCH_FIXTURE)
    monkeypatch.setattr(risk_free_us, "KEN_FRENCH_PATH", path)
    monkeypatch.setattr(risk_free_us, "_KEN_FRENCH_HEADER_LINE", 4)
    monkeypatch.setattr(risk_free_us, "_KEN_FRENCH_LAST_LINE", 6)
    return path


def test_load_us_rf_ken_french_stops_before_annual_block(fixture_csv):
    result = risk_free_us.load_us_rf_ken_french()

    # 2 monthly rows only -- the annual block's 1927/1928 "years" must
    # NEVER appear as parsed dates. This is the exact bug the config's
    # parse trap warns about: a naive read_csv silently appends them.
    assert result.height == 2
    assert result["date"].dt.year().to_list() == [1926, 1926]


def test_load_us_rf_ken_french_converts_percent_to_decimal(fixture_csv):
    result = risk_free_us.load_us_rf_ken_french()

    row = result.filter(pl.col("date") == datetime.date(1926, 7, 31))
    assert row.height == 1
    assert row["rf"][0] == pytest.approx(0.0022, rel=1e-9)


def test_load_us_rf_ken_french_raises_if_annual_marker_missing(tmp_path, monkeypatch):
    # File truncated so the blank-line/annual-marker boundary this loader
    # checks for isn't where config says -- must refuse rather than guess.
    bad_fixture = tmp_path / "truncated.csv"
    bad_fixture.write_text(",Mkt-RF,SMB,HML,RF\n192607,   2.89,  -2.42,  -2.75,   0.22\n")
    monkeypatch.setattr(risk_free_us, "KEN_FRENCH_PATH", bad_fixture)
    monkeypatch.setattr(risk_free_us, "_KEN_FRENCH_HEADER_LINE", 4)
    monkeypatch.setattr(risk_free_us, "_KEN_FRENCH_LAST_LINE", 6)

    with pytest.raises(AssertionError, match="Annual Factors"):
        risk_free_us.load_us_rf_ken_french()


def test_load_us_rf_ken_french_real_file_anchors():
    result = risk_free_us.load_us_rf_ken_french()

    assert result["date"].min() == datetime.date(1926, 7, 31)
    # 202607 row's RF is 0.33 (percent) -> 0.0033 (decimal).
    row = result.filter(pl.col("date") == datetime.date(2026, 7, 31))
    assert row.height == 1
    assert row["rf"][0] == pytest.approx(0.0033, rel=1e-9)
    # 1927 has 12 genuine monthly rows. If the annual block's bare-year
    # keys leaked in as parsed dates, this would be 13 (12 months + 1
    # annual row misparsed as e.g. 1927-01-01).
    assert result.filter(pl.col("date").dt.year() == 1927).height == 12
    # Annual block's RF value for 1927 is 3.12 (%) -- if it leaked in as
    # a decimal 0.0312 row, no real month has that exact value.
    assert 0.0312 not in result["rf"].to_list()


def test_load_us_rf_aqr_gate4_real_file_anchors():
    result = risk_free_us.load_us_rf_aqr_gate4()

    assert set(result.columns) == {"date", "rf"}
    row = result.filter(pl.col("date") == datetime.date(1926, 7, 31))
    assert row.height == 1
    # AQR's RF sheet is already decimal (config: units: decimal) -- 0.0022,
    # NOT 0.22 and not 0.000022.
    assert row["rf"][0] == pytest.approx(0.0022, rel=1e-6)


def test_us_rf_ken_french_and_aqr_gate4_are_close_but_not_identical():
    # Config explicitly warns these are "close but NOT identical" -- a
    # loader bug that made them byte-identical (e.g. accidentally reading
    # the same column twice) would be invisible to a naive equality check
    # but is exactly the kind of error this project's bugs have looked
    # like before. Assert they're correlated but not exactly equal on the
    # overlap.
    kf = risk_free_us.load_us_rf_ken_french()
    aqr = risk_free_us.load_us_rf_aqr_gate4()
    joined = kf.join(aqr, on="date", suffix="_aqr")
    assert joined.height > 1000
    diffs = (joined["rf"] - joined["rf_aqr"]).abs()
    assert diffs.max() > 0.0  # not byte-identical
    assert diffs.mean() < 0.001  # but close, as the config states
