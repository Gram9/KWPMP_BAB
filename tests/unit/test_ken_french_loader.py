"""Tests for src/data/ken_french_loader.py -- Ken French data library
CSV parsing for Gate 2 (docs/superpowers/specs/2026-09-08-gate2-size-deciles-design.md).

Uses small synthetic fixture files replicating each real file's exact
stacked-table layout (header block + one data block + a second block
that must be ignored) rather than the real multi-thousand-row files --
this test is about the PARSING LOGIC, not about the real data's values.
"""

import datetime

import polars as pl
import pytest

from src.data import ken_french_loader

_DAILY_FIXTURE = """This file was created using the 202607 CRSP database.  It contains
value- and equal-weighted returns for size portfolios.  Each record contains returns for:

Negative (not used)  30%  40%  30%     5 Quintiles    10 Deciles

The portfolios are constructed at the end of Jun.  The annual returns are from January
to December.

Missing data are indicated by -99.99 or -999.


  Average Value Weighted Returns -- Daily
,<= 0,Lo 30,Med 40,Hi 30,Lo 20,Qnt 2,Qnt 3,Qnt 4,Hi 20,Lo 10,Dec 2,Dec 3,Dec 4,Dec 5,Dec 6,Dec 7,Dec 8,Dec 9,Hi 10
20150102,  -99.99,   -0.54,   -0.39,   -0.07,   -0.47,   -0.62,   -0.47,   -0.09,   -0.07,   -0.15,   -0.77,   -0.62,   -0.63,   -0.62,   -0.37,   -0.18,   -0.03,    0.00,   -0.08
20150105,  -999,   -1.00,   -1.00,   -1.00,   -1.00,   -1.00,   -1.00,   -1.00,   -1.00,   -999,   -999,   -999,   -999,   -999,   -999,   -999,   -999,   -999,   -999

  Average Equal Weighted Returns -- Daily
,<= 0,Lo 30,Med 40,Hi 30,Lo 20,Qnt 2,Qnt 3,Qnt 4,Hi 20,Lo 10,Dec 2,Dec 3,Dec 4,Dec 5,Dec 6,Dec 7,Dec 8,Dec 9,Hi 10
20150102,  -99.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99,    9.99

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_load_size_decile_daily_returns_parses_vw_block_only(tmp_path, monkeypatch):
    fixture_path = tmp_path / "Portfolios_Formed_on_ME_daily.csv"
    fixture_path.write_text(_DAILY_FIXTURE)
    monkeypatch.setattr(ken_french_loader, "DAILY_RETURNS_PATH", fixture_path)

    result = ken_french_loader.load_size_decile_daily_returns()

    assert set(result.columns) == {"date", "decile", "ret"}
    assert result.schema["date"] == pl.Date
    assert result.schema["decile"] == pl.Int8
    assert result.schema["ret"] == pl.Float64
    # 2 real trading days x 10 deciles -- and NOT the EW block's 9.99 sentinel row.
    assert result.height == 20
    assert result["ret"].max() < 1.0  # EW block's 9.99 would fail this if leaked through


def test_load_size_decile_daily_returns_converts_percent_to_decimal(tmp_path, monkeypatch):
    fixture_path = tmp_path / "Portfolios_Formed_on_ME_daily.csv"
    fixture_path.write_text(_DAILY_FIXTURE)
    monkeypatch.setattr(ken_french_loader, "DAILY_RETURNS_PATH", fixture_path)

    result = ken_french_loader.load_size_decile_daily_returns()

    # "Lo 10" column on 20150102 is -0.15 (percent) -> -0.0015 (decimal), decile 1.
    row = result.filter(
        (pl.col("date") == pl.date(2015, 1, 2)) & (pl.col("decile") == 1)
    )
    assert row.height == 1
    assert row["ret"][0] == pytest.approx(-0.0015, rel=1e-9)

    # "Hi 10" column on 20150102 is -0.08 (percent) -> -0.0008 (decimal), decile 10.
    row = result.filter(
        (pl.col("date") == pl.date(2015, 1, 2)) & (pl.col("decile") == 10)
    )
    assert row.height == 1
    assert row["ret"][0] == pytest.approx(-0.0008, rel=1e-9)


def test_load_size_decile_daily_returns_maps_missing_codes_to_null(tmp_path, monkeypatch):
    fixture_path = tmp_path / "Portfolios_Formed_on_ME_daily.csv"
    fixture_path.write_text(_DAILY_FIXTURE)
    monkeypatch.setattr(ken_french_loader, "DAILY_RETURNS_PATH", fixture_path)

    result = ken_french_loader.load_size_decile_daily_returns()

    # 20150105's row uses the -999 sentinel on every decile column.
    day = result.filter(pl.col("date") == pl.date(2015, 1, 5))
    assert day.height == 10
    assert day["ret"].null_count() == 10


def test_load_size_decile_daily_returns_raises_if_header_not_found(tmp_path, monkeypatch):
    fixture_path = tmp_path / "Portfolios_Formed_on_ME_daily.csv"
    fixture_path.write_text("not the expected file format at all\n")
    monkeypatch.setattr(ken_french_loader, "DAILY_RETURNS_PATH", fixture_path)

    with pytest.raises(AssertionError, match="Average Value Weighted Returns -- Daily"):
        ken_french_loader.load_size_decile_daily_returns()


_BREAKPOINTS_FIXTURE = """This file was created using the 202607 CRSP database.  It contains every 5th NYSE ME percentile (divided by 1000000).

201505,   500,        5.00,       10.00,       15.00,       20.00,       25.00,       30.00,       35.00,       40.00,       45.00,       50.00,       55.00,       60.00,       65.00,       70.00,       75.00,       80.00,       85.00,       90.00,       95.00,      100.00
201506,   505,        6.00,       12.00,       18.00,       24.00,       30.00,       36.00,       42.00,       48.00,       54.00,       60.00,       66.00,       72.00,       78.00,       84.00,       90.00,       96.00,      102.00,      108.00,      114.00,      120.00
201606,   510,        7.00,       14.00,       21.00,       28.00,       35.00,       42.00,       49.00,       56.00,       63.00,       70.00,       77.00,       84.00,       91.00,       98.00,      105.00,      112.00,      119.00,      126.00,      133.00,      140.00

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_load_nyse_breakpoints_parses_all_rows(tmp_path, monkeypatch):
    fixture_path = tmp_path / "ME_Breakpoints.csv"
    fixture_path.write_text(_BREAKPOINTS_FIXTURE)
    monkeypatch.setattr(ken_french_loader, "BREAKPOINTS_PATH", fixture_path)

    result = ken_french_loader.load_nyse_breakpoints()

    expected_cols = {
        "month_end", "n_firms",
        "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90",
    }
    assert set(result.columns) == expected_cols
    assert result.height == 3
    assert result.schema["month_end"] == pl.Date


def test_load_nyse_breakpoints_converts_millions_to_dollars(tmp_path, monkeypatch):
    fixture_path = tmp_path / "ME_Breakpoints.csv"
    fixture_path.write_text(_BREAKPOINTS_FIXTURE)
    monkeypatch.setattr(ken_french_loader, "BREAKPOINTS_PATH", fixture_path)

    result = ken_french_loader.load_nyse_breakpoints()

    row = result.filter(pl.col("month_end") == pl.date(2015, 6, 30))
    assert row.height == 1
    # p10 for 201506 is 12.00 ($M) -> 12,000,000.
    assert row["p10"][0] == pytest.approx(12_000_000.0, rel=1e-9)
    # p90 for 201506 is 108.00 ($M) -> 108,000,000.
    assert row["p90"][0] == pytest.approx(108_000_000.0, rel=1e-9)


def test_load_nyse_breakpoints_month_end_is_last_calendar_day(tmp_path, monkeypatch):
    fixture_path = tmp_path / "ME_Breakpoints.csv"
    fixture_path.write_text(_BREAKPOINTS_FIXTURE)
    monkeypatch.setattr(ken_french_loader, "BREAKPOINTS_PATH", fixture_path)

    result = ken_french_loader.load_nyse_breakpoints()

    # 201506 -> June 2015 -> 2015-06-30 (30 days in June).
    assert datetime.date(2015, 6, 30) in result["month_end"].to_list()
    # 201505 -> May 2015 -> 2015-05-31 (31 days in May).
    assert datetime.date(2015, 5, 31) in result["month_end"].to_list()
