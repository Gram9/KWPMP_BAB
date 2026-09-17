"""
Point-in-time Canadian universe membership tests (CHASS-based). Ground-
truths the row-presence membership design and its two corroborating
checks (terminal-row NaN pattern, CUSIP-continuity rename detection)
investigated and validated this session -- see src/data/chass_universe.py's
module docstring for the full derivation, and docs/01_data_notes.md
section 6 for the related, separately-documented delisting-return gap.

Ported to polars API (2026-09-05, Gate 1 plan Phase A) -- same test names
and pinned assertions as the original pandas version, preserved at
src/data/chass_universe_pandas_reference.py and exercised by
tests/unit/test_chass_polars_parity.py's dedicated parity suite. This file
is the ordinary regression suite for the polars module.

Named ground-truth cases:
- Pembina Pipeline Corporation (ticker PPL, usage 2): confirmed real,
  currently-listed, continuously present with zero monthly gaps from
  1997-10-31 through 2025-12-31 (339 rows, max gap between consecutive
  rows 33 days -- one ordinary calendar month).
- Morgan Hydrocarbons Inc. (ticker MHI, usage 0): confirmed genuine
  delisting -- acquired by Talisman Energy, Oct 1996. Monthly rows stop
  entirely after 1996-10-31 (last row: shares_out and return both NaN,
  the terminal-row corroboration pattern). Its CUSIP (617900105) never
  resurfaces under any other ticker -- confirms this is a real exit, not
  a rename.
- National Bank of Canada (cusip 633067103, symbol-Ticker is null in the
  source parquet): a real, currently-listed major bank whose null-symbol
  group was found (via the polars port's parity testing, 2026-09-05) to
  be silently dropped by the pandas original's _terminal_row_flags(),
  because pandas's groupby() defaults to dropna=True. The polars port
  intentionally does NOT reproduce this drop -- see
  docs/01_data_notes.md section 18 and
  tests/unit/test_chass_polars_parity.py's
  test_terminal_row_flags_parity_excluding_known_null_symbol_divergence.
"""

import datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.chass_loader import load_daily, load_monthly
from src.data.chass_universe import (
    _check_cusip_continuity,
    _terminal_row_flags,
    market_cap_at,
    universe_at,
)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "CHASS_Data" / "parquet"


def _parquet_available() -> bool:
    return (RAW_DATA_DIR / "monthly.parquet").exists()


pytestmark = pytest.mark.skipif(
    not _parquet_available(),
    reason="data/raw/CHASS_Data/parquet/ not present -- run "
    "src/data/convert_chass_csv.py first",
)


def _synthetic_monthly(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal synthetic monthly frame with the columns
    universe_at()'s helpers touch, for the helpers that don't depend on
    apply_chass_fund_mlp_reit_exclusion (which asserts its input's
    (symbol, usage) keys are a superset of config/chass_fund_
    classification.csv's keys -- a real production safety check, but one
    a small synthetic frame can never satisfy). universe_at() itself is
    tested against the real monthly panel below instead, since it always
    runs the fund exclusion step."""
    defaults = {
        "usage-Usage Number": 0,
        "cusip-CUSIP": "000000001",
        "business-Business": "HOLDING COMPANY",
        "foreign_flag-Foreign Flag": "Domestic Firm",
        "shares_out-Monthly Shares outstanding (100s of shares)": 1000.0,
        "return-Monthly Return": 0.01,
    }
    merged_rows = [{**defaults, **row} for row in rows]
    df = pl.DataFrame(merged_rows)
    return df.with_columns(pl.col("trdate-Trade Date").cast(pl.Datetime("ns")))


def test_universe_at_includes_name_present_that_month():
    """Real monthly panel, injected with an extra synthetic row for a
    ticker guaranteed not to collide with any real name, to isolate the
    row-presence boundary logic without needing the whole classification
    CSV to describe a fabricated frame."""
    monthly = pl.concat(
        [
            load_monthly(),
            _synthetic_monthly(
                [
                    {
                        "symbol-Ticker": "ZZTEST1",
                        "name-Name": "ZZTEST1 CO",
                        "trdate-Trade Date": datetime.date(2015, 6, 30),
                    },
                ]
            ),
        ],
        how="diagonal",
    )
    result = universe_at(monthly, month_end=datetime.date(2015, 6, 30))
    assert result.filter(pl.col("symbol-Ticker") == "ZZTEST1").height > 0


def test_universe_at_excludes_name_absent_that_month():
    """A name with rows only before month_end is not a member that
    month -- absence of a row IS the delisting/not-yet-listed bound, by
    construction of the row-presence design (no separate date-bound
    field needed, unlike the CRSP securitybegdt/securityenddt approach)."""
    monthly = pl.concat(
        [
            load_monthly(),
            _synthetic_monthly(
                [
                    {
                        "symbol-Ticker": "ZZTEST2",
                        "name-Name": "ZZTEST2 CO",
                        "trdate-Trade Date": datetime.date(2015, 5, 31),
                    },
                ]
            ),
        ],
        how="diagonal",
    )
    result = universe_at(monthly, month_end=datetime.date(2015, 6, 30))
    assert result.filter(pl.col("symbol-Ticker") == "ZZTEST2").height == 0


def test_universe_at_excludes_foreign_firm():
    monthly = pl.concat(
        [
            load_monthly(),
            _synthetic_monthly(
                [
                    {
                        "symbol-Ticker": "ZZTEST3",
                        "name-Name": "ZZTEST3 FOREIGN CO",
                        "trdate-Trade Date": datetime.date(2015, 6, 30),
                        "foreign_flag-Foreign Flag": "Foreign Firm",
                    },
                ]
            ),
        ],
        how="diagonal",
    )
    result = universe_at(monthly, month_end=datetime.date(2015, 6, 30))
    assert result.filter(pl.col("symbol-Ticker") == "ZZTEST3").height == 0


def test_universe_at_excludes_fund_product():
    monthly = pl.concat(
        [
            load_monthly(),
            _synthetic_monthly(
                [
                    {
                        "symbol-Ticker": "ZZTEST4",
                        "name-Name": "ZZTEST4 FUND",
                        "trdate-Trade Date": datetime.date(2015, 6, 30),
                        "business-Business": "INVESTMENT FUND",
                    },
                ]
            ),
        ],
        how="diagonal",
    )
    result = universe_at(monthly, month_end=datetime.date(2015, 6, 30))
    assert result.filter(pl.col("symbol-Ticker") == "ZZTEST4").height == 0


def test_terminal_row_flags_true_for_nan_shares_out_and_return():
    monthly = _synthetic_monthly(
        [
            {
                "symbol-Ticker": "EEE",
                "name-Name": "EEE CO",
                "trdate-Trade Date": datetime.date(2015, 5, 31),
            },
            {
                "symbol-Ticker": "EEE",
                "name-Name": "EEE CO",
                "trdate-Trade Date": datetime.date(2015, 6, 30),
                "shares_out-Monthly Shares outstanding (100s of shares)": None,
                "return-Monthly Return": None,
            },
        ]
    )
    flags = _terminal_row_flags(monthly)
    row = flags.filter(
        (pl.col("symbol-Ticker") == "EEE") & (pl.col("usage-Usage Number") == 0)
    )
    assert row["terminal_nan_corroboration"][0]


def test_terminal_row_flags_false_for_normal_terminal_row():
    monthly = _synthetic_monthly(
        [
            {
                "symbol-Ticker": "FFF",
                "name-Name": "FFF CO",
                "trdate-Trade Date": datetime.date(2015, 6, 30),
            },
        ]
    )
    flags = _terminal_row_flags(monthly)
    row = flags.filter(
        (pl.col("symbol-Ticker") == "FFF") & (pl.col("usage-Usage Number") == 0)
    )
    assert not row["terminal_nan_corroboration"][0]


def test_cusip_continuity_finds_rename_within_window():
    monthly = _synthetic_monthly(
        [
            {
                "symbol-Ticker": "OLD",
                "name-Name": "OLD NAME CO",
                "trdate-Trade Date": datetime.date(2015, 5, 31),
                "cusip-CUSIP": "999999999",
            },
            {
                "symbol-Ticker": "NEW",
                "name-Name": "NEW NAME CO",
                "trdate-Trade Date": datetime.date(2015, 6, 30),
                "cusip-CUSIP": "999999999",
            },
        ]
    )
    match = _check_cusip_continuity(monthly, ticker="OLD", usage=0)
    assert match == ("NEW", 0)


def test_cusip_continuity_returns_none_when_no_match():
    monthly = _synthetic_monthly(
        [
            {
                "symbol-Ticker": "GGG",
                "name-Name": "GGG CO",
                "trdate-Trade Date": datetime.date(2015, 5, 31),
                "cusip-CUSIP": "111111111",
            },
        ]
    )
    match = _check_cusip_continuity(monthly, ticker="GGG", usage=0)
    assert match is None


def test_pembina_pipeline_is_continuously_a_member():
    """339 monthly rows, 1997-10-31 through 2025-12-31, zero gaps
    (live-measured this session)."""
    monthly = load_monthly()
    ppl_dates = monthly.filter(
        (pl.col("symbol-Ticker") == "PPL") & (pl.col("usage-Usage Number") == 2)
    )["trdate-Trade Date"]
    assert len(ppl_dates) == 339

    sample_month = datetime.date(2015, 6, 30)
    universe = universe_at(monthly, month_end=sample_month)
    assert universe.filter(pl.col("symbol-Ticker") == "PPL").height > 0


def test_morgan_hydrocarbons_absent_after_delisting_month():
    monthly = load_monthly()

    still_member = universe_at(monthly, month_end=datetime.date(1996, 10, 31))
    assert (
        still_member.filter(
            (pl.col("symbol-Ticker") == "MHI") & (pl.col("usage-Usage Number") == 0)
        ).height
        > 0
    )

    delisted = universe_at(monthly, month_end=datetime.date(1996, 11, 30))
    assert (
        delisted.filter(
            (pl.col("symbol-Ticker") == "MHI") & (pl.col("usage-Usage Number") == 0)
        ).height
        == 0
    )


def test_morgan_hydrocarbons_shows_terminal_nan_corroboration():
    monthly = load_monthly()
    flags = _terminal_row_flags(monthly)
    row = flags.filter(
        (pl.col("symbol-Ticker") == "MHI") & (pl.col("usage-Usage Number") == 0)
    )
    assert row["terminal_nan_corroboration"][0]


def test_morgan_hydrocarbons_cusip_never_resurfaces():
    """Confirms MHI is a genuine delisting, not a ticker rename -- its
    CUSIP (617900105) never appears under a different (ticker, usage)."""
    monthly = load_monthly()
    match = _check_cusip_continuity(monthly, ticker="MHI", usage=0)
    assert match is None


def test_national_bank_of_canada_null_symbol_included_in_terminal_row_flags():
    """The real bug this port fixed (docs/01_data_notes.md section 18):
    National Bank of Canada has a null symbol-Ticker in the source data.
    Under the polars port, its group must still appear in
    _terminal_row_flags() output -- unlike the pandas original, which
    silently dropped it via groupby(dropna=True)."""
    monthly = load_monthly()
    flags = _terminal_row_flags(monthly)
    nbc_rows = flags.filter(pl.col("symbol-Ticker").is_null())
    assert nbc_rows.height == 1


def _synthetic_daily(rows: list[dict]) -> pl.DataFrame:
    defaults = {"usage-Usage Number": 0, "volume-Daily Volume": 1000.0}
    merged_rows = [{**defaults, **row} for row in rows]
    df = pl.DataFrame(merged_rows)
    return df.with_columns(pl.col("trdate-Trade Date").cast(pl.Datetime("ns")))


def test_market_cap_uses_prior_day_close_not_same_day():
    """The core leakage check, mirroring universe_panel.py's
    test_market_cap_never_uses_month_end_or_later_price: a daily row
    dated AT month_end must never be used, even if it's the only
    same-day row available and an earlier row exists."""
    monthly = pl.concat(
        [
            load_monthly(),
            _synthetic_monthly(
                [
                    {
                        "symbol-Ticker": "ZZTEST5",
                        "name-Name": "ZZTEST5 CO",
                        "trdate-Trade Date": datetime.date(2015, 6, 30),
                    },
                ]
            ),
        ],
        how="diagonal",
    )
    daily = _synthetic_daily(
        [
            {
                "symbol-Ticker": "ZZTEST5",
                "trdate-Trade Date": datetime.date(2015, 6, 29),
                "closeprice-Daily Closing price": 10.0,
            },
            {
                "symbol-Ticker": "ZZTEST5",
                "trdate-Trade Date": datetime.date(2015, 6, 30),
                "closeprice-Daily Closing price": 999.0,
            },
        ]
    )
    market_cap_df, _coverage = market_cap_at(monthly, daily, month_end=datetime.date(2015, 6, 30))
    row = market_cap_df.filter(pl.col("symbol-Ticker") == "ZZTEST5")
    assert row.height == 1
    # shares_out default is 1000.0 (100s of shares) -> 1000.0 * 100 = 100,000 shares
    assert row["mkt_cap"][0] == 10.0 * 1000.0 * 100


def test_market_cap_excludes_null_prior_day_close():
    monthly = pl.concat(
        [
            load_monthly(),
            _synthetic_monthly(
                [
                    {
                        "symbol-Ticker": "ZZTEST6",
                        "name-Name": "ZZTEST6 CO",
                        "trdate-Trade Date": datetime.date(2015, 6, 30),
                    },
                ]
            ),
        ],
        how="diagonal",
    )
    daily = _synthetic_daily(
        [
            {
                "symbol-Ticker": "ZZTEST6",
                "trdate-Trade Date": datetime.date(2015, 6, 29),
                "closeprice-Daily Closing price": None,
            },
        ]
    )
    market_cap_df, coverage = market_cap_at(monthly, daily, month_end=datetime.date(2015, 6, 30))
    assert market_cap_df.filter(pl.col("symbol-Ticker") == "ZZTEST6").height == 0

    coverage_row = coverage.filter(pl.col("symbol-Ticker") == "ZZTEST6")
    assert coverage_row.height == 1
    assert not coverage_row["matched"][0]


def test_market_cap_excludes_null_shares_out():
    monthly = pl.concat(
        [
            load_monthly(),
            _synthetic_monthly(
                [
                    {
                        "symbol-Ticker": "ZZTEST7",
                        "name-Name": "ZZTEST7 CO",
                        "trdate-Trade Date": datetime.date(2015, 6, 30),
                        "shares_out-Monthly Shares outstanding (100s of shares)": None,
                    },
                ]
            ),
        ],
        how="diagonal",
    )
    daily = _synthetic_daily(
        [
            {
                "symbol-Ticker": "ZZTEST7",
                "trdate-Trade Date": datetime.date(2015, 6, 29),
                "closeprice-Daily Closing price": 10.0,
            },
        ]
    )
    market_cap_df, coverage = market_cap_at(monthly, daily, month_end=datetime.date(2015, 6, 30))
    assert market_cap_df.filter(pl.col("symbol-Ticker") == "ZZTEST7").height == 0
    coverage_row = coverage.filter(pl.col("symbol-Ticker") == "ZZTEST7")
    assert not coverage_row["matched"][0]


def test_pembina_pipeline_market_cap_matches_hand_computed_value():
    """Prior trading day before 2015-06-30 is 2015-06-29 (close $40.58,
    live-measured this session); June 2015 shares_out is 3,404,490.0
    (100s of shares). Expected mkt_cap = 40.58 * 3,404,490.0 * 100 =
    13,819,829,220 -- in the right order of magnitude for Pembina's real
    known ~$14B market cap in mid-2015."""
    monthly = load_monthly()
    daily = load_daily(years=[2015])
    market_cap_df, _coverage = market_cap_at(monthly, daily, month_end=datetime.date(2015, 6, 30))
    row = market_cap_df.filter(
        (pl.col("symbol-Ticker") == "PPL") & (pl.col("usage-Usage Number") == 2)
    )
    assert row.height == 1
    expected = 40.58 * 3_404_490.0 * 100
    assert row["mkt_cap"][0] == pytest.approx(expected, rel=1e-6)
