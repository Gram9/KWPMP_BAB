"""
CHASS (CFMRC) loader tests. Ground-truths the exact counts live-measured
this session against the real converted parquet files
(data/raw/CHASS_Data/parquet/) -- see src/data/chass_loader.py's module
docstring for the full investigation.

Ported to polars API (2026-09-05, Gate 1 plan Phase A) -- same test names
and pinned assertions as the original pandas version, which is preserved
at tests/unit/test_chass_loader_pandas_reference.py and exercised via
src/data/chass_loader_pandas_reference.py by
tests/unit/test_chass_polars_parity.py's dedicated parity suite. This file
is no longer a parity check; it is the ordinary regression suite for the
polars module, so it uses polars idioms directly rather than mirroring
pandas call shapes.

Named ground-truth cases:
- TWE (ticker) usage 0 = TRANS-WESTERN EXPLORATION INC. and usage 1 =
  TRANSWEST ENERGY INC. both trade live under the same CUSIP
  (893921106) on overlapping dates in May-June 1983 -- confirms
  (ticker, usage) must be kept as two distinct rows, not deduped as if
  CUSIP were the identity key.
- BCC (ticker) usage 0, CUSIP 087239109, 1980-02-25: two dividend
  events (regular + extra) on the same day produce two rows identical
  on every trading field, differing only in dividend detail -- confirms
  the dividend fan-out collapses to exactly one row.
"""

from pathlib import Path

import polars as pl
import pytest

from src.data.chass_loader import (
    SECURITY_ID_COLUMNS,
    apply_chass_fund_mlp_reit_exclusion,
    chass_daily_trade_status,
    filter_chass_daily_traded,
    load_daily,
    load_monthly,
)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "CHASS_Data" / "parquet"


def _parquet_available() -> bool:
    return (RAW_DATA_DIR / "monthly.parquet").exists() and any(
        RAW_DATA_DIR.glob("daily/year=*/part.parquet")
    )


pytestmark = pytest.mark.skipif(
    not _parquet_available(),
    reason="data/raw/CHASS_Data/parquet/ not present -- run "
    "src/data/convert_chass_csv.py first",
)

_KEY = SECURITY_ID_COLUMNS + ["trdate-Trade Date"]


def test_load_monthly_row_count_matches_live_measured_total():
    monthly = load_monthly()
    assert monthly.height == 807_453


def test_load_monthly_ticker_usage_date_is_unique():
    monthly = load_monthly()
    assert monthly.height == monthly.unique(subset=_KEY).height


def test_load_daily_full_row_count_matches_live_measured_total():
    """16,715,944 raw rows - 10 exact full-row duplicates - 2,735 rows
    removed collapsing the dividend fan-out = 16,713,209. Live-measured
    this session, not an estimate."""
    daily = load_daily()
    assert daily.height == 16_713_209


def test_load_daily_ticker_usage_date_is_unique():
    daily = load_daily(years=[1980])
    assert daily.height == daily.unique(subset=_KEY).height


def test_load_daily_keeps_distinct_ticker_usage_sharing_a_cusip():
    """TWE usage 0 and usage 1 both trade live under CUSIP 893921106 on
    1983-05-26 -- must remain two separate rows, not be collapsed as if
    CUSIP were the identity key."""
    daily = load_daily(years=[1983])
    rows = daily.filter(
        (pl.col("symbol-Ticker") == "TWE")
        & (pl.col("trdate-Trade Date") == pl.datetime(1983, 5, 26))
    )
    assert rows.height == 2
    assert set(rows["usage-Usage Number"].to_list()) == {0, 1}
    assert set(rows["name-Name"].to_list()) == {
        "TRANS-WESTERN EXPLORATION INC.",
        "TRANSWEST ENERGY INC.",
    }


def test_load_daily_collapses_dividend_fanout():
    """BCC (CUSIP 087239109) has two dividend-event rows on 1980-02-25
    (regular + extra) that are identical on every trading field -- must
    collapse to exactly one row."""
    daily = load_daily(years=[1980])
    rows = daily.filter(
        (pl.col("cusip-CUSIP") == "087239109")
        & (pl.col("trdate-Trade Date") == pl.datetime(1980, 2, 25))
    )
    assert rows.height == 1


def test_load_daily_single_year_matches_full_load_slice():
    """Loading one year in isolation must match that year's slice of the
    full load -- the dedup logic must not depend on cross-year state."""
    full = load_daily()
    single = load_daily(years=[1980])
    full_1980 = full.filter(pl.col("trdate-Trade Date").dt.year() == 1980)
    assert single.height == full_1980.height


def test_exclusion_removes_clean_categories_entirely():
    """INVESTMENT FUND, INVESTMENT TRUST, LIMITED PARTNERSHIP are clean
    (every sampled name is a genuine fund/REIT/MLP product, see
    config/universe.yaml's chass_fund_mlp_reit_exclusion block) -- must
    be fully removed."""
    monthly = load_monthly()
    filtered = apply_chass_fund_mlp_reit_exclusion(monthly)
    clean_categories = {"INVESTMENT FUND", "INVESTMENT TRUST", "LIMITED PARTNERSHIP"}
    assert filtered.filter(pl.col("business-Business").is_in(clean_categories)).height == 0


def test_exclusion_row_count_matches_live_measured_total():
    """807,453 raw rows - 190,177 clean-category rows - 19,025 hand-
    classified FUND_PRODUCT rows (from config/chass_fund_classification.csv)
    = 598,251. Live-measured this session against the real data."""
    monthly = load_monthly()
    filtered = apply_chass_fund_mlp_reit_exclusion(monthly)
    assert filtered.height == 598_251


def test_exclusion_keeps_real_estate_category():
    """REAL ESTATE is operating real-estate companies (Brookfield Office
    Properties, Trizec), not REITs -- confirmed by inspection this
    session, must NOT be excluded (same trap as CRSP's icbindustry,
    documented in config/universe.yaml)."""
    monthly = load_monthly()
    filtered = apply_chass_fund_mlp_reit_exclusion(monthly)
    assert filtered.filter(pl.col("business-Business") == "REAL ESTATE").height > 0


@pytest.mark.parametrize(
    "name",
    [
        "PEMBINA PIPELINE CORPORATION",
        "ALGONQUIN POWER & UTILITIES CORP.",
        "FRANCO-NEVADA CORPORATION",
        "CI FINANCIAL CORP.",
        "AGF MANAGEMENT LTD. CL 'B' NV",
    ],
)
def test_exclusion_keeps_hand_confirmed_real_companies(name):
    """These were misfiled under contaminated business-Business
    categories (INVESTMENT COMPANY/MUTUAL FUNDS) but confirmed this
    session to be real, large, currently-listed operating companies --
    must survive exclusion."""
    monthly = load_monthly()
    filtered = apply_chass_fund_mlp_reit_exclusion(monthly)
    assert filtered.filter(pl.col("name-Name") == name).height > 0


@pytest.mark.parametrize(
    "name",
    [
        "DIVIDEND 15 SPLIT CORP. CL 'A'",
        "FIDELITY PARTNERSHIP 1993 UNITS",
        "ATRIUM MORTGAGE INVESTMENT CORPORATION",
        "ENERPLUS RESOURCES CORP. SER 'C' ROYALTY UN",
    ],
)
def test_exclusion_removes_hand_confirmed_fund_products(name):
    """A split-share corp, a fund LP unit, and the two group decisions
    made this session (mortgage investment corporations excluded as
    fund-like pooled capital; royalty-unit/income-trust share classes
    excluded regardless of the underlying issuer) -- all must be
    removed."""
    monthly = load_monthly()
    filtered = apply_chass_fund_mlp_reit_exclusion(monthly)
    assert filtered.filter(pl.col("name-Name") == name).height == 0


def test_exclusion_keeps_unknown_names_but_labels_them():
    """UNKNOWN names (confidence too low to classify -- mostly obscure
    microcaps) are included for now per user decision (a later market-
    cap filter will likely remove them), but the classification must
    still say UNKNOWN, not be silently collapsed into REAL_COMPANY --
    this is an open item, not a resolved decision."""
    from src.data.chass_loader import chass_fund_mlp_reit_classification

    monthly = load_monthly()
    filtered = apply_chass_fund_mlp_reit_exclusion(monthly)
    assert filtered.filter(pl.col("name-Name") == "HARROWSTON INC. CL 'A'").height > 0

    classification = chass_fund_mlp_reit_classification(monthly)
    row = classification.filter(pl.col("name-Name") == "HARROWSTON INC. CL 'A'")
    assert (row["classification"] == "UNKNOWN").all()


def test_daily_trade_status_partitions_1980_exactly():
    """volume-Daily Volume > 0 / == 0 / NaN must sum exactly to the total
    row count with zero overlap. Live-measured this session: 140,563
    traded / 68,984 no-trade / 7,341 missing of 216,888 total rows in
    year=1980."""
    daily = load_daily(years=[1980])
    status = chass_daily_trade_status(daily)
    counts = status.value_counts()
    counts_dict = dict(zip(counts["trade_status"].to_list(), counts["count"].to_list()))
    assert counts_dict["TRADED"] == 140_563
    assert counts_dict["NO_TRADE"] == 68_984
    assert counts_dict["MISSING"] == 7_341
    assert sum(counts_dict.values()) == daily.height == 216_888


def test_filter_chass_daily_traded_keeps_only_positive_volume():
    daily = load_daily(years=[1980])
    traded = filter_chass_daily_traded(daily)
    assert traded.height == 140_563
    assert (traded["volume-Daily Volume"] > 0).all()


def test_filter_chass_daily_traded_excludes_no_trade_and_missing():
    """A no-trade row (real info, zero trades) and a missing row (no
    information at all) must both be excluded from the traded filter,
    but chass_daily_trade_status must still distinguish them rather than
    collapsing both into a single 'excluded' bucket."""
    daily = load_daily(years=[1980])
    status = chass_daily_trade_status(daily)

    daily_with_status = daily.with_columns(status)
    no_trade_rows = daily_with_status.filter(pl.col("trade_status") == "NO_TRADE")
    missing_rows = daily_with_status.filter(pl.col("trade_status") == "MISSING")
    assert no_trade_rows.height > 0
    assert missing_rows.height > 0

    traded = filter_chass_daily_traded(daily)
    no_trade_keys = set(
        zip(
            no_trade_rows["symbol-Ticker"].to_list(),
            no_trade_rows["usage-Usage Number"].to_list(),
            no_trade_rows["trdate-Trade Date"].to_list(),
        )
    )
    missing_keys = set(
        zip(
            missing_rows["symbol-Ticker"].to_list(),
            missing_rows["usage-Usage Number"].to_list(),
            missing_rows["trdate-Trade Date"].to_list(),
        )
    )
    traded_keys = set(
        zip(
            traded["symbol-Ticker"].to_list(),
            traded["usage-Usage Number"].to_list(),
            traded["trdate-Trade Date"].to_list(),
        )
    )
    assert traded_keys.isdisjoint(no_trade_keys)
    assert traded_keys.isdisjoint(missing_keys)
