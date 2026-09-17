"""Tests for src/portfolio/longonly_data.py -- the per-leg wrapper that
assembles betas_by_date/fringe_by_date/quarterly_monthly_rets from real
US/CHASS data for src/portfolio/longonly.py's pure core.

Real-data tests share ONE built set of inputs per leg via module-scoped
fixtures -- building a multi-year monthly panel + returns + multi-variant
betas is expensive (measured this session: ~8-25s for a 1.5-3 year US
window), so re-fetching independently in every test would multiply that
cost across the file for no benefit. Windows are kept as SHORT as still
proves each property, not maximally long -- a 3-year US window and a
6-year Canada window are enough to clear a 24m estimator window with
room to test multiple formation dates.
"""

import datetime

import polars as pl
import pytest

from src.data.chass_loader import MONTHLY_PATH
from src.data.universe_panel import RAW_DATA_DIR
from src.portfolio import longonly, longonly_data

pytestmark_realdata = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)
pytestmark_chass = pytest.mark.skipif(
    not MONTHLY_PATH.exists(),
    reason="data/raw/CHASS_Data/parquet/monthly.parquet not present",
)


# ---------------------------------------------------------------------------
# _quarterly_month_ends -- pure function, no data needed.
# ---------------------------------------------------------------------------


def test_quarterly_month_ends_full_year():
    result = longonly_data._quarterly_month_ends(
        datetime.date(2015, 1, 1), datetime.date(2015, 12, 31)
    )
    assert result == [
        datetime.date(2015, 3, 31),
        datetime.date(2015, 6, 30),
        datetime.date(2015, 9, 30),
        datetime.date(2015, 12, 31),
    ]


def test_quarterly_month_ends_partial_range_mid_quarter():
    """A range starting/ending mid-quarter should only include
    quarter-ends that fall INSIDE [start, end], never one outside it."""
    result = longonly_data._quarterly_month_ends(
        datetime.date(2015, 4, 15), datetime.date(2016, 1, 15)
    )
    assert result == [
        datetime.date(2015, 6, 30),
        datetime.date(2015, 9, 30),
        datetime.date(2015, 12, 31),
    ]


def test_quarterly_month_ends_empty_range_before_any_quarter_end():
    result = longonly_data._quarterly_month_ends(
        datetime.date(2015, 10, 1), datetime.date(2015, 10, 15)
    )
    assert result == []


# ---------------------------------------------------------------------------
# build_us_inputs -- ONE real-data build shared across several assertions.
# 2013-2015 (3 years) is enough for a 24m-window formation date to clear
# by 2015 while keeping the build itself under ~25s (measured this
# session), instead of the far more expensive decade-plus span an
# independent-per-test design would need to guarantee overlap.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def us_inputs_small():
    return longonly_data.build_us_inputs(
        datetime.date(2013, 1, 31),
        datetime.date(2015, 12, 31),
        variants=("window_24m", "window_60m"),
    )


@pytestmark_realdata
def test_build_us_inputs_schema(us_inputs_small):
    betas_by_variant, fringe_by_date, quarterly_rets = us_inputs_small
    assert set(betas_by_variant.keys()) == {"window_24m", "window_60m"}
    # window_60m's min_obs=36 only needs 36 REAL observations, not a full
    # 60-month span -- with a 3-year (36-month) input range, at most the
    # VERY LAST formation date (2015-12-31) can clear it (names with a
    # complete run back to 2013-01), and only barely. Measured: exactly 1
    # formation date clears window_60m here -- fewer dates than
    # window_24m (2-year requirement), never more.
    assert len(betas_by_variant["window_60m"]) <= len(betas_by_variant["window_24m"])
    # window_24m needs 2 years -- should clear by the tail of this range.
    assert len(betas_by_variant["window_24m"]) >= 1

    for formation_date, betas in betas_by_variant["window_24m"].items():
        assert set(betas.columns) == {"id", "beta"}
        assert betas.schema["id"] == pl.Utf8
        assert formation_date.month in (3, 6, 9, 12)

    assert len(fringe_by_date) >= 1
    for formation_date, fringe in fringe_by_date.items():
        assert set(fringe.columns) == {"id", "mkt_cap", "price"}
        assert fringe.schema["id"] == pl.Utf8

    assert set(quarterly_rets.columns) == {"id", "month", "ret"}
    assert quarterly_rets.schema["id"] == pl.Utf8


@pytestmark_realdata
def test_build_us_inputs_fringe_matches_market_cap_at_directly(us_inputs_small):
    """The fringe snapshot for a real formation date must be identical
    (mkt_cap, price) to calling universe_panel.market_cap_at() directly
    -- this wrapper should not be silently transforming the values, only
    relabeling permno -> id."""
    from src.data import universe_panel

    _betas, fringe_by_date, _rets = us_inputs_small
    target = datetime.date(2015, 6, 30)
    assert target in fringe_by_date

    direct, _coverage = universe_panel.market_cap_at(target)
    direct_relabeled = direct.select(
        pl.col("permno").cast(pl.Utf8).alias("id"), "mkt_cap", "price"
    ).sort("id")

    wrapped = fringe_by_date[target].sort("id")
    assert wrapped.height == direct_relabeled.height
    assert (wrapped["mkt_cap"] - direct_relabeled["mkt_cap"]).abs().max() < 1e-6


@pytestmark_realdata
def test_us_inputs_feed_run_longonly_backtest_end_to_end(us_inputs_small):
    """Confirms the wrapper's output actually satisfies longonly.py's
    real input contract end-to-end, not just looks plausible in
    isolation. min_survivors relaxed to 5 (not the production 20) --
    this is a 3-year window testing plumbing, not a realistic backtest."""
    betas_by_variant, fringe_by_date, quarterly_rets = us_inputs_small
    result = longonly.run_longonly_backtest(
        betas_by_variant["window_24m"],
        fringe_by_date,
        quarterly_rets,
        n_holdings=5,
        turnover_k=1,
        min_survivors=5,
        mkt_cap_floor=1_000_000_000.0,
        price_floor=1.0,
        id_col="id",
    )
    assert result.returns.height >= 1, (
        f"run_longonly_backtest produced no held quarters at all from "
        f"real US wrapper output -- skips: {result.skips}"
    )


# ---------------------------------------------------------------------------
# build_us_inputs multi-variant independence -- a SEPARATE, slightly wider
# window (needs 5+ years for window_60m to clear at all), kept as its own
# fixture since the schema tests above deliberately use a window too short
# for 60m to produce anything (proving min_obs is enforced).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def us_inputs_wide_enough_for_60m():
    return longonly_data.build_us_inputs(
        datetime.date(2010, 1, 31),
        datetime.date(2015, 12, 31),
        variants=("window_24m", "window_60m"),
    )


@pytestmark_realdata
def test_build_us_inputs_multiple_variants_are_independent(us_inputs_wide_enough_for_60m):
    """24m and 60m variants at the SAME formation date must be genuinely
    different betas for at least some names (different lookback windows
    -> different regressions) -- proves the multi-variant loop actually
    computes distinct windows, not the same thing 3 times."""
    betas_by_variant, _fringe, _rets = us_inputs_wide_enough_for_60m
    common_dates = set(betas_by_variant["window_24m"]) & set(betas_by_variant["window_60m"])
    assert len(common_dates) >= 1
    some_date = max(common_dates)

    b24 = betas_by_variant["window_24m"][some_date]
    b60 = betas_by_variant["window_60m"][some_date]
    joined = b24.join(b60, on="id", suffix="_60m")
    assert joined.height >= 1
    # At least one name's beta must differ meaningfully between windows
    # (exact equality across ALL names would be suspicious -- different
    # windows over real, noisy data essentially never agree exactly).
    diffs = (joined["beta"] - joined["beta_60m"]).abs()
    assert diffs.max() > 1e-6


# ---------------------------------------------------------------------------
# build_canada_inputs -- ONE real-data build shared across several
# assertions, mirroring the US pattern above. 2010-2015 (6 years, well
# past the Canada leg's own 1986 sample start) clears a 24m window with
# room to spare while staying far cheaper than a multi-decade span.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canada_inputs_small():
    return longonly_data.build_canada_inputs(
        datetime.date(2010, 1, 1), datetime.date(2015, 12, 31), variants=("window_24m",)
    )


@pytestmark_chass
def test_build_canada_inputs_schema(canada_inputs_small):
    betas_by_variant, fringe_by_date, quarterly_rets = canada_inputs_small
    assert set(betas_by_variant.keys()) == {"window_24m"}
    assert len(betas_by_variant["window_24m"]) >= 1
    for betas in betas_by_variant["window_24m"].values():
        assert set(betas.columns) == {"id", "beta"}
        assert betas.schema["id"] == pl.Utf8

    assert len(fringe_by_date) >= 1
    for fringe in fringe_by_date.values():
        assert set(fringe.columns) == {"id", "mkt_cap", "price"}

    assert set(quarterly_rets.columns) == {"id", "month", "ret"}


@pytestmark_chass
def test_build_canada_inputs_formation_dates_are_real_chass_trdates(canada_inputs_small):
    """Formation dates must be CHASS's own real trading dates (e.g.
    2015-06-29 or -30, whichever CHASS actually reports), not a naive
    calendar quarter-end that might not exist in the CHASS calendar."""
    from src.data.chass_loader import load_monthly

    betas_by_variant, _fringe, _rets = canada_inputs_small
    monthly = load_monthly()
    real_trdates = {d.date() for d in monthly["trdate-Trade Date"].unique().to_list()}

    for formation_date in betas_by_variant["window_24m"]:
        assert formation_date in real_trdates, (
            f"formation_date {formation_date} is not a real CHASS trdate"
        )


@pytestmark_chass
def test_build_canada_inputs_returned_month_is_calendar_not_trdate(canada_inputs_small):
    """Regression test for a real leakage-adjacent defect found by
    leakage-auditor this session: quarterly_monthly_rets' `month` column
    was CHASS's own real trdate (e.g. 2013-08-30, not calendar month-end
    2013-08-31) while longonly.py's _quarterly_return looks up EXACT
    CALENDAR month-ends. Measured impact before the fix: 161/552 (29%)
    of CHASS trdates are not the calendar last day, causing 125/156
    (80%) of Canadian held quarters to silently compound only 1-2 of
    their 3 real months (the third looking "missing" and implicitly
    treated as 0% cash -- correct for a genuine mid-quarter delisting,
    wrong here since it fired on healthy names from a pure date-key
    mismatch).

    2013-08-30 is confirmed (live, this session) to be August 2013's
    real CHASS trdate, NOT the calendar month-end 2013-08-31 -- exactly
    the mismatch case. This test asserts quarterly_monthly_rets carries
    the CALENDAR month-end 2013-08-31, not the raw trdate 2013-08-30."""
    _betas, _fringe, quarterly_rets = canada_inputs_small
    august_2013_months = quarterly_rets.filter(
        (pl.col("month") >= datetime.date(2013, 8, 1))
        & (pl.col("month") <= datetime.date(2013, 8, 31))
    )["month"].unique().to_list()

    assert datetime.date(2013, 8, 30) not in august_2013_months, (
        "quarterly_monthly_rets still carries the raw CHASS trdate "
        "(2013-08-30) instead of the calendar month-end -- the fix did "
        "not take effect"
    )
    assert datetime.date(2013, 8, 31) in august_2013_months, (
        "quarterly_monthly_rets is missing the calendar-month-end-keyed "
        "August 2013 row entirely"
    )
