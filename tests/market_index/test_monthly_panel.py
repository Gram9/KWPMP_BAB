"""Tests for src/market_index/monthly_panel.py -- monthly-grain
common-schema (id, date, mkt_cap, ret) panels feeding into
build.build_index() for docs/04_handoff_lowbeta_longonly.md Sec 4.2's
monthly beta variants.
"""

import datetime

import polars as pl
import pytest

from src.data.chass_loader import MONTHLY_PATH
from src.data.universe_panel import RAW_DATA_DIR
from src.market_index import build, monthly_panel

pytestmark_realdata = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)
pytestmark_chass = pytest.mark.skipif(
    not MONTHLY_PATH.exists(),
    reason="data/raw/CHASS_Data/parquet/monthly.parquet not present",
)


# ---------------------------------------------------------------------------
# Schema checks -- both builders must emit exactly the common schema
# build_index() consumes, regardless of leg-specific internals.
# ---------------------------------------------------------------------------


@pytestmark_realdata
def test_us_monthly_panel_schema():
    panel = monthly_panel.us_monthly_panel(
        datetime.date(2015, 1, 31), datetime.date(2015, 3, 31)
    )
    assert set(panel.columns) == {"id", "date", "mkt_cap", "ret"}
    assert panel.schema["id"] == pl.Utf8
    assert panel.schema["date"] == pl.Date
    assert panel.height > 0


@pytestmark_chass
def test_canada_monthly_panel_schema():
    panel = monthly_panel.canada_monthly_panel(
        datetime.date(2015, 1, 1), datetime.date(2015, 3, 31)
    )
    assert set(panel.columns) == {"id", "date", "mkt_cap", "ret"}
    assert panel.schema["id"] == pl.Utf8
    assert panel.schema["date"] == pl.Date
    assert panel.height > 0


def test_us_monthly_panel_empty_range_returns_typed_empty_frame():
    """A range before the panel's own coverage should return an empty
    but correctly-typed frame, not raise -- matches universe_panel's own
    convention for out-of-range dates."""
    panel = monthly_panel.us_monthly_panel(
        datetime.date(1900, 1, 31), datetime.date(1900, 3, 31)
    )
    assert panel.height == 0
    assert set(panel.columns) == {"id", "date", "mkt_cap", "ret"}


# ---------------------------------------------------------------------------
# Real-data feed-through: the monthly panel must actually work as input to
# build_index() -- the entire point of matching the common schema.
# ---------------------------------------------------------------------------


@pytestmark_realdata
def test_us_monthly_panel_feeds_build_index():
    panel = monthly_panel.us_monthly_panel(
        datetime.date(2014, 6, 30), datetime.date(2015, 6, 30)
    )
    index = build.build_index(panel, method="vw_uncapped")
    assert set(index.columns) == {"date", "index_ret"}
    assert index.height >= 12
    # Every index return should be a plausible monthly equity return --
    # not a units/scale sanity check masquerading as a real assertion,
    # just a floor that would catch a gross error (e.g. percent-vs-decimal).
    assert index["index_ret"].abs().max() < 1.0


@pytestmark_chass
def test_canada_monthly_panel_feeds_build_index():
    panel = monthly_panel.canada_monthly_panel(
        datetime.date(2014, 1, 1), datetime.date(2015, 12, 31)
    )
    index = build.build_index(panel, method="vw_capped_10pct")
    assert set(index.columns) == {"date", "index_ret"}
    assert index.height >= 12
    assert index["index_ret"].abs().max() < 1.0


# ---------------------------------------------------------------------------
# Regression test for the two real leaks the leakage-auditor found and this
# session fixed (see module docstring): (1) mkt_cap was resolved at
# month_end's OWN close, a same-period weight/return correlation that
# inflated the 2015 US index from +0.12% to +5.32% cumulative; (2)
# membership was ALSO resolved at month_end's own close, which silently
# dropped every name that delisted mid-month from that month's row entirely
# (survivorship bias stacking in the same direction). Anchored on a real
# permno, not a synthetic fixture -- CLAUDE.md: an absolute check outside
# the dataset being validated.
# ---------------------------------------------------------------------------


@pytestmark_realdata
def test_delisted_name_still_contributes_its_final_month_return():
    """permno 11999: eligible in universe_panel.universe_at(2015-05-31)
    (confirmed live), but delisted before 2015-06-30 (absent from
    universe_at(2015-06-30) -- confirmed live). Its real June 2015
    return is -0.7736283362013889 (from monthly_returns.
    monthly_arithmetic_returns, which folds the delisting return in --
    independently confirmed, not re-derived from this module).

    Pre-fix, this permno would be ABSENT from the panel's June row
    entirely (membership resolved at June's own close, which excludes
    it) -- exactly the survivorship-bias mechanism the leakage-auditor
    found (17 of 17 real June 2015 delistings were dropped this way).
    Post-fix, membership is resolved at May's close (where 11999 WAS a
    member), so its June return must appear."""
    panel = monthly_panel.us_monthly_panel(datetime.date(2015, 4, 30), datetime.date(2015, 6, 30))
    june_row = panel.filter(
        (pl.col("id") == "11999") & (pl.col("date") == datetime.date(2015, 6, 30))
    )
    assert june_row.height == 1, (
        "permno 11999 (eligible at 2015-05-31, delisted before 2015-06-30) "
        "must still contribute a June 2015 row -- its absence is exactly "
        "the survivorship-bias defect this test guards against"
    )
    assert june_row["ret"][0] == pytest.approx(-0.7736283362013889, rel=1e-9)
    assert june_row["mkt_cap"][0] is not None


@pytestmark_realdata
def test_mkt_cap_is_lagged_one_full_month_not_contemporaneous():
    """The mkt_cap for permno 11999's June-2015 row must equal its
    market_cap_at(2015-05-31) snapshot (the PRIOR month-end) -- NOT
    market_cap_at(2015-06-30) (which would be a same-period,
    contemporaneous weight, the mechanism that inflated the 2015 US
    index by 5.2 percentage points pre-fix). 11999 isn't even IN
    market_cap_at(2015-06-30)'s eligible set (it delisted), so a
    same-period lookup would either be null or absent -- this test pins
    the exact prior-month value instead."""
    from src.data import universe_panel as up

    expected_mkt_cap_df, _coverage = up.market_cap_at(datetime.date(2015, 5, 31))
    expected = expected_mkt_cap_df.filter(pl.col("permno") == 11999)["mkt_cap"][0]

    panel = monthly_panel.us_monthly_panel(datetime.date(2015, 4, 30), datetime.date(2015, 6, 30))
    june_row = panel.filter(
        (pl.col("id") == "11999") & (pl.col("date") == datetime.date(2015, 6, 30))
    )
    assert june_row.height == 1
    assert june_row["mkt_cap"][0] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Sanity cross-check against the daily-compounded index (build_index_chunked)
# -- not required to match exactly (different weighting frequency: monthly
# panel reweights once a month, daily panel reweights every day), but
# should not wildly diverge over the same real calendar months. This is the
# CLAUDE.md-required absolute/anchor-adjacent check: an independently-built
# series, not just an internal self-consistency check.
# ---------------------------------------------------------------------------


@pytestmark_realdata
def test_us_monthly_index_roughly_tracks_daily_compounded_index():
    """Over 2015, the monthly-panel-built vw_uncapped index and the
    daily-compounded-to-monthly vw_uncapped index should have correlated,
    same-sign-majority monthly returns -- both are value-weighted total
    return on (approximately) the same universe. A large disagreement
    would mean the monthly panel's mkt_cap or ret construction has a real
    bug (e.g. wrong lag, wrong return convention), not just weighting-
    frequency noise."""
    from src.data import gate_adapters
    from src.gates import gate1_index

    start, end = datetime.date(2015, 1, 1), datetime.date(2015, 12, 31)

    monthly = monthly_panel.us_monthly_panel(
        datetime.date(2014, 12, 31), end
    )
    monthly_index = build.build_index(monthly, method="vw_uncapped").filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )

    daily_panel = gate_adapters.us_gate1_panel(start, end)
    daily_index = gate1_index.build_vw_index(daily_panel)
    # compound_daily_index_to_monthly returns {month_end, monthly_ret},
    # not {date, index_ret} -- different column names from build_index()'s
    # output, confirmed against the source (gate1_index.py) rather than
    # assumed.
    daily_monthly = gate1_index.compound_daily_index_to_monthly(daily_index)

    assert monthly_index.height >= 10
    assert daily_monthly.height >= 10

    # Both indexed by month -- join on the month number (year, month)
    # rather than exact date, since the two use different month-end
    # resolution conventions.
    m = monthly_index.with_columns(
        pl.col("date").dt.year().alias("y"), pl.col("date").dt.month().alias("m")
    )
    d = daily_monthly.with_columns(
        pl.col("month_end").dt.year().alias("y"),
        pl.col("month_end").dt.month().alias("m"),
    )
    joined = m.join(d, on=["y", "m"], how="inner")
    assert joined.height >= 8, (
        f"only {joined.height} overlapping months found -- expected at "
        "least 8 of 12 months in 2015 to be present in both series"
    )

    same_sign = ((joined["index_ret"] > 0) == (joined["monthly_ret"] > 0)).sum()
    assert same_sign >= joined.height * 0.6, (
        f"monthly-panel index and daily-compounded index agree on sign "
        f"in only {same_sign}/{joined.height} months -- expected general "
        "directional agreement between two value-weighted total-return "
        "series over roughly the same universe"
    )

    # ABSOLUTE anchor (CLAUDE.md: correlation/sign checks alone cannot
    # catch a level/scale bug -- see the mkt_cap-lag defect this test
    # caught pre-fix, where correlation was 0.9997 and sign agreement
    # was 12/12 despite the index being inflated by 5.2 percentage
    # points over 2015). Cumulative 2015 return of the two series must
    # agree within a stated tolerance in PERCENTAGE POINTS, not a
    # relative/correlation measure. Measured post-fix: gap is ~0.03pp;
    # 1.0pp is a wide, deliberately loose bound (weighting-frequency
    # noise between monthly and daily reweighting is real and expected)
    # that the pre-fix ~5.2pp defect would still have failed by 5x.
    cum_monthly = float((1.0 + monthly_index["index_ret"]).product() - 1.0)
    cum_daily = float((1.0 + daily_monthly["monthly_ret"]).product() - 1.0)
    gap_pp = abs(cum_monthly - cum_daily) * 100.0
    assert gap_pp < 1.0, (
        f"monthly-panel cumulative 2015 return ({cum_monthly:.4%}) vs "
        f"daily-compounded ({cum_daily:.4%}) differ by {gap_pp:.2f} "
        "percentage points -- expected < 1.0pp. A gap this large "
        "previously indicated a mkt_cap lag defect (contemporaneous "
        "weight/return correlation), not weighting-frequency noise."
    )
