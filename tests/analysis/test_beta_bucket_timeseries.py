"""Tests for src/analysis/beta_bucket_timeseries.py -- the time-
stability view of the beta-bucket analysis,
docs/04_handoff_lowbeta_longonly.md Sec 10.
"""

import datetime

import polars as pl
import pytest

from src.analysis import beta_bucket_timeseries as bbts


def _quarterly_rows(rows: list[tuple[str, datetime.date, float, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "bucket": [r[0] for r in rows],
            "quarter": [r[1] for r in rows],
            "ret": [r[2] for r in rows],
            "beta": [r[3] for r in rows],
            "n": [r[4] for r in rows],
        }
    )


def _quarter_range(start: datetime.date, n: int) -> list[datetime.date]:
    dates = [start]
    d = start
    for _ in range(n - 1):
        year, month = d.year, d.month + 3
        if month > 12:
            month -= 12
            year += 1
        if month == 12:
            d = datetime.date(year, 12, 31)
        elif month in (6, 9):
            d = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
        else:
            d = datetime.date(year, 4, 1) - datetime.timedelta(days=1)
        dates.append(d)
    return dates


# ---------------------------------------------------------------------------
# _period_label -- boundary correctness, anchored to the series' own start.
# ---------------------------------------------------------------------------


def test_period_label_5y_boundary_cases():
    anchor = datetime.date(1965, 3, 31)
    assert bbts._period_label(datetime.date(1965, 3, 31), 60, anchor) == "1965-1970"
    assert bbts._period_label(datetime.date(1969, 12, 31), 60, anchor) == "1965-1970"
    assert bbts._period_label(datetime.date(1970, 3, 31), 60, anchor) == "1970-1975"


def test_period_label_1y_from_january_anchor_gives_single_year_labels():
    """A 1-year period anchored to a January start aligns with calendar
    years, so labels collapse to a single year. Anchored to a
    NON-January start (the more common real case -- formation dates
    follow the sample's own first quarter-end, not Jan 1), a 12-month
    window legitimately spans two calendar years and the label reflects
    that (see the companion test below) -- this is correct, not a bug,
    since period boundaries are anchored to the series' own start
    (_period_label's own docstring), not forced to calendar-year
    alignment."""
    anchor = datetime.date(2000, 1, 31)
    assert bbts._period_label(datetime.date(2000, 1, 31), 12, anchor) == "2000"
    assert bbts._period_label(datetime.date(2000, 12, 31), 12, anchor) == "2000"
    assert bbts._period_label(datetime.date(2001, 1, 31), 12, anchor) == "2001"


def test_period_label_1y_from_march_anchor_spans_two_calendar_years():
    """Anchored to a March start (this project's real US formation-date
    convention -- quarters end Mar/Jun/Sep/Dec), a 12-month period runs
    Mar-through-Feb, genuinely spanning two calendar years -- the label
    must reflect that, not silently collapse to one year."""
    anchor = datetime.date(2000, 3, 31)
    assert bbts._period_label(datetime.date(2000, 3, 31), 12, anchor) == "2000-2001"
    assert bbts._period_label(datetime.date(2000, 12, 31), 12, anchor) == "2000-2001"
    assert bbts._period_label(datetime.date(2001, 3, 31), 12, anchor) == "2001-2002"


# ---------------------------------------------------------------------------
# bucket_returns_by_period -- the core regime-recovery test (the shape of
# test this module exists for: 3 DISTINCT 5-year regimes must produce 3
# DISTINCT period rows with the right signs, not one averaged-away value).
# ---------------------------------------------------------------------------


def test_bucket_returns_by_period_recovers_three_distinct_regimes():
    """15 years (60 quarters) of V01 data: +2%/qtr for 5y, -1%/qtr for
    5y, +2%/qtr for 5y. Must produce exactly 3 period rows with
    ann_ret = quarterly_ret * 4 in each -- NOT a single full-sample
    average that would hide the middle regime's sign flip entirely
    (the exact failure mode a full-sample-only view has, which is the
    whole reason this module exists)."""
    quarters = _quarter_range(datetime.date(1965, 3, 31), 60)
    rows = []
    for i, q in enumerate(quarters):
        ret = 0.02 if i < 20 else (-0.01 if i < 40 else 0.02)
        rows.append(("V01", q, ret, 0.5, 10))
    quarterly = _quarterly_rows(rows)

    result = bbts.bucket_returns_by_period(quarterly, period="5y")
    assert result.height == 3
    periods = result["period"].to_list()
    ann_rets = result["ann_ret"].to_list()

    assert periods == ["1965-1970", "1970-1975", "1975-1980"]
    assert ann_rets[0] == pytest.approx(0.08, rel=1e-9)
    assert ann_rets[1] == pytest.approx(-0.04, rel=1e-9)
    assert ann_rets[2] == pytest.approx(0.08, rel=1e-9)
    assert ann_rets[1] < 0 < ann_rets[0], (
        "the middle regime's negative sign must survive into its own "
        "period row, not be averaged away with the positive regimes on "
        "either side"
    )


def test_bucket_returns_by_period_n_quarters_sums_to_total():
    quarters = _quarter_range(datetime.date(1965, 3, 31), 60)
    rows = [("V01", q, 0.01, 0.5, 10) for q in quarters]
    quarterly = _quarterly_rows(rows)

    result = bbts.bucket_returns_by_period(quarterly, period="5y")
    assert result["n_quarters"].sum() == 60


def test_bucket_returns_by_period_multiple_buckets_kept_separate():
    """Two buckets with DIFFERENT regimes must never be pooled together
    -- V01 and D10 sharing the same quarters but opposite signs must
    both appear, each with its own correct sign."""
    quarters = _quarter_range(datetime.date(1965, 3, 31), 20)
    rows = []
    for q in quarters:
        rows.append(("V01", q, 0.02, 0.3, 10))
        rows.append(("D10", q, -0.02, 2.0, 10))
    quarterly = _quarterly_rows(rows)

    result = bbts.bucket_returns_by_period(quarterly, period="5y")
    v01_row = result.filter(pl.col("bucket") == "V01")
    d10_row = result.filter(pl.col("bucket") == "D10")
    assert v01_row.height == 1
    assert d10_row.height == 1
    assert v01_row["ann_ret"][0] > 0
    assert d10_row["ann_ret"][0] < 0


def test_bucket_returns_by_period_unknown_period_raises():
    quarterly = _quarterly_rows([("V01", datetime.date(2015, 3, 31), 0.01, 0.5, 10)])
    with pytest.raises(KeyError):
        bbts.bucket_returns_by_period(quarterly, period="2y")


def test_bucket_returns_by_period_empty_input_returns_typed_empty_frame():
    empty = pl.DataFrame(
        schema={"bucket": pl.Utf8, "quarter": pl.Date, "ret": pl.Float64, "beta": pl.Float64, "n": pl.Int64}
    )
    result = bbts.bucket_returns_by_period(empty, period="5y")
    assert result.height == 0
    assert set(result.columns) == {"bucket", "period", "ann_ret", "n_quarters", "mean_realized_beta"}


def test_bucket_returns_by_period_decade_gives_coarser_cells():
    """Same 15-year fixture as the regime-recovery test, but period=
    'decade' (120mo) instead of '5y' (60mo) -- must give FEWER, wider
    cells (2, not 3), confirming the period parameter actually changes
    the aggregation width rather than being ignored."""
    quarters = _quarter_range(datetime.date(1965, 3, 31), 60)
    rows = []
    for i, q in enumerate(quarters):
        ret = 0.02 if i < 20 else (-0.01 if i < 40 else 0.02)
        rows.append(("V01", q, ret, 0.5, 10))
    quarterly = _quarterly_rows(rows)

    result_5y = bbts.bucket_returns_by_period(quarterly, period="5y")
    result_decade = bbts.bucket_returns_by_period(quarterly, period="decade")
    assert result_5y.height == 3
    assert result_decade.height < result_5y.height
