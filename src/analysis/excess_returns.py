"""Excess-return assembly for docs/06_handoff_section6.md items A/B/C:
re-key a monthly series onto calendar month-ends, compound a risk-free
rate to quarterly, and subtract it from a raw quarterly return series.

WHY THIS EXISTS. The cached long-only backtest returns
(_scratch/longonly_{us,ca}_window_*_returns.parquet) are RAW quarterly
returns, but docs/05_report_spec.md assumes excess throughout -- every
Sharpe and alpha figure for the report has to be recomputed against the
risk-free rate. src.portfolio.diagnostics.annualized_sharpe takes an
ALREADY-EXCESS series and deliberately accepts no rf argument (to make a
silent double-subtraction impossible), so converting the series is a
caller's job. This module is that caller's shared implementation.

THE RE-KEYING TRAP, measured rather than assumed. Both legs' cached
quarter labels are CALENDAR quarter-ends. But:

  - 158 of 552 Canadian risk-free months (28.6%)
  - 137 of 480 Canadian market-index months

are keyed on CHASS real trading days (e.g. 2025-11-28), not calendar
month-ends. src.portfolio.longonly._quarterly_return looks up EXACT
calendar month-ends, and treats a month it cannot find as 0% cash. So
joining a trading-day-keyed series against calendar quarter-ends
silently compounds part of each quarter:

    months matched per quarter, RAW trdate keys: {2: 110, 3: 30, 1: 8}
    months matched per quarter, RE-KEYED:        {3: 148}

(_scratch/_probe_ca_align_out.txt). All 148 quarters returned a
plausible number; none raised. Understating compounded rf OVERSTATES
excess return, alpha and Sharpe -- the report's headline figures.

This exact trap was already hit and fixed once, for Canadian STOCK
returns -- see src/portfolio/longonly_data.py's "TWO representations of
'month'" comment block. The risk-free and market-index series have the
same shape and had no such fix. This module supplies it, and pairs it
with quarterly_compounding.assert_full_quarters so that a future
regression fails loudly instead of returning a slightly-wrong number.

DIRECTION IS ONE-WAY. Everything is re-keyed INTO calendar-month-end
space, never the reverse, because the cached quarter labels the whole
analysis is keyed on are calendar quarter-ends (verified: 0 of 231 US
and 0 of 148 Canadian labels are non-calendar).
"""

import datetime

import polars as pl

from src.analysis import quarterly_compounding

# Suffix for the column to_excess emits, so a frame carrying both a raw
# and an excess series can never have the two confused by a reader or a
# downstream join. CLAUDE.md: no magic values in src/ -- named here
# rather than written inline at the one call site that builds it.
_EXCESS_SUFFIX = "_excess"

# The column name quarterly_risk_free emits for the compounded rate.
# Distinct from the input "rf" (monthly) on purpose: a frame holding
# both would otherwise silently shadow one with the other, and a
# quarterly rate mistaken for a monthly one is a 3x error.
RF_QUARTERLY_COLUMN = "rf_q"


def rekey_to_calendar_month_end(
    monthly: pl.DataFrame,
    *,
    date_col: str,
) -> pl.DataFrame:
    """Re-key a monthly series onto calendar month-ends.

    A series keyed on real trading days (CHASS trdate: 2025-11-28) is
    moved onto the calendar month-end containing it (2025-11-30), which
    is the key space every quarter lookup in this analysis uses. A
    series already keyed on calendar month-ends is returned unchanged --
    dt.month_end() of a month-end is itself -- so this is safe (and
    worth) applying uniformly to both legs.

    RAISES rather than dedupes if two input rows collapse onto the same
    calendar month-end. The map is lossy; two rows sharing a key would
    be compounded TWICE by the log-space sum downstream, inflating the
    compounded rate. Deduping would instead pick an arbitrary surviving
    row, which src/data/risk_free_canada.py already refuses to do for
    exactly this reason ("refusing to silently pick an arbitrary
    value"). Measured today: the Canadian rf series maps 552 rows onto
    552 distinct month-ends, zero collisions -- this guard exists for
    the next data refresh, not for today's data.
    """
    if date_col not in monthly.columns:
        raise ValueError(
            f"rekey_to_calendar_month_end: date_col={date_col!r} is not a column "
            f"of the input frame (has {monthly.columns})"
        )

    rekeyed = monthly.with_columns(pl.col(date_col).dt.month_end().alias(date_col))

    n_in = rekeyed.height
    n_distinct = rekeyed[date_col].n_unique()
    if n_distinct != n_in:
        collisions = (
            rekeyed.group_by(date_col)
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
            .sort(date_col)
        )
        colliding_dates = collisions[date_col].to_list()
        raise ValueError(
            f"rekey_to_calendar_month_end: {n_in} input rows collapsed onto "
            f"{n_distinct} distinct calendar month-ends -- "
            f"{len(colliding_dates)} month-end(s) received more than one row "
            f"(first: {colliding_dates[0]}). Re-keying is a lossy map and a "
            "duplicated key would be compounded twice by the downstream "
            "log-space sum, inflating the compounded rate. Refusing to "
            "deduplicate, because that silently picks an arbitrary row."
        )

    return rekeyed


def quarterly_risk_free(
    rf_monthly: pl.DataFrame,
    *,
    quarters: list[datetime.date],
    rekey: bool,
) -> pl.DataFrame:
    """Compound a monthly risk-free series to the named quarters.

    rf_monthly: [date, rf, ...] as returned by
    src.data.risk_free_us.load_us_rf_ken_french or
    src.data.risk_free_canada.load_canada_rf_chass. Any further columns
    (Canada's rf_source marker) are dropped here -- the compounder takes
    a single value column, and the driver records rf_source separately
    for the report's footnote on the two derived months.

    rekey: keyword-only and WITHOUT A DEFAULT, deliberately. The caller
    must state, per leg, whether the series it is handing over is keyed
    on real trading days or calendar month-ends. A default would let one
    leg silently inherit the other's behaviour, which is precisely the
    mistake src/portfolio/longonly_data.py documents having already
    made once. Passing True on an already-calendar series is a harmless
    no-op that also buys that leg the collision guard.

    Compounding goes through quarterly_compounding, hence through
    longonly._quarterly_return -- the SAME log-space
    expm1(sum(log1p(r))) path the portfolio and market returns use. Not
    a 3-month arithmetic sum: at this project's rate levels the two
    differ by roughly 1bp, small enough to look right, which CLAUDE.md
    names as the worst possible outcome.

    Returns [quarter, rf_q, n_months]. n_months is NOT asserted here --
    the caller applies quarterly_compounding.assert_full_quarters, so
    that a partial quarter is caught at the point where the label
    identifying WHICH series is short is known.
    """
    if "rf" not in rf_monthly.columns:
        raise ValueError(
            f"quarterly_risk_free: expected an 'rf' column, got {rf_monthly.columns}"
        )

    trimmed = rf_monthly.select(["date", "rf"])
    if rekey:
        trimmed = rekey_to_calendar_month_end(trimmed, date_col="date")

    compounded = quarterly_compounding.compound_monthly_series_to_quarters(
        trimmed.rename({"date": "month"}), quarters=quarters, value_col="rf"
    )
    return compounded.rename({"rf": RF_QUARTERLY_COLUMN})


def to_excess(
    raw_quarterly: pl.DataFrame,
    rf_quarterly: pl.DataFrame,
    *,
    value_col: str,
) -> pl.DataFrame:
    """excess = raw - rf_q, joined on quarter.

    Arithmetic subtraction, matching src/portfolio/legs.py's own
    convention for the BAB payoff -- NOT a geometric de-rating
    raw/(1+rf_q), which is defensible but is not what the rest of this
    codebase does, and mixing the two across sections would be
    undetectable in any correlation-based check.

    INNER join, deliberately. A quarter present in the return series but
    absent from the risk-free series drops out rather than being
    left-joined and null-filled to zero -- a null-filled rf would report
    a RAW return in an excess column, understating the rate to zero for
    that quarter.

    Returns [quarter, <value_col>_excess].
    """
    if value_col not in raw_quarterly.columns:
        raise ValueError(
            f"to_excess: value_col={value_col!r} is not a column of the raw "
            f"frame (has {raw_quarterly.columns})"
        )
    if RF_QUARTERLY_COLUMN not in rf_quarterly.columns:
        raise ValueError(
            f"to_excess: expected a {RF_QUARTERLY_COLUMN!r} column on the "
            f"risk-free frame, got {rf_quarterly.columns}"
        )

    joined = raw_quarterly.join(
        rf_quarterly.select(["quarter", RF_QUARTERLY_COLUMN]), on="quarter", how="inner"
    )
    return joined.select(
        pl.col("quarter"),
        (pl.col(value_col) - pl.col(RF_QUARTERLY_COLUMN)).alias(f"{value_col}{_EXCESS_SUFFIX}"),
    ).sort("quarter")
