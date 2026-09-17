"""Per-permno monthly ARITHMETIC return compounding for Gate 4
(docs/02_validation_gates.md) and src.portfolio.rebalance.run_backtest's
monthly_rets input.

Mirrors src.estimation.beta_fp._daily_log_returns's file-reading pattern
(same _year_partition_files union across every calendar year in range,
not just the two endpoints -- see that function's own docstring for the
regression this avoids), but compounds to MONTHLY ARITHMETIC returns
rather than leaving a daily LOG series, since
src.portfolio.rebalance.run_backtest's monthly_rets contract is
arithmetic (legs.asymmetric_inverse_beta operates on realized holding-
period returns, not log returns).

DELISTING ROWS ARE COMPOUNDED EXACTLY WHERE CRSP STAMPED THEM, deliberately.

CRSP v2 folds a name's delisting return into `dlyret` on a row dated 1-4
TRADING DAYS after its last real trade (dlydelflg == 'Y'), with
descriptors blanked. An earlier version of this module tried a
"fold_back" mode that reassigned an orphaned delisting row (one landing
in a LATER CALENDAR MONTH than the permno's last real trade) to that
earlier month, on the theory that the loss economically belongs to
whoever held the position last. THIS WAS WRONG, discovered while
smoke-testing Gate 4 (2026-09-13), and deliberately not re-added:

Two real cases are STRUCTURALLY IDENTICAL at the daily-panel level --
"last real trade, then a delisting row exactly ONE TRADING DAY later,
crossing a calendar-month boundary" -- with OPPOSITE correct answers:

  - permno 22518 (2008): last trade 2008-08-29 (Fri), delisting row
    2008-09-02 (Tue, Labor Day between). The position is NOT held into
    September by any real formation date in this project's backtest --
    the loss should be excluded from the return series entirely for
    whatever formation date would otherwise claim it, not charged to
    August.
  - permno 87899 (1995): last trade 1995-01-31 (Tue, itself a formation
    month-end), delisting row 1995-02-01 (Wed). A portfolio formed on
    1995-01-31 genuinely HOLDS this position into February -- the +3%
    delisting return on 1995-02-01 is a real February holding-period
    return, correctly stamped, and folding it back into January would
    hand it to a month before the position was ever held.

Nothing in the daily panel distinguishes these two rows -- both are "one
row, one trading day after the last real trade, in the following
calendar month." The only thing that tells them apart is a PORTFOLIO'S
OWN FORMATION CALENDAR (was this permno actually weighted into a
formation cross-section whose held month is the delisting row's month?),
which this module cannot see and must not guess at. Threading formation
dates into this module would couple the data layer to portfolio
construction -- exactly the coupling F1's frame-in/frame-out design
avoided, and the coupling that let the truncation test prove something
about a PURE loop.

The correct place to resolve this is where formation dates already live:
src.portfolio.rebalance.run_backtest's existing missing-coverage skip.
When a weighted name has no real return for its held month (because this
module correctly produces NO ROW for a month with zero real trading
activity), run_backtest skips that formation date with a recorded reason
rather than fabricating a return -- this already handles permno 22518's
case correctly (no August row exists to wrongly inherit the September
loss; the formation date that would have held it skips, visibly), while
permno 87899's real February row flows through unaltered. One mechanism,
correct on both cases, already built (F2c) for exactly the "don't
fabricate a return over missing coverage" reason.

src.gates.gate4_bab quantifies how often this skip fires and whether it
concentrates in high-beta names (delisting is not decade- or beta-
uniform -- see docs/02_validation_gates.md) as the measured finding this
was originally meant to produce, rather than reporting a fold_back-vs-
as_stamped delta between one correct and one silently-wrong convention.
"""

import datetime

import polars as pl

from src.data import universe_panel


def _month_end(d: datetime.date) -> datetime.date:
    """The calendar month-end containing d. Matches
    src.portfolio.rebalance._next_month_end's convention exactly --
    monthly_rets' `month` values must be the SAME month-end convention
    betas_by_date's keys use (that function's own docstring)."""
    if d.month == 12:
        next_month_first = datetime.date(d.year + 1, 1, 1)
    else:
        next_month_first = datetime.date(d.year, d.month + 1, 1)
    return next_month_first - datetime.timedelta(days=1)


def monthly_arithmetic_returns(
    start_date: datetime.date,
    end_date: datetime.date,
    *,
    leg: str,
) -> pl.DataFrame:
    """Every US permno's monthly arithmetic return for every calendar
    month overlapping [start_date, end_date], compounded from daily
    dlyret via expm1(sum(log1p(dlyret))) -- log-space summation, the
    same convention _daily_log_returns uses, avoiding the float-rounding
    difference from chaining (1+r).prod()-1 across a large panel.

    A dlydelflg=='Y' delisting row is compounded into whatever calendar
    month CRSP stamped it in -- no reassignment (see module docstring
    for why an earlier fold_back attempt was wrong and removed).

    leg: "us" only, REQUIRED keyword-only, no default -- mirrors
    _daily_log_returns(leg=...). "canada" raises NotImplementedError:
    the Canadian stock-side identity key (CHASS ids are strings, not
    permno Int64) is not built, same gap as beta_fp's per-name path.

    Returns permno, month (calendar month-end), ret (arithmetic),
    n_days (real trading-day count compounded into this row),
    has_delisting (bool: this row includes a dlydelflg=='Y' return). A
    permno-month with ZERO real observations produces NO ROW -- never a
    fabricated 0.0 -- so that rebalance.run_backtest's missing-coverage
    skip can fire instead of silently absorbing an absent return as a
    fabricated one (the exact bug F2c's truncation test caught).
    """
    if leg == "canada":
        raise NotImplementedError(
            "monthly_arithmetic_returns(leg='canada') -- the Canadian "
            "stock-side identity path is not built (same gap as "
            "beta_fp._daily_log_returns(leg='canada'); CHASS ids are "
            "strings, not permno Int64)."
        )
    if leg != "us":
        raise KeyError(f"leg must be 'us' or 'canada', got {leg!r}")

    years_needed = sorted(set(range(start_date.year, end_date.year + 1)))
    files: list[str] = []
    for year in years_needed:
        files.extend(universe_panel._year_partition_files(year))
    files = sorted(set(files))
    if not files:
        return pl.DataFrame(
            schema={
                "permno": pl.Int64,
                "month": pl.Date,
                "ret": pl.Float64,
                "n_days": pl.UInt32,
                "has_delisting": pl.Boolean,
            }
        )

    daily = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlyret", "dlydelflg"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(start_date).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(end_date).cast(pl.Datetime("ns")))
        )
        .filter(pl.col("dlyret").is_not_null())
        .collect(engine="streaming")
        .with_columns(
            pl.col("dlycaldt").cast(pl.Date).alias("date"),
            (pl.col("dlydelflg") == "Y").alias("has_delisting"),
        )
        .select("permno", "date", "dlyret", "has_delisting")
    )

    daily = daily.with_columns(
        pl.col("date").map_elements(_month_end, return_dtype=pl.Date).alias("month")
    )

    grouped = (
        daily.group_by(["permno", "month"])
        .agg(
            (pl.col("dlyret") + 1.0).log().sum().alias("_log_sum"),
            pl.len().alias("n_days"),
            pl.col("has_delisting").any().alias("has_delisting"),
        )
        .with_columns((pl.col("_log_sum").exp() - 1.0).alias("ret"))
        .select(["permno", "month", "ret", "n_days", "has_delisting"])
        .sort(["permno", "month"])
    )
    return grouped
