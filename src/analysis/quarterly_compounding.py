"""Monthly -> quarterly compounding for a SINGLE series (a market index,
a risk-free rate), plus the partial-quarter guard that makes a silently
incomplete quarter detectable.

WHY THIS MODULE EXISTS. The compounding logic below was previously
inline in src/analysis/_grouped_quarterly_analysis.py (its market-
quarterly block), where it had exactly one caller. Item A of
docs/06_handoff_section6.md -- regressing the 20+20 long-only book's
return series on its own market index -- needs the same operation, and
CLAUDE.md is explicit that a second implementation of quarterly
compounding is how the two paths diverge. So it is extracted here and
_grouped_quarterly_analysis.py now calls it, rather than the two
carrying near-identical loops.

This is an EXTRACTION OF ALREADY-AUDITED CODE, not new logic. Every
compounding decision still belongs to src.portfolio.longonly.
_quarterly_return, which this module delegates to unchanged -- including
its log-space expm1(sum(log1p(r))) convention and its contract that a
month absent from the input contributes 0% cash rather than voiding the
quarter. Nothing about temporal ordering changed in the extraction.

WHY n_months IS A FIRST-CLASS OUTPUT, not an internal detail. Measured
on this project's own Canadian data (_scratch/_probe_ca_align_out.txt),
compounding the Canadian market index onto calendar quarter-end labels
without first re-keying its CHASS real-trading-day month keys gives:

    months matched per quarter, RAW trdate keys: {2: 110, 3: 30, 1: 8}
    months matched per quarter, RE-KEYED:        {3: 148}

-- i.e. 118 of 148 quarters silently compounded only one or two of their
three months, and _quarterly_return returned a plausible-looking number
for all 148 with no error and no missing rows. The compounded VALUE
cannot distinguish the correct series from the corrupt one; only the
month count can. That is the entire justification for returning
n_months and for assert_full_quarters existing at all. CLAUDE.md: "a
wrong number that looks right is the worst possible outcome."

WHAT n_months DOES **NOT** CATCH -- stated explicitly so no future
reader over-trusts it (gate-verifier audit, 2026-09-15):

  - A month PRESENT but carrying a WRONG VALUE. n_months is a
    cardinality statistic and is invariant to every value transform: an
    rf series in percent rather than decimal gives a 100x-wrong rf_q
    with a perfectly clean n_months of 3. Magnitude has to be bounded
    elsewhere; this column cannot do it.
  - A bug in _prior_month_end itself. _quarter_months deliberately
    reuses the same helper _quarterly_return uses, so the count and the
    compounding window would shift TOGETHER and still agree. That
    deliberate coupling is what makes n_months trustworthy about the
    window actually used -- and is exactly why the unit test
    reconstructs the month window INDEPENDENTLY rather than importing
    it (tests/analysis/test_quarterly_compounding.py::_months_of_quarter).

A duplicated month WAS in this blind spot until 2026-09-15: .height
counted rows, so (Jan, Jan, Feb) with March missing reported n_months=3
and compounded January twice. It is now a hard error -- see below.
"""

import datetime
import itertools

import numpy as np
import polars as pl

from src.portfolio.longonly import _prior_month_end, _quarterly_return
from src.portfolio.quarterly import next_quarter_end

# A calendar quarter has exactly three constituent calendar months.
# Named rather than written as a bare 3 (CLAUDE.md: no magic numbers in
# src/) because this is the value every caller asserts n_months against
# -- it is the threshold that separates a complete quarter from the
# silently-partial one documented in this module's docstring.
MONTHS_PER_QUARTER = 3

# The single-series id this module tags onto its input before handing it
# to _quarterly_return, which is a per-name function. Leading dunder so
# it cannot collide with a real permno/CHASS id if a caller ever passes a
# frame that already carries one.
_SERIES_ID = "__series__"


def _quarter_months(quarter_end: datetime.date) -> list[datetime.date]:
    """The three calendar month-ends constituting `quarter_end`'s
    quarter, walking backward via longonly._prior_month_end.

    Deliberately the SAME helper _quarterly_return uses internally to
    build its own month window, so n_months cannot disagree with the set
    of months actually compounded. Re-deriving this with independent
    date arithmetic would create precisely the divergence this module
    exists to prevent.
    """
    month_2 = quarter_end
    month_1 = _prior_month_end(month_2)
    month_0 = _prior_month_end(month_1)
    return [month_0, month_1, month_2]


def compound_monthly_series_to_quarters(
    monthly: pl.DataFrame,
    *,
    quarters: list[datetime.date],
    value_col: str,
) -> pl.DataFrame:
    """Compound a single monthly series to the named quarters.

    monthly: [month (Date), <value_col> (Float64)] -- ONE series, one row
    per month. Not a per-name panel; callers with a panel want
    _grouped_quarterly_analysis instead.

    quarters: the calendar quarter-ends to produce. Both legs' cached
    long-only return frames are keyed on calendar quarter-ends, so
    everything else is re-keyed INTO that space (see
    src.analysis.excess_returns.rekey_to_calendar_month_end), never the
    reverse.

    value_col: the column to compound. Named explicitly rather than
    assumed to be "ret" because the risk-free path compounds "rf"
    through this same function -- one implementation, two series.

    Returns [quarter, <value_col>, n_months], one row per requested
    quarter that had AT LEAST ONE matching month, in the order
    `quarters` was given.

    A quarter with ZERO matching months produces NO ROW -- mirroring
    _quarterly_return's own contract (a fabricated 0% is never
    invented) and preserving the behaviour _grouped_quarterly_analysis
    relied on before this extraction, where such a quarter simply
    dropped out of the downstream inner join.

    n_months is the count of the quarter's three constituent months
    actually present with a non-null value. It is NOT validated here --
    see assert_full_quarters for the opt-in guard, and this module's
    docstring for why the distinction between "compounded 3 months" and
    "compounded 2 months and looked fine" is load-bearing.
    """
    if value_col not in monthly.columns:
        raise ValueError(
            f"compound_monthly_series_to_quarters: value_col={value_col!r} is not "
            f"a column of the input frame (has {monthly.columns}) -- refusing to "
            "compound a column that does not exist rather than propagate nulls "
            "into a reported return"
        )
    if "month" not in monthly.columns:
        raise ValueError(
            f"compound_monthly_series_to_quarters: input frame needs a 'month' "
            f"column, has {monthly.columns}"
        )

    # _quarterly_return is a per-name function keyed on (id_col, month,
    # ret). Tag the single series with a constant id and present
    # value_col under the name it expects; the returned column is
    # renamed back to value_col below.
    as_panel = monthly.select(
        pl.lit(_SERIES_ID).alias(_SERIES_ID),
        pl.col("month"),
        pl.col(value_col).alias("ret"),
    )

    rows: list[dict] = []
    for quarter_end in quarters:
        compounded = _quarterly_return(as_panel, _SERIES_ID, quarter_end)
        if compounded.height == 0:
            continue

        matched = as_panel.filter(
            pl.col("month").is_in(_quarter_months(quarter_end)) & pl.col("ret").is_not_null()
        )
        # DISTINCT months, not row count. A duplicated month would
        # otherwise count toward MONTHS_PER_QUARTER while being
        # compounded TWICE by the log-space sum -- gate-verifier finding
        # 2026-09-15, reproduced: input (Jan, Jan, Feb) with March
        # absent returned 0.3310000000000004 and n_months=3, byte-identical
        # to what a correct (Jan, Feb, Mar) quarter returns, and
        # assert_full_quarters passed it clean. That defeats the whole
        # justification for this column.
        n_months = matched["month"].n_unique()
        if matched.height != n_months:
            duplicated = (
                matched.group_by("month")
                .agg(pl.len().alias("n"))
                .filter(pl.col("n") > 1)
                .sort("month")["month"]
                .to_list()
            )
            raise ValueError(
                f"compound_monthly_series_to_quarters: quarter {quarter_end} has "
                f"{matched.height} rows spanning only {n_months} distinct month(s) "
                f"-- duplicated: {duplicated}. The log-space sum would compound "
                "each duplicated month more than once, inflating the result while "
                "leaving the month count looking complete. Refusing to "
                "deduplicate, because that silently picks an arbitrary row."
            )

        rows.append(
            {
                "quarter": quarter_end,
                value_col: float(compounded["ret"][0]),
                "n_months": n_months,
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={"quarter": pl.Date, value_col: pl.Float64, "n_months": pl.Int64}
        )
    return pl.DataFrame(rows)


def assert_clean_quarterly_series(
    frame: pl.DataFrame,
    *,
    value_col: str,
    label: str,
    require_contiguous: bool = False,
) -> None:
    """Reject a quarterly series that would produce a plausible-looking
    but wrong statistic. Added 2026-09-15 after a gate-verifier audit
    found all three of these propagating silently to reported output.

    Checks, each reproduced as a real defect before being added:

    1. NON-FINITE VALUES. A single null in a cached return parquet gave
       realized_beta=nan, alpha=nan and port_sharpe=nan with NO
       exception and n_quarters still reporting 5 -- a row that looks
       structurally complete gets written to parquet as `nan`.
    2. DUPLICATE QUARTERS. Measured: 6 rows over 5 distinct quarters
       reported n_periods=6, and on a drawdown series a duplicated
       quarter moved max_drawdown from -0.44 to -0.61. An inner join
       against such a frame fans out silently.
    3. GAPS (opt-in via require_contiguous). Removing one quarter from
       the middle of a drawdown series moved max_drawdown from -0.44 to
       -0.30 with no signal. Off by default because a legitimately
       skipped formation quarter is not an error everywhere -- callers
       whose statistic is path-dependent (drawdown) must opt in.

    Deliberately NOT folded into the analysis functions' own bodies:
    each raises with `label`, so a failure names WHICH series is bad
    rather than leaving a caller to bisect several call sites.
    """
    if "quarter" not in frame.columns:
        raise ValueError(
            f"assert_clean_quarterly_series({label!r}): frame has no 'quarter' "
            f"column (has {frame.columns})"
        )
    if value_col not in frame.columns:
        raise ValueError(
            f"assert_clean_quarterly_series({label!r}): value_col={value_col!r} is "
            f"not a column of the frame (has {frame.columns})"
        )

    values = np.asarray(frame[value_col].to_numpy(), dtype=float)
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        first_bad = frame["quarter"].to_list()[int(np.argmax(~np.isfinite(values)))]
        raise ValueError(
            f"assert_clean_quarterly_series({label!r}): {n_bad} of {values.size} "
            f"{value_col!r} value(s) are null/NaN/inf (first: {first_bad}). These "
            "propagate silently through OLS and Sharpe as nan and would be "
            "written to the output as a structurally complete-looking row."
        )

    n_rows = frame.height
    n_distinct = frame["quarter"].n_unique()
    if n_rows != n_distinct:
        duplicated = (
            frame.group_by("quarter")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
            .sort("quarter")["quarter"]
            .to_list()
        )
        raise ValueError(
            f"assert_clean_quarterly_series({label!r}): {n_rows} rows span only "
            f"{n_distinct} distinct quarters -- duplicated: {duplicated[:5]}"
            f"{' ...' if len(duplicated) > 5 else ''}. A duplicated quarter is "
            "double-counted by every mean, and fans out any inner join."
        )

    if require_contiguous and n_rows > 1:
        ordered = frame["quarter"].sort().to_list()
        gaps = [
            (a, b) for a, b in itertools.pairwise(ordered) if next_quarter_end(a) != b
        ]
        if gaps:
            raise ValueError(
                f"assert_clean_quarterly_series({label!r}): {len(gaps)} gap(s) in the "
                f"quarter sequence (first: {gaps[0][0]} -> {gaps[0][1]}). A "
                "path-dependent statistic such as a drawdown compounds straight "
                "across a gap and reports a shallower decline than really occurred."
            )


def assert_full_quarters(
    compounded: pl.DataFrame,
    *,
    quarters: list[datetime.date],
    label: str,
) -> None:
    """Raise ValueError unless EVERY requested quarter is present in
    `compounded` with n_months == MONTHS_PER_QUARTER.

    Deliberately NOT folded into compound_monthly_series_to_quarters.
    src.analysis._grouped_quarterly_analysis tolerates partial quarters
    today (its market block simply inner-joins whatever compounded), and
    making the compounder itself assert would change that module's
    behaviour and therefore the already-published Sec 4 bucket numbers.
    Item A/B/C callers opt in to the guard instead.

    label: names the series being checked ("us_market", "ca_rf", ...) so
    a failure says WHICH of the several compounded series is short,
    rather than leaving a caller to bisect four call sites.

    This is the assertion that catches the defect class measured in this
    module's docstring: a series that returns a plausible number for
    every quarter while having silently compounded only part of each.
    Its own ability to fire is tested directly (tests/analysis/
    test_quarterly_compounding.py::
    test_assert_full_quarters_raises_on_partial_quarter), per CLAUDE.md's
    rule that a check must be demonstrated capable of failing.
    """
    present = dict(zip(compounded["quarter"].to_list(), compounded["n_months"].to_list()))

    missing = [q for q in quarters if q not in present]
    partial = [
        (q, present[q]) for q in quarters if q in present and present[q] != MONTHS_PER_QUARTER
    ]

    if not missing and not partial:
        return

    problems = []
    if missing:
        problems.append(
            f"{len(missing)} quarter(s) absent entirely (first: {missing[0]}, "
            f"last: {missing[-1]})"
        )
    if partial:
        counts = sorted({n for _, n in partial})
        problems.append(
            f"{len(partial)} quarter(s) compounded fewer than {MONTHS_PER_QUARTER} "
            f"months (observed month counts: {counts}; first: {partial[0][0]} with "
            f"{partial[0][1]})"
        )

    raise ValueError(
        f"assert_full_quarters({label!r}): " + "; ".join(problems) + ". "
        "A partially-compounded quarter still returns a plausible-looking "
        "number, so this is the only check that can detect it -- most likely "
        "cause is a monthly series keyed on real trading days joined against "
        "calendar quarter-ends without re-keying (see "
        "src.analysis.excess_returns.rekey_to_calendar_month_end), or a "
        "market/risk-free series that does not cover the requested span."
    )
