"""Tests for src/analysis/quarterly_compounding.py -- the monthly->
quarterly compounding helper extracted from
src/analysis/_grouped_quarterly_analysis.py so that item A of
docs/06_handoff_section6.md (long-only book vs market) and the existing
bucket/target-beta callers share ONE implementation.

WRITTEN BEFORE THE MODULE (CLAUDE.md workflow).

The anchors here are ABSOLUTE, not correlational. CLAUDE.md: correlation
is location- and scale-invariant, so it cannot detect a units bug, a
constant bias, or a scale error -- the defect classes a compounding
helper is most likely to carry. Every anchor below pins an exact number
derived by hand outside the dataset.
"""

import datetime

import polars as pl
import pytest

from src.analysis import quarterly_compounding
from src.portfolio import diagnostics
from src.portfolio.longonly import _quarterly_return


def _monthly(rows: list[tuple[datetime.date, float]], value_col: str = "ret") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "month": [r[0] for r in rows],
            value_col: [r[1] for r in rows],
        }
    )


def _months_of_quarter(quarter_end: datetime.date) -> list[datetime.date]:
    """The three calendar month-ends constituting `quarter_end`'s quarter.
    Local to this test module -- deliberately an INDEPENDENT
    reconstruction rather than an import from the module under test, so a
    defect in that module's own month-window logic cannot hide behind a
    shared helper."""
    year = quarter_end.year
    end_month = quarter_end.month
    months = []
    for offset in (2, 1, 0):
        m = end_month - offset
        y = year
        if m <= 0:
            m += 12
            y -= 1
        nxt = datetime.date(y + 1, 1, 1) if m == 12 else datetime.date(y, m + 1, 1)
        months.append(nxt - datetime.timedelta(days=1))
    return months


# The three calendar months constituting the quarter ending 1990-03-31.
_Q1_1990 = datetime.date(1990, 3, 31)
_Q1_1990_MONTHS = _months_of_quarter(_Q1_1990)


# ---------------------------------------------------------------------------
# compound_monthly_series_to_quarters -- ABSOLUTE ANCHORS
# ---------------------------------------------------------------------------


def test_three_months_of_ten_percent_compound_to_exactly_0_331():
    """ABSOLUTE ANCHOR (CLAUDE.md): three months of +10% compound to
    1.10**3 - 1 = 0.331 EXACTLY, a number derived by hand outside any
    dataset.

    This single assertion discriminates between every plausible wrong
    implementation:
      - arithmetic SUM of returns      -> 0.300  (fails)
      - arithmetic MEAN of returns     -> 0.100  (fails)
      - a x3 scale error               -> 0.300  (fails)
      - log-space sum left un-exp'd    -> 0.2859 (fails)
      - correct log-space compounding  -> 0.331  (passes)

    A correlation-based check would score 1.0 against all five.
    """
    monthly = _monthly([(m, 0.10) for m in _Q1_1990_MONTHS])

    got = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=[_Q1_1990], value_col="ret"
    )

    assert got.height == 1, f"expected exactly one quarter row, got {got.height}"
    assert got["ret"][0] == pytest.approx(0.331, rel=1e-12), (
        f"three months of +10% compounded to {got['ret'][0]!r}, not 0.331 -- "
        "0.300 means an arithmetic sum or a x3 scale error, 0.100 a mean, "
        "0.2859 an un-exponentiated log sum"
    )


def test_market_on_market_through_helper_gives_beta_exactly_one():
    """ABSOLUTE ANCHOR #2 (CLAUDE.md, docs/05_report_spec.md's global
    rule 2): a series compounded by this helper, regressed on ITSELF
    through the same full_sample_market_loading code path the report
    uses, must give beta = 1.0 and alpha = 0.0.

    This idiom already exists in _scratch/longonly_real_run.py's
    _market_on_market_check but has never been a pytest assertion. It
    catches a misaligned or shifted join INSIDE the helper -- a defect
    that leaves the returned series looking entirely plausible.
    """
    quarters = [
        datetime.date(1990, 3, 31),
        datetime.date(1990, 6, 30),
        datetime.date(1990, 9, 30),
        datetime.date(1990, 12, 31),
        datetime.date(1991, 3, 31),
    ]
    rows = []
    value = 0.01
    for q in quarters:
        for m in _months_of_quarter(q):
            rows.append((m, value))
            value += 0.003
    monthly = _monthly(rows)

    got = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=quarters, value_col="ret"
    )
    series = got["ret"].to_list()

    loading = diagnostics.full_sample_market_loading(series, series)

    assert loading["beta"] == pytest.approx(1.0, abs=1e-9), (
        f"series regressed on itself gave beta={loading['beta']!r}, not 1.0 -- "
        "the helper has a level, scale or alignment defect"
    )
    assert loading["alpha"] == pytest.approx(0.0, abs=1e-12), (
        f"series regressed on itself gave alpha={loading['alpha']!r}, not 0.0"
    )


# ---------------------------------------------------------------------------
# n_months -- the partial-quarter detector. This column is the ONLY thing
# distinguishing a correct series from the silently-corrupt one measured
# in _scratch/_probe_ca_align_out.txt (Canada: 2 of 3 months in 110 of
# 148 quarters, yet every quarter returned a plausible number).
# ---------------------------------------------------------------------------


def test_n_months_is_three_for_a_complete_quarter():
    monthly = _monthly([(m, 0.01) for m in _Q1_1990_MONTHS])

    got = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=[_Q1_1990], value_col="ret"
    )

    assert got["n_months"][0] == 3, (
        f"a quarter with all 3 months reported n_months={got['n_months'][0]}"
    )


def test_n_months_is_two_when_one_month_is_absent():
    """The exact fingerprint of the measured Canadian re-keying defect:
    the compounded value still looks like a plausible return, and only
    n_months reveals that a third of the quarter is missing."""
    monthly = _monthly([(_Q1_1990_MONTHS[0], 0.01), (_Q1_1990_MONTHS[2], 0.01)])

    got = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=[_Q1_1990], value_col="ret"
    )

    assert got.height == 1, "a 2-of-3-month quarter must still produce a row"
    assert got["n_months"][0] == 2, (
        f"a quarter missing one month reported n_months={got['n_months'][0]}, not 2 "
        "-- the partial-quarter detector is blind, which is the whole defect "
        "this column exists to catch"
    )
    # The value is plausible, which is precisely why n_months is needed.
    assert got["ret"][0] == pytest.approx(1.01 * 1.01 - 1.0, rel=1e-12)


def test_quarter_with_no_months_yields_no_row():
    """Mirrors longonly._quarterly_return's own contract (a name absent
    entirely produces NO row, never a fabricated 0%) and preserves
    _grouped_quarterly_analysis's existing behaviour, which skips such a
    quarter rather than joining a zero against it."""
    monthly = _monthly([(datetime.date(1995, 6, 30), 0.01)])

    got = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=[_Q1_1990], value_col="ret"
    )

    assert got.height == 0, (
        f"a quarter with zero matching months produced {got.height} row(s) -- "
        "a fabricated 0% quarter is exactly what this contract forbids"
    )


# ---------------------------------------------------------------------------
# Delegation -- proof there is only ONE compounding implementation.
# ---------------------------------------------------------------------------


def test_matches_quarterly_return_directly_on_same_input():
    """CLAUDE.md: a second implementation of quarterly compounding is
    how the two paths diverge. This asserts EXACT float equality against
    a direct longonly._quarterly_return call, so any future edit that
    reimplements rather than delegates turns this test red."""
    rows = [
        (_Q1_1990_MONTHS[0], 0.0234),
        (_Q1_1990_MONTHS[1], -0.0157),
        (_Q1_1990_MONTHS[2], 0.0411),
    ]
    monthly = _monthly(rows)

    got = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=[_Q1_1990], value_col="ret"
    )

    direct_input = monthly.with_columns(pl.lit("__x__").alias("_id"))
    direct = _quarterly_return(direct_input, "_id", _Q1_1990)

    assert got["ret"][0] == direct["ret"][0], (
        f"helper gave {got['ret'][0]!r}, direct _quarterly_return gave "
        f"{direct['ret'][0]!r} -- the helper is no longer delegating"
    )


def test_value_col_other_than_ret_is_compounded_and_named():
    """The risk-free path passes value_col='rf'. A helper that hardcoded
    'ret' internally would either crash or silently compound the wrong
    column."""
    monthly = _monthly([(m, 0.01) for m in _Q1_1990_MONTHS], value_col="rf")

    got = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=[_Q1_1990], value_col="rf"
    )

    assert "rf" in got.columns, f"expected an 'rf' column, got {got.columns}"
    assert got["rf"][0] == pytest.approx(1.01**3 - 1.0, rel=1e-12)


def test_missing_value_col_raises():
    """A typo'd column name must fail loudly rather than propagate nulls
    into a reported return."""
    monthly = _monthly([(m, 0.01) for m in _Q1_1990_MONTHS])

    with pytest.raises((ValueError, pl.exceptions.ColumnNotFoundError)):
        quarterly_compounding.compound_monthly_series_to_quarters(
            monthly, quarters=[_Q1_1990], value_col="not_a_column"
        )


def test_quarters_are_returned_in_requested_order():
    """A regression joins this frame against another on `quarter`; a
    shuffled return order would still join correctly, but any caller
    zipping two series positionally would silently misalign them."""
    quarters = [
        datetime.date(1990, 3, 31),
        datetime.date(1990, 6, 30),
        datetime.date(1990, 9, 30),
    ]
    rows = []
    for i, q in enumerate(quarters):
        for m in _months_of_quarter(q):
            rows.append((m, 0.01 * (i + 1)))
    monthly = _monthly(rows)

    got = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=quarters, value_col="ret"
    )

    assert got["quarter"].to_list() == quarters, (
        f"quarters returned as {got['quarter'].to_list()}, not in requested order"
    )


# ---------------------------------------------------------------------------
# assert_full_quarters -- the opt-in guard. Its OWN ability to fire is
# tested here, per CLAUDE.md's rule that a check must be demonstrated
# capable of failing.
# ---------------------------------------------------------------------------


def test_assert_full_quarters_passes_on_complete_quarters():
    monthly = _monthly([(m, 0.01) for m in _Q1_1990_MONTHS])
    compounded = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=[_Q1_1990], value_col="ret"
    )

    quarterly_compounding.assert_full_quarters(
        compounded, quarters=[_Q1_1990], label="synthetic"
    )


def test_assert_full_quarters_raises_on_partial_quarter():
    """Demonstrates the guard CAN fail -- this is the assertion that
    would have caught the measured Canadian defect (110 of 148 quarters
    compounding only 2 months while returning plausible numbers)."""
    monthly = _monthly([(_Q1_1990_MONTHS[0], 0.01), (_Q1_1990_MONTHS[2], 0.01)])
    compounded = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=[_Q1_1990], value_col="ret"
    )

    with pytest.raises(ValueError, match="ca_rf_synthetic"):
        quarterly_compounding.assert_full_quarters(
            compounded, quarters=[_Q1_1990], label="ca_rf_synthetic"
        )


def test_assert_full_quarters_raises_on_missing_quarter():
    """A requested quarter absent from the compounded frame entirely --
    distinct from a partial quarter, and equally fatal to a regression
    that assumes aligned series."""
    monthly = _monthly([(m, 0.01) for m in _Q1_1990_MONTHS])
    compounded = quarterly_compounding.compound_monthly_series_to_quarters(
        monthly, quarters=[_Q1_1990], value_col="ret"
    )

    with pytest.raises(ValueError, match="us_market_synthetic"):
        quarterly_compounding.assert_full_quarters(
            compounded,
            quarters=[_Q1_1990, datetime.date(1990, 6, 30)],
            label="us_market_synthetic",
        )


# ---------------------------------------------------------------------------
# Duplicate-month detection and assert_clean_quarterly_series. Added
# 2026-09-15 after a gate-verifier audit found n_months counted ROWS,
# not distinct months, so a duplicated month reported a clean 3 while
# being compounded twice.
# ---------------------------------------------------------------------------


def test_duplicated_month_raises_rather_than_counting_as_three():
    """THE AUDIT FINDING, as a regression test. Before the fix, input
    (Jan, Jan, Feb) with March absent returned ret=0.3310000000000004
    and n_months=3 -- BYTE-IDENTICAL to what a correct (Jan, Feb, Mar)
    quarter returns -- and assert_full_quarters passed it clean.

    That defeated the module's entire justification: the docstring
    claims only the month count can distinguish a correct series from a
    corrupt one, and for this corruption class the count agreed with the
    corrupt series.
    """
    monthly = _monthly(
        [
            (_Q1_1990_MONTHS[0], 0.10),
            (_Q1_1990_MONTHS[0], 0.10),  # January duplicated
            (_Q1_1990_MONTHS[1], 0.10),  # March absent entirely
        ]
    )

    with pytest.raises(ValueError, match="distinct month"):
        quarterly_compounding.compound_monthly_series_to_quarters(
            monthly, quarters=[_Q1_1990], value_col="ret"
        )


def test_clean_series_guard_rejects_a_null_value():
    """A single null gave realized_beta=nan, alpha=nan and
    port_sharpe=nan downstream with no exception, while n_quarters still
    reported the full count -- a structurally complete-looking row
    headed for the output parquet."""
    frame = pl.DataFrame(
        {"quarter": [_Q1_1990, datetime.date(1990, 6, 30)], "ret": [0.01, None]}
    )

    with pytest.raises(ValueError, match="null/NaN/inf"):
        quarterly_compounding.assert_clean_quarterly_series(
            frame, value_col="ret", label="nulls"
        )


def test_clean_series_guard_rejects_a_duplicated_quarter():
    """Measured before the fix: 6 rows over 5 distinct quarters reported
    n_quarters=6, and on a drawdown series a duplicated quarter moved
    max_drawdown from -0.44 to -0.61."""
    frame = pl.DataFrame({"quarter": [_Q1_1990, _Q1_1990], "ret": [0.01, 0.02]})

    with pytest.raises(ValueError, match="distinct quarters"):
        quarterly_compounding.assert_clean_quarterly_series(
            frame, value_col="ret", label="dupes"
        )


def test_clean_series_guard_rejects_a_gap_only_when_contiguity_required():
    """Removing one quarter from the middle of a drawdown series moved
    max_drawdown from -0.44 to -0.30 with no signal, because a
    path-dependent statistic compounds straight across the gap.

    Opt-in, because a legitimately skipped formation quarter is not an
    error for a regression -- only for a path-dependent statistic.
    """
    frame = pl.DataFrame(
        {
            "quarter": [_Q1_1990, datetime.date(1990, 12, 31)],  # Jun/Sep missing
            "ret": [0.01, 0.02],
        }
    )

    # Default: gaps tolerated.
    quarterly_compounding.assert_clean_quarterly_series(
        frame, value_col="ret", label="gappy"
    )

    with pytest.raises(ValueError, match="gap"):
        quarterly_compounding.assert_clean_quarterly_series(
            frame, value_col="ret", label="gappy", require_contiguous=True
        )


def test_clean_series_guard_passes_a_well_formed_series():
    frame = pl.DataFrame(
        {
            "quarter": [
                _Q1_1990,
                datetime.date(1990, 6, 30),
                datetime.date(1990, 9, 30),
            ],
            "ret": [0.01, 0.02, -0.01],
        }
    )

    quarterly_compounding.assert_clean_quarterly_series(
        frame, value_col="ret", label="clean", require_contiguous=True
    )
