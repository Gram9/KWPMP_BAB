"""Tests for src/portfolio/rebalance.py -- the monthly formation/holding
loop, F2b (docs/03_roadmap.md). Ties weights.rank_deviation_from_median,
legs.asymmetric_inverse_beta/leg_ex_ante_beta, and
diagnostics.compute_diagnostics together across a date range.

run_backtest() is a PURE frame-in/frame-out core: it never fetches data
(no WRDS, no universe_at, no market_index). Production wires
estimate_beta_fp() into it; tests/leakage/test_no_lookahead.py's
run_pipeline adapter wires the E2 fixture + precomputed_betas into the
SAME loop -- one code path, two callers, which is what makes the
leakage tests exercise production's actual construction rather than a
parallel test-only implementation.
"""

import datetime

import polars as pl
import pytest

from src.portfolio import legs, rebalance, weights


def _cross_section(permnos, betas, mkt_caps=None):
    data = {"permno": permnos, "beta_shrunk": betas}
    if mkt_caps is not None:
        data["mkt_cap"] = mkt_caps
    return pl.DataFrame(data)


def _monthly_rets(rows):
    """rows: list of (permno, month, ret)."""
    permnos, months, rets = zip(*rows)
    return pl.DataFrame({"permno": list(permnos), "month": list(months), "ret": list(rets)})


def _monthly_rf(rows):
    """rows: list of (month, rf)."""
    months, rfs = zip(*rows)
    return pl.DataFrame({"month": list(months), "rf": list(rfs)})


# ---------------------------------------------------------------------------
# Formation timing -- the rule that overrides everything (CLAUDE.md)
# ---------------------------------------------------------------------------


def test_formation_date_beta_is_used_for_the_FOLLOWING_months_return():
    """Betas estimated through t-1 must be used for month t's HELD return
    -- CLAUDE.md's core rule, spec section 10. Constructed so using the
    WRONG month's return (same-month instead of next-month) would change
    the output: month 1's return is 0.0 everywhere (a no-op), month 2's
    return is where all the signal lives. If the loop ever indexes
    monthly_rets at the FORMATION month rather than the HELD month, the
    January formation would incorrectly earn the January (zero) return
    instead of February's, and the resulting BAB return would be 0.0
    instead of nonzero.
    """
    jan = datetime.date(2020, 1, 31)
    feb = datetime.date(2020, 2, 29)

    betas_by_date = {
        jan: _cross_section([1, 2, 3, 4], [0.4, 0.8, 1.2, 1.8]),
    }
    monthly_rets = _monthly_rets(
        [
            (1, jan, 0.0), (2, jan, 0.0), (3, jan, 0.0), (4, jan, 0.0),
            (1, feb, 0.05), (2, feb, 0.02), (3, feb, -0.01), (4, feb, -0.05),
        ]
    )
    monthly_rf = _monthly_rf([(jan, 0.0), (feb, 0.0)])

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=1, min_names_per_leg=1, max_consecutive_skips=10,
    )

    assert feb in result.returns["month"].to_list(), (
        "the beta formed at jan's month-end must produce a return for "
        "FEBRUARY (the held month), not January"
    )
    feb_ret = _get_month_return(result, feb)
    assert feb_ret != pytest.approx(0.0, abs=1e-9), (
        "using January's (all-zero) return instead of February's for the "
        "jan-formed beta would produce exactly 0.0 -- got 0.0, which is "
        "the exact signature of the formation/holding timing bug this "
        "test exists to catch"
    )


def test_run_backtest_never_uses_formation_months_own_return():
    """Direct structural check: the loop must not be capable of indexing
    monthly_rets at the formation month for the held return. Two formation
    dates in adjacent months, each month's return engineered to a
    DISTINCT, recognizable value -- if month t's beta ever earned month
    t's own return, the output would match that month's engineered value
    rather than the following month's.
    """
    jan = datetime.date(2020, 1, 31)
    feb = datetime.date(2020, 2, 29)
    mar = datetime.date(2020, 3, 31)

    betas_by_date = {
        jan: _cross_section([1, 2, 3, 4], [0.4, 0.8, 1.2, 1.8]),
        feb: _cross_section([1, 2, 3, 4], [0.5, 0.9, 1.1, 1.7]),
    }
    # Distinct, recognizable per-month values -- 100x apart so a same-
    # month bug is unmistakable in the output.
    monthly_rets = _monthly_rets(
        [
            (1, jan, 1.00), (2, jan, 1.00), (3, jan, -1.00), (4, jan, -1.00),
            (1, feb, 0.01), (2, feb, 0.01), (3, feb, -0.01), (4, feb, -0.01),
            (1, mar, 0.02), (2, mar, 0.02), (3, mar, -0.02), (4, mar, -0.02),
        ]
    )
    monthly_rf = _monthly_rf([(jan, 0.0), (feb, 0.0), (mar, 0.0)])

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=1, min_names_per_leg=1, max_consecutive_skips=10,
    )

    jan_formed_ret = _get_month_return(result, feb)
    feb_formed_ret = _get_month_return(result, mar)

    # jan's engineered return magnitude is 1.00 -- two orders of magnitude
    # larger than feb's (0.01) or mar's (0.02). If jan's beta incorrectly
    # earned jan's own return, the output would be of order 1.0, not
    # order 0.01.
    assert abs(jan_formed_ret) < 0.5, (
        f"jan-formed beta's return is {jan_formed_ret} -- of the same "
        "order as JANUARY's own engineered return (1.00), suggesting the "
        "loop used the formation month's own return instead of the held "
        "month's (February's, order 0.01)"
    )


def _get_month_return(result, month):
    returns = result.returns
    if isinstance(returns, pl.DataFrame):
        row = returns.filter(pl.col("month") == month)
        assert row.height == 1
        return row["ret"].item()
    # pandas-like fallback
    return returns.loc[month]


# ---------------------------------------------------------------------------
# Weight/leg-scaling composition -- correct wiring of existing modules
# ---------------------------------------------------------------------------


def test_run_backtest_composes_existing_weights_legs_diagnostics_not_reimplemented():
    """rebalance.py must WIRE weights.rank_deviation_from_median and
    legs.asymmetric_inverse_beta/leg_ex_ante_beta, not re-derive their
    math. Verified by hand-computing the expected BAB return via those
    same functions directly and comparing to run_backtest()'s output for
    a single formation/holding pair.
    """
    month1 = datetime.date(2020, 1, 31)
    month2 = datetime.date(2020, 2, 29)

    permnos = [1, 2, 3, 4, 5]
    beta_vals = [0.3, 0.6, 1.0, 1.5, 2.2]
    betas_by_date = {month1: _cross_section(permnos, beta_vals)}

    rets = [0.03, -0.01, 0.02, -0.04, 0.05]
    monthly_rets = _monthly_rets([(p, month2, r) for p, r in zip(permnos, rets)]
                                  + [(p, month1, 0.0) for p in permnos])
    rf = 0.001
    monthly_rf = _monthly_rf([(month1, 0.0), (month2, rf)])

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=1, min_names_per_leg=1, max_consecutive_skips=10,
    )

    # Hand-derive via the SAME primitives rebalance.py must call.
    weighted = weights.rank_deviation_from_median(_cross_section(permnos, beta_vals))
    beta_long = legs.leg_ex_ante_beta(weighted, weight_col="weight_long", beta_col="beta_shrunk")
    beta_high = legs.leg_ex_ante_beta(weighted, weight_col="weight_short", beta_col="beta_shrunk")

    ret_by_permno = dict(zip(permnos, rets))
    w = weighted.to_dicts()
    r_long = sum(row["weight_long"] * ret_by_permno[row["permno"]] for row in w)
    r_short = sum(row["weight_short"] * ret_by_permno[row["permno"]] for row in w)

    expected = legs.asymmetric_inverse_beta(
        r_long=r_long, r_short=r_short, beta_long=beta_long, beta_high=beta_high, r_f=rf
    )

    actual = _get_month_return(result, month2)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_run_backtest_emits_diagnostics_with_realized_market_loading_none():
    """D-b: realized_market_loading is inherently a t+1 quantity (a
    regression of REALIZED returns against the market, only knowable
    after the holding period ends). Per-month diagnostic rows must carry
    None/null for it -- it is computed post-loop, full-sample, as a
    reporting-only statistic that nothing inside the loop reads."""
    month1 = datetime.date(2020, 1, 31)
    month2 = datetime.date(2020, 2, 29)
    permnos = [1, 2, 3, 4]
    betas_by_date = {month1: _cross_section(permnos, [0.4, 0.8, 1.2, 1.8])}
    monthly_rets = _monthly_rets(
        [(p, month1, 0.0) for p in permnos] + [(p, month2, 0.01 * i) for i, p in enumerate(permnos)]
    )
    monthly_rf = _monthly_rf([(month1, 0.0), (month2, 0.0)])

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=1, min_names_per_leg=1, max_consecutive_skips=10,
    )

    diag_row = _get_diagnostics_row(result, month2)
    assert diag_row["realized_market_loading"] is None


def _get_diagnostics_row(result, month):
    diag = result.diagnostics
    if isinstance(diag, pl.DataFrame):
        row = diag.filter(pl.col("holding_month") == month)
        assert row.height == 1
        return row.to_dicts()[0]
    return dict(diag.loc[month])


# ---------------------------------------------------------------------------
# Skip handling -- D-c: two thresholds, first-class asserted output
# ---------------------------------------------------------------------------


def test_thin_universe_below_min_names_total_is_skipped():
    """A formation date with fewer names than min_names_total must emit
    NO return for that month and record a skip reason -- not a degenerate
    2-3 name 'portfolio'."""
    month1 = datetime.date(2020, 1, 31)
    month2 = datetime.date(2020, 2, 29)
    betas_by_date = {month1: _cross_section([1, 2], [0.4, 1.8])}  # only 2 names
    monthly_rets = _monthly_rets([(1, month1, 0.0), (2, month1, 0.0),
                                   (1, month2, 0.01), (2, month2, -0.01)])
    monthly_rf = _monthly_rf([(month1, 0.0), (month2, 0.0)])

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=10, min_names_per_leg=1, max_consecutive_skips=10,
    )

    assert month1 in result.skips
    assert "total" in result.skips[month1].lower()
    returns_months = (
        result.returns["month"].to_list()
        if isinstance(result.returns, pl.DataFrame)
        else list(result.returns.index)
    )
    assert month2 not in returns_months, (
        "a formation date skipped for thin universe must emit NO return "
        "for its held month"
    )


def test_empty_leg_below_min_names_per_leg_is_skipped_with_distinct_reason():
    """An empty/thin LEG is a different failure than a thin universe and
    must be distinguishable in the skip reason string. All betas on one
    side of the median (rank-deviation weighting can produce a
    near-empty short leg when many names tie or cluster) must be
    caught by min_names_per_leg with a reason mentioning the leg, not
    reused generically."""
    month1 = datetime.date(2020, 1, 31)
    month2 = datetime.date(2020, 2, 29)
    # 10 names total (clears min_names_total) but constructed so one
    # side of the rank-deviation split has very few names with nonzero
    # weight -- use min_names_per_leg high enough to force the skip.
    betas_by_date = {
        month1: _cross_section(list(range(1, 11)), [float(i) for i in range(1, 11)])
    }
    monthly_rets = _monthly_rets(
        [(p, month1, 0.0) for p in range(1, 11)]
        + [(p, month2, 0.01) for p in range(1, 11)]
    )
    monthly_rf = _monthly_rf([(month1, 0.0), (month2, 0.0)])

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=1, min_names_per_leg=100, max_consecutive_skips=10,
    )

    assert month1 in result.skips
    assert "leg" in result.skips[month1].lower()


def test_max_consecutive_skips_raises():
    """A long run of skipped months means something is broken upstream
    (a universe filter gone wrong, a config threshold misconfigured) --
    the loop must raise rather than silently return a mostly-empty
    return series. A few thin months at a sample's start is normal and
    must NOT raise (proven by the passing case in
    test_thin_universe_below_min_names_total_is_skipped, which uses the
    same skip path with max_consecutive_skips=10 and does not raise)."""
    months = [datetime.date(2020, m, 1) + datetime.timedelta(days=27) for m in range(1, 9)]
    betas_by_date = {m: _cross_section([1, 2], [0.4, 1.8]) for m in months}  # always thin
    rows = []
    for m in months:
        rows.append((1, m, 0.0))
        rows.append((2, m, 0.0))
    monthly_rets = _monthly_rets(rows)
    monthly_rf = _monthly_rf([(m, 0.0) for m in months])

    with pytest.raises(RuntimeError, match="consecutive"):
        rebalance.run_backtest(
            betas_by_date, monthly_rets, monthly_rf,
            min_names_total=10, min_names_per_leg=1, max_consecutive_skips=3,
        )


def test_max_consecutive_skips_resets_after_a_successful_month():
    """The circuit breaker counts CONSECUTIVE skips -- a successful month
    in between must reset the counter, so a sample with occasional thin
    months scattered throughout (not clustered) never trips it."""
    thin_month_a = datetime.date(2020, 1, 31)
    good_month = datetime.date(2020, 2, 29)
    thin_month_b = datetime.date(2020, 3, 31)
    held_a = datetime.date(2020, 2, 29)
    held_good = datetime.date(2020, 3, 31)
    held_b = datetime.date(2020, 4, 30)

    betas_by_date = {
        thin_month_a: _cross_section([1, 2], [0.4, 1.8]),
        good_month: _cross_section(list(range(1, 11)), [float(i) for i in range(1, 11)]),
        thin_month_b: _cross_section([1, 2], [0.4, 1.8]),
    }
    rows = [(1, thin_month_a, 0.0), (2, thin_month_a, 0.0)]
    rows += [(p, good_month, 0.0) for p in range(1, 11)]
    rows += [(1, thin_month_b, 0.0), (2, thin_month_b, 0.0)]
    rows += [(1, held_a, 0.01), (2, held_a, -0.01)]
    rows += [(p, held_good, 0.01) for p in range(1, 11)]
    rows += [(1, held_b, 0.01), (2, held_b, -0.01)]
    monthly_rets = _monthly_rets(rows)
    monthly_rf = _monthly_rf(
        [(thin_month_a, 0.0), (good_month, 0.0), (thin_month_b, 0.0),
         (held_a, 0.0), (held_good, 0.0), (held_b, 0.0)]
    )

    # max_consecutive_skips=1 would trip on two ADJACENT skips but must
    # NOT trip here since a good month sits between the two thin ones.
    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=10, min_names_per_leg=1, max_consecutive_skips=1,
    )
    assert thin_month_a in result.skips
    assert thin_month_b in result.skips
    assert good_month not in result.skips


# ---------------------------------------------------------------------------
# Missing held-month data -- a fabricated 0.0 is worse than a skip
# ---------------------------------------------------------------------------


def test_formation_with_no_data_for_held_month_is_skipped_not_zero():
    """CLAUDE.md: 'a wrong number that looks right is the worst possible
    outcome.' If a formation date's held month has NO rows at all in
    monthly_rets (e.g. the caller truncated monthly_rets at a boundary T
    that falls one month before this formation's held month -- exactly
    tests/leakage/test_no_lookahead.py's own truncation-test shape), the
    loop must SKIP that formation date and record why, never silently
    sum an empty ret_by_permno to 0.0 and emit a fabricated
    exactly-zero return. A real return is essentially never bit-exact
    0.0; a missing-data return computed as sum([]) always is -- this is
    precisely the "wrong number that looks right" failure mode.

    Reproduces a REAL bug found via tests/leakage/test_no_lookahead.py's
    truncation test: the last formation date in a truncated daily panel
    produces a held month past the truncation boundary, for which
    monthly_rets has no data -- and the pre-fix loop emitted
    ret=0.000000 for that month instead of skipping it.
    """
    month1 = datetime.date(2020, 1, 31)
    # Deliberately NO row for month2 (the held month) anywhere in
    # monthly_rets/monthly_rf -- simulates a caller-truncated panel
    # whose last formation's held month falls just past the cutoff.
    month2 = datetime.date(2020, 2, 29)
    permnos = [1, 2, 3, 4]
    betas_by_date = {month1: _cross_section(permnos, [0.4, 0.8, 1.2, 1.8])}
    monthly_rets = _monthly_rets([(p, month1, 0.0) for p in permnos])
    monthly_rf = _monthly_rf([(month1, 0.0)])

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=1, min_names_per_leg=1, max_consecutive_skips=10,
    )

    assert month1 in result.skips, (
        "a formation date whose held month has NO data at all must be "
        "skipped, not silently produce a fabricated 0.0 return"
    )
    assert "held" in result.skips[month1].lower() or "data" in result.skips[month1].lower()
    returns_months = (
        result.returns["month"].to_list()
        if isinstance(result.returns, pl.DataFrame)
        else list(result.returns.index)
    )
    assert month2 not in returns_months


def test_formation_with_partial_leg_coverage_for_held_month_is_skipped():
    """A held month with SOME but not ALL of a leg's names present (a
    partial-coverage month, distinct from zero-coverage) must also be
    caught -- not silently renormalize weights over the names that
    happen to have data, which would change what fraction of the leg
    each surviving name represents without recording that anything
    was amiss. Constructed so most of one leg's names are missing from
    monthly_rets for the held month."""
    month1 = datetime.date(2020, 1, 31)
    month2 = datetime.date(2020, 2, 29)
    permnos = [1, 2, 3, 4, 5, 6]
    betas_by_date = {month1: _cross_section(permnos, [0.2, 0.4, 0.6, 1.4, 1.6, 1.8])}
    # Only permno 1 (of the long leg) and permno 6 (of the short leg)
    # have a held-month return -- 2,3 and 4,5 are missing entirely.
    monthly_rets = _monthly_rets(
        [(p, month1, 0.0) for p in permnos] + [(1, month2, 0.01), (6, month2, -0.01)]
    )
    monthly_rf = _monthly_rf([(month1, 0.0), (month2, 0.0)])

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=1, min_names_per_leg=1, max_consecutive_skips=10,
    )

    assert month1 in result.skips, (
        "a held month missing returns for most of a leg's names must be "
        "skipped, not silently renormalize weights over whichever names "
        "happen to have data"
    )


# ---------------------------------------------------------------------------
# Date tracking -- prerequisite for open item (d)
# ---------------------------------------------------------------------------


def test_run_backtest_emits_per_period_date_tracking():
    """Open item (d) needs max(estimation_date) < min(holding_return_date)
    assertable on REAL pipeline dates. The loop must carry
    (formation_date, estimation_end, holding_start, holding_end) per
    period somewhere in its output -- diagnostics is the natural place,
    since it is already emitted every period unconditionally (D8)."""
    month1 = datetime.date(2020, 1, 31)
    month2 = datetime.date(2020, 2, 29)
    permnos = [1, 2, 3, 4]
    betas_by_date = {month1: _cross_section(permnos, [0.4, 0.8, 1.2, 1.8])}
    monthly_rets = _monthly_rets(
        [(p, month1, 0.0) for p in permnos] + [(p, month2, 0.01) for p in permnos]
    )
    monthly_rf = _monthly_rf([(month1, 0.0), (month2, 0.0)])

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets, monthly_rf,
        min_names_total=1, min_names_per_leg=1, max_consecutive_skips=10,
    )

    diag_row = _get_diagnostics_row(result, month2)
    for key in ("formation_date", "holding_month"):
        assert key in diag_row, f"diagnostics row missing date-tracking field {key!r}"
    assert diag_row["formation_date"] == month1
    assert diag_row["holding_month"] == month2
