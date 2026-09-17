"""
Leakage tests. These are the difference between a backtest you can present and
one you can't.

Fill in the four adapter functions at the top against your actual pipeline.
Everything below them is generic and should not need changing.

Run: pytest tests/leakage/ -x
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# ADAPTERS — wire these to your pipeline, leave the tests alone
# ---------------------------------------------------------------------------

def run_pipeline(daily_rets, daily_mkt, monthly_rets, monthly_rf, **kwargs):
    """Full pipeline: returns a monthly Series of BAB returns.

    F4 (docs/03_roadmap.md): a pandas<->polars ADAPTER over
    src.portfolio.rebalance.run_backtest -- all portfolio math (weighting,
    leg scaling, diagnostics, skip handling) lives there, not here. This
    function does exactly three things: (1) get a beta cross-section per
    calendar month (via estimate_betas(), unless the caller supplies
    precomputed_betas -- the shuffled-signal test's own null-test
    convention), picking the LAST available beta date within each month
    as that month's formation date; (2) convert daily_rets'/daily_mkt's
    columns into the long-format polars frames run_backtest expects; (3)
    convert the polars BacktestResult.returns back to a pandas Series,
    since that is the type both leakage tests assert on
    (pd.testing.assert_series_equal, .mean(), .std()).

    precomputed_betas (accepted via **kwargs, not a named parameter --
    test_shuffled_signal_produces_no_alpha's own calling convention):
    a pandas DataFrame, date x security, values beta_shrunk -- EXACTLY
    estimate_betas()'s own return shape. When given, used AS-IS instead
    of calling estimate_betas() again -- this is what lets the shuffled-
    signal test destroy cross-sectional signal while preserving every
    other part of construction (panel structure, weighting, leg scaling).

    monthly_rets/monthly_rf may be None (test_shuffled_signal_produces_
    no_alpha's own load_test_panel() call passes them through unused
    when... no: as of F4 both are always real, load_test_panel() no
    longer returns None/None. Accepting None defensively here anyway
    costs nothing and matches this adapter's own one-line docstring,
    which promises nothing about non-null inputs.
    """

    import polars as pl

    from src.portfolio import rebalance

    betas = kwargs.get("precomputed_betas")
    if betas is None:
        betas = estimate_betas(daily_rets, daily_mkt)

    if monthly_rets is None or monthly_rf is None:
        raise ValueError(
            "run_pipeline requires real monthly_rets/monthly_rf -- "
            "load_test_panel() no longer returns None/None as of F4"
        )

    # One formation date per CALENDAR MONTH: the LAST beta-bearing date
    # within that month. betas' index is a DatetimeIndex of trading
    # dates (estimate_betas() emits one row per date with enough
    # trailing history); grouping by month-end period and taking the
    # max index value per group picks the latest trading day in each
    # month that actually has a beta row -- never a date past the
    # month's own trading calendar, and never more than one formation
    # per calendar month (multiple formations in one month would let
    # rebalance.run_backtest's _next_month_end lookup collide).
    month_of = betas.index.to_period("M")
    formation_dates = betas.groupby(month_of).apply(lambda g: g.index.max())

    betas_by_date: dict = {}
    for formation_date in formation_dates.to_list():
        row = betas.loc[formation_date].dropna()
        if row.empty:
            continue
        # rebalance.run_backtest keys betas_by_date by a MONTH-END date
        # (it advances by calendar month, per its own _next_month_end),
        # not by the trading date the beta happened to be estimated on
        # -- reindex the formation key to that trading date's own
        # month-end so run_backtest's month-advance lands on the next
        # REAL calendar month, matching monthly_rets'/monthly_rf's own
        # month-end index exactly.
        key = (formation_date + pd.offsets.MonthEnd(0)).date()
        betas_by_date[key] = pl.DataFrame(
            {"permno": row.index.to_list(), "beta_shrunk": row.to_numpy()}
        )

    monthly_rets_long = monthly_rets.reset_index().melt(
        id_vars=monthly_rets.index.name or "index", var_name="permno", value_name="ret"
    ).rename(columns={monthly_rets.index.name or "index": "month"})
    monthly_rets_long = monthly_rets_long.dropna(subset=["ret"])
    monthly_rets_pl = pl.DataFrame(
        {
            "permno": monthly_rets_long["permno"].to_list(),
            "month": [d.date() for d in monthly_rets_long["month"]],
            "ret": monthly_rets_long["ret"].to_numpy(),
        }
    )

    monthly_rf_df = monthly_rf.reset_index()
    monthly_rf_df.columns = ["month", "rf"]
    monthly_rf_pl = pl.DataFrame(
        {
            "month": [d.date() for d in monthly_rf_df["month"]],
            "rf": monthly_rf_df["rf"].to_numpy(),
        }
    )

    result = rebalance.run_backtest(
        betas_by_date,
        monthly_rets_pl,
        monthly_rf_pl,
        # Measured 2026-09-12, re-measured after extending the window to
        # 2018-12-31 (see load_test_panel's own docstring for why): this
        # 40-name fixture produces a real 24-consecutive-skip run around
        # 2010-2011 under run_backtest's full-leg-coverage rule (any
        # weighted name missing a held return skips the month, closing a
        # fabricated-exactly-0.0-return bug). The run is 24, not the
        # original window's 20 -- the SAME 2010-2011 attrition cluster,
        # unaffected by extending the window's tail; the count moved
        # only because more total formation dates exist to skip around
        # it (108 attempted vs. 66 before). 18 of the fixture's 40 names
        # delist at real, NON-clustered dates spread across 2005-2018 --
        # this fixture's own deliberate design (its docstring: names
        # "are free to have gaps or delist partway through"), not a
        # data defect. Dropping the affected names to shorten the run
        # would reintroduce the exact survivorship filter a prior
        # leakage-auditor finding (2026-09-10) already fixed here.
        # config/portfolio.yaml's production default (6) is unchanged
        # and stays appropriate for a real, large universe where a
        # 24-month clustered gap would be a genuine red flag. 25 still
        # has margin (24 < 25); if a future window change pushes the
        # run to 25+, re-measure rather than bumping this blindly.
        max_consecutive_skips=25,
    )

    returns_pd = result.returns.to_pandas()
    if returns_pd.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    returns_pd["month"] = pd.to_datetime(returns_pd["month"])
    series = returns_pd.set_index("month")["ret"].sort_index()
    series.attrs["skips"] = {
        pd.Timestamp(k): v for k, v in result.skips.items()
    }
    return series


def estimate_betas(daily_rets, daily_mkt, **kwargs):
    """Returns DataFrame of daily-updated betas (date x security).

    E2 leakage adapter (docs/03_roadmap.md): thin pandas-facing wrapper
    around src.estimation.beta_fp._estimate_beta_fp_core -- the SAME
    formation-timing-correct math Gate 3's own tests exercise, not a
    second independent implementation. daily_rets/daily_mkt are ARITHMETIC
    returns (this test file's own convention, e.g. rng.normal(0.0003,
    0.01, ...) fed straight in as daily_mkt) -- converted to log returns
    here (ln(1+r)) to match _estimate_beta_fp_core's expected input,
    mirroring src/estimation/beta_fp.py's _daily_log_returns() convention
    for real data.

    Returns a REAL date x security panel, not a single most-recent row:
    every date with enough trailing history (independently, per security:
    sigma>=120 obs, rho>=750 obs, straight from config/beta_estimator.yaml's
    fp_spec block -- rho is the binding constraint, so a security's first
    beta appears once its trailing 750-day window clears) gets its own
    beta, computed using ONLY data strictly before that date. This is NOT
    a simplification for test convenience -- test_betas_are_point_in_time
    truncates two calls' outputs by date and intersects their indices; a
    single-row (most-recent-date-only) output would make that
    intersection vacuously empty, so the test would pass without checking
    anything. A real multi-date panel is the only way that test is a
    genuine check rather than a structurally blind one.
    """
    import datetime as _dt

    import polars as pl

    from src.estimation import beta_fp

    cfg = beta_fp._load_beta_config()
    min_days_needed = cfg["rho_window_days"]  # rho is the binding constraint

    dates = list(daily_rets.index)
    security_cols = list(daily_rets.columns)

    # Pre-convert the whole panel to log returns and long-format polars
    # ONCE -- the per-date loop below only slices this, it never
    # re-derives log returns or re-touches the pandas input per date.
    # errstate suppresses the benign divide-by-zero warning ln(1+r) hits
    # for a genuine r=-1.0 (a real -100% return, e.g. a delisting-to-zero
    # event) -- log_ret=-inf there is mathematically correct, not a bug.
    with np.errstate(divide="ignore"):
        market_log_ret = np.log1p(daily_mkt.to_numpy())
        stock_log_ret = np.log1p(daily_rets.to_numpy())  # shape (n_days, n_securities)
    market_returns_pl = pl.DataFrame(
        {"date": [d.date() if hasattr(d, "date") else d for d in dates],
         "log_ret": market_log_ret}
    )

    daily_returns_pl = pl.concat(
        [
            pl.DataFrame(
                {
                    "permno": [i] * len(dates),
                    "date": [d.date() if hasattr(d, "date") else d for d in dates],
                    "log_ret": stock_log_ret[:, i],
                }
            )
            for i in range(len(security_cols))
        ]
    )
    # Reversible int-id <-> original-column-name mapping -- _estimate_beta_fp_core's
    # joins are Int64-typed (real permnos), but this test file's column
    # names are arbitrary strings ("S0", "S1", ...) or ints depending on
    # the caller. Map by POSITION, not by parsing the name, so this works
    # regardless of the caller's column-naming scheme.
    id_to_column = dict(enumerate(security_cols))

    rows = {}
    for t, as_of in enumerate(dates):
        if t < min_days_needed:
            continue  # cannot possibly clear rho_min_obs yet
        as_of_date = as_of.date() if hasattr(as_of, "date") else as_of
        month_end = as_of_date + _dt.timedelta(days=1)  # boundary is STRICTLY before as_of_date+1,
        # i.e. strictly before "tomorrow" -- so as_of_date's own return is the
        # last one included, never anything after it.

        betas_row = beta_fp._estimate_beta_fp_core(
            month_end, daily_returns_pl, market_returns_pl
        )
        if betas_row.height == 0:
            continue
        row_values = {
            id_to_column[permno]: beta_shrunk
            for permno, beta_shrunk in zip(
                betas_row["permno"].to_list(), betas_row["beta_shrunk"].to_list()
            )
        }
        rows[as_of] = row_values

    if not rows:
        return pd.DataFrame(columns=security_cols)
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=security_cols)


def build_universe(as_of_date):
    """Returns the set of security ids (CRSP permnos, as ints) eligible at
    as_of_date. Thin shim over src/data/universe_panel.py::universe_at() --
    see that module for the point-in-time resolution logic (trading-day
    backward resolution, listing/delisting bounds, exchange/REIT filters)."""
    from src.data.universe_panel import universe_at

    universe = universe_at(as_of_date.date())
    return set(universe["permno"].to_list())


def load_test_panel():
    """Small cached slice of real data for integration-style tests.

    E2 (docs/03_roadmap.md): built from a fixed real window
    (2005-01-01..2018-12-31, ~14 years) via src.estimation.beta_fp's
    existing _daily_log_returns()/_market_log_return_series(), converted
    from log back to arithmetic returns (this test file's own convention)
    and pivoted long-to-wide.

    Window extended from 2015-06-30 to 2018-12-31 (2026-09-12), same
    reason as the ORIGINAL 2013-2015-to-2005-2015 extension below: the
    end date grew, not the start, and the START is what the 40-permno
    selection reads (first_60_days.notna().all()) -- re-measured
    directly, the extension selects the IDENTICAL 40 permnos, not a
    different sample. Needed because test_point_in_time_reproducibility's
    T = index[int(len(monthly_rets)*0.7)] landed at 2012-05-31 on the old
    window, and of this fixture's real, non-clustered name attrition (18
    of 40 permnos delist at distinct dates 2005-2018 -- this fixture's
    own deliberate design, see below), only 7 formation dates before T
    survived run_backtest's full-leg-coverage rule (F4, 2026-09-12) --
    below the test's own `>12` bar. T now lands at 2014-10-31 (measured,
    not estimated), clear of the 2010-2011 attrition cluster that
    caused the shortfall; 17 valid pre-T months confirmed by direct
    measurement, not calendar arithmetic.

    F4 (docs/03_roadmap.md): monthly_rets/monthly_rf are now real, not
    None/None. monthly_rets is the ARITHMETIC monthly compounding of the
    SAME 40-permno daily_rets panel -- (1+r).prod()-1 per calendar month,
    per name, via pandas' own month-end resampling (not a second
    independent data pull; same names, same window). monthly_rf is
    src.data.risk_free_us.load_us_rf_ken_french(), which already emits
    true calendar month-end dates in decimal units -- verified to align
    with monthly_rets' resampled index without any period-based
    re-alignment (both use the same last-calendar-day-of-month
    convention). Building these was deferred as long as possible (this
    function's original docstring: "no in-scope test uses them") --
    test_point_in_time_reproducibility (which needs all four) was E3,
    blocked on Phase F. Phase F is done; this is that promised step, not
    scope creep.

    The window must be long enough for estimate_betas() to actually
    clear rho_min_obs (750 overlapping-return observations over a
    trailing rho_window_days=1260-day window) somewhere inside it -- a
    2013-2015 window (the first version of this function) was too short
    (only ~628 trading days total, never reaching the ~1260-day mark),
    so estimate_betas() would silently produce zero rows for every date,
    an EMPTY output rather than a real point-in-time comparison. 10.5
    years leaves ~1300+ trading days of real rolling-beta coverage after
    the initial 5-year window is consumed.

    Subsampled to a small FIXED set of permnos (the first N with a
    complete return history across the whole window, sorted for
    determinism) -- estimate_betas()'s per-date loop calls the full
    sigma/rho computation once per date, and doing that across the
    entire real universe (several thousand permnos) over ~1300 candidate
    dates would take tens of minutes per test run. A meaningful
    point-in-time check needs enough real names to be non-trivial, not
    the full universe -- matching this function's own "small cached
    slice" docstring intent.
    """
    import datetime as _dt

    from src.data import risk_free_us
    from src.estimation import beta_fp

    start, end = _dt.date(2005, 1, 1), _dt.date(2018, 12, 31)
    daily_log = beta_fp._daily_log_returns(start, end, leg="us")
    market_log = beta_fp._market_log_return_series(
        start, end, leg="us", market_index="vw_uncapped"
    )

    market_wide = market_log.to_pandas().set_index("date")["log_ret"]
    market_wide.index = pd.to_datetime(market_wide.index)
    daily_mkt = np.expm1(market_wide).sort_index()

    daily_wide = daily_log.to_pandas().pivot(index="date", columns="permno", values="log_ret")
    daily_wide.index = pd.to_datetime(daily_wide.index)
    daily_wide = daily_wide.reindex(daily_mkt.index).sort_index()

    # leakage-auditor finding (2026-09-10): selecting permnos on
    # notna().all(axis=0) -- complete history across the WHOLE window --
    # is a survivorship filter. It doesn't invalidate the two in-scope
    # tests (both compare the SAME fixture against itself, so the filter
    # cancels), but it silently removes the suite's ability to ever
    # catch a bug that only manifests on a name with a gap or an ending
    # (a rolling_sum bridging a delisting gap, a join dropping a name
    # mid-window, an off-by-one at a series boundary). Fixed: select on
    # history available at the START of the window only (real at
    # formation), not persistence across it -- names are free to have
    # gaps or delist partway through, which is what real data looks
    # like, and the consuming code already handles the resulting nulls
    # (_estimate_beta_fp_core filters is_not_null and gates on counts).
    n_permnos = 40
    first_60_days = daily_wide.iloc[:60]
    real_at_formation = sorted(daily_wide.columns[first_60_days.notna().all(axis=0)])[:n_permnos]
    daily_rets = np.expm1(daily_wide[real_at_formation])

    # Arithmetic monthly compounding of the SAME panel, per name --
    # (1+r).prod()-1 over each calendar month's trading days. A name
    # with ALL-NaN returns in a given month (not yet listed, or already
    # gapped out) must compound to NaN for that month, not silently to
    # 0.0 -- min_count=1 on the underlying sum-of-logs makes an
    # all-missing group NaN rather than treating "no data" as "no
    # return". Uses log-space summation (equivalent to the arithmetic
    # product, cheaper and avoids a subtle float rounding difference
    # from chaining `.apply(lambda x: (1+x).prod()-1)` across ~2600 rows)
    # then converts back with expm1, mirroring this fixture's own
    # log<->arithmetic round-trip convention used just above.
    monthly_log = np.log1p(daily_rets).resample("ME").sum(min_count=1)
    monthly_rets = np.expm1(monthly_log)

    monthly_rf_pl = risk_free_us.load_us_rf_ken_french()
    monthly_rf_pd = monthly_rf_pl.to_pandas().set_index("date")["rf"]
    monthly_rf_pd.index = pd.to_datetime(monthly_rf_pd.index)
    monthly_rf_pd = monthly_rf_pd.reindex(monthly_rets.index)

    return daily_rets, daily_mkt, monthly_rets, monthly_rf_pd


# ---------------------------------------------------------------------------
# 1. THE TRUNCATION TEST — the single most valuable test in this file
# ---------------------------------------------------------------------------

def test_point_in_time_reproducibility():
    """
    Output for dates <= T must be IDENTICAL whether or not data after T exists.

    This is the general-case lookahead detector. Any leak from the future --
    a rolling window that peeks ahead, a full-sample percentile, a filter that
    uses end-of-sample information, a merge that forward-fills across the
    boundary -- changes the pre-T output when post-T data is present.

    If this passes, most classes of temporal leakage are ruled out at once.
    """
    daily_rets, daily_mkt, monthly_rets, monthly_rf = load_test_panel()
    T = monthly_rets.index[int(len(monthly_rets) * 0.7)]

    # Run A: pipeline never sees anything after T
    run_a = run_pipeline(
        daily_rets[daily_rets.index <= T],
        daily_mkt[daily_mkt.index <= T],
        monthly_rets[monthly_rets.index <= T],
        monthly_rf[monthly_rf.index <= T],
    )

    # Run B: pipeline sees everything, then we truncate the OUTPUT
    run_b = run_pipeline(daily_rets, daily_mkt, monthly_rets, monthly_rf)
    run_b_trunc = run_b[run_b.index <= T]

    common = run_a.index.intersection(run_b_trunc.index)
    assert len(common) > 12, "not enough overlap to be a meaningful test"

    pd.testing.assert_series_equal(
        run_a.loc[common], run_b_trunc.loc[common],
        check_names=False, rtol=1e-10,
    )

    # Skip-set equality over the SAME range, not just matching returns
    # on months both runs happened to emit (design decision, F4,
    # 2026-09-12). run_backtest() can SKIP a formation date (thin
    # universe, thin leg, missing held-month data) instead of emitting
    # a return -- and a leak that changes universe/coverage MEMBERSHIP
    # rather than a return's numeric value could change WHICH months
    # qualify without ever touching a value assert_series_equal checks.
    # Comparing only run_a.index.intersection(run_b_trunc.index) would
    # let such months silently drop out of `common` and pass -- a
    # membership-changing leak has to hide EXACTLY here, in the set
    # subtracted out before the numeric comparison ever runs. Scoped to
    # formation dates <= T (not <= T's held-month boundary): a
    # formation dated exactly T's own month legitimately sees less
    # trailing data in run_a than in run_b's SAME formation once
    # truncated post-hoc, since run_a's held-month lookup for that one
    # date can fall just past T where run_a has no data and run_b (pre-
    # truncation) does -- this is the SAME boundary effect run_b_trunc's
    # own value-level comparison already tolerates by scoping to
    # `common`, not a leak; confirmed by direct inspection this is the
    # single boundary-adjacent formation date where the two skip
    # REASONS legitimately diverge while the skip KEYS still agree.
    skips_a = {k: v for k, v in run_a.attrs.get("skips", {}).items() if k <= T}
    skips_b = {k: v for k, v in run_b.attrs.get("skips", {}).items() if k <= T}
    assert set(skips_a.keys()) == set(skips_b.keys()), (
        "run_a and run_b_trunc skipped a DIFFERENT set of formation "
        "dates on or before T -- a leak changed which months qualify "
        "for a return, not just what value a qualifying month produced. "
        f"only in run_a: {set(skips_a) - set(skips_b)}; "
        f"only in run_b: {set(skips_b) - set(skips_a)}"
    )


def test_betas_are_point_in_time():
    """Same idea, one layer down: isolates whether a leak is in the estimator."""
    daily_rets, daily_mkt, _, _ = load_test_panel()
    T = daily_rets.index[int(len(daily_rets) * 0.7)]

    b_a = estimate_betas(daily_rets[daily_rets.index <= T],
                         daily_mkt[daily_mkt.index <= T])
    b_b = estimate_betas(daily_rets, daily_mkt)
    b_b = b_b[b_b.index <= T]

    common_idx = b_a.index.intersection(b_b.index)
    common_col = b_a.columns.intersection(b_b.columns)
    pd.testing.assert_frame_equal(
        b_a.loc[common_idx, common_col],
        b_b.loc[common_idx, common_col],
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# 2. NULL TESTS — catch leaks in construction rather than in the signal
# ---------------------------------------------------------------------------

def test_shuffled_signal_produces_no_alpha():
    """
    Permute betas cross-sectionally within each month, destroying all signal
    while preserving the panel structure, weighting scheme, and leg scaling.

    A shuffled signal must produce alpha indistinguishable from zero. If it
    doesn't, the return is coming from the CONSTRUCTION, not the beta sort --
    e.g. leg scaling using realized rather than ex-ante quantities.
    """
    daily_rets, daily_mkt, monthly_rets, monthly_rf = load_test_panel()
    betas = estimate_betas(daily_rets, daily_mkt)

    rng = np.random.default_rng(42)
    shuffled = betas.copy()
    for dt in shuffled.index:
        row = shuffled.loc[dt].dropna()
        if len(row) > 1:
            shuffled.loc[dt, row.index] = rng.permutation(row.values)

    ret = run_pipeline(daily_rets, daily_mkt, monthly_rets, monthly_rf,
                       precomputed_betas=shuffled)

    t_stat = ret.mean() / (ret.std() / np.sqrt(len(ret)))
    assert abs(t_stat) < 2.5, (
        f"shuffled signal produced t={t_stat:.2f} -- return is coming from "
        "construction, not from the beta sort"
    )


def test_synthetic_known_betas_recovered():
    """
    Simulate stocks with KNOWN betas. The estimator must recover them.
    Catches alignment and off-by-one errors that real data hides completely,
    because on real data every estimator produces plausible-looking output.
    """
    rng = np.random.default_rng(0)
    n_days, n_stocks = 2000, 200
    dates = pd.bdate_range("2000-01-01", periods=n_days)

    true_betas = rng.uniform(0.3, 2.0, n_stocks)
    mkt = pd.Series(rng.normal(0.0003, 0.01, n_days), index=dates)
    idio = rng.normal(0, 0.015, (n_days, n_stocks))
    rets = pd.DataFrame(
        np.outer(mkt.values, true_betas) + idio,
        index=dates, columns=[f"S{i}" for i in range(n_stocks)],
    )

    est = estimate_betas(rets, mkt).iloc[-1]
    # account for shrinkage toward 1 -- read weight/target from config
    # rather than hardcoding 0.6/0.4, so a config change can't silently
    # desync this un-shrinking from what estimate_betas() actually applied.
    from src.estimation.beta_fp import _load_beta_config

    cfg = _load_beta_config()
    implied = (est - (1 - cfg["shrinkage_weight"]) * cfg["shrinkage_target"]) / cfg["shrinkage_weight"]
    corr = np.corrcoef(implied.values, true_betas)[0, 1]
    assert corr > 0.95, f"estimator recovers true betas at only corr={corr:.3f}"
    # Correlation alone is scale-invariant -- a uniform 2x (or 0.5x) error
    # in every recovered beta scores an unchanged corr (gate-verifier
    # finding, 2026-09-10: measured identical to 10 decimal places under
    # a 2x injection). MAE is an absolute, non-ratio anchor that a scale
    # error actually moves.
    mae = np.mean(np.abs(implied.values - true_betas))
    assert mae < 0.15, f"estimator recovers true betas at only mae={mae:.4f}"


# ---------------------------------------------------------------------------
# 3. SURVIVORSHIP & UNIVERSE
# ---------------------------------------------------------------------------

def test_universe_contains_eventually_delisted_names():
    """
    A universe built at date t must include names that were alive at t but died
    later. If delisted names are absent from historical universes, the universe
    was built from an end-of-sample name list -- classic survivorship bias.

    NOTE on what this test does and does not catch (investigated 2026-09-09
    during A1 verification): this checks membership via securitybegdt/
    securityenddt spell bounds only. Confirmed directly that those bounds are
    IDENTICAL for Lehman between the pre-fix and post-fix CRSP panels (the
    2026-09-08 dc4e92e fix recovered the delisting-RETURN row, dlyret/
    dlydelflg on a row dated one trading day after securityenddt -- it does
    not change securityenddt itself). So this test passes on both panels and
    does not discriminate the fix; it guards a different, real bug class
    (end-of-sample name-list construction) and is kept for that reason. The
    delisting-row recovery itself is checked by
    test_delisting_return_row_recovered below.
    """
    as_of = pd.Timestamp("2007-06-30")
    universe = build_universe(as_of)
    # Lehman, permno 80599, was alive in 2007, delisted Sept 2008.
    # build_universe() is keyed on CRSP permnos (see the adapter above) --
    # gvkey ("030128") is not a value universe_at() ever returns, so the
    # gvkey branch below is dead by construction. Kept, rather than deleted,
    # so this stays honest about the identifier space instead of silently
    # narrowing the check to "80599 in universe" with no explanation.
    assert "030128" in universe or 80599 in universe, (
        "a name alive at the as-of date but delisted later is missing -- "
        "universe is likely built from surviving names only"
    )


def test_delisting_return_row_recovered():
    """
    The actual A1 discriminator: the 2026-09-08 fix (dc4e92e) recovered
    delisting-RETURN rows that CRSP v2 blanks the base security descriptors
    on, which the original pull's WHERE clause silently dropped (31,114 rows
    project-wide). That fix is invisible to universe_at()/build_universe()
    membership -- confirmed directly that Lehman's securitybegdt/securityenddt
    spell bounds are identical pre- and post-fix, so
    test_universe_contains_eventually_delisted_names above cannot detect it.
    This test reads the recovered row directly instead.

    Known case (docs/03_roadmap.md A1 checkpoint): Lehman, permno 80599,
    delisting return row dated 2008-09-18 (one trading day after
    securityenddt=2008-09-17, matching stkdelists.deldlydt vs delistingdt),
    dlyret approx -0.6, dlydelflg='Y'. Entirely absent from the pre-fix panel.
    """
    import polars as pl

    from src.data.universe_panel import RAW_DATA_DIR

    lf = pl.scan_parquet(str(RAW_DATA_DIR / "year=2008" / "part.parquet"))
    row = (
        lf.filter(pl.col("permno") == 80599)
        .filter(pl.col("dlycaldt") == pl.date(2008, 9, 18))
        .select(["dlyret", "dlydelflg"])
        .collect()
    )
    assert row.height == 1, (
        "Lehman's recovered delisting-return row (2008-09-18) is missing -- "
        "the delisting-return fix did not make it into this panel"
    )
    assert row["dlydelflg"].item() == "Y"
    assert row["dlyret"].item() == pytest.approx(-0.6, abs=1e-6)


def test_universe_excludes_not_yet_listed_names():
    """The mirror test: no name may appear before it existed.

    Originally specified against Shopify (gvkey "023650"), IPO May 2015.
    Reconciled 2026-09-09: universe_at() is CRSP-primary and returns
    permnos, not gvkeys, so the identifier alone would need translating --
    but Shopify is Canadian-incorporated (usincflg='N', permno 15358) and
    is not in the panel at all (never pulled -- fails the base filter
    unconditionally, not by date). Swapped to Fitbit (permno 15390), a
    US-incorporated name (securitybegdt=2015-06-18, confirmed live against
    crsp.wrds_dsfv2_query) that genuinely IPO'd after this test's as_of date.

    NOTE (leakage-auditor finding, 2026-09-09): despite the swap, this test
    still cannot fail, for a reason unrelated to the identifier choice.
    universe_at() snapshots a single trading day (_snapshot_on) BEFORE
    _apply_pit_bounds ever runs -- a not-yet-listed permno simply has no row
    on that day (verified: zero rows in the whole panel have
    dlycaldt < securitybegdt), so it's excluded by the snapshot regardless of
    whether the securitybegdt bound itself is correct. Mutation-tested:
    deleting or inverting _apply_pit_bounds entirely does not change this
    test's outcome. Kept as a real (if redundant) snapshot-path check, but
    the securitybegdt bound itself is exercised directly by
    test_pit_bounds_exclude_not_yet_listed below.
    """
    as_of = pd.Timestamp("2014-06-30")
    universe = build_universe(as_of)
    assert 15390 not in universe, "a name appears before its IPO date"


def test_pit_bounds_exclude_not_yet_listed():
    """Direct unit test of the securitybegdt bound test_universe_excludes_
    not_yet_listed_names above cannot actually exercise (see its docstring):
    the real panel never has a trading row before securitybegdt, so the
    snapshot mechanism excludes not-yet-listed names before the bound is
    ever consulted. This synthesizes a row that WOULD reach the bound and
    checks the bound itself catches it, so a future regression in the
    point-in-time filter logic doesn't go silently unnoticed."""
    import datetime

    import polars as pl

    from src.data.universe_panel import _apply_pit_bounds

    frame = pl.DataFrame({
        "permno": [15390],
        "securitybegdt": [datetime.date(2015, 6, 18)],
        "securityenddt": [datetime.date(2021, 1, 13)],
    })
    assert _apply_pit_bounds(frame, datetime.date(2014, 6, 30)).height == 0, (
        "a name is admitted before its securitybegdt -- the point-in-time "
        "bound is not actually filtering"
    )
    assert _apply_pit_bounds(frame, datetime.date(2015, 6, 18)).height == 1, (
        "a name is excluded ON its own securitybegdt -- bound is off by one"
    )


def test_terminated_series_have_delisting_records():
    """
    Systematic delisting-coverage check. For every security whose return series
    stops before the sample end, assert a delisting record exists.

    Do NOT QA this by spot-checking famous bankruptcies -- parent-shell survival
    through reorganization is common (WaMu's PERMNO ran continuously into Mr.
    Cooper Group), so some 'known failures' aren't delistings at all.
    """
    pytest.skip("implement once the delisting merge exists")


# ---------------------------------------------------------------------------
# 4. FORMATION TIMING
# ---------------------------------------------------------------------------

def test_formation_strictly_precedes_holding_period():
    """
    The last date feeding a beta estimate must be strictly before the first date
    of the return it's used to earn. Assert on actual dates, not on the intent
    of the code.

    Open item (d) (docs/03_roadmap.md F1/F2b): decided last session --
    PRODUCTION's convention is canonical (lookback_end = the last market
    date strictly before month_end, resolved by
    src.estimation.beta_fp._resolve_window_starts -- see that function's
    own docstring), NOT the E2 leakage adapter's inverted
    `month_end = as_of_date + 1 day` convention (estimate_betas() in
    this file), which labels a beta at `t` using `t`'s own return. That
    inversion is deliberately isolated to estimate_betas() (a synthetic-
    security-compatible thin wrapper over the SAME core math, per its
    own docstring) and never reaches production's real month_end
    resolution -- so this test asserts against
    beta_fp._resolve_window_starts directly, on rebalance.run_backtest's
    OWN real per-period output, not against estimate_betas().

    max(estimation_date) < min(holding_return_date), asserted on ACTUAL
    dates emitted by a real run (rebalance.run_backtest's diagnostics,
    fed by beta_fp on real US data), not on the intent of the code --
    the roadmap's own instruction for closing this item.
    """
    import datetime as _dt

    import polars as pl

    from src.data.universe_panel import RAW_DATA_DIR
    from src.estimation import beta_fp
    from src.portfolio import rebalance

    if not RAW_DATA_DIR.exists():
        pytest.skip("data/raw/us_panel_crsp_full/ not present")

    # rho_window_days=1260 (~5 years) of trailing history must exist
    # before the FIRST month_end tested, or estimate_beta_fp() produces
    # zero rows for every candidate (rho is the binding constraint --
    # same reasoning as load_test_panel()'s own window-length comment
    # above). Fetch from 2009-01-01 so 2014-2015's month-ends have a
    # full trailing window; only 2014-2015's month-ends are used as
    # FORMATION dates below, the earlier years exist purely to feed
    # their rolling windows.
    fetch_start, fetch_end = _dt.date(2009, 1, 1), _dt.date(2015, 6, 30)
    market_returns = beta_fp._market_log_return_series(
        fetch_start, fetch_end, leg="us", market_index="vw_uncapped"
    )
    daily_returns = beta_fp._daily_log_returns(fetch_start, fetch_end, leg="us")
    cfg = beta_fp._load_beta_config()

    test_start, test_end = _dt.date(2014, 1, 1), _dt.date(2015, 6, 30)
    month_ends = sorted(
        d for d in set(market_returns["date"].to_list()) if test_start <= d <= test_end
    )
    # One real calendar month-end per month in range, matching
    # rebalance.run_backtest's own key convention (a true month-end
    # date, e.g. 2014-01-31) -- NOT every trading day, since that is
    # what a real monthly rebalance actually forms on.
    candidate_month_ends = sorted(
        {
            (d + _dt.timedelta(days=32)).replace(day=1) - _dt.timedelta(days=1)
            for d in month_ends
        }
    )

    betas_by_date = {}
    lookback_end_by_formation = {}
    for month_end in candidate_month_ends:
        betas = beta_fp.estimate_beta_fp(month_end, daily_returns, market_returns)
        if betas.height == 0:
            continue
        _sigma_start, _rho_start, lookback_end = beta_fp._resolve_window_starts(
            month_end, market_returns, cfg
        )
        if lookback_end is None:
            continue
        betas_by_date[month_end] = betas
        lookback_end_by_formation[month_end] = lookback_end

    assert len(betas_by_date) > 0, (
        "no real formation dates produced betas in this window -- "
        "test setup assumption failed, not the assertion under test"
    )

    # monthly_rets/monthly_rf keyed to the SAME real month-ends -- built
    # from real US data via the same daily panel already fetched above,
    # collapsed to arithmetic monthly returns per permno, exactly
    # load_test_panel()'s own convention.
    all_permnos = sorted(
        {p for betas in betas_by_date.values() for p in betas["permno"].to_list()}
    )
    daily_pd = daily_returns.to_pandas().pivot(index="date", columns="permno", values="log_ret")
    daily_pd.index = pd.to_datetime(daily_pd.index)
    daily_pd = daily_pd.reindex(columns=all_permnos)
    monthly_log = daily_pd.resample("ME").sum(min_count=1)
    monthly_rets_pd = np.expm1(monthly_log)
    monthly_rets_long = monthly_rets_pd.reset_index().melt(
        id_vars="date", var_name="permno", value_name="ret"
    ).dropna(subset=["ret"])
    monthly_rets_pl = pl.DataFrame(
        {
            "permno": monthly_rets_long["permno"].to_list(),
            "month": [d.date() for d in monthly_rets_long["date"]],
            "ret": monthly_rets_long["ret"].to_numpy(),
        }
    )
    monthly_rf_pl = pl.DataFrame(
        {"month": list(betas_by_date.keys()), "rf": [0.0] * len(betas_by_date)}
    )

    result = rebalance.run_backtest(
        betas_by_date, monthly_rets_pl, monthly_rf_pl,
        min_names_total=1, min_names_per_leg=1, max_consecutive_skips=100,
    )

    assert result.diagnostics.height > 0, (
        "no real held periods survived the loop -- test setup assumption "
        "failed, not the assertion under test"
    )

    # max(estimation_date) < min(holding_return_date), on REAL dates --
    # per-period, not just once, and reading lookback_end straight from
    # _resolve_window_starts (independently re-derived here, not read
    # back from a field rebalance.run_backtest happens to store) so this
    # test cannot be satisfied by a diagnostics field that merely
    # RESTATES formation_date without the code path actually respecting
    # the t-1 boundary.
    for row in result.diagnostics.to_dicts():
        formation_date = row["formation_date"]
        holding_month = row["holding_month"]
        estimation_date = lookback_end_by_formation[formation_date]
        # First real trading day of the held month, from the SAME real
        # market calendar -- never assumed to be the 1st of the month.
        holding_candidates = [
            d for d in month_ends
            if d.year == holding_month.year and d.month == holding_month.month
        ]
        if not holding_candidates:
            continue  # held month outside the fetched market calendar
        holding_return_date = min(holding_candidates)
        assert estimation_date < holding_return_date, (
            f"formation {formation_date}: estimation_date={estimation_date} "
            f"is NOT strictly before holding_return_date={holding_return_date} "
            "-- the last date feeding this beta did not precede the first "
            "date of the return it earns"
        )


def test_no_zero_volume_days_in_estimation_window():
    """
    Stale quotes have zero realized variance, which understates sigma_i and
    pushes defunct shells into the LOW-beta long leg.

    F2b (docs/03_roadmap.md): the liquidity filter now exists
    (src/portfolio/liquidity.py), wired into estimate_beta_fp() via
    liquidity.liquid_permnos()'s min_nonzero_volume_days gate
    (config/liquidity.yaml). Real observed case, verified directly against
    the cached CRSP panel (the originally-cited Lehman 2014 case is not
    reachable -- Lehman delisted in September 2008 and has zero rows in
    2014; Watsco Inc Class B common, permno 46068, is a genuine, real
    substitute found by searching the actual data rather than assuming the
    anecdote transfers): 46068 traded on only 94 of 252 trading days in
    2014 (158 zero-volume rows), below the min_nonzero_volume_days=120
    threshold, and is confirmed excluded from estimate_beta_fp()'s real
    output for a month_end whose sigma window covers that year.
    """
    import datetime

    from src.estimation import beta_fp

    month_end = datetime.date(2015, 6, 30)
    daily_returns = beta_fp._daily_log_returns(
        datetime.date(2010, 6, 1), month_end, leg="us"
    )
    market_returns = beta_fp._market_log_return_series(
        datetime.date(2010, 6, 1), month_end, leg="us", market_index="vw_uncapped"
    )
    result = beta_fp.estimate_beta_fp(month_end, daily_returns, market_returns)

    assert 46068 not in result["permno"].to_list(), (
        "permno 46068 (Watsco Inc Class B, 94 traded days in 2014, below "
        "the 120-day minimum) appears in estimate_beta_fp()'s output -- "
        "the liquidity filter's volume gate is not excluding a real "
        "insufficiently-traded name"
    )


def test_sub_penny_names_are_filtered():
    """
    The opposite failure mode: a name trading GENUINELY (nonzero volume) at
    a sub-dollar price, where a one-tick move is a large percentage
    'return', inflating sigma_i and pushing the name into the HIGH-beta
    short leg.

    F2b (docs/03_roadmap.md): verified directly against the cached CRSP
    panel (the originally-cited Circuit City case is not reachable in this
    panel's coverage; Kowabunga Inc, permno 90608, is a genuine, real
    substitute found by searching the actual data): traded genuinely
    (nonzero volume, e.g. 688,800 shares on 2009-12-31) at $0.34-$0.35
    through December 2009 and into January 2010 -- below
    config/liquidity.yaml's min_price=1.00 floor applied to the
    formation-date lagged close. Confirmed excluded from
    estimate_beta_fp()'s real output for a month_end whose formation date
    falls in that window.

    Window note: needs a full ~7-year lookback (not just enough to cover
    the sigma window), or this permno fails the CORE sigma/rho min-obs
    gates before ever reaching the liquidity gate -- confirmed directly:
    with only a 2-year lookback this permno is absent from
    _estimate_beta_fp_core's own output, which would make this test pass
    for the wrong reason (excluded by rho_min_obs, not by the price
    floor) regardless of whether the liquidity gate works at all.
    """
    import datetime

    from src.estimation import beta_fp

    month_end = datetime.date(2010, 1, 31)
    daily_returns = beta_fp._daily_log_returns(
        datetime.date(2003, 1, 1), month_end, leg="us"
    )
    market_returns = beta_fp._market_log_return_series(
        datetime.date(2003, 1, 1), month_end, leg="us", market_index="vw_uncapped"
    )

    # Confirm the setup assumption directly: this permno must clear the
    # CORE sigma/rho gates on this window, so its absence from
    # estimate_beta_fp()'s output below is attributable to the liquidity
    # gate specifically, not to insufficient history.
    core_betas = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)
    assert 90608 in core_betas["permno"].to_list(), (
        "test setup assumption failed -- permno 90608 does not clear the "
        "core sigma/rho gates on this window, so this test would not "
        "actually be exercising the liquidity filter"
    )

    result = beta_fp.estimate_beta_fp(month_end, daily_returns, market_returns)

    assert 90608 not in result["permno"].to_list(), (
        "permno 90608 (Kowabunga Inc, genuinely traded at $0.34-$0.35 in "
        "December 2009) appears in estimate_beta_fp()'s output -- the "
        "liquidity filter's sub-penny price gate is not excluding a real "
        "sub-dollar name"
    )
