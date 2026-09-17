"""
Gate 4 (docs/02_validation_gates.md): "THE GATE THAT MATTERS." Runs the US
BAB strategy monthly under FP's exact specification (config/runs.yaml's
fp_baseline_us cell) and correlates the resulting monthly return series
against AQR's own published US BAB factor.

Every gate before this one compared the dataset against itself. Gate 3's
open item (b) is the material one -- the beta estimator has no anchor
external to the dataset: a +50bp/day common-mode bias, a 1.5x market scale
error, and a 1-day market misalignment ALL scored bit-identical or higher
on every prior gate's own statistic. AQR's series is the project's first
genuine external anchor.

Correlation alone is NOT enough (CLAUDE.md: correlation is location- and
scale-invariant, and cannot detect a units bug, a constant bias, or a
scale error). Three further, absolute checks are required alongside it --
mean monthly return, Sharpe, and realized market loading -- each derived
from measured values, not round numbers. See run_gate4_us()'s docstring
and tests/gates/test_gate4_bab.py for where each band comes from.

Debug order on failure (docs/02_validation_gates.md): universe definition
-> delisting returns -> estimator windows/minimums -> weighting scheme ->
leg scaling. Check realized market loading FIRST -- if it isn't near
zero, the estimator is the problem, not the portfolio construction.
"""

import datetime
import gc
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.data import aqr_bab_loader, monthly_returns, risk_free_us
from src.estimation import beta_fp
from src.portfolio import diagnostics, rebalance

# The FP-spec rho window is 1260 trading days (~5 years). A formation date
# needs this much daily history BEFORE it to produce an unstunted
# cross-section (beta_fp._resolve_window_starts falls back to the
# earliest available market date otherwise, per that function's own
# docstring) -- so daily/market data must be fetched starting well before
# the first month_end this gate ever forms a portfolio on. 6 calendar
# years covers 1260 trading days with room for weekends/holidays.
_LOOKBACK_BUFFER_YEARS = 6

_GATE4_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "gate4.yaml"


def _load_gate4_config() -> dict:
    """config/gate4.yaml's band floors/multipliers and exact pins
    (CLAUDE.md: no magic numbers in src/). Re-read on every call rather
    than cached at import time -- this module is gate code, not a hot
    loop, and re-reading means an edited config takes effect without a
    process restart, matching rebalance._load_rebalance_config's own
    convention."""
    with open(_GATE4_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_GATE4_CFG = _load_gate4_config()
EXPECTED_N_DATES_1970_2025: int = _GATE4_CFG["expected_n_dates_1970_2025"]
EXPECTED_N_SKIPS_1970_2025: int = _GATE4_CFG["expected_n_skips_1970_2025"]
LOADING_DIFF_T_BOUND: float = _GATE4_CFG["loading_diff_t_bound"]


def mean_return_band(result: dict) -> float:
    """FROZEN band: max(2*SE_diff, 1.5*measured_gap), evaluated ONCE
    against the real 1970-2025 baseline and pinned in config/gate4.yaml
    -- NOT recomputed from whatever `result` is passed in.

    This is deliberate, not an oversight: a live recomputation from
    `result` is tautologically self-satisfying, because a x100 scale
    injection inflates its own gap term by the same x100, widening the
    band to match the very defect it should catch. Verified directly
    this session -- a live-recomputed version of this function PASSED
    under a x100 injection. `result` is accepted only so every band
    function's call signature matches the test call sites
    (`gate4_bab.mean_return_band(result)`); nothing defect-sensitive is
    read out of it here. See config/gate4.yaml for the frozen value and
    the measured inputs it was derived from."""
    return _GATE4_CFG["bands"]["mean_return"]["frozen_value"]


def sharpe_band(result: dict) -> float:
    """FROZEN band (see mean_return_band's docstring for why): the same
    goalpost-inflation failure applies here -- a beta-swap injection that
    should blow out the Sharpe gap would instead widen a live-recomputed
    band by exactly the swap's own magnitude. Value pinned in
    config/gate4.yaml, derived once from the real baseline gap (0.0305)
    against a 0.15 floor."""
    return _GATE4_CFG["bands"]["sharpe"]["frozen_value"]


def market_loading_band(result: dict) -> float:
    """FROZEN band (see mean_return_band's docstring). Bounds the
    MAGNITUDE of our own realized market loading -- catches a gross
    neutralization break (e.g. long-only). Separate from
    loading_diff_band, which is the real external-anchor comparison.
    Value pinned in config/gate4.yaml."""
    return _GATE4_CFG["bands"]["market_loading_magnitude"]["frozen_value"]


def loading_diff_band(result: dict) -> float:
    """FROZEN band (see mean_return_band's docstring). Replaces the
    original |t_stat_ours| < 2.0 bar, which rejected AQR's own published
    US BAB factor (t=-2.165 against the same market series) and FP's own
    published -0.06 evaluated at our SE (t=-2.17) -- it measured sample
    size, not defect presence, and was written before any real number
    existed (docs/02_validation_gates.md's Gate 4 section has the full
    reasoning). This bar instead asks whether OUR loading differs
    significantly from the BENCHMARK's own loading -- the actual purpose
    of an external-anchor gate. Value pinned in config/gate4.yaml,
    derived once from the real baseline diff (-0.01267) and its
    regression's own SE (0.010066)."""
    return _GATE4_CFG["bands"]["loading_diff"]["frozen_value"]


def _calendar_month_ends(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    """Every calendar month-end in [start, end], inclusive -- the SAME
    convention src.portfolio.rebalance._next_month_end uses (pure
    calendar arithmetic, never a trading day). start/end need not
    themselves be month-ends."""
    month_ends = []
    year, month = start.year, start.month
    while True:
        if month == 12:
            next_month_first = datetime.date(year + 1, 1, 1)
        else:
            next_month_first = datetime.date(year, month + 1, 1)
        month_end = next_month_first - datetime.timedelta(days=1)
        if month_end > end:
            break
        if month_end >= start:
            month_ends.append(month_end)
        year, month = next_month_first.year, next_month_first.month
    return month_ends


def build_bab_series(
    formation_start: datetime.date,
    formation_end: datetime.date,
    *,
    variant: str = "fp_spec",
    market_index: str = "vw_uncapped",
) -> rebalance.BacktestResult:
    """Runs config/runs.yaml's fp_baseline_us cell over every calendar
    month-end in [formation_start, formation_end]: loads daily returns
    and the market series ONCE for the whole span (estimate_beta_fp takes
    both as arguments precisely so the caller controls this -- re-reading
    per month-end turns a ~12-minute loop into hours), estimates a beta
    cross-section at each month-end, builds monthly arithmetic returns
    via src.data.monthly_returns (delisting rows compounded exactly where
    CRSP stamped them -- see that module's docstring for why a fold_back
    alternative was tried and removed), loads AQR's own RF (docs/02_
    validation_gates.md: "AQR built their BAB series with it, so it is
    the faithful comparison input for this gate" -- NOT Ken French), and
    calls src.portfolio.rebalance.run_backtest, whose missing-coverage
    skip is what correctly excludes a formation date when a weighted
    name has no real held-month return (e.g. an orphaned delisting row
    landing in a month no formation date ever holds).

    variant/market_index are exposed as keyword arguments, not hardcoded,
    so the SAME function serves both the fp_baseline_us default run and
    every failure-injection demonstration this gate's tests run.

    No caching -- every call recomputes. A cached result could silently
    serve a pre-injection value during a failure-injection demonstration
    (a config-hash cache key would not detect a change to src/ itself),
    which would make a broken assertion look like it fired correctly.
    Runtime is real (~15-25 minutes for the full 1965-2025 span,
    measured) -- callers needing a fast path should narrow
    formation_start/formation_end instead of expecting a cache.
    """
    daily_start = formation_start - datetime.timedelta(days=365 * _LOOKBACK_BUFFER_YEARS)
    # The LAST formation month's held return falls in the calendar month
    # AFTER formation_end (rebalance._next_month_end advances one month
    # forward) -- monthly_rets must cover that following month too, or
    # run_backtest's missing-coverage skip fires for every name in the
    # final formation date, discarding it entirely rather than just
    # trimming one month off the tail as intended.
    returns_end = _next_month_end_of(formation_end) + datetime.timedelta(days=31)

    daily_returns = beta_fp._daily_log_returns(daily_start, formation_end, leg="us")
    market_returns = beta_fp._market_log_return_series(
        daily_start, formation_end, leg="us", market_index=market_index
    )

    betas_by_date: dict[datetime.date, pl.DataFrame] = {}
    for month_end in _calendar_month_ends(formation_start, formation_end):
        betas = beta_fp.estimate_beta_fp(month_end, daily_returns, market_returns, variant)
        if betas.height > 0:
            betas_by_date[month_end] = betas

    # Explicit free before the next large allocation (monthly_rets, ~3.5M
    # rows over the full 1970-2025 span) -- daily_returns (73M rows) and
    # market_returns are never read again after the loop above, but
    # Python keeps them alive until this function returns otherwise.
    # Measured (2026-09-13): a stage-wise script with this exact del +
    # gc.collect() between the beta loop and monthly_rets completed the
    # full 56-year run clean (904.6s); the same pipeline without it
    # crashed with a Rust memory allocation failure on more than one
    # attempt, despite 14-17GB of free system memory each time -- a
    # long-running polars/Rust process accumulating 672 iterations'
    # worth of intermediate allocations without a chance to consolidate,
    # not a genuine exhaustion. This is not a fix for a bug so much as a
    # hint to the allocator; keep it even though the underlying frames
    # are individually well within available memory.
    del daily_returns, market_returns
    gc.collect()

    monthly_rets = monthly_returns.monthly_arithmetic_returns(
        daily_start, returns_end, leg="us",
    ).select(["permno", "month", "ret"])

    monthly_rf = risk_free_us.load_us_rf_aqr_gate4().rename({"date": "month"})

    return rebalance.run_backtest(betas_by_date, monthly_rets, monthly_rf)


def characterize_skips(result: rebalance.BacktestResult) -> dict:
    """Quantifies HOW OFTEN and WHERE run_backtest's missing-coverage
    skip fires, and whether it concentrates in high-beta names -- the
    measured finding the fold_back/as_stamped comparison was originally
    meant to produce (see src.data.monthly_returns' module docstring for
    why that comparison was abandoned as convention-vs-convention once
    fold_back was found to be simply wrong on some cases).

    "Missing return data" skip reasons only -- thin-universe and
    thin-leg skips are a different phenomenon (data coverage at
    formation, not a delisting-return placement effect) and are excluded
    from this count.

    Returns {"n_skips_missing_coverage", "by_decade": {decade: count}}.
    Decade concentration matters because delisting is not uniform across
    the sample (docs/02_validation_gates.md: the 2020s are this project's
    worst delisting decade, mean -5.55%) -- a skip count concentrated in
    a small number of high-volatility decades is a different finding
    than one spread evenly across the sample.
    """
    coverage_skips = {
        d: reason
        for d, reason in result.skips.items()
        if "missing return data" in reason
    }
    by_decade: dict[int, int] = {}
    for d in coverage_skips:
        decade = (d.year // 10) * 10
        by_decade[decade] = by_decade.get(decade, 0) + 1

    return {
        "n_skips_missing_coverage": len(coverage_skips),
        "by_decade": dict(sorted(by_decade.items())),
    }


def run_gate4_us(
    formation_start: datetime.date,
    formation_end: datetime.date,
    *,
    variant: str = "fp_spec",
    market_index: str = "vw_uncapped",
) -> dict:
    """Builds the BAB series (build_bab_series) and joins it against
    AQR's published US factor (src.data.aqr_bab_loader.load_aqr_bab_factors).

    Returns:
    - "comparison": month, ours, aqr, diff -- an inner join, so a month
      present in only one series is silently absent here rather than
      raising; check n_ours_only/n_aqr_only before trusting correlation
      (same join-asymmetry convention as gate1_index.run_gate1_us).
    - "correlation": Pearson correlation between ours and aqr.
    - "n_dates", "n_ours_only", "n_aqr_only".
    - "skips": src.portfolio.rebalance.BacktestResult.skips, passed
      straight through -- a long skip run must be visible, never merely
      inferred from a row count.
    - "skip_characterization": characterize_skips(result) -- how many
      skips are the missing-return-coverage kind (as opposed to a thin
      universe/leg) and their decade distribution.
    - "mean_ours" / "mean_aqr": arithmetic mean monthly return.
    - "sharpe_ours" / "sharpe_aqr": diagnostics.annualized_sharpe. Both
      series are already excess returns (BAB is long-short; AQR's own
      published factor is likewise already excess by construction), so
      no risk-free subtraction happens here.
    - "market_loading" / "market_loading_t" / "market_loading_alpha":
      diagnostics.full_sample_market_loading(ours, market_excess), where
      market_excess is THIS project's own self-built market index
      (spec section 7: estimation and evaluation benchmarks must be the
      same object), compounded to monthly and re-keyed from its native
      last-real-trading-day convention to calendar month-ends, minus
      AQR's own RF -- matching this gate's own comparison sample exactly.
    - "market_loading_aqr" / "market_loading_aqr_t": the SAME regression,
      run on AQR's own published series against the SAME market_excess --
      the benchmark's own loading, needed because market_loading_band's
      predecessor (|t_stat_ours| < 2.0) rejected AQR's own factor
      (t=-2.165) and is retired for that reason (see loading_diff_band).
    - "market_loading_diff" / "market_loading_diff_t" /
      "market_loading_diff_se": full_sample_market_loading(ours - aqr,
      market_excess) -- whether OUR loading differs significantly from
      the BENCHMARK's own loading, which is what an external-anchor gate
      should actually test, rather than whether either loading differs
      from zero (a bar the benchmark itself fails at this sample size).
    - "n_months_backtested": len(comparison) after all joins.
    """
    result = build_bab_series(
        formation_start, formation_end, variant=variant, market_index=market_index,
    )
    skips = result.skips
    skip_characterization = characterize_skips(result)
    ours = result.returns.rename({"ret": "ours"})
    # result.positions/result.diagnostics are never read in this function
    # -- freed before _market_excess_monthly's own separate multi-decade
    # panel build, for the same allocator-headroom reason build_bab_series
    # frees daily_returns/market_returns before its own large allocation
    # (see that function's comment).
    del result
    gc.collect()

    aqr = aqr_bab_loader.load_aqr_bab_factors().select(
        pl.col("date").alias("month"), pl.col("usa_ret").alias("aqr")
    )

    ours_months = set(ours["month"].to_list())
    aqr_months = set(aqr.filter(pl.col("aqr").is_not_null())["month"].to_list())

    comparison = ours.join(aqr, on="month", how="inner").filter(
        pl.col("aqr").is_not_null()
    ).with_columns((pl.col("ours") - pl.col("aqr")).alias("diff")).sort("month")

    correlation = comparison.select(pl.corr("ours", "aqr")).item()

    mean_ours = float(np.mean(comparison["ours"].to_numpy()))
    mean_aqr = float(np.mean(comparison["aqr"].to_numpy()))
    sharpe_ours = diagnostics.annualized_sharpe(comparison["ours"].to_list())
    sharpe_aqr = diagnostics.annualized_sharpe(comparison["aqr"].to_list())

    market_excess = _market_excess_monthly(
        formation_start, formation_end, market_index, comparison["month"].to_list()
    )
    loading = diagnostics.full_sample_market_loading(
        comparison["ours"].to_list(), market_excess
    )
    loading_aqr = diagnostics.full_sample_market_loading(
        comparison["aqr"].to_list(), market_excess
    )
    loading_diff = diagnostics.full_sample_market_loading(
        (comparison["ours"] - comparison["aqr"]).to_list(), market_excess
    )

    return {
        "comparison": comparison,
        "correlation": correlation,
        "n_dates": comparison.height,
        "n_ours_only": len(ours_months - aqr_months),
        "n_aqr_only": len(aqr_months - ours_months),
        "skips": skips,
        "skip_characterization": skip_characterization,
        "mean_ours": mean_ours,
        "mean_aqr": mean_aqr,
        "sharpe_ours": sharpe_ours,
        "sharpe_aqr": sharpe_aqr,
        "market_loading": loading["beta"],
        "market_loading_t": loading["t_stat"],
        "market_loading_alpha": loading["alpha"],
        "market_loading_aqr": loading_aqr["beta"],
        "market_loading_aqr_t": loading_aqr["t_stat"],
        "market_loading_diff": loading_diff["beta"],
        "market_loading_diff_t": loading_diff["t_stat"],
        "market_loading_diff_se": abs(loading_diff["beta"] / loading_diff["t_stat"]),
        "n_months_backtested": comparison.height,
    }


def _market_excess_monthly(
    formation_start: datetime.date,
    formation_end: datetime.date,
    market_index: str,
    months: list[datetime.date],
) -> list[float]:
    """The project's OWN market index (spec section 7 -- never SPX/
    vwretd/TSX Composite), compounded to monthly and minus AQR's own RF,
    aligned to `months` (the SAME calendar month-ends the BAB/AQR
    comparison uses). gate1_index.compound_daily_index_to_monthly keys
    its output by the last REAL TRADING DAY of each month, not a calendar
    month-end -- re-keyed here via _calendar_month_ends' own convention
    so this aligns with rebalance.run_backtest's month keys exactly.

    `months` can include one calendar month PAST formation_end (the last
    formation date's held-period month, same reason build_bab_series
    extends its own monthly_rets fetch) -- the panel fetch below extends
    to cover max(months), not just formation_end, or a KeyError on the
    final lookup below would follow (caught while pinning this gate's
    smoke-test numbers, 2026-09-13).

    Uses market_index_build.build_index_chunked, not build_index() on a
    single gate_adapters.us_gate1_panel() call -- a single call over
    Gate 4's full 56-year span returns ~71M per-name rows before ever
    collapsing to the ~14,000-row per-date index this function actually
    needs, which crashed with a Rust memory allocation failure (isolated
    and confirmed, 2026-09-13; see build_index_chunked's own docstring).
    """
    from src.gates import gate1_index
    from src.market_index import build as market_index_build

    panel_end = max(formation_end, max(months)) if months else formation_end
    daily_start = formation_start - datetime.timedelta(days=365 * _LOOKBACK_BUFFER_YEARS)
    index = market_index_build.build_index_chunked(
        "us", daily_start, panel_end, market_index
    )
    monthly_index = gate1_index.compound_daily_index_to_monthly(index)

    monthly_index = monthly_index.with_columns(
        pl.col("month_end").map_elements(_next_month_end_of, return_dtype=pl.Date).alias("month")
    )

    rf = risk_free_us.load_us_rf_aqr_gate4().rename({"date": "month", "rf": "rf"})
    joined = monthly_index.join(rf, on="month", how="left").with_columns(
        (pl.col("monthly_ret") - pl.col("rf")).alias("market_excess")
    )
    by_month = dict(zip(joined["month"].to_list(), joined["market_excess"].to_list()))
    return [by_month[m] for m in months]


def _next_month_end_of(d: datetime.date) -> datetime.date:
    """The calendar month-end containing d -- used to re-key
    gate1_index.compound_daily_index_to_monthly's real-last-trading-day
    output (e.g. 2015-01-30) onto rebalance.run_backtest's calendar
    month-end convention (2015-01-31), so the two can be joined on
    `month` exactly. Same calendar arithmetic as
    rebalance._next_month_end, applied to d itself rather than advancing
    one month forward from it."""
    if d.month == 12:
        next_month_first = datetime.date(d.year + 1, 1, 1)
    else:
        next_month_first = datetime.date(d.year, d.month + 1, 1)
    return next_month_first - datetime.timedelta(days=1)
