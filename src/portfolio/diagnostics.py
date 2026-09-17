"""
Portfolio diagnostics, spec 00_spec.md section 8 / schema decision D8:
"Monthly logging, every run" -- six diagnostics, non-optional.
config/portfolio.yaml's diagnostics.always_emitted block lists them for
documentation only; THIS module is what actually emits them, and it takes
no toggle parameter that could suppress any one of the six (D8: "the
emitting code must not read a toggle").

realized_market_loading, dollars_long, dollars_short are provided by the
caller (they depend on the realized holding-period return and leg
scaling, computed in src/portfolio/legs.py and the rebalance loop) --
this module's job is name_count, ex_ante_beta_spread, and
weight_fraction_smallest_size_decile, which are all derivable from the
formation-date weighted beta frame alone, plus passing the
caller-provided three through unchanged into one dict so every caller
reads all six diagnostics from one place.

realized_market_loading is inherently a t+1 quantity -- a regression of
REALIZED returns against the market, only knowable after the holding
period ends, and only meaningful as a full-sample statistic rather than
a per-month one. src/portfolio/rebalance.py's per-month loop always
passes None for it (see that module's own test,
test_run_backtest_emits_diagnostics_with_realized_market_loading_none);
it is computed post-loop, full-sample, by whatever calls run_backtest
across a full date range, not by this function or the loop itself.
"""

import numpy as np
import polars as pl

from src.portfolio import legs


def compute_diagnostics(
    weighted_betas: pl.DataFrame,
    *,
    realized_market_loading: float | None,
    r_long: float,
    r_short: float,
    dollars_long: float,
    dollars_short: float,
    full_universe_mkt_cap: pl.DataFrame | None = None,
) -> dict:
    """The six spec section 8 diagnostics for one formation date.

    weighted_betas: the formation-date frame with beta_shrunk,
    weight_long, weight_short columns (weights.rank_deviation_from_median()'s
    output, or value weighting's once F2c builds it).

    realized_market_loading, r_long, r_short, dollars_long, dollars_short:
    caller-provided -- these depend on the realized holding-period return
    and the leg-scaling choice (src/portfolio/legs.py), which this
    formation-date-only function has no way to compute itself. r_long/
    r_short are accepted but currently only used for interface
    completeness with the future rebalance loop; not read by this
    function's own math (name_count/ex_ante_beta_spread/
    weight_fraction_smallest_size_decile depend only on weighted_betas
    and full_universe_mkt_cap).

    full_universe_mkt_cap: the FULL candidate universe's permno/mkt_cap
    cross-section at the SAME formation date -- required for
    weight_fraction_smallest_size_decile, since the bottom-decile
    breakpoint (N-M&V's finding, bab-methodology skill: FP's BAB commits
    ~$1.05 per $1 to stocks in the bottom 1% of market cap) is a property
    of the full universe's size distribution, not just the names
    ultimately selected into the portfolio. If omitted,
    weight_fraction_smallest_size_decile is None rather than silently 0
    -- a caller must not be able to mistake "not computed" for "computed
    as zero."

    Returns a dict with exactly the six spec section 8 keys, every call,
    unconditionally -- no parameter here can suppress one.
    """
    beta_l = legs.leg_ex_ante_beta(
        weighted_betas, weight_col="weight_long", beta_col="beta_shrunk"
    )
    beta_h = legs.leg_ex_ante_beta(
        weighted_betas, weight_col="weight_short", beta_col="beta_shrunk"
    )
    ex_ante_beta_spread = (beta_h - beta_l) / (beta_h * beta_l)

    weight_fraction_smallest_decile = None
    if full_universe_mkt_cap is not None:
        weight_fraction_smallest_decile = _weight_fraction_smallest_size_decile(
            weighted_betas, full_universe_mkt_cap
        )

    return {
        "realized_market_loading": realized_market_loading,
        "dollars_long": dollars_long,
        "dollars_short": dollars_short,
        "ex_ante_beta_spread": ex_ante_beta_spread,
        "name_count": weighted_betas.height,
        "weight_fraction_smallest_size_decile": weight_fraction_smallest_decile,
    }


def _weight_fraction_smallest_size_decile(
    weighted_betas: pl.DataFrame, full_universe_mkt_cap: pl.DataFrame
) -> float:
    """Total (weight_long + weight_short) held in names whose mkt_cap
    falls in the bottom decile of `full_universe_mkt_cap` -- the FULL
    candidate universe at this formation date, not the selected-names
    subset. The decile cutoff is computed cross-sectionally on the full
    universe (CLAUDE.md: never pooled over the full sample, never
    computed from a post-selection subset that could shift the
    breakpoint)."""
    cutoff = full_universe_mkt_cap["mkt_cap"].quantile(0.10, interpolation="linear")
    in_bottom_decile = weighted_betas.filter(pl.col("mkt_cap") <= cutoff)
    return float((in_bottom_decile["weight_long"] + in_bottom_decile["weight_short"]).sum())


def annualized_sharpe(excess_rets, *, periods_per_year: int = 12) -> float:
    """Annualized Sharpe ratio of an ALREADY-EXCESS return series:
    mean(excess)/std(excess, ddof=1) * sqrt(periods_per_year).

    periods_per_year DEFAULTS TO 12 (monthly) -- this is Gate 4's BAB
    series' own grain (config/portfolio.yaml's rebalance: monthly), and
    every existing caller (src.gates.gate4_bab, this file's own tests)
    relies on that default, so it is preserved unchanged. Pass
    periods_per_year=4 explicitly for a QUARTERLY return series (e.g.
    docs/04_handoff_lowbeta_longonly.md's long-only backtest and beta-
    bucket module, both quarterly-rebalanced) -- silently reusing the
    monthly default on quarterly data would misannualize by a factor of
    sqrt(12)/sqrt(4) = sqrt(3), a real, silent-looking error CLAUDE.md
    calls out ("a wrong number that looks right is the worst possible
    outcome").

    Takes no risk-free argument, deliberately (Gate 4,
    docs/02_validation_gates.md): src.portfolio.legs.asymmetric_inverse_beta
    already subtracts r_f inside the BAB payoff formula, so the series
    this function receives is already an excess return. Accepting an rf
    parameter here would invite a silent double-subtraction -- a 40-60%
    Sharpe error that would still look like a plausible number.
    """
    values = np.asarray(list(excess_rets), dtype=float)
    if values.size < 2:
        raise ValueError(
            f"annualized_sharpe needs at least 2 observations to compute "
            f"a sample standard deviation, got {values.size}"
        )
    return float(values.mean() / values.std(ddof=1) * np.sqrt(periods_per_year))


def full_sample_market_loading(bab_rets, market_excess) -> dict:
    """OLS of a full BAB return series on the project's OWN market index
    (compounded to monthly, minus risk-free) -- Gate 4's realized_market_
    loading, computed post-loop and full-sample (see this module's own
    docstring on why it cannot be a per-month diagnostic).

    bab_rets = alpha + beta * market_excess + residual. Returns
    {"alpha", "beta", "t_stat", "alpha_t_stat", "n"} -- t_stat is BETA's
    own t-statistic (beta_hat / SE(beta_hat)), the self-normalizing form
    Gate 4 asserts on, since it does not require guessing a magnitude
    band the way the point estimate alone would. alpha_t_stat is the
    SEPARATE statistic for the intercept (alpha_hat / SE(alpha_hat)) --
    do not use `t_stat` for an alpha significance claim (see
    docs/04_handoff_lowbeta_longonly.md Sec 10 for a real instance of
    exactly that mislabelling).

    Spec section 7: the market series must be THIS project's own
    self-built index, never SPX/vwretd/TSX Composite -- estimation and
    evaluation benchmarks must be the same object, or a non-zero realized
    loading cannot be distinguished from a bug. Re-keying that index's
    daily-compounded-to-monthly output to the caller's month convention
    is the caller's job (src.gates.gate4_bab), not this function's --
    this function is pure numerics over two aligned sequences.

    Raises ValueError if the two inputs have different lengths, or if
    there are fewer than 3 observations (dof = n - 2 must be >= 1, or the
    residual variance -- and therefore SE(beta) and t_stat -- is
    undefined, not merely noisy)."""
    bab = np.asarray(list(bab_rets), dtype=float)
    market = np.asarray(list(market_excess), dtype=float)
    if bab.shape != market.shape:
        raise ValueError(
            f"bab_rets and market_excess must have the same length, got "
            f"{bab.shape[0]} and {market.shape[0]}"
        )
    n = bab.shape[0]
    dof = n - 2
    if dof < 1:
        raise ValueError(
            f"full_sample_market_loading needs at least 3 observations "
            f"(dof = n - 2 >= 1) to estimate a residual variance, got n={n}"
        )

    design = np.column_stack([np.ones(n), market])
    coefs, *_ = np.linalg.lstsq(design, bab, rcond=None)
    alpha_hat, beta_hat = coefs
    residuals = bab - design @ coefs
    sigma2 = float((residuals**2).sum() / dof)
    xtx_inv = np.linalg.inv(design.T @ design)
    se_beta = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    t_stat = float(beta_hat / se_beta)
    # alpha_t_stat -- xtx_inv[0, 0] is the ALPHA row (design's intercept
    # column is column 0), same OLS variance formula as se_beta/t_stat
    # above but for the intercept instead of the slope. Added per
    # docs/04_handoff_lowbeta_longonly.md Sec 10: an earlier probe
    # mislabelled beta's own t-stat (this function's `t_stat`) as an
    # alpha t-stat -- the tell was market-on-market showing t=6.8e16,
    # which is exactly what beta's t-stat does when beta=1 exactly and
    # residuals are ~0, not what an alpha t-stat would ever show. This
    # key did not exist before; callers needing a real alpha significance
    # test had nothing to use.
    se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    alpha_t_stat = float(alpha_hat / se_alpha)

    return {
        "alpha": float(alpha_hat),
        "beta": float(beta_hat),
        "t_stat": t_stat,
        "alpha_t_stat": alpha_t_stat,
        "n": n,
    }
