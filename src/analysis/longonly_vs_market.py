"""The 20+20 long-only book measured against its own market index --
docs/06_handoff_section6.md item A, report Sec 6.3 / Table 6.1.

WHY THIS TAKES A RETURN SERIES, NOT betas_by_date. The 20+20 book is
TURNOVER-LIMITED: it holds 20 names and replaces only the worst-ranked 5
each quarter (config/lowbeta_longonly.yaml's turnover_k), so its holdings
depend on what was held last quarter, not only on this quarter's ranking.
src.analysis._grouped_quarterly_analysis re-picks its groups from scratch
at every formation date, which is a DIFFERENT portfolio -- running a
"lowest 20 by beta" assign_fn through it would produce a full-re-pick
book whose returns are not the ones the report quotes.

So item A regresses the CACHED return series that
src.portfolio.longonly.run_longonly_backtest actually produced. This
function's signature enforces that: it accepts two return frames and has
no access to a cross-section, no assign_fn, and no way to form a
portfolio. The constraint is structural, not a matter of discipline.

What it DOES reuse is the market-compounding path
(src.analysis.quarterly_compounding, hence longonly._quarterly_return)
and the regression path (src.portfolio.diagnostics), so there is exactly
one implementation of each in the codebase.

BOTH HALVES OF THE RESULT ARE REPORTED. The handoff is explicit: the US
BAB long leg showed significant alpha (t=2.99) alongside INSIGNIFICANT
raw outperformance (t=1.17), and both belong in the report. A
risk-adjusted win that is not a raw win is a real and reportable
finding, not something to resolve in favour of whichever is prettier.
"""


import numpy as np
import polars as pl

from src.analysis import quarterly_compounding
from src.portfolio import diagnostics

# Quarterly series: 4 periods per year. Named and passed explicitly at
# every call site because diagnostics.annualized_sharpe DEFAULTS to 12
# (monthly, Gate 4's grain) -- silently inheriting that default on a
# quarterly series misannualizes by sqrt(12)/sqrt(4) = sqrt(3), which
# that function's own docstring calls out as a real, plausible-looking
# error. CLAUDE.md: no magic numbers in src/.
QUARTERS_PER_YEAR = 4


def _annualize_return(mean_quarterly: float) -> float:
    """GEOMETRIC annualization: (1 + mean_q)**4 - 1.

    Deliberately NOT the arithmetic mean_q * 4 that
    src/analysis/beta_bucket_timeseries.py uses for its period means. At
    this project's return levels the two differ by ~42bp (a 2.633%
    quarterly mean gives 10.955% geometric vs 10.532% arithmetic), and
    the measured headline figures already in
    docs/06_handoff_section6.md (2.633%/qtr -> "ann 10.96%") are
    geometric. Sec 6 must not silently disagree with Sec 4 over a
    convention; the difference is recorded here and flagged in the
    report.
    """
    return (1.0 + mean_quarterly) ** QUARTERS_PER_YEAR - 1.0


def _annualize_vol(quarterly_vol: float) -> float:
    """Quarterly standard deviation scaled by sqrt(4)."""
    return quarterly_vol * np.sqrt(QUARTERS_PER_YEAR)


def longonly_vs_market_stats(
    port_excess: pl.DataFrame,
    market_excess: pl.DataFrame,
    *,
    label: str,
) -> dict:
    """Full-sample performance and market loading for one book.

    port_excess / market_excess: [quarter, <one value column>] -- the
    EXCESS quarterly return series for the book and for the market index
    its betas were estimated against (spec Sec 7: estimation and
    evaluation benchmark must be the same object; US vw_uncapped, Canada
    vw_capped_10pct). The value column may be named anything (the
    excess_returns module emits "ret_excess"); whichever non-quarter
    column is present is used.

    Inner-joined on quarter, so a coverage gap shows up as a smaller
    n_quarters rather than as a misaligned regression.

    Returns, per the handoff's Sec 6.3 list:
      label, n_quarters, first_quarter, last_quarter
      port_mean_excess, mkt_mean_excess
      port_ann_excess, mkt_ann_excess        (GEOMETRIC -- see above)
      port_ann_vol, mkt_ann_vol
      port_sharpe, mkt_sharpe                (periods_per_year=4)
      realized_beta, alpha, alpha_t_stat, beta_t_stat
      raw_outperf_mean, raw_outperf_se, raw_outperf_t

    ALPHA SIGNIFICANCE USES alpha_t_stat, NEVER beta_t_stat. Both are
    surfaced under unambiguous names because a prior probe in this
    project mislabelled one as the other (docs/
    04_handoff_lowbeta_longonly.md Sec 10).

    raw_outperf_* is the simple mean(port - mkt) t-test,
    mean(d)/(std(d, ddof=1)/sqrt(n)). Note the risk-free rate CANCELS in
    the difference, so this statistic is identical whether computed on
    excess or raw returns -- it cannot be changed by the excess
    convention, which is worth knowing when reconciling against any
    earlier raw-basis figure.
    """
    port_col = _value_column(port_excess, "port_excess")
    mkt_col = _value_column(market_excess, "market_excess")

    # Reject a null/NaN value or a duplicated quarter BEFORE regressing.
    # gate-verifier finding 2026-09-15, both reproduced: one null in the
    # portfolio series returned realized_beta=nan, alpha=nan and
    # port_sharpe=nan with no exception and n_quarters still reporting 5
    # -- a structurally complete-looking row headed for the parquet. A
    # duplicated quarter fans out the inner join below (measured: 6 rows
    # from a 5-quarter join) and double-counts in every mean.
    quarterly_compounding.assert_clean_quarterly_series(
        port_excess, value_col=port_col, label=f"{label}:portfolio"
    )
    quarterly_compounding.assert_clean_quarterly_series(
        market_excess, value_col=mkt_col, label=f"{label}:market"
    )

    joined = (
        port_excess.select(pl.col("quarter"), pl.col(port_col).alias("_port"))
        .join(
            market_excess.select(pl.col("quarter"), pl.col(mkt_col).alias("_mkt")),
            on="quarter",
            how="inner",
        )
        .sort("quarter")
    )

    if joined.height < 3:
        raise ValueError(
            f"longonly_vs_market_stats({label!r}): only {joined.height} quarter(s) "
            "survive the portfolio/market join -- a market loading needs at "
            "least 3 observations (dof = n - 2 >= 1). Most likely the two "
            "series are keyed on different date conventions; see "
            "src.analysis.excess_returns for the calendar re-keying this "
            "project's Canadian series requires."
        )

    port = np.asarray(joined["_port"].to_numpy(), dtype=float)
    mkt = np.asarray(joined["_mkt"].to_numpy(), dtype=float)
    quarters = joined["quarter"].to_list()

    loading = diagnostics.full_sample_market_loading(port.tolist(), mkt.tolist())

    diff = port - mkt
    n = diff.shape[0]
    diff_se = float(diff.std(ddof=1) / np.sqrt(n))

    return {
        "label": label,
        "n_quarters": n,
        "first_quarter": quarters[0],
        "last_quarter": quarters[-1],
        "port_mean_excess": float(port.mean()),
        "mkt_mean_excess": float(mkt.mean()),
        "port_ann_excess": _annualize_return(float(port.mean())),
        "mkt_ann_excess": _annualize_return(float(mkt.mean())),
        "port_ann_vol": _annualize_vol(float(port.std(ddof=1))),
        "mkt_ann_vol": _annualize_vol(float(mkt.std(ddof=1))),
        "port_sharpe": diagnostics.annualized_sharpe(
            port.tolist(), periods_per_year=QUARTERS_PER_YEAR
        ),
        "mkt_sharpe": diagnostics.annualized_sharpe(
            mkt.tolist(), periods_per_year=QUARTERS_PER_YEAR
        ),
        "realized_beta": loading["beta"],
        "alpha": loading["alpha"],
        "alpha_t_stat": loading["alpha_t_stat"],
        "beta_t_stat": loading["t_stat"],
        "raw_outperf_mean": float(diff.mean()),
        "raw_outperf_se": diff_se,
        "raw_outperf_t": float(diff.mean() / diff_se) if diff_se > 0 else None,
    }


def _value_column(frame: pl.DataFrame, what: str) -> str:
    """The single non-quarter column of a return frame."""
    candidates = [c for c in frame.columns if c != "quarter"]
    if len(candidates) != 1:
        raise ValueError(
            f"longonly_vs_market_stats: expected {what} to have exactly one "
            f"value column besides 'quarter', found {candidates}"
        )
    return candidates[0]


def longonly_vs_market_table(rows: list[dict]) -> pl.DataFrame:
    """Several stat dicts as one frame with a pinned column order, so the
    driver and the tests agree on layout."""
    columns = [
        "label",
        "n_quarters",
        "first_quarter",
        "last_quarter",
        "port_mean_excess",
        "mkt_mean_excess",
        "port_ann_excess",
        "mkt_ann_excess",
        "port_ann_vol",
        "mkt_ann_vol",
        "port_sharpe",
        "mkt_sharpe",
        "realized_beta",
        "alpha",
        "alpha_t_stat",
        "beta_t_stat",
        "raw_outperf_mean",
        "raw_outperf_se",
        "raw_outperf_t",
    ]
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 for c in columns})
    return pl.DataFrame(rows).select(columns)
