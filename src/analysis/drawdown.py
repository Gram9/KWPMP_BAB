"""Max drawdown, worst single period, and recovery timing --
docs/06_handoff_section6.md item B, for report Sec 6.3's Table 6.1.

BASIS: TOTAL RETURNS. User decision this session, and already the
settled convention in docs/05_report_spec.md ("Publish the total-return
figures -- they are the ones a reader would reconcile against a
statement"). Excess-return drawdowns overstate total-return drawdowns by
1.1-5.1pp in this project's own data, the gap widening with beta and
concentrating in high-rate periods, because compounding a series with
the risk-free rate stripped out understates the recovery. This module is
basis-agnostic -- it compounds whatever series it is handed -- so the
CALLER must pass total returns and the exhibit must be labelled
accordingly.

A NOTE THE REPORT MUST CARRY: Sec 5's drawdown figures (market -50.37%,
low-beta -52.59%) come from the MONTHLY Gate-4 BAB series. Sec 6's come
from the QUARTERLY long-only book. These are different objects measured
at different frequencies over different samples -- differing numbers are
expected and are NOT a discrepancy to reconcile. A quarterly series
cannot see an intra-quarter trough, so it will generally report a
shallower drawdown than a monthly one on the same underlying strategy.

Nothing in src/ computed a drawdown before this module; the figures
quoted in the docs came from throwaway numpy one-liners in _scratch/.
"""


import numpy as np
import polars as pl

from src.analysis import quarterly_compounding

# The wealth index is seeded at 1.0, so `wealth` is directly readable as
# "value of $1 invested at the start" and every drawdown is a fraction of
# a peak anchored outside the dataset. Named rather than written inline
# (CLAUDE.md: no magic numbers in src/) because it is the absolute
# anchor the test suite pins -- a scale defect shows up as a seed that
# is not 1.0, which no ratio-based check could detect.
WEALTH_SEED = 1.0

# Reported for `quarters_to_recover` when the series never regains its
# pre-drawdown peak within the sample.
#
# DELIBERATELY None, NOT a sentinel integer. A -1 or 999 in a "quarters
# to recover" column is precisely a wrong number that looks right, which
# CLAUDE.md names as the worst possible outcome; and a censored count
# presented under this name would silently answer a different question
# ("how long so far") with the label of the real one.
# src/portfolio/diagnostics.py sets this precedent already, returning
# None rather than 0 for weight_fraction_smallest_size_decile
# specifically so "not computed" cannot be mistaken for "computed as
# zero". `quarters_since_trough` carries the censored count under its
# own, honest name.
RECOVERY_NOT_ACHIEVED: int | None = None


def drawdown_path(
    returns: pl.DataFrame,
    *,
    value_col: str,
    label: str = "drawdown",
) -> pl.DataFrame:
    """The full wealth/peak/drawdown path -- the chartable series that
    drawdown_stats reduces.

    returns: [quarter (Date), <value_col> (Float64)], one row per
    period, sorted ascending by quarter internally.

    Returns [quarter, wealth, running_peak, drawdown] where:

        wealth_t       = WEALTH_SEED * prod(1 + r_i) for i <= t
        running_peak_t = max(WEALTH_SEED, max wealth_i for i <= t)
        drawdown_t     = wealth_t / running_peak_t - 1     (always <= 0)

    running_peak includes WEALTH_SEED itself, so a series that falls from
    its very first period is measured against the seed rather than
    against its own first (already-depressed) value -- otherwise an
    opening decline would report a drawdown of 0.

    Exposed as a public function, not an internal step, so the Fig 6.1
    drawdown chart and the Table 6.1 headline number are provably the
    same computation (see this module's test
    test_stats_are_a_reduction_over_the_path).
    """
    if value_col not in returns.columns:
        raise ValueError(
            f"drawdown_path: value_col={value_col!r} is not a column of the "
            f"input frame (has {returns.columns})"
        )
    if returns.height == 0:
        raise ValueError(
            "drawdown_path: empty return series -- refusing to report a "
            "drawdown of 0.0, which would read as 'never fell' rather than "
            "'no data'"
        )

    # A drawdown is PATH-DEPENDENT, so it is uniquely sensitive to the
    # shape of the input rather than only its values -- hence
    # require_contiguous here, where the regression in
    # longonly_vs_market does not need it. gate-verifier findings
    # 2026-09-15, all three reproduced on this function:
    #   null value      -> max_drawdown = nan, np.argmin returns index 0
    #   duplicated qtr  -> n_periods 8 -> 9, max_drawdown -0.44 -> -0.61
    #   one qtr removed -> max_drawdown -0.44 -> -0.30 (compounds across
    #                      the gap and reports a shallower decline)
    # The item-B input is the cached backtest parquet, which passes
    # through no other completeness guard at all.
    quarterly_compounding.assert_clean_quarterly_series(
        returns, value_col=value_col, label=label, require_contiguous=True
    )

    ordered = returns.sort("quarter")
    rets = np.asarray(ordered[value_col].to_numpy(), dtype=float)

    wealth = WEALTH_SEED * np.cumprod(1.0 + rets)
    # The seed is a legitimate peak: prepend it, take the running max,
    # then drop it, so period 0 is compared against WEALTH_SEED.
    running_peak = np.maximum.accumulate(np.concatenate(([WEALTH_SEED], wealth)))[1:]
    dd = wealth / running_peak - 1.0

    return pl.DataFrame(
        {
            "quarter": ordered["quarter"].to_list(),
            "wealth": wealth,
            "running_peak": running_peak,
            "drawdown": dd,
        }
    )


def drawdown_stats(
    returns: pl.DataFrame,
    *,
    value_col: str,
    label: str,
) -> dict:
    """Scalar drawdown statistics, reduced from drawdown_path.

    label: names the series ("us_longonly_total", "us_market_total") so
    a table row is self-identifying.

    Returns:
      label, n_periods
      max_drawdown           most negative peak-to-trough decline
      peak_quarter           last quarter at/before the trough where
                             wealth == running peak; None if that peak
                             is the seed itself (see peak_is_seed)
      peak_is_seed           True when the drawdown began before the
                             first observation
      trough_quarter         where max_drawdown occurred (first, on ties)
      quarters_peak_to_trough
      recovery_quarter       first quarter after the trough where wealth
                             regains the pre-drawdown peak; None if never
      quarters_to_recover    trough -> recovery, or RECOVERY_NOT_ACHIEVED
      recovered              bool
      quarters_since_trough  ALWAYS populated -- the censored count, so a
                             non-recovery can be reported as "N quarters
                             and counting" rather than as a number that
                             reads like a completed recovery
      worst_period_ret       the single most negative period return
      worst_period_quarter
      final_wealth

    worst_period_ret and max_drawdown answer DIFFERENT questions and the
    report states both: a -50% quarter followed by another -50% quarter
    is a worst period of -0.50 but a max drawdown of -0.75.
    """
    path = drawdown_path(returns, value_col=value_col, label=label)

    wealth = np.asarray(path["wealth"].to_numpy(), dtype=float)
    peak = np.asarray(path["running_peak"].to_numpy(), dtype=float)
    dd = np.asarray(path["drawdown"].to_numpy(), dtype=float)
    quarters = path["quarter"].to_list()

    trough_idx = int(np.argmin(dd))
    max_dd = float(dd[trough_idx])
    peak_at_trough = float(peak[trough_idx])

    # The peak being drawn down FROM: the last index at or before the
    # trough whose wealth equals the running peak there. If none, the
    # peak is the seed itself -- flagged rather than silently reported
    # as the first quarter, which would misdate the drawdown.
    peak_idx = None
    for i in range(trough_idx, -1, -1):
        if wealth[i] == peak_at_trough:
            peak_idx = i
            break
    peak_is_seed = peak_idx is None

    recovery_idx = None
    for i in range(trough_idx + 1, len(wealth)):
        if wealth[i] >= peak_at_trough:
            recovery_idx = i
            break

    ordered_rets = np.asarray(returns.sort("quarter")[value_col].to_numpy(), dtype=float)
    worst_idx = int(np.argmin(ordered_rets))

    return {
        "label": label,
        "n_periods": len(quarters),
        "max_drawdown": max_dd,
        "peak_quarter": quarters[peak_idx] if peak_idx is not None else None,
        "peak_is_seed": peak_is_seed,
        "trough_quarter": quarters[trough_idx],
        "quarters_peak_to_trough": (
            trough_idx - peak_idx if peak_idx is not None else trough_idx + 1
        ),
        "recovery_quarter": quarters[recovery_idx] if recovery_idx is not None else None,
        "quarters_to_recover": (
            recovery_idx - trough_idx if recovery_idx is not None else RECOVERY_NOT_ACHIEVED
        ),
        "recovered": recovery_idx is not None,
        "quarters_since_trough": len(quarters) - 1 - trough_idx,
        "worst_period_ret": float(ordered_rets[worst_idx]),
        "worst_period_quarter": quarters[worst_idx],
        "final_wealth": float(wealth[-1]),
    }


def drawdown_table(rows: list[dict]) -> pl.DataFrame:
    """Several drawdown_stats dicts as one frame with a pinned column
    order, so the driver and the tests agree on layout."""
    columns = [
        "label",
        "n_periods",
        "max_drawdown",
        "peak_quarter",
        "peak_is_seed",
        "trough_quarter",
        "quarters_peak_to_trough",
        "recovery_quarter",
        "quarters_to_recover",
        "recovered",
        "quarters_since_trough",
        "worst_period_ret",
        "worst_period_quarter",
        "final_wealth",
    ]
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 for c in columns})
    return pl.DataFrame(rows).select(columns)
