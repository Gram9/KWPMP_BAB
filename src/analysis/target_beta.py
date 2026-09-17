"""Target-beta portfolios, docs/04_handoff_lowbeta_longonly.md's spirit
extended per user request this session: instead of the 15 SML buckets
(src.analysis.beta_buckets), which slice the universe by CROSS-SECTIONAL
RANK (bottom 5%, next 5%, ... -- a bucket's realized beta is whatever
that rank happens to average, drifting quarter to quarter, never pinned
to a chosen number), this module asks a different question at each
formation date: "of all eligible names, which 20 are closest to a
SPECIFIC target beta?" -- for a fixed grid of target levels.

Shares the exact same per-formation-date aggregation core as
beta_buckets.py (src.analysis._grouped_quarterly_analysis) -- only the
group-assignment rule differs. See that module's docstring for why the
core was extracted rather than duplicated.

TARGET_BETAS and N_HOLDINGS_PER_TARGET are user-confirmed this session,
not invented defaults.
"""

import polars as pl

from src.analysis._grouped_quarterly_analysis import (
    GroupedQuarterlyResult,
    run_grouped_quarterly_analysis,
)

# 0.25..1.25 in 0.10 steps (11 targets), then 1.50/1.75/2.00 in 0.25
# steps (3 targets) -- 14 total. User-confirmed this session: finer,
# consistent resolution through market beta (1.0) and slightly above it,
# coarser further out where the report's thesis (low-beta outperformance)
# matters less.
TARGET_BETAS: list[float] = [round(0.25 + 0.10 * i, 2) for i in range(11)] + [1.50, 1.75, 2.00]
assert TARGET_BETAS == [
    0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.50, 1.75, 2.00,
], f"TARGET_BETAS drifted from the user-confirmed grid: {TARGET_BETAS}"

# 20 names per target portfolio -- user-confirmed this session, same
# size as the main 20+20 long-only book (src.portfolio.longonly), for
# direct comparability.
N_HOLDINGS_PER_TARGET = 20


def _target_label(target: float) -> str:
    """'T0.25', 'T2.00' -- a bucket label distinct from beta_buckets.py's
    'V01'/'D06' style, so the two group-label spaces can never collide
    if ever compared side by side."""
    return f"T{target:.2f}"


def assign_target_beta_groups(cross_section: pl.DataFrame, beta_col: str) -> pl.DataFrame:
    """For EACH target in TARGET_BETAS, independently: rank all rows in
    `cross_section` by |beta_col - target| ascending (ties broken by
    mkt_cap DESCENDING -- user-confirmed this session: prefer the
    larger, more liquid name when two names are exactly equidistant from
    a target), take the N_HOLDINGS_PER_TARGET closest, and tag those
    rows with bucket = f"T{target:.2f}".

    Computed cross-sectionally WITHIN this single call (never pooled
    across formation dates -- CLAUDE.md), exactly like
    beta_buckets.assign_buckets.

    UNLIKE assign_buckets (which partitions the universe exclusively,
    one row per input row), this function's output can have MORE rows
    than its input: a name simultaneously among the 20 closest to two
    different targets (e.g. a name at beta=0.80 could be closest-20 for
    both T0.75 and T0.85) appears once per target it qualifies for. The
    shared aggregation core (src.analysis._grouped_quarterly_analysis)
    is explicitly designed to handle this correctly -- see that
    module's own docstring.

    Requires `mkt_cap` to be a column of `cross_section` (present via
    the fringe join every caller already performs before this function
    is invoked, in run_grouped_quarterly_analysis's own loop).

    Returns a DataFrame with columns: everything in `cross_section`
    (including beta_col, mkt_cap, price, id) plus `bucket`. Fewer than
    N_HOLDINGS_PER_TARGET rows for a given target if `cross_section`
    itself has fewer eligible names than that -- never pads with
    fabricated rows.
    """
    if cross_section.height == 0:
        return cross_section.with_columns(pl.lit(None, dtype=pl.Utf8).alias("bucket"))

    group_frames = []
    for target in TARGET_BETAS:
        ranked = cross_section.with_columns(
            (pl.col(beta_col) - target).abs().alias("_dist")
        ).sort(["_dist", "mkt_cap"], descending=[False, True], nulls_last=True)
        closest = ranked.head(N_HOLDINGS_PER_TARGET).drop("_dist")
        group_frames.append(closest.with_columns(pl.lit(_target_label(target)).alias("bucket")))

    return pl.concat(group_frames, how="vertical")


def run_target_beta_analysis(
    betas_by_date,
    fringe_by_date,
    quarterly_monthly_rets: pl.DataFrame,
    market_monthly: pl.DataFrame,
    *,
    mkt_cap_floor: float | None,
    price_floor: float | None,
    id_col: str = "id",
    beta_col: str = "beta",
) -> GroupedQuarterlyResult:
    """Forms 14 target-beta portfolios (TARGET_BETAS) at every formation
    date, holds each one quarter forward equal-weighted, then aggregates
    full-sample per target -- see
    src.analysis._grouped_quarterly_analysis.run_grouped_quarterly_analysis
    for the full field-by-field docstring (this function is a thin
    wrapper naming assign_target_beta_groups as the grouping rule and
    N_HOLDINGS_PER_TARGET as the minimum cross-section size -- a
    formation date needs at least N_HOLDINGS_PER_TARGET eligible names
    for even a single target's top-20 to be meaningful).

    `summary.bucket` values are "T0.25".."T2.00" (see _target_label);
    `summary.ex_ante_beta` is each portfolio's own mean formation-date
    beta -- compare against the target itself (parse the numeric part of
    `bucket`, or zip against TARGET_BETAS in caller code) to see how
    closely the achieved beta tracks what was asked for.

    mkt_cap_floor/price_floor: the SAME fringe filter the 20+20 backtest
    and the SML buckets use, applied cross-sectionally at each formation
    date (CLAUDE.md: never pooled).
    """
    return run_grouped_quarterly_analysis(
        betas_by_date,
        fringe_by_date,
        quarterly_monthly_rets,
        market_monthly,
        mkt_cap_floor=mkt_cap_floor,
        price_floor=price_floor,
        id_col=id_col,
        beta_col=beta_col,
        assign_fn=assign_target_beta_groups,
        min_names_for_assignment=N_HOLDINGS_PER_TARGET,
    )
