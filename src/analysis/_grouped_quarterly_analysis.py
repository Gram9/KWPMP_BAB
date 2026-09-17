"""Shared quarterly-grouping aggregation core, extracted this session
from src/analysis/beta_buckets.py so two callers -- the percentile-based
SML buckets (beta_buckets.py) and the target-beta portfolios
(target_beta.py) -- share ONE tested, leakage-audited implementation of
"assign names to groups at each formation date, hold one quarter
equal-weighted, aggregate full-sample" rather than diverging copies.

The ONLY thing that differs between callers is HOW names get assigned to
group labels at each formation date (`assign_fn`) -- everything else
(fringe filter, per-quarter equal-weighted return, quarterly market
compounding, full-sample aggregation with mean_ret/sharpe/se_ret/
realized_beta/alpha/alpha_t_stat) is identical regardless of what a
"group" means to the caller. `assign_fn` may return MORE rows than its
input (one row per (name, group) match, not one row per name) -- this
module's own group_by("bucket") calls are agnostic to that; a name
appearing under multiple group labels (e.g. one of the 20 nearest to two
different target betas) is handled correctly, just like a name
appearing under exactly one label (percentile buckets, which partition
the universe exclusively) is.

THIS IS A REFACTOR OF ALREADY-AUDITED CODE, not new logic -- every line
below is beta_buckets.py's own pre-extraction implementation, with
`assign_buckets(cross_section, beta_col=beta_col)` replaced by a passed-
in `assign_fn(cross_section, beta_col)` and the bucket-count minimum
replaced by a passed-in `min_names_for_assignment`. beta_buckets.py's
own 17-test suite (and 26 combined with beta_bucket_timeseries.py)
passing unmodified against this module, plus a real-data byte-identical
regression check, is the acceptance bar for this extraction -- not a
fresh leakage audit of "new" logic, since none of the temporal-ordering
behavior changed.
"""

import datetime
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.analysis import quarterly_compounding
from src.portfolio import diagnostics
from src.portfolio.longonly import _quarterly_return
from src.portfolio.quarterly import next_quarter_end


@dataclass
class GroupedQuarterlyResult:
    """Two views of the same run -- see src.analysis.beta_buckets.
    BetaBucketResult (kept as an alias of this class, so existing
    imports/tests are unaffected by this extraction):

    - summary: one row per group, the full-sample aggregate (mean
      return, Sharpe, realized beta, alpha, alpha_t_stat, se_ret).
    - quarterly: one row per (group, held quarter) -- bucket, quarter,
      ret, beta, n. The per-formation-date detail `summary` is built
      from, exposed so a caller can build a time-stability view or
      confidence bars without a second backtest.
    """

    summary: pl.DataFrame
    quarterly: pl.DataFrame


def run_grouped_quarterly_analysis(
    betas_by_date: dict[datetime.date, pl.DataFrame],
    fringe_by_date: dict[datetime.date, pl.DataFrame],
    quarterly_monthly_rets: pl.DataFrame,
    market_monthly: pl.DataFrame,
    *,
    mkt_cap_floor: float | None,
    price_floor: float | None,
    id_col: str,
    beta_col: str,
    assign_fn: Callable[[pl.DataFrame, str], pl.DataFrame],
    min_names_for_assignment: int,
) -> GroupedQuarterlyResult:
    """Forms groups at every formation date via `assign_fn`, holds each
    one quarter forward equal-weighted, then aggregates full-sample per
    group: mean quarterly return, annualized Sharpe (periods_per_year=4
    -- QUARTERLY, not the diagnostics module's monthly default),
    realized beta, alpha, and alpha_t_stat (via diagnostics.
    full_sample_market_loading against market_monthly -- NEVER t_stat
    for an alpha claim).

    assign_fn(cross_section, beta_col) -> DataFrame with an added
    `bucket` column (string group label). May return MORE rows than
    `cross_section` if a name qualifies for multiple groups (e.g.
    target-beta portfolios, where one name can be among the 20 closest
    to more than one target) -- this function's own group_by("bucket")
    calls handle that correctly either way.

    mkt_cap_floor/price_floor: fringe filter applied cross-sectionally
    AT EACH FORMATION DATE (CLAUDE.md: never pooled). Pass BOTH as None
    to skip filtering entirely.

    market_monthly: columns [month, ret] -- the SAME market index used
    for beta estimation (spec Sec 7: estimation and evaluation
    benchmarks must be the same object). Caller's responsibility to pass
    the correct leg's index.

    min_names_for_assignment: skip a formation date's cross-section
    entirely if it has fewer than this many eligible names -- group
    assignment isn't meaningful below this floor. Caller-specified
    (percentile buckets need enough names for 15 non-trivial bands;
    target-beta portfolios need enough for a real top-20 per target),
    never hardcoded here.

    `summary` has one row per group: bucket, mean_ret, sharpe,
    ex_ante_beta (the group's own mean beta at formation, AVERAGED OVER
    THE SAME SURVIVING POPULATION as mean_ret/realized_beta -- never
    over the full pre-survival membership, see
    _equal_weighted_group_return's docstring for why that distinction
    matters), realized_beta, alpha, alpha_t_stat, n_quarters (held
    quarters this group has a real return in), se_ret (standard error
    of the quarterly mean return, std(ret, ddof=1)/sqrt(n_quarters) --
    the input to a 95% CI band, `mean_ret +/- 1.96*se_ret`. NOTE this
    treats held quarters as independent draws, which is an
    approximation -- group membership is a full re-sort each quarter
    here, closer to independent than a turnover-limited real portfolio's
    quarters, but still not exactly independent).

    `quarterly` has one row per (group, held quarter): bucket, quarter,
    ret, beta, n.
    """
    quarter_rows: list[pl.DataFrame] = []

    for formation_date in sorted(betas_by_date.keys()):
        betas = betas_by_date[formation_date]
        fringe = fringe_by_date.get(formation_date)
        if fringe is None or fringe.height == 0:
            continue

        if mkt_cap_floor is not None or price_floor is not None:
            filt = pl.lit(True)
            if mkt_cap_floor is not None:
                filt = filt & (pl.col("mkt_cap") > mkt_cap_floor)
            if price_floor is not None:
                filt = filt & (pl.col("price") > price_floor)
            fringe = fringe.filter(filt)

        cross_section = fringe.join(betas, on=id_col, how="inner")
        if cross_section.height < min_names_for_assignment:
            # Fewer names than the caller-specified floor -- group
            # assignment is meaningless at this formation date (mirrors
            # longonly.py's min_survivors guard in spirit, though this
            # module has no equivalent skip-with-reason output since it
            # is an analysis pass, not a portfolio construction loop).
            continue

        grouped = assign_fn(cross_section, beta_col)

        holding_quarter_end = next_quarter_end(formation_date)
        group_rets = _equal_weighted_group_return(
            grouped, quarterly_monthly_rets, id_col, beta_col, holding_quarter_end
        )
        if group_rets.height > 0:
            quarter_rows.append(
                group_rets.with_columns(pl.lit(holding_quarter_end).alias("quarter"))
            )

    empty_quarterly_schema = {
        "bucket": pl.Utf8,
        "ret": pl.Float64,
        "beta": pl.Float64,
        "n": pl.Int64,
        "quarter": pl.Date,
    }
    if not quarter_rows:
        return GroupedQuarterlyResult(
            summary=pl.DataFrame(
                schema={
                    "bucket": pl.Utf8,
                    "mean_ret": pl.Float64,
                    "sharpe": pl.Float64,
                    "ex_ante_beta": pl.Float64,
                    "realized_beta": pl.Float64,
                    "alpha": pl.Float64,
                    "alpha_t_stat": pl.Float64,
                    "n_quarters": pl.Int64,
                    "se_ret": pl.Float64,
                }
            ),
            quarterly=pl.DataFrame(schema=empty_quarterly_schema),
        )

    all_quarters = pl.concat(quarter_rows, how="vertical")

    # The market's own return must be QUARTERLY-COMPOUNDED the SAME way
    # group returns are (via _quarterly_return's log-space compounding),
    # not left at monthly grain and joined against quarterly group
    # returns on a shared "quarter" label -- that would regress a
    # quarterly return against a single MONTH's return (whichever month
    # happened to share the quarter-end's date value), not against the
    # market's real quarterly return.
    #
    # EXTRACTED 2026-09-15 to src.analysis.quarterly_compounding so that
    # docs/06_handoff_section6.md item A (the 20+20 long-only book
    # regressed on its own market index) compounds the market through
    # the SAME code rather than a second near-identical loop. Behaviour
    # is unchanged: that helper delegates to the same _quarterly_return,
    # and emits no row for a quarter with zero matching months, exactly
    # as the loop this replaced did. Its extra n_months column is
    # dropped here -- this module deliberately does NOT assert on it
    # (see quarterly_compounding.assert_full_quarters' own docstring for
    # why that guard is opt-in), so the downstream join below is
    # identical to before.
    held_quarters_needed = sorted(all_quarters["quarter"].unique().to_list())
    market_by_quarter = quarterly_compounding.compound_monthly_series_to_quarters(
        market_monthly, quarters=held_quarters_needed, value_col="ret"
    ).select(pl.col("quarter"), pl.col("ret").alias("mkt_ret"))

    result_rows = []
    for bucket_name in sorted(all_quarters["bucket"].unique().to_list()):
        bucket_series = all_quarters.filter(pl.col("bucket") == bucket_name).sort("quarter")
        n_quarters = bucket_series.height
        ret_values = np.asarray(bucket_series["ret"].to_numpy(), dtype=float)
        mean_ret = float(ret_values.mean())
        sharpe = (
            diagnostics.annualized_sharpe(bucket_series["ret"].to_list(), periods_per_year=4)
            if n_quarters >= 2
            else None
        )
        se_ret = (
            float(ret_values.std(ddof=1) / np.sqrt(n_quarters)) if n_quarters >= 2 else None
        )

        joined = bucket_series.join(market_by_quarter, on="quarter", how="inner")
        if joined.height >= 3:
            loading = diagnostics.full_sample_market_loading(
                joined["ret"].to_list(), joined["mkt_ret"].to_list()
            )
            realized_beta = loading["beta"]
            alpha = loading["alpha"]
            alpha_t_stat = loading["alpha_t_stat"]
        else:
            realized_beta = None
            alpha = None
            alpha_t_stat = None

        ex_ante_beta = float(np.asarray(bucket_series["beta"].to_numpy(), dtype=float).mean())

        result_rows.append(
            {
                "bucket": bucket_name,
                "mean_ret": mean_ret,
                "sharpe": sharpe,
                "ex_ante_beta": ex_ante_beta,
                "realized_beta": realized_beta,
                "alpha": alpha,
                "alpha_t_stat": alpha_t_stat,
                "n_quarters": n_quarters,
                "se_ret": se_ret,
            }
        )

    summary = pl.DataFrame(result_rows).sort("ex_ante_beta")
    quarterly = all_quarters.select(["bucket", "quarter", "ret", "beta", "n"])
    return GroupedQuarterlyResult(summary=summary, quarterly=quarterly)


def _equal_weighted_group_return(
    grouped: pl.DataFrame,
    monthly_rets: pl.DataFrame,
    id_col: str,
    beta_col: str,
    holding_quarter_end: datetime.date,
) -> pl.DataFrame:
    """For every group present in `grouped`, the equal-weighted mean of
    its members' quarterly returns (via longonly._quarterly_return --
    the SAME tested compounding, including partial-quarter handling for
    delisted names). A group with zero members having a real return
    this quarter is simply absent from the output -- no fabricated 0%.

    ALSO returns each group's mean beta over the SAME surviving
    population as `ret` -- leakage-auditor finding from this session's
    beta_buckets.py work, preserved here: an earlier version computed
    ex_ante_beta over the FULL group membership (before the return-
    survival join) while ret/realized_beta used the SURVIVING subset
    only. On real (non-balanced) panels this is a real ex-ante/ex-post
    population mismatch, not a temporal leak -- but it biases the x/y
    pairing of any chart built from this output (measured, in the SML
    case, to flatten the fitted line in exactly the direction the BAB
    thesis predicts -- an artifact that would look like confirming
    evidence). Fixed by computing BOTH ret and beta over the identical
    `merged` (survived) population, never over `grouped` directly.

    `grouped` may have MORE rows than one-per-name (assign_fn's
    contract allows a name to appear under multiple group labels) --
    the join below and group_by("bucket") handle this correctly:
    each (name, group) row independently joins to that name's own
    quarterly return, so a name in two groups contributes its return to
    both groups' means, not split between them."""
    quarter_rets = _quarterly_return(monthly_rets, id_col, holding_quarter_end)
    merged = grouped.select(["bucket", id_col, beta_col]).join(quarter_rets, on=id_col, how="inner")
    if merged.height == 0:
        return pl.DataFrame(
            schema={"bucket": pl.Utf8, "ret": pl.Float64, "beta": pl.Float64, "n": pl.Int64}
        )
    return (
        merged.group_by("bucket")
        .agg(
            pl.col("ret").mean().alias("ret"),
            pl.col(beta_col).mean().alias("beta"),
            pl.len().alias("n"),
        )
        .sort("bucket")
    )
