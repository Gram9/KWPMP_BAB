"""The long-only, quarterly, equal-weight low-beta backtester,
docs/04_handoff_lowbeta_longonly.md Sec 4.1. KWPMP is a margin-
constrained, long-only investor (handoff Sec 1) -- no legs, no shorting,
no leverage, no leg scaling. This is deliberately NOT built on
src/portfolio/rebalance.py: that module is inherently two-legged and
BAB-shaped (asymmetric inverse-beta leg scaling, permno-hardcoded joins,
a stateless monthly loop with no turnover machinery) -- see the approved
plan's exploration findings for the full comparison. This module reuses
that file's PATTERNS (skip-with-reason, circuit breaker, a
dataclass-shaped result), not its code.

PURE frame-in/frame-out core, same discipline as rebalance.run_backtest:
no WRDS, no universe_at, no market_index calls inside this module. A
caller (per-leg wrapper, not yet built -- see handoff Sec 8 step 4's
remaining wiring) is responsible for assembling betas_by_date,
fringe-filter snapshots, and quarterly_rets from the real data layer
(src.data.universe_panel / src.data.chass_universe / src.estimation.
beta_monthly / src.market_index.monthly_panel) and passing them in here.

Formation timing (CLAUDE.md's core rule): betas_by_date's keys are
formation dates -- the quarter-end whose LAST TRADING DAY's data the
beta was estimated through (one month before the quarter being held,
per handoff Sec 3.4). This loop looks up the FOLLOWING quarter's return
via src.portfolio.quarterly.next_quarter_end, NEVER the formation
quarter's own return -- see that module's docstring for the exact
mistake (handoff Sec 7) this discipline exists to prevent.

TURNOVER is the one genuinely new piece of state this build needs that
rebalance.run_backtest's stateless monthly loop has no equivalent of:
the director's spec is "drop the worst-ranked names, buy the
best-ranked replacements" (handoff Sec 1), which requires knowing what
was held LAST quarter, not just what ranks best THIS quarter.
"""

import datetime
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from src.portfolio.quarterly import next_quarter_end


@dataclass
class LongOnlyResult:
    """Mirrors rebalance.BacktestResult's shape (returns/positions/
    skips), minus the diagnostics field (BAB-specific: dollars_long/
    short, ex_ante_beta_spread have no meaning for an unlevered,
    unscaled equal-weight book) plus turnover, a first-class output
    tracking what was dropped/added each quarter -- the director's own
    spec item, not a footnote."""

    returns: pl.DataFrame
    positions: pl.DataFrame
    turnover: pl.DataFrame
    skips: dict[datetime.date, str] = field(default_factory=dict)


def _quarterly_return(
    monthly_rets: pl.DataFrame, id_col: str, holding_quarter_end: datetime.date
) -> pl.DataFrame:
    """Compound the 3 calendar months ending at holding_quarter_end into
    one quarterly arithmetic return per name, via expm1(sum(log1p(r)))
    -- same log-space-summation convention monthly_returns.py and
    beta_fp._daily_log_returns both use, avoiding float-rounding drift
    from chaining (1+r).prod()-1 directly.

    A name missing ONE OR TWO of the 3 constituent months (delisted
    mid-quarter -- the missing months are simply the ones AFTER its last
    real month within the quarter, since monthly_arithmetic_returns
    produces no row past a name's last real trade) gets its REAL months
    compounded (this correctly captures a delisting return, which
    monthly_returns.py already folds into whatever calendar month CRSP
    stamped it) with the missing remainder of the quarter treated as 0%
    cash -- a partial-quarter return, not a fabricated full one and not
    a silent exclusion either. user-confirmed this session, replacing an
    earlier "exclude entirely" design after leakage-auditor flagged that
    excluding delisted names understates losses specifically in
    downturns, exactly the period handoff Sec 1's "performance in
    downturns" analysis cares about.

    A name missing ALL 3 months (never listed in this quarter at all, or
    every one of its monthly rows for the quarter is null) produces NO
    ROW -- there is nothing to compound, real or partial; this is
    distinct from "delisted mid-quarter" and must not be treated the
    same way (a name absent from the input entirely is a data-coverage
    gap the caller should investigate, not a 0%-cash quarter)."""
    month_2 = holding_quarter_end
    month_1 = _prior_month_end(month_2)
    month_0 = _prior_month_end(month_1)
    quarter_months = [month_0, month_1, month_2]

    window = monthly_rets.filter(
        pl.col("month").is_in(quarter_months) & pl.col("ret").is_not_null()
    )

    has_any = window.select(id_col).unique()
    if has_any.height == 0:
        return pl.DataFrame(schema={id_col: monthly_rets.schema[id_col], "ret": pl.Float64})

    # Real months compounded in log space; missing months within the
    # quarter contribute log(1 + 0.0) = 0.0 -- i.e. simply don't add to
    # the sum. A missing month is therefore already correctly a 0%-cash
    # contribution without needing an explicit fill/join against
    # quarter_months, since the log-sum only ever sums over rows that
    # exist.
    return (
        window.group_by(id_col)
        .agg((pl.col("ret") + 1.0).log().sum().alias("_log_sum"))
        .with_columns((pl.col("_log_sum").exp() - 1.0).alias("ret"))
        .select([id_col, "ret"])
    )


def _prior_month_end(d: datetime.date) -> datetime.date:
    """The calendar month-end immediately before d (d itself assumed a
    month-end). Local to this module -- only used to walk backward
    within a single quarter to find its 3 constituent months, not a
    general-purpose date utility (that role is
    src.portfolio.quarterly.next_quarter_end, which only ever advances)."""
    first_of_month = d.replace(day=1)
    return first_of_month - datetime.timedelta(days=1)


def run_longonly_backtest(
    betas_by_date: dict[datetime.date, pl.DataFrame],
    fringe_by_date: dict[datetime.date, pl.DataFrame],
    quarterly_monthly_rets: pl.DataFrame,
    *,
    n_holdings: int,
    turnover_k: int,
    min_survivors: int,
    mkt_cap_floor: float,
    price_floor: float,
    id_col: str = "id",
    max_consecutive_skips: int = 6,
) -> LongOnlyResult:
    """The quarterly formation/holding loop.

    betas_by_date: formation_date (a quarter-end, the last trading day of
    the month before the held quarter) -> cross-section with [id_col,
    "beta"] (RAW/unshrunk -- handoff Sec 3.2). Any other columns pass
    through unused.

    fringe_by_date: SAME keys as betas_by_date -> cross-section with
    [id_col, "mkt_cap", "price"] -- that formation date's OWN lagged
    snapshot. The mkt_cap_floor/price_floor filter is applied HERE, per
    formation date, cross-sectionally -- CLAUDE.md: never a pooled/
    full-sample filter. A name absent from this frame, or present but
    failing either floor, is not eligible for selection at that date.

    quarterly_monthly_rets: long format [id_col, "month", "ret"] --
    MONTHLY arithmetic returns (not pre-compounded to quarterly; this
    function compounds each held quarter's 3 months itself via
    _quarterly_return, so a name missing one of the 3 months is excluded
    from that quarter rather than silently partial).

    n_holdings/turnover_k/min_survivors/mkt_cap_floor/price_floor:
    config/lowbeta_longonly.yaml values, passed explicitly (never read
    from the config file directly inside this pure core -- matches
    rebalance.run_backtest's own convention of accepting thresholds as
    kwargs, testable without a config file on disk).

    Turnover mechanic (director's spec, handoff Sec 1): at the FIRST
    formation date, select the n_holdings lowest-beta eligible names
    outright. At every SUBSEQUENT formation date: rank all eligible
    names by beta ascending; a currently-held name is dropped ONLY if it
    falls into the bottom turnover_k of the currently-held set BY RANK
    (i.e. is among the turnover_k worst-ranked of last quarter's
    holdings); it is replaced by the best-ranked eligible name not
    already held. This directly implements "drop the worst-ranked
    names, buy the best-ranked replacements" rather than a full re-pick
    every quarter (which would defeat the entire point of a turnover
    parameter).
    """
    held: list = []  # currently held ids, in no particular order
    return_rows: list[dict] = []
    position_rows: list[dict] = []
    turnover_rows: list[dict] = []
    skips: dict[datetime.date, str] = {}
    consecutive_skips = 0

    for formation_date in sorted(betas_by_date.keys()):
        holding_quarter_end = next_quarter_end(formation_date)
        betas = betas_by_date[formation_date]
        fringe = fringe_by_date.get(formation_date)

        if fringe is None or fringe.height == 0:
            reason = f"no fringe-filter snapshot for formation_date={formation_date}"
            skips[formation_date] = reason
            consecutive_skips += 1
            if consecutive_skips > max_consecutive_skips:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive skipped formation dates "
                    f"exceeds max_consecutive_skips={max_consecutive_skips} "
                    f"-- most recent reason: {reason!r}"
                )
            continue

        fringe_passing = fringe.filter(
            (pl.col("mkt_cap") > mkt_cap_floor) & (pl.col("price") > price_floor)
        )
        survivors = fringe_passing.join(betas, on=id_col, how="inner")
        n_survivors = survivors.height

        if n_survivors < min_survivors:
            reason = (
                f"only {n_survivors} names cleared the mkt_cap>{mkt_cap_floor}/"
                f"price>{price_floor} fringe filter AND had a beta, below "
                f"min_survivors={min_survivors}"
            )
            skips[formation_date] = reason
            consecutive_skips += 1
            if consecutive_skips > max_consecutive_skips:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive skipped formation dates "
                    f"exceeds max_consecutive_skips={max_consecutive_skips} "
                    f"-- most recent reason: {reason!r}"
                )
            continue

        ranked = survivors.sort("beta")  # ascending -- lowest beta first
        ranked_ids = ranked[id_col].to_list()
        rank_of = {name: i for i, name in enumerate(ranked_ids)}

        if not held:
            selected = ranked_ids[:n_holdings]
            dropped: list = []
            added = list(selected)
        else:
            still_eligible_held = [h for h in held if h in rank_of]
            no_longer_eligible = [h for h in held if h not in rank_of]

            still_eligible_held_sorted = sorted(
                still_eligible_held, key=lambda h: rank_of[h]
            )
            # Worst-ranked (highest beta rank) of the currently-held,
            # still-eligible names -- candidates for replacement, up to
            # turnover_k of them. Names that fell OUT of eligibility
            # entirely (delisted, failed the fringe filter) are ALWAYS
            # dropped regardless of turnover_k -- turnover_k caps
            # discretionary replacement, it cannot force this build to
            # hold a name with no current beta or price.
            k = min(turnover_k, len(still_eligible_held_sorted))
            worst_of_held = (
                still_eligible_held_sorted[-k:] if k > 0 else []
            )
            dropped = no_longer_eligible + worst_of_held
            kept = [h for h in held if h not in dropped]

            replacement_pool = [n for n in ranked_ids if n not in kept]
            n_needed = n_holdings - len(kept)
            added = replacement_pool[:n_needed]
            selected = kept + added

        held = selected

        quarter_rets = _quarterly_return(quarterly_monthly_rets, id_col, holding_quarter_end)
        held_rets = quarter_rets.filter(pl.col(id_col).is_in(selected))

        if held_rets.height == 0:
            reason = (
                f"no held name had ANY real monthly return for holding "
                f"quarter {holding_quarter_end} -- not even a partial one"
            )
            skips[formation_date] = reason
            consecutive_skips += 1
            if consecutive_skips > max_consecutive_skips:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive skipped formation dates "
                    f"exceeds max_consecutive_skips={max_consecutive_skips} "
                    f"-- most recent reason: {reason!r}"
                )
            continue

        # Equal-weight over names with a REALIZED (possibly partial-
        # quarter) return. A name that delisted mid-quarter contributes
        # its real months' compounded return, including the delisting
        # return itself (_quarterly_return's docstring) -- it is NOT
        # excluded and renormalized away, which would understate losses
        # concentrated in downturns (user-confirmed fix this session,
        # after leakage-auditor flagged the original exclude-and-
        # renormalize design as biasing the exact "performance in
        # downturns" analysis handoff Sec 1 asks for). Only a name with
        # ZERO real months this quarter is absent from held_rets at all
        # (handled by the height==0 skip above at the PORTFOLIO level,
        # not per-name here -- a single such name inside an otherwise-
        # healthy quarter simply isn't in quarter_rets and is silently
        # absent from this mean, which is correct: there is no return,
        # partial or otherwise, to average in).
        portfolio_ret = float(np.asarray(held_rets["ret"].to_numpy(), dtype=float).mean())

        return_rows.append({"quarter": holding_quarter_end, "ret": portfolio_ret})
        for row in selected:
            position_rows.append({"formation_date": formation_date, id_col: row})
        turnover_rows.append(
            {
                "formation_date": formation_date,
                "n_dropped": len(dropped),
                "n_added": len(added),
                "n_held": len(selected),
            }
        )

        consecutive_skips = 0

    return LongOnlyResult(
        returns=pl.DataFrame(return_rows) if return_rows else pl.DataFrame(schema={"quarter": pl.Date, "ret": pl.Float64}),
        positions=pl.DataFrame(position_rows) if position_rows else pl.DataFrame(schema={"formation_date": pl.Date, id_col: pl.Utf8}),
        turnover=pl.DataFrame(turnover_rows) if turnover_rows else pl.DataFrame(
            schema={"formation_date": pl.Date, "n_dropped": pl.Int64, "n_added": pl.Int64, "n_held": pl.Int64}
        ),
        skips=skips,
    )
