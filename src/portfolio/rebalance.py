"""The monthly formation/holding loop, F2b (docs/03_roadmap.md). Wires
weights.rank_deviation_from_median -> legs.asymmetric_inverse_beta/
leg_ex_ante_beta -> diagnostics.compute_diagnostics across a date range.

PURE frame-in/frame-out core: no WRDS, no universe_at, no market_index,
no I/O of any kind. This is deliberate -- production wires
estimate_beta_fp()'s output into betas_by_date; tests/leakage/
test_no_lookahead.py's run_pipeline adapter wires the E2 fixture +
precomputed_betas into this SAME function. One code path, two callers,
so the leakage tests exercise production's actual construction rather
than a parallel test-only implementation.

Formation timing (CLAUDE.md's core rule, spec section 10): betas
estimated through the last trading day of month t-1 are used for the
return HELD in month t. betas_by_date's keys are formation dates (the
month-end each cross-section was estimated through); this loop looks up
the FOLLOWING month's return in monthly_rets, never the formation
month's own return. formation_lag_days (config/runs.yaml) composes
BEFORE this function ever sees a formation date -- it is threaded into
beta estimation upstream (src.estimation.beta_fp._resolve_window_starts),
never into this loop, which only ever advances forward by one calendar
month from whatever formation date it is given.
"""

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import yaml

from src.portfolio import diagnostics as diagnostics_module
from src.portfolio import legs, weights

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "portfolio.yaml"


def _load_rebalance_config() -> dict:
    """The min_names_total/min_names_per_leg/max_consecutive_skips
    thresholds from config/portfolio.yaml's rebalance block (CLAUDE.md:
    no magic numbers in src/)."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["rebalance"]


@dataclass
class BacktestResult:
    """returns/diagnostics are polars DataFrames, one row per HELD month
    that was not skipped. positions is one row per (formation_date,
    permno) actually weighted. skips maps a SKIPPED formation date to a
    human-readable reason -- a first-class, asserted output (not a
    diagnostics footnote), per the design decision that a shortened
    return series must be visible at the top level, never merely
    inferred from a row count."""

    returns: pl.DataFrame
    positions: pl.DataFrame
    diagnostics: pl.DataFrame
    skips: dict[datetime.date, str] = field(default_factory=dict)


def _next_month_end(d: datetime.date) -> datetime.date:
    """The calendar month-end immediately following d (d is itself
    assumed to be a month-end, since betas_by_date's keys are formation
    month-ends). Held-month lookups in monthly_rets use this, never d
    itself -- that is the one thing standing between this loop and a
    same-month formation/holding leak."""
    if d.month == 12:
        next_month_first = datetime.date(d.year + 1, 1, 1)
    else:
        next_month_first = datetime.date(d.year, d.month + 1, 1)
    following_month_first = (
        datetime.date(next_month_first.year + 1, 1, 1)
        if next_month_first.month == 12
        else datetime.date(next_month_first.year, next_month_first.month + 1, 1)
    )
    return following_month_first - datetime.timedelta(days=1)


def run_backtest(
    betas_by_date: dict[datetime.date, pl.DataFrame],
    monthly_rets: pl.DataFrame,
    monthly_rf: pl.DataFrame,
    *,
    min_names_total: int | None = None,
    min_names_per_leg: int | None = None,
    max_consecutive_skips: int | None = None,
    full_universe_mkt_cap: dict[datetime.date, pl.DataFrame] | None = None,
) -> BacktestResult:
    """The monthly formation/holding loop.

    betas_by_date: formation_date (a month-end) -> cross-section with a
    `permno` column and a `beta_shrunk` column (any others, e.g.
    `mkt_cap`, pass through). One entry per formation date the caller
    wants a held return for.

    monthly_rets: long-format permno, month, ret (arithmetic). `month`
    values must be the SAME month-end convention betas_by_date's keys
    use, advanced by exactly one calendar month for the held return.

    monthly_rf: month, rf.

    Threshold kwargs default to config/portfolio.yaml's rebalance block
    when omitted -- explicit kwargs (as the leakage-adapter tests use)
    override rather than requiring every caller to read the config file.

    full_universe_mkt_cap, if given, is keyed the SAME way as
    betas_by_date (per formation date) and passed through to
    diagnostics.compute_diagnostics for weight_fraction_smallest_size_
    decile; omitted, that diagnostic is None for every month (the
    documented behaviour of compute_diagnostics(), not a workaround).
    """
    cfg = _load_rebalance_config()
    if min_names_total is None:
        min_names_total = cfg["min_names_total"]
    if min_names_per_leg is None:
        min_names_per_leg = cfg["min_names_per_leg"]
    if max_consecutive_skips is None:
        max_consecutive_skips = cfg["max_consecutive_skips"]

    return_rows: list[dict] = []
    position_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    skips: dict[datetime.date, str] = {}
    consecutive_skips = 0

    rf_by_month = dict(zip(monthly_rf["month"].to_list(), monthly_rf["rf"].to_list()))

    for formation_date in sorted(betas_by_date.keys()):
        holding_month = _next_month_end(formation_date)
        betas = betas_by_date[formation_date]

        if betas.height < min_names_total:
            reason = (
                f"total universe too thin: {betas.height} names < "
                f"min_names_total={min_names_total}"
            )
            skips[formation_date] = reason
            consecutive_skips += 1
            if consecutive_skips > max_consecutive_skips:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive skipped formation "
                    f"dates exceeds max_consecutive_skips={max_consecutive_skips} "
                    f"-- most recent: {formation_date} ({reason}). A long "
                    "run of skips means something is broken upstream, not "
                    "an ordinary thin-sample-start effect."
                )
            continue

        weighted = weights.rank_deviation_from_median(betas)

        n_long = weighted.filter(pl.col("weight_long") > 0.0).height
        n_short = weighted.filter(pl.col("weight_short") > 0.0).height
        if n_long < min_names_per_leg or n_short < min_names_per_leg:
            reason = (
                f"leg too thin: {n_long} long / {n_short} short names < "
                f"min_names_per_leg={min_names_per_leg}"
            )
            skips[formation_date] = reason
            consecutive_skips += 1
            if consecutive_skips > max_consecutive_skips:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive skipped formation "
                    f"dates exceeds max_consecutive_skips={max_consecutive_skips} "
                    f"-- most recent: {formation_date} ({reason}). A long "
                    "run of skips means something is broken upstream, not "
                    "an ordinary thin-sample-start effect."
                )
            continue

        held_rets = monthly_rets.filter(pl.col("month") == holding_month)
        ret_by_permno = dict(zip(held_rets["permno"].to_list(), held_rets["ret"].to_list()))

        long_leg = weighted.filter(pl.col("weight_long") > 0.0)
        short_leg = weighted.filter(pl.col("weight_short") > 0.0)
        long_missing = [p for p in long_leg["permno"].to_list() if p not in ret_by_permno]
        short_missing = [p for p in short_leg["permno"].to_list() if p not in ret_by_permno]
        if long_missing or short_missing:
            # CLAUDE.md: "a wrong number that looks right is the worst
            # possible outcome." Silently summing over only the names
            # that happen to have a held return (a) would let a
            # zero-coverage month (e.g. a caller-truncated monthly_rets
            # whose last formation's held month falls past the cutoff --
            # tests/leakage/test_no_lookahead.py's own truncation-test
            # shape) fabricate an exactly-0.0 "return" via sum([]), and
            # (b) even with partial coverage, would silently change what
            # FRACTION of the leg each surviving name represents without
            # recording that anything was amiss -- weight_long/
            # weight_short are normalized to sum to 1.0 over the FULL
            # weighted leg (weights.rank_deviation_from_median's own
            # invariant), so dropping a subset changes that invariant
            # for the remaining names without ever un-normalizing first.
            reason = (
                f"held month {holding_month} missing return data for "
                f"{len(long_missing)} long-leg / {len(short_missing)} "
                "short-leg names -- refusing to silently renormalize or "
                "fabricate a return over an incomplete leg"
            )
            skips[formation_date] = reason
            consecutive_skips += 1
            if consecutive_skips > max_consecutive_skips:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive skipped formation "
                    f"dates exceeds max_consecutive_skips={max_consecutive_skips} "
                    f"-- most recent: {formation_date} ({reason}). A long "
                    "run of skips means something is broken upstream, not "
                    "an ordinary thin-sample-start effect."
                )
            continue

        consecutive_skips = 0

        beta_long = legs.leg_ex_ante_beta(
            weighted, weight_col="weight_long", beta_col="beta_shrunk"
        )
        beta_high = legs.leg_ex_ante_beta(
            weighted, weight_col="weight_short", beta_col="beta_shrunk"
        )

        r_long = sum(
            w * ret_by_permno[p]
            for p, w in zip(long_leg["permno"].to_list(), long_leg["weight_long"].to_list())
        )
        r_short = sum(
            w * ret_by_permno[p]
            for p, w in zip(short_leg["permno"].to_list(), short_leg["weight_short"].to_list())
        )
        r_f = rf_by_month.get(holding_month, 0.0)

        bab_ret = legs.asymmetric_inverse_beta(
            r_long=r_long, r_short=r_short, beta_long=beta_long, beta_high=beta_high, r_f=r_f
        )
        return_rows.append({"month": holding_month, "ret": bab_ret})

        for row in weighted.to_dicts():
            position_rows.append({"formation_date": formation_date, **row})

        universe_mkt_cap = (
            full_universe_mkt_cap.get(formation_date) if full_universe_mkt_cap else None
        )
        diag = diagnostics_module.compute_diagnostics(
            weighted,
            realized_market_loading=None,
            r_long=r_long,
            r_short=r_short,
            dollars_long=1.0 / beta_long,
            dollars_short=1.0 / beta_high,
            full_universe_mkt_cap=universe_mkt_cap,
        )
        diag["formation_date"] = formation_date
        diag["holding_month"] = holding_month
        diagnostic_rows.append(diag)

    returns_df = pl.DataFrame(return_rows, schema={"month": pl.Date, "ret": pl.Float64}) \
        if return_rows else pl.DataFrame(schema={"month": pl.Date, "ret": pl.Float64})
    positions_df = pl.DataFrame(position_rows) if position_rows else pl.DataFrame()
    diagnostics_df = pl.DataFrame(diagnostic_rows) if diagnostic_rows else pl.DataFrame()

    return BacktestResult(
        returns=returns_df,
        positions=positions_df,
        diagnostics=diagnostics_df,
        skips=skips,
    )
