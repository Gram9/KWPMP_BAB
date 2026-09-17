"""Per-leg data-layer wiring for src/portfolio/longonly.py's pure
frame-in/frame-out core, docs/04_handoff_lowbeta_longonly.md Sec 4.1 and
docs/05_report_spec.md's Day-1 build order. longonly.py itself takes no
WRDS/CHASS calls (its own docstring); this module IS the caller that
assembles betas_by_date/fringe_by_date/quarterly_monthly_rets from real
US/CHASS data.

Parallel-sibling dispatch (build_us_inputs / build_canada_inputs), not a
unified function -- matches this codebase's established convention
(gate_adapters.py's us_gate1_panel/canada_gate1_panel, beta_fp.py's
_LEG_PANELS dict): the two legs' identity keys, data sources, and market
index defaults already diverge everywhere else in this project (recorded
lesson: feedback_adapter_over_full_unification memory).

FORMATION DATES are quarterly month-ends (Mar/Jun/Sep/Dec-31): "the last
trading day of the month before the quarter being held" (handoff Sec 3.4)
IS a calendar quarter-end already, since a quarterly rebalance holds the
FOLLOWING quarter (src.portfolio.quarterly.next_quarter_end). This module
builds one betas_by_date per NAMED VARIANT (e.g. "window_24m",
"window_36m", "window_60m") at every such formation date in range --
computing multiple variants together is NOT a leakage risk: each
variant's OLS window inside estimate_beta_monthly is independently
bounded to month <= formation_date, so a 24m and a 60m regression at the
same formation_date are simply two independent historical regressions,
neither can see the other's data or anything post-formation_date.

Each leg's stock_monthly and market_monthly panels are built ONCE for the
whole date range (not per formation date) -- measured this session:
building US's full-history monthly returns costs ~90s, the monthly
market-index panel ~170s; a fresh per-date rebuild would multiply that by
the ~240 US formation dates for no benefit, since estimate_beta_monthly
already does its own point-in-time windowing internally.
"""

import datetime

import polars as pl

from src.data import chass_loader, chass_universe, universe_panel
from src.data.chass_loader import SECURITY_ID_COLUMNS
from src.data.monthly_returns import monthly_arithmetic_returns
from src.estimation.beta_monthly import estimate_beta_monthly
from src.market_index import monthly_panel
from src.market_index.build import build_index

DEFAULT_VARIANTS = ("window_24m", "window_36m", "window_60m")

# US market index for beta estimation/evaluation -- spec Sec 7's settled
# default (vw_uncapped), the SAME object used everywhere else in this
# project's US leg (matches beta_fp.py's own convention).
US_MARKET_INDEX_METHOD = "vw_uncapped"

# Canada market index -- spec Sec 7's settled default (vw_capped_10pct),
# required because of Nortel's ~28% peak weight in the uncapped index
# (config/market_index.yaml, src/market_index/build.py's own docstring).
CANADA_MARKET_INDEX_METHOD = "vw_capped_10pct"


def _quarterly_month_ends(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    """Every calendar quarter-end (Mar/Jun/Sep/Dec-31) in [start, end]."""
    out = []
    year, month = start.year, ((start.month - 1) // 3) * 3 + 3
    while True:
        if month == 12:
            qend = datetime.date(year, 12, 31)
        elif month == 6:
            qend = datetime.date(year, 6, 30)
        elif month == 9:
            qend = datetime.date(year, 9, 30)
        else:
            qend = datetime.date(year, 3, 31)
        if qend > end:
            break
        if qend >= start:
            out.append(qend)
        month += 3
        if month > 12:
            month -= 12
            year += 1
    return out


def build_us_inputs(
    start: datetime.date,
    end: datetime.date,
    *,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
) -> tuple[dict[str, dict[datetime.date, pl.DataFrame]], dict[datetime.date, pl.DataFrame], pl.DataFrame]:
    """US-leg inputs for longonly.run_longonly_backtest, id_col="id"
    (str(permno), matching longonly's default id_col so a caller does not
    need to override it for the US leg).

    Returns (betas_by_date_per_variant, fringe_by_date, quarterly_monthly_rets):
    - betas_by_date_per_variant: {variant_name: {formation_date: DataFrame[id, beta]}}
    - fringe_by_date: {formation_date: DataFrame[id, mkt_cap, price]} -- the
      SAME lagged snapshot universe_panel.market_cap_at() already produces,
      just relabeled id=str(permno) to match beta output's id column.
    - quarterly_monthly_rets: DataFrame[id, month, ret] -- monthly
      arithmetic returns, passed straight through to
      longonly.run_longonly_backtest's own quarterly-compounding logic.
    """
    formation_dates = _quarterly_month_ends(start, end)

    stock_monthly = monthly_arithmetic_returns(start, end, leg="us").rename({"permno": "id"})
    stock_monthly = stock_monthly.with_columns(pl.col("id").cast(pl.Utf8))

    market_panel = monthly_panel.us_monthly_panel(start, end)
    market_index = build_index(market_panel, method=US_MARKET_INDEX_METHOD)
    market_monthly = market_index.rename({"date": "month", "index_ret": "ret"})

    betas_by_date_per_variant: dict[str, dict[datetime.date, pl.DataFrame]] = {
        variant: {} for variant in variants
    }
    fringe_by_date: dict[datetime.date, pl.DataFrame] = {}

    for formation_date in formation_dates:
        mkt_cap_df, _coverage = universe_panel.market_cap_at(formation_date)
        if mkt_cap_df.height > 0:
            fringe_by_date[formation_date] = mkt_cap_df.select(
                pl.col("permno").cast(pl.Utf8).alias("id"), "mkt_cap", "price"
            )

        for variant in variants:
            betas = estimate_beta_monthly(
                stock_monthly, market_monthly, formation_date, variant=variant, id_col="id"
            )
            if betas.height > 0:
                betas_by_date_per_variant[variant][formation_date] = betas.select(["id", "beta"])

    return (
        betas_by_date_per_variant,
        fringe_by_date,
        stock_monthly.select(["id", "month", "ret"]),
    )


def build_canada_inputs(
    start: datetime.date,
    end: datetime.date,
    *,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
) -> tuple[dict[str, dict[datetime.date, pl.DataFrame]], dict[datetime.date, pl.DataFrame], pl.DataFrame]:
    """Canada-leg inputs for longonly.run_longonly_backtest, id_col="id"
    (f"{symbol}_{usage}", matching longonly's default id_col). Formation
    dates are CHASS's own real trading quarter-ends closest to (on or
    before) each calendar quarter-end -- chass_universe.universe_at()'s
    own docstring: CHASS trdate values are already each month's actual
    last trading day, not a naive calendar month-end.

    Returns the same 3-tuple shape as build_us_inputs.
    """
    monthly = chass_loader.load_monthly()
    all_trdates = sorted(d.date() for d in monthly["trdate-Trade Date"].unique().to_list())

    calendar_quarter_ends = _quarterly_month_ends(start, end)

    def _resolve_trdate(calendar_qend: datetime.date) -> datetime.date | None:
        candidates = [d for d in all_trdates if d <= calendar_qend]
        return max(candidates) if candidates else None

    formation_dates = sorted(
        {
            resolved
            for calendar_qend in calendar_quarter_ends
            if (resolved := _resolve_trdate(calendar_qend)) is not None
        }
    )

    # TWO representations of "month" are needed and must NOT be conflated
    # (leakage-auditor finding, this session): the beta path
    # (estimate_beta_monthly) joins stock_monthly against market_monthly
    # on exact date -- both use CHASS's own real trdate keys internally
    # and are self-consistent with each other, so stock_monthly_trdate
    # below stays trdate-keyed. But the RETURNED quarterly_monthly_rets
    # feeds longonly.py's _quarterly_return, which looks up exact
    # CALENDAR month-ends (via _prior_month_end) -- 29% of CHASS
    # trdates are NOT the calendar last day (measured: 161/552), so
    # joining trdate-keyed returns against calendar-month-end lookup
    # keys silently missed 1-2 of 3 months in 80% of Canadian held
    # quarters before this fix (verified: _quarterly_return's log-sum
    # design treats a "missing" month as 0% cash, exactly the mid-quarter-
    # delisting convention, firing instead on healthy names purely from a
    # date-key mismatch). quarterly_monthly_rets_calendar re-keys each
    # trdate's return to the CALENDAR month-end containing it.
    stock_monthly_trdate = (
        monthly.select(
            SECURITY_ID_COLUMNS
            + [
                pl.col("trdate-Trade Date").cast(pl.Date).alias("month"),
                pl.col("return-Monthly Return").alias("ret"),
            ]
        )
        .filter((pl.col("month") >= start) & (pl.col("month") <= end))
        .with_columns(
            (
                pl.col(SECURITY_ID_COLUMNS[0]).cast(pl.Utf8)
                + "_"
                + pl.col(SECURITY_ID_COLUMNS[1]).cast(pl.Utf8)
            ).alias("id")
        )
        .select(["id", "month", "ret"])
    )
    stock_monthly = stock_monthly_trdate
    quarterly_monthly_rets_calendar = stock_monthly_trdate.with_columns(
        pl.col("month").dt.month_end().alias("month")
    )

    market_panel = monthly_panel.canada_monthly_panel(start, end)
    market_index = build_index(market_panel, method=CANADA_MARKET_INDEX_METHOD)
    market_monthly = market_index.rename({"date": "month", "index_ret": "ret"})

    betas_by_date_per_variant: dict[str, dict[datetime.date, pl.DataFrame]] = {
        variant: {} for variant in variants
    }
    fringe_by_date: dict[datetime.date, pl.DataFrame] = {}

    daily_cache: dict[int, pl.DataFrame] = {}

    def _daily_for_year(year: int) -> pl.DataFrame:
        frames = []
        for y in (year - 1, year):
            if y not in daily_cache:
                try:
                    daily_cache[y] = chass_loader.load_daily(years=[y])
                except FileNotFoundError:
                    daily_cache[y] = pl.DataFrame()
            if daily_cache[y].height > 0:
                frames.append(daily_cache[y])
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical")

    for formation_date in formation_dates:
        daily = _daily_for_year(formation_date.year)
        mkt_cap_df, _coverage = chass_universe.market_cap_at(monthly, daily, formation_date)
        if mkt_cap_df.height > 0:
            fringe_by_date[formation_date] = mkt_cap_df.select(
                (
                    pl.col(SECURITY_ID_COLUMNS[0]).cast(pl.Utf8)
                    + "_"
                    + pl.col(SECURITY_ID_COLUMNS[1]).cast(pl.Utf8)
                ).alias("id"),
                "mkt_cap",
                "price",
            )

        for variant in variants:
            betas = estimate_beta_monthly(
                stock_monthly, market_monthly, formation_date, variant=variant, id_col="id"
            )
            if betas.height > 0:
                betas_by_date_per_variant[variant][formation_date] = betas.select(["id", "beta"])

        for y in list(daily_cache.keys()):
            if y < formation_date.year - 1:
                del daily_cache[y]

    return (betas_by_date_per_variant, fringe_by_date, quarterly_monthly_rets_calendar)
