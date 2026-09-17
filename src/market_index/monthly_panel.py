"""Monthly-grain common-schema panels (id, date, mkt_cap, ret), one row
per name per calendar/CHASS month-end -- feeds directly into
build.build_index(), which is grain-agnostic (it only ever aggregates
within a `date` group; nothing in it assumes daily rows).

Built for docs/04_handoff_lowbeta_longonly.md Sec 4.2's monthly beta
variants (36m/60m rolling OLS): those need a monthly market index as the
regressor, and building one by compounding build_index_chunked()'s DAILY
output down to monthly would pay the exact daily-panel cost the handoff
says to avoid (build_index_chunked calls gate_adapters.us_gate1_panel /
canada_gate1_panel, both daily-grain, chunked specifically because a
multi-decade daily span OOMs -- see that module's own docstring).

Instead this reads directly from each leg's already-monthly sources:
US per-permno monthly returns (monthly_returns.monthly_arithmetic_
returns, leg="us") and CHASS's OWN native monthly return column
(no compounding needed at all on the Canada side -- CHASS ships
`return-Monthly Return` directly).

MARKET-CAP LAG IS ONE MONTH, NOT ONE TRADING DAY. This is a real defect
that was found and fixed here (leakage-auditor, this session): each
leg's market_cap_at(month_end) already lags by one TRADING DAY -- the
correct lag for a DAILY panel, where "prior trading day's close" is
"prior period's close." Pairing that with a MONTHLY return is wrong: on
a monthly grain, "prior period" means the PRIOR MONTH-END, not the prior
trading day within the same month. The prior-trading-day mkt_cap for
month M's row already embeds ~all of month M's own price return (e.g.
market_cap_at(2015-06-30) uses the 2015-06-29 close, which already
reflects essentially all of June's return) -- pairing that contemporaneous
weight with month M's own return is a same-period weight/return
correlation, not a leak of FUTURE information exactly, but a real bias:
it overweights whichever names had the biggest return in the very month
being weighted (measured on real 2015 data: US index cumulative return
inflated from +0.12% to +5.32% before this fix, all 12 months biased in
the same direction; Canada biased +0.29 to +1.34pp every month). The
same mechanism also drops delisted names' final month of return entirely
(membership resolved at month M's own close excludes a name that
delisted mid-month, even though it has a real, usually large negative,
return for month M) -- a survivorship bias stacking in the same
direction as the weight bias.

FIXED: both membership and mkt_cap are now resolved at the PRIOR
month-end (one full month back), then joined to the CURRENT month's own
return. This is the direct monthly analogue of what every daily-grain
consumer in this project already does correctly (gate_adapters.py's
daily panels shift by one full period, not by a fraction of one) --
CLAUDE.md's core rule, applied at the right grain.

Deliberately per-leg, not unified into one function: the two legs already
diverge structurally everywhere else in this codebase (different id
space, different month-end conventions, different data sources for
price vs. return) -- see gate_adapters.py's own module docstring for the
prior decision not to force a shared implementation, and the
project's own recorded lesson (feedback_adapter_over_full_unification)
that this has already been tried and rejected once.
"""

import datetime

import polars as pl

from src.data import chass_loader, chass_universe, universe_panel
from src.data.chass_loader import SECURITY_ID_COLUMNS
from src.data.monthly_returns import monthly_arithmetic_returns

_COMMON_SCHEMA = ["id", "date", "mkt_cap", "ret"]


def _us_month_ends(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    """Every calendar month-end from start's month through end's month
    (inclusive), NOT clipped to [start, end] itself -- matches
    gate_adapters._month_ends_in_range's convention of covering full
    months. universe_panel.market_cap_at() resolves each of these back
    to the real last trading day on or before it internally, so a
    calendar month-end that isn't a trading day (e.g. a weekend) still
    resolves correctly."""
    month_ends = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        if month == 12:
            next_first = datetime.date(year + 1, 1, 1)
        else:
            next_first = datetime.date(year, month + 1, 1)
        month_ends.append(next_first - datetime.timedelta(days=1))
        month = month + 1
        if month > 12:
            month = 1
            year += 1
    return month_ends


def _prior_month_end(d: datetime.date) -> datetime.date:
    """The calendar month-end immediately before d (d itself assumed a
    month-end)."""
    first_of_month = d.replace(day=1)
    return first_of_month - datetime.timedelta(days=1)


def us_monthly_panel(start: datetime.date, end: datetime.date) -> pl.DataFrame:
    """US monthly common-schema panel: id=str(permno), date=calendar
    month-end, mkt_cap=lagged ONE FULL MONTH (the PRIOR month-end's
    universe_panel.market_cap_at() snapshot -- see module docstring on
    why one trading day is the wrong lag at monthly grain), ret=that
    permno's monthly arithmetic return for the month ENDING at date (from
    monthly_returns.monthly_arithmetic_returns -- same convention used
    everywhere else in this project: a `month` value is the month whose
    return this row describes, not a forward-looking label).

    ONE call to monthly_arithmetic_returns() for the whole range (a range
    builder already), but market_cap_at() is a single-date snapshot
    function -- called once per PRIOR month-end needed (month_ends[0]'s
    own prior month through month_ends[-1]'s prior month), same pattern
    this session's universe census script used. mkt_cap at each prior
    month-end is then relabeled to the FOLLOWING month-end's date before
    joining to that month's own return -- this relabeling is the lag.
    """
    month_ends = _us_month_ends(start, end)
    if not month_ends:
        return pl.DataFrame(
            schema={"id": pl.Utf8, "date": pl.Date, "mkt_cap": pl.Float64, "ret": pl.Float64}
        )

    rets = monthly_arithmetic_returns(month_ends[0], month_ends[-1], leg="us")

    mkt_cap_frames = []
    for month_end in month_ends:
        weight_asof = _prior_month_end(month_end)
        mkt_cap_df, _coverage = universe_panel.market_cap_at(weight_asof)
        if mkt_cap_df.height == 0:
            continue
        # Relabel the PRIOR month-end's snapshot to THIS month_end's date
        # -- this relabeling IS the one-month lag: mkt_cap for the row
        # dated `month_end` is a snapshot taken at `month_end`'s own
        # PRIOR month-end, never at month_end itself.
        mkt_cap_frames.append(
            mkt_cap_df.select(["permno", "mkt_cap"]).with_columns(pl.lit(month_end).alias("date"))
        )

    if not mkt_cap_frames:
        return pl.DataFrame(
            schema={"id": pl.Utf8, "date": pl.Date, "mkt_cap": pl.Float64, "ret": pl.Float64}
        )

    mkt_cap = pl.concat(mkt_cap_frames, how="vertical")

    merged = mkt_cap.join(
        rets.select(["permno", "month", "ret"]),
        left_on=["permno", "date"],
        right_on=["permno", "month"],
        how="left",
    )

    return (
        merged.with_columns(pl.col("permno").cast(pl.Utf8).alias("id"))
        .select(_COMMON_SCHEMA)
        .sort(["date", "id"])
    )


def canada_monthly_panel(start: datetime.date, end: datetime.date) -> pl.DataFrame:
    """Canada monthly common-schema panel: id=f"{symbol}_{usage}",
    date=CHASS's own real trading month-end (NOT a naive calendar
    month-end -- chass_universe.universe_at()'s own docstring: CHASS
    trdate values ARE each month's actual last trading day already),
    mkt_cap=lagged ONE FULL CHASS MONTH (membership AND mkt_cap both
    resolved at the PRIOR CHASS trading month-end -- see module docstring
    on why a one-trading-day lag, correct at daily grain, is wrong at
    monthly grain), ret=that name's OWN `return-Monthly Return` value AT
    `date` itself (the CURRENT month, read directly from `monthly` -- not
    from the prior month's `eligible` frame, since a name can be a member
    at the prior month-end, delist mid-month, and still carry a real
    (usually large negative) return-Monthly Return for the CURRENT
    month -- CHASS ships this natively, no compounding from daily needed,
    the entire point of this module vs. build_index_chunked's daily
    path).

    daily is loaded per-calendar-year and cached across the loop (same
    pattern as this session's universe census script) -- market_cap_at's
    own PRICE join needs the trading day PRIOR TO the weight_asof date
    (a second, trading-day-level lag internal to market_cap_at itself,
    unchanged by this fix), which for an early-year weight_asof can
    reach back into the prior calendar year, so both `year` and
    `year - 1` are loaded whenever a new year is first needed.
    """
    monthly = chass_loader.load_monthly()
    all_trdates = sorted(monthly["trdate-Trade Date"].unique().to_list())
    all_trdate_list = [d.date() for d in all_trdates]
    month_ends = [d for d in all_trdate_list if start <= d <= end]

    if not month_ends:
        return pl.DataFrame(
            schema={"id": pl.Utf8, "date": pl.Date, "mkt_cap": pl.Float64, "ret": pl.Float64}
        )

    def _prior_trdate(d: datetime.date) -> datetime.date | None:
        earlier = [t for t in all_trdate_list if t < d]
        return max(earlier) if earlier else None

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

    panel_frames = []
    for month_end in month_ends:
        weight_asof = _prior_trdate(month_end)
        if weight_asof is None:
            continue
        eligible = chass_universe.universe_at(monthly, weight_asof)
        if eligible.height == 0:
            continue
        daily = _daily_for_year(weight_asof.year)
        mkt_cap_df, _coverage = chass_universe.market_cap_at(monthly, daily, weight_asof)
        if mkt_cap_df.height == 0:
            continue

        # ret comes from `monthly` AT month_end (the CURRENT month) --
        # NOT from `eligible` (which is the PRIOR month-end's membership
        # frame) -- see docstring: a name eligible at weight_asof may
        # delist before month_end and still carry a real return-Monthly
        # Return row for month_end itself.
        current_month_rets = monthly.filter(
            pl.col("trdate-Trade Date") == pl.lit(month_end).cast(pl.Datetime("ns"))
        ).select(SECURITY_ID_COLUMNS + ["return-Monthly Return"])

        merged = mkt_cap_df.join(current_month_rets, on=SECURITY_ID_COLUMNS, how="left")
        panel_frames.append(
            merged.with_columns(
                (
                    pl.col(SECURITY_ID_COLUMNS[0]).cast(pl.Utf8)
                    + "_"
                    + pl.col(SECURITY_ID_COLUMNS[1]).cast(pl.Utf8)
                ).alias("id"),
                pl.lit(month_end).alias("date"),
                pl.col("return-Monthly Return").alias("ret"),
            ).select(_COMMON_SCHEMA)
        )

        # Bound daily_cache memory the same way the census script does --
        # dates are processed in ascending order here too.
        for y in list(daily_cache.keys()):
            if y < month_end.year - 1:
                del daily_cache[y]

    if not panel_frames:
        return pl.DataFrame(
            schema={"id": pl.Utf8, "date": pl.Date, "mkt_cap": pl.Float64, "ret": pl.Float64}
        )

    return pl.concat(panel_frames, how="vertical").sort(["date", "id"])
