"""
Output-normalizing adapters for Gate 1 (and later estimation/portfolio
code) -- see the Gate 1 implementation plan, Phase B
(C:\\Users\\spite\\.claude\\plans\\given-the-overall-goal-crystalline-hoare.md).

US CRSP-primary (universe_panel.py) and Canadian CHASS (chass_universe.py)
have genuinely different internal shapes -- different identity keys
(permno vs. (symbol-Ticker, usage-Usage Number)), different libraries
were true before the CHASS polars port, still-different structural
content (CHASS has no delisting-return field, no native trading-day
resolution the way CRSP's securitybegdt/securityenddt does). Rather than
force one shared universe_at()/market_cap_at() signature across both
(considered and rejected -- see docs/01_data_notes.md section 18's
sibling discussion and the Gate 1 plan's Context section), this module
adds a thin adapter PER COUNTRY that normalizes only the OUTPUT to one
common schema, leaving each country's loader/universe module untouched.

Common schema (both adapters):
    id: str        -- country-specific identity key as a string surrogate
                       (US: str(permno); Canada: f"{symbol}_{usage}").
                       Not a claim the two id spaces are comparable --
                       just a shared column name so downstream code
                       doesn't branch on country.
    date: datetime.date
    price: float    -- that date's own close/price (NOT lagged -- this is
                       the raw observation for that day).
    shares: float   -- shares outstanding as of that date's own listing.
    mkt_cap: float  -- LAGGED market cap: prior trading day's price x
                       shares, matching how universe_panel.py's and
                       chass_universe.py's own market_cap_at() compute it.
                       This is the portfolio-WEIGHT input; never derived
                       from the same day's own price (CLAUDE.md's core
                       rule -- no t+1 information at t, and symmetrically
                       here, no same-day price standing in for "prior
                       day's close").
    ret: float      -- that date's own total return (US: dlyret, NOT
                       dlyretx -- confirmed 2026-09-07 these genuinely
                       diverge on real ex-dividend dates, and dlyret
                       matches vwretd's total-return convention, the
                       Gate 1 comparison target; Canada: return-Daily
                       Return).

Design (date-range panel builder, not a single-date snapshot -- confirmed
with the user 2026-09-07): membership is resolved once per month via each
country's universe_at(), and a name eligible at month-end m is held
through month m+1 (the same formation-timing convention docs/00_spec.md
section 10 describes for the real BAB portfolio -- betas/membership from
month t-1, held through month t). Daily price/return/shares are then read
directly from the underlying daily panel for every trading day in that
membership window. This is the shape Gate 1's VW index construction
needs (sum weight*ret over every trading day in a range), and the shape
later estimation/portfolio code will need too for holding-period returns
-- building it once here avoids duplicating the monthly-membership +
daily-detail join elsewhere.
"""

import datetime

import polars as pl

from src.data import chass_loader, chass_universe, universe_panel

_COMMON_SCHEMA_ORDER = ["id", "date", "price", "shares", "mkt_cap", "ret"]

# CORRECTED 2026-09-08 (Gate 2 Task 6 investigation): shrout in the US
# CRSP-primary panel is in THOUSANDS of shares (matches
# universe_panel.SHROUT_UNITS_MULTIPLIER's own correction and its
# module-level comment for the full real-world-anchor derivation) -- not
# raw shares as this module originally assumed. Applied at read-time here
# so both the "shares" output column and every mkt_cap derived from it
# stay in real, consistent units.


def _month_ends_in_range(
    start_date: datetime.date, end_date: datetime.date
) -> list[datetime.date]:
    """Every calendar month-end from the month containing start_date
    through the month containing end_date, inclusive. A caller wanting
    the panel for [start_date, end_date] needs membership resolved for
    each of these -- a name eligible at month-end m is held through
    month m+1, so the month BEFORE start_date's month is also needed to
    cover start_date's own early days."""
    month_ends = []
    year, month = start_date.year, start_date.month
    # Include the prior month-end too: start_date's own month is "held"
    # based on the PRIOR month's membership snapshot (formation timing,
    # spec section 10), so a caller starting mid-month still needs that
    # prior snapshot resolved.
    prior_month = month - 1 if month > 1 else 12
    prior_year = year if month > 1 else year - 1
    year, month = prior_year, prior_month

    while (year, month) <= (end_date.year, end_date.month):
        if month == 12:
            next_day = datetime.date(year + 1, 1, 1)
        else:
            next_day = datetime.date(year, month + 1, 1)
        month_ends.append(next_day - datetime.timedelta(days=1))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return month_ends


def _us_gate1_membership(start_date: datetime.date, end_date: datetime.date) -> pl.DataFrame:
    """Every permno's holding-month membership for [start_date, end_date]
    -- factored out of us_gate1_panel so _us_gate1_panel_chunk (one
    calendar year) can resolve membership against the SAME month-end
    convention a direct un-chunked call would use, without re-deriving
    the month-end/holding-month logic per chunk."""
    month_ends = _month_ends_in_range(start_date, end_date)

    # One universe_at() call per month-end covering the range -- reuses
    # the existing, already-tested membership resolution rather than
    # re-deriving point-in-time logic here.
    membership_frames = []
    for month_end in month_ends:
        eligible = universe_panel.universe_at(month_end)
        if eligible.height == 0:
            continue
        # A permno eligible at month_end m is held for the NEXT calendar
        # month (m's trading days come after m, through the next
        # month-end) -- tag each membership row with the holding month
        # (year, month) it covers.
        if month_end.month == 12:
            holding_year, holding_month = month_end.year + 1, 1
        else:
            holding_year, holding_month = month_end.year, month_end.month + 1
        membership_frames.append(
            eligible.select("permno").with_columns(
                pl.lit(holding_year, dtype=pl.Int32).alias("_holding_year"),
                pl.lit(holding_month, dtype=pl.Int8).alias("_holding_month"),
            )
        )

    if not membership_frames:
        return pl.DataFrame(schema={"permno": pl.Int64, "_holding_year": pl.Int32, "_holding_month": pl.Int8})

    return pl.concat(membership_frames, how="vertical").unique()


# Calendar-day carry applied at EVERY chunk boundary, matching the ORIGINAL
# un-chunked function's own convention at its single start_date boundary
# (a long weekend or holiday cluster is never more than a few calendar
# days) -- not a 1-day carry, which would be too short across a holiday
# cluster and would silently drop a real prior-trading-day price, handing
# one name a null mkt_cap on the first trading day of a year. Same named
# constant both call sites (a bare "10" reappearing at two sites would
# invite exactly the drift a shared constant prevents).
_LAG_CARRY_DAYS = 10


def _us_gate1_panel_chunk(
    membership: pl.DataFrame,
    chunk_start: datetime.date,
    chunk_end: datetime.date,
    files: list[str],
) -> pl.DataFrame:
    """One calendar-bounded slice of us_gate1_panel's daily join/shift
    logic -- identical math to the original un-chunked function, applied
    to [chunk_start, chunk_end] instead of the full requested range, so
    that memory scales with ONE chunk's row count rather than the whole
    span's (measured: the 56-year 1970-2025 span's un-chunked .collect()
    + .shift().over("permno") crashed with a Rust memory allocation
    failure on a 72.9M-row frame; a single year is ~1.3M rows).

    `files` is the CALLER's file list (spanning the chunk's own year plus
    whatever is needed for _LAG_CARRY_DAYS lookback) -- passed in rather
    than re-resolved per chunk so a caller iterating many chunks against
    the same underlying files does the file-existence check once.
    """
    read_start = chunk_start - datetime.timedelta(days=_LAG_CARRY_DAYS)
    daily = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlyprc", "shrout", "dlyret"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(read_start).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(chunk_end).cast(pl.Datetime("ns")))
        )
        .collect(engine="streaming")
    )

    # Compute the lag over the FULL unfiltered per-permno history in
    # THIS chunk's widened window first, matching
    # universe_panel._mkt_cap_from_panel's own convention (lag, then
    # filter to eligible names -- not the reverse). Filtering to
    # membership BEFORE shifting would drop the prior trading day's row
    # whenever it falls in a different holding-month than the target day
    # (e.g. the last day of June feeding July 1's lag) -- that prior row
    # is a real trading day and a real price, just not itself a
    # "held" day, so it must stay available for the shift even though it
    # won't survive the final membership filter itself.
    daily = daily.sort(["permno", "dlycaldt"])
    daily = daily.with_columns(
        (pl.col("shrout") * universe_panel.SHROUT_UNITS_MULTIPLIER).alias("shrout")
    )
    daily = daily.with_columns(
        (pl.col("dlyprc").abs() * pl.col("shrout")).alias("_same_day_mkt_cap")
    )
    daily = daily.with_columns(
        pl.col("_same_day_mkt_cap").shift(1).over("permno").alias("mkt_cap")
    )

    daily = daily.with_columns(
        pl.col("dlycaldt").dt.year().alias("_holding_year"),
        pl.col("dlycaldt").dt.month().alias("_holding_month"),
    )
    daily = daily.join(
        membership, on=["permno", "_holding_year", "_holding_month"], how="inner"
    )

    daily = daily.filter(
        (pl.col("dlycaldt") >= pl.lit(chunk_start).cast(pl.Datetime("ns")))
        & (pl.col("dlycaldt") <= pl.lit(chunk_end).cast(pl.Datetime("ns")))
    )
    daily = daily.filter(pl.col("mkt_cap").is_not_null())

    result = daily.select(
        pl.col("permno").cast(pl.Utf8).alias("id"),
        pl.col("dlycaldt").cast(pl.Date).alias("date"),
        pl.col("dlyprc").alias("price"),
        pl.col("shrout").cast(pl.Float64).alias("shares"),
        pl.col("mkt_cap"),
        pl.col("dlyret").alias("ret"),
    )
    return result.select(_COMMON_SCHEMA_ORDER)


def us_gate1_panel(start_date: datetime.date, end_date: datetime.date) -> pl.DataFrame:
    """US CRSP-primary panel normalized to the common Gate 1 schema, for
    every trading day in [start_date, end_date].

    Membership: for each trading day, the eligible permno set is
    universe_panel.universe_at() resolved at the PRECEDING month-end
    (formation timing -- a permno eligible at month-end m is held
    through month m+1's trading days). mkt_cap is each permno's own
    LAGGED price x shares (prior trading day, resolved from the panel's
    own dates -- reuses universe_panel._mkt_cap_from_panel's convention
    directly rather than re-deriving it).

    PROCESSED IN YEARLY CHUNKS internally (2026-09-13, Gate 4): every
    prior caller of this function (Gate 1, Gate 2) only ever requested a
    single calendar year, so the original implementation collected the
    ENTIRE requested range eagerly and ran .shift(1).over("permno")
    across all of it in one pass -- fine at 1-year scale, but Gate 4's
    56-year US sample (1970-2025, ~72.9M daily rows) crashed with a Rust
    memory allocation failure on this exact operation, confirmed by
    isolating this function alone from every other piece of the Gate 4
    pipeline. Each chunk re-applies the SAME _LAG_CARRY_DAYS=10-calendar-
    day widen-then-filter trick the original code already used at its
    single start_date boundary -- now applied at every year boundary, not
    only the first -- so the lag computation still sees a genuine prior
    trading day across a year-end weekend/holiday cluster, never a null
    manufactured by a chunk boundary landing too close to it.

    Bit-identical to the pre-chunking implementation over the same
    [start_date, end_date] -- see
    tests/unit/test_gate_adapters.py::test_us_gate1_panel_chunked_matches_reference_implementation,
    which pins this against an inline reproduction of the original
    single-pass logic over a real multi-year span (the boundary carry is
    the only thing chunking could break; a single-year test cannot
    exercise it, since there is only one chunk).
    """
    membership = _us_gate1_membership(start_date, end_date)
    if membership.height == 0:
        return pl.DataFrame(schema=_empty_schema())

    month_ends = _month_ends_in_range(start_date, end_date)
    years_needed = sorted({d.year for d in month_ends} | {start_date.year, end_date.year})
    files = []
    for year in years_needed:
        path = universe_panel.RAW_DATA_DIR / f"year={year}" / "part.parquet"
        if path.exists():
            files.append(str(path))
    if not files:
        return pl.DataFrame(schema=_empty_schema())

    chunk_results = []
    for year in range(start_date.year, end_date.year + 1):
        chunk_start = max(start_date, datetime.date(year, 1, 1))
        chunk_end = min(end_date, datetime.date(year, 12, 31))
        if chunk_start > chunk_end:
            continue
        # This chunk's own file plus the PRIOR year's (never more than
        # _LAG_CARRY_DAYS calendar days are ever needed for the lag
        # carry, but the prior year's partition file is what actually
        # holds those late-December rows).
        chunk_files = [
            f for f in files
            if f"year={year}" in f or f"year={year - 1}" in f
        ]
        if not chunk_files:
            continue
        chunk_results.append(
            _us_gate1_panel_chunk(membership, chunk_start, chunk_end, chunk_files)
        )

    if not chunk_results:
        return pl.DataFrame(schema=_empty_schema())

    return pl.concat(chunk_results, how="vertical").sort(["id", "date"])


def canada_gate1_panel(start_date: datetime.date, end_date: datetime.date) -> pl.DataFrame:
    """Canadian CHASS panel normalized to the common Gate 1 schema, for
    every trading day in [start_date, end_date].

    Structurally different from us_gate1_panel(), not just relabeled --
    see module docstring's Context on why these two stay separate rather
    than unified:

    - CHASS's own real month-end trading dates (e.g. 2015-05-29, not the
      calendar 2015-05-31) must be read from the monthly file itself --
      unlike universe_panel.universe_at(), chass_universe.universe_at()
      does no month-end resolution of its own (confirmed in that
      function's docstring), so this adapter must supply real dates.
    - shares_out lives in the MONTHLY file, price/return in the DAILY
      file (chass_universe.market_cap_at()'s own documented daily<->
      monthly join, extended here to every day instead of just each
      month's own snapshot day). shares_out is held constant across a
      holding month at that month's own value (matching
      chass_universe.market_cap_at()'s convention exactly) while price
      and return vary daily.
    - Identity key is (symbol-Ticker, usage-Usage Number), surrogated to
      a single string id as f"{symbol}_{usage}".
    """
    monthly = chass_loader.load_monthly()
    # Every calendar year in range, not just the two endpoints -- the
    # SAME bug class beta_fp._daily_log_returns's own docstring documents
    # and fixes (a bare {start.year, end.year} silently drops any year
    # strictly in between, since load_daily(years=...) loads only the
    # named years' partition files). Found 2026-09-13 while validating
    # Gate 4's chunked market-index builder: a direct 1995-2000 call
    # returned only 252 unique dates (one year's worth) instead of
    # ~1260 (five years) -- every prior test of this function used a
    # single-month span, so {start.year, end.year} was always a
    # single-element set and this was invisible until now. See
    # tests/unit/test_gate_adapters.py::test_canada_panel_covers_every_year_in_multi_year_range.
    years_needed = sorted(set(range(start_date.year, end_date.year + 1)))
    daily = chass_loader.load_daily(years=years_needed)

    # Real CHASS trading month-ends within the window, plus one prior
    # month-end so start_date's own early days can resolve a holding
    # month from the month before it (same formation-timing reasoning as
    # us_gate1_panel's _month_ends_in_range).
    all_month_ends = (
        monthly.select("trdate-Trade Date")
        .unique()
        .sort("trdate-Trade Date")["trdate-Trade Date"]
        .to_list()
    )
    all_month_ends = [d.date() for d in all_month_ends]
    window_month_ends = [d for d in all_month_ends if d <= end_date]
    # Keep every month-end from the last one before start_date's month
    # through end_date -- mirrors us_gate1_panel's "prior month-end
    # included" logic, but against CHASS's own real dates rather than
    # naive calendar arithmetic.
    prior_candidates = [d for d in window_month_ends if d < start_date]
    month_ends = (prior_candidates[-1:] if prior_candidates else []) + [
        d for d in window_month_ends if d >= start_date
    ]
    if not month_ends:
        return pl.DataFrame(schema=_empty_schema())

    membership_frames = []
    shares_frames = []
    for i, month_end in enumerate(month_ends):
        eligible = chass_universe.universe_at(monthly, month_end)
        if eligible.height == 0:
            continue
        # Holding month = the calendar month immediately after month_end
        # (same convention as us_gate1_panel: a name eligible at
        # month-end m is held through month m+1's trading days).
        if month_end.month == 12:
            holding_year, holding_month = month_end.year + 1, 1
        else:
            holding_year, holding_month = month_end.year, month_end.month + 1
        membership_frames.append(
            eligible.select(chass_loader.SECURITY_ID_COLUMNS).with_columns(
                pl.lit(holding_year, dtype=pl.Int32).alias("_holding_year"),
                pl.lit(holding_month, dtype=pl.Int8).alias("_holding_month"),
            )
        )
        # shares_out for the holding month is THIS month_end's own value
        # (chass_universe.market_cap_at()'s documented convention:
        # June's shares_out pairs with June's own snapshot, used for
        # June's holding period).
        shares_frames.append(
            eligible.select(
                chass_loader.SECURITY_ID_COLUMNS
                + ["shares_out-Monthly Shares outstanding (100s of shares)"]
            ).with_columns(
                pl.lit(holding_year, dtype=pl.Int32).alias("_holding_year"),
                pl.lit(holding_month, dtype=pl.Int8).alias("_holding_month"),
            )
        )

    if not membership_frames:
        return pl.DataFrame(schema=_empty_schema())

    membership = pl.concat(membership_frames, how="vertical").unique()
    shares_by_holding_month = pl.concat(shares_frames, how="vertical").unique(
        subset=chass_loader.SECURITY_ID_COLUMNS + ["_holding_year", "_holding_month"]
    )

    # Widen the read window before start_date so the first in-range
    # trading day can resolve a genuine prior-trading-day close for its
    # lagged mkt_cap -- same reasoning as us_gate1_panel's read_start.
    read_start = start_date - datetime.timedelta(days=10)
    daily = daily.filter(
        (pl.col("trdate-Trade Date") >= pl.lit(read_start).cast(pl.Datetime("ns")))
        & (pl.col("trdate-Trade Date") <= pl.lit(end_date).cast(pl.Datetime("ns")))
    )

    # Lag computed over the FULL unfiltered per-name daily history first
    # (same reasoning as us_gate1_panel: the prior trading day may fall
    # in a different holding-month than the target day, e.g. the last
    # day of a holding month feeding the next month's first day's lag,
    # so it must survive to be shiftable even though it won't itself
    # pass the eventual membership filter).
    daily = daily.sort(chass_loader.SECURITY_ID_COLUMNS + ["trdate-Trade Date"])
    daily = daily.with_columns(
        pl.col("closeprice-Daily Closing price")
        .shift(1)
        .over(chass_loader.SECURITY_ID_COLUMNS)
        .alias("_prior_close")
    )

    daily = daily.with_columns(
        pl.col("trdate-Trade Date").dt.year().alias("_holding_year"),
        pl.col("trdate-Trade Date").dt.month().alias("_holding_month"),
    )
    daily = daily.join(
        membership,
        on=chass_loader.SECURITY_ID_COLUMNS + ["_holding_year", "_holding_month"],
        how="inner",
    )
    daily = daily.join(
        shares_by_holding_month,
        on=chass_loader.SECURITY_ID_COLUMNS + ["_holding_year", "_holding_month"],
        how="left",
    )

    daily = daily.with_columns(
        (
            pl.col("_prior_close").abs()
            * pl.col("shares_out-Monthly Shares outstanding (100s of shares)")
            * chass_universe._SHARES_OUT_UNITS_MULTIPLIER
        ).alias("mkt_cap")
    )

    daily = daily.filter(
        (pl.col("trdate-Trade Date") >= pl.lit(start_date).cast(pl.Datetime("ns")))
        & (pl.col("trdate-Trade Date") <= pl.lit(end_date).cast(pl.Datetime("ns")))
    )
    daily = daily.filter(pl.col("mkt_cap").is_not_null())

    result = daily.select(
        (pl.col("symbol-Ticker") + pl.lit("_") + pl.col("usage-Usage Number").cast(pl.Utf8)).alias(
            "id"
        ),
        pl.col("trdate-Trade Date").cast(pl.Date).alias("date"),
        pl.col("closeprice-Daily Closing price").alias("price"),
        (
            pl.col("shares_out-Monthly Shares outstanding (100s of shares)")
            * chass_universe._SHARES_OUT_UNITS_MULTIPLIER
        ).alias("shares"),
        pl.col("mkt_cap"),
        pl.col("return-Daily Return").alias("ret"),
    )
    return result.select(_COMMON_SCHEMA_ORDER)


def _empty_schema() -> pl.Schema:
    return pl.Schema(
        {
            "id": pl.Utf8,
            "date": pl.Date,
            "price": pl.Float64,
            "shares": pl.Float64,
            "mkt_cap": pl.Float64,
            "ret": pl.Float64,
        }
    )
