"""
Gate 2 (docs/02_validation_gates.md, design:
docs/superpowers/specs/2026-09-08-gate2-size-deciles-design.md): build
Fama-French-style value-weighted size decile portfolios from our own
CRSP-primary universe/market-cap data (US leg only -- Ken French's data
library has no Canadian equivalent) and correlate each decile's daily
returns against Ken French's own published series.

Methodology (confirmed with user during brainstorming, matches FF's
documented convention exactly):
- NYSE-only breakpoints: the 10th/20th/.../90th percentile of market
  cap among NYSE-listed names only (primaryexch == "N"), computed each
  June 30.
- Applied to the full major-exchange universe (NYSE+AMEX+NASDAQ, i.e.
  universe_panel.universe_at()'s own existing filter) for decile
  assignment.
- SAME-DAY market cap (June 30's own close x shares) for both
  breakpoints and assignment -- NOT the lagged value
  universe_panel.market_cap_at() returns (that function is correct for
  daily VW weighting, wrong for a formation-day snapshot; see the
  design spec's "Resolved: same-day vs. lagged market cap" section).
  Using June 30's own close for a June-30 decision is not a CLAUDE.md
  lookahead violation -- no t+1 information is used.
- Decile assignment locks at each June 30, held for the following
  July-June holding year; within that year, VW weights drift daily via
  LAGGED market cap (same mechanism as gate1_index.build_vw_index()).
"""

import datetime

import polars as pl

from src.data import ken_french_loader, universe_panel
from src.gates import gate1_index

_N_DECILES = 10
_FORMATION_MONTH = 6
_FORMATION_DAY = 30
_HOLDING_MONTHS_PER_YEAR = 12
_FIRST_HOLDING_MONTH = 7

# CORRECTED 2026-09-08: shrout in this panel is in THOUSANDS of shares, not
# raw shares -- see universe_panel.SHROUT_UNITS_MULTIPLIER's module-level
# comment for the full real-world-anchor derivation (Apple/ExxonMobil,
# 2015-06-30, both off by ~1000x under the original "no multiplier"
# assumption). Applied at read-time in both _same_day_mkt_cap_at() and
# build_decile_daily_panel() below.


def _same_day_mkt_cap_at(month_end) -> pl.DataFrame:
    """Market cap (that day's OWN price x shares, not lagged) for every
    permno eligible per universe_at(month_end). Deliberately separate
    from universe_panel.market_cap_at(), which is LAGGED by design (see
    module docstring) -- reusing it here would shift the entire
    breakpoint/assignment snapshot back one trading day, a real
    deviation from FF's documented methodology.

    Resolves the same trading day universe_at() snapshots on (via
    _resolve_last_trading_day), matching that function's own
    memory-safe, single-resolution pattern -- not a second independent
    date resolution.
    """
    month_end_date = universe_panel._to_date(month_end)
    files = universe_panel._year_partition_files(month_end_date.year)
    snapshot_day = universe_panel._resolve_last_trading_day(month_end_date, files)

    if snapshot_day is None or not files:
        return pl.DataFrame(schema={"permno": pl.Int64, "mkt_cap": pl.Float64})

    snapshot_day_dt = pl.lit(snapshot_day).cast(pl.Datetime("ns"))
    panel = (
        pl.scan_parquet(files)
        .select(universe_panel._MKT_CAP_COLUMNS)
        .filter(pl.col("dlycaldt") == snapshot_day_dt)
        .collect(engine="streaming")
    )
    panel = panel.filter(pl.col("dlyprc").is_not_null() & pl.col("shrout").is_not_null())
    return panel.with_columns(
        (
            pl.col("dlyprc").abs()
            * pl.col("shrout")
            * universe_panel.SHROUT_UNITS_MULTIPLIER
        ).alias("mkt_cap")
    ).select(["permno", "mkt_cap"])


def _nyse_company_caps_at(month_end) -> pl.DataFrame:
    """Company-level (permco) same-day market cap among NYSE-listed
    names, summing across every eligible permno (share class) that
    rolls up to the same permco -- matches Ken French's convention of
    aggregating multiple share classes to one company before computing
    NYSE breakpoints (see docs/03_roadmap.md Phase C: Berkshire A/B is
    the canonical example). Returns permco, mkt_cap."""
    eligible = universe_panel.universe_at(month_end)
    nyse_only = eligible.filter(pl.col("primaryexch") == "N")
    same_day_caps = _same_day_mkt_cap_at(month_end)

    nyse_caps = nyse_only.select(["permno", "permco"]).join(
        same_day_caps, on="permno", how="inner"
    )
    return nyse_caps.group_by("permco").agg(pl.col("mkt_cap").sum().alias("mkt_cap"))


def nyse_breakpoints_at(month_end) -> list[float]:
    """The 10th/20th/.../90th percentile of same-day market cap among
    NYSE-listed COMPANIES (permco, not permno) -- 9 cutpoints defining
    10 buckets, ascending order. Ken French sums share classes to
    company level before taking NYSE percentiles (docs/03_roadmap.md
    Phase C); computing this per-permno instead double-counts
    multi-share-class companies as if each class were its own firm."""
    nyse_caps = _nyse_company_caps_at(month_end)
    quantiles = [i / _N_DECILES for i in range(1, _N_DECILES)]
    return [
        nyse_caps.select(pl.col("mkt_cap").quantile(q, interpolation="linear")).item()
        for q in quantiles
    ]


def assign_deciles(month_end) -> pl.DataFrame:
    """Every permno eligible per universe_at(month_end) (full
    major-exchange universe), bucketed 1-10 against NYSE-only
    breakpoints from nyse_breakpoints_at(). A value exactly equal to a
    breakpoint falls in the LOWER decile (standard convention: bucket i
    is (breakpoint[i-1], breakpoint[i]], decile 1 is
    (-inf, breakpoint[0]]).

    Assignment is COMPANY-level (permco), matching nyse_breakpoints_at():
    every permno belonging to the same permco is bucketed by that
    company's SUMMED market cap, not its own individual security's --
    otherwise a multi-share-class company (e.g. Berkshire A/B) could
    have its classes split across different deciles from each other,
    which is not how Ken French's methodology (or this gate's benchmark)
    works. The daily return panel downstream stays per-permno (prices
    and returns are security-specific); only breakpoints and bucket
    assignment are company-level."""
    eligible = universe_panel.universe_at(month_end)
    same_day_caps = _same_day_mkt_cap_at(month_end)
    breakpoints = nyse_breakpoints_at(month_end)

    permco_caps = (
        eligible.select(["permno", "permco"])
        .join(same_day_caps, on="permno", how="inner")
        .group_by("permco")
        .agg(pl.col("mkt_cap").sum().alias("mkt_cap"))
    )
    caps = eligible.select(["permno", "permco"]).join(
        permco_caps, on="permco", how="inner"
    )

    # Built from the HIGHEST cutpoint down to the lowest: polars
    # when/otherwise chains evaluate the LAST `.when()` added first (each
    # new `.otherwise()` wraps the previous expression), so building from
    # i=len-1 down to 0 makes decile 1's condition (<= breakpoints[0])
    # the outermost/final check -- the one that wins for the smallest
    # market caps, correctly overriding the higher-decile default it
    # would otherwise inherit.
    decile_expr = pl.lit(_N_DECILES, dtype=pl.Int8)
    for i in reversed(range(len(breakpoints))):
        decile_expr = (
            pl.when(pl.col("mkt_cap") <= breakpoints[i])
            .then(pl.lit(i + 1, dtype=pl.Int8))
            .otherwise(decile_expr)
        )

    return caps.with_columns(decile_expr.alias("decile")).select(["permno", "decile"])


def _june_30_formation_dates(start_year: int, end_year: int) -> list[datetime.date]:
    """Every June 30 from start_year through end_year, inclusive --
    each is a formation date whose decile assignment holds for the
    following July-June year."""
    return [
        datetime.date(year, _FORMATION_MONTH, _FORMATION_DAY)
        for year in range(start_year, end_year + 1)
    ]


def build_decile_membership(start_year: int, end_year: int) -> pl.DataFrame:
    """One row per (permno, holding_year, holding_month) for every
    permno assigned a decile at each June 30 in [start_year, end_year],
    tagged with the holding month it covers (July of the formation year
    through June of the following year) -- same _holding_year/
    _holding_month tagging convention as
    src/data/gate_adapters.py::us_gate1_panel(), so the daily join in
    build_decile_daily_panel() can mirror that function's join logic
    directly."""
    formation_dates = _june_30_formation_dates(start_year, end_year)

    frames = []
    for formation_date in formation_dates:
        assigned = assign_deciles(formation_date)
        if assigned.height == 0:
            continue
        formation_year = formation_date.year
        for month_offset in range(_HOLDING_MONTHS_PER_YEAR):
            holding_month_num = _FIRST_HOLDING_MONTH + month_offset
            if holding_month_num > 12:
                holding_year = formation_year + 1
                holding_month = holding_month_num - 12
            else:
                holding_year = formation_year
                holding_month = holding_month_num
            frames.append(
                assigned.with_columns(
                    pl.lit(holding_year, dtype=pl.Int32).alias("_holding_year"),
                    pl.lit(holding_month, dtype=pl.Int8).alias("_holding_month"),
                )
            )

    if not frames:
        return pl.DataFrame(
            schema={
                "permno": pl.Int64,
                "decile": pl.Int8,
                "_holding_year": pl.Int32,
                "_holding_month": pl.Int8,
            }
        )
    return pl.concat(frames, how="vertical").unique(
        subset=["permno", "_holding_year", "_holding_month"]
    )


def build_decile_daily_panel(start_date: datetime.date, end_date: datetime.date) -> pl.DataFrame:
    """Daily CRSP rows joined against annual decile membership, common
    schema (id, date, price, shares, mkt_cap, ret) plus decile --
    mirrors src/data/gate_adapters.py::us_gate1_panel()'s exact join
    convention: mkt_cap lag computed over the FULL unfiltered per-permno
    history first, filtered to membership only afterward (never the
    reverse -- see us_gate1_panel's own docstring for why filtering
    first breaks the lag at holding-period boundaries)."""
    formation_years_needed = sorted({start_date.year - 1, start_date.year, end_date.year})
    membership = build_decile_membership(
        start_year=min(formation_years_needed), end_year=max(formation_years_needed)
    )
    empty_schema = {
        "id": pl.Utf8, "date": pl.Date, "price": pl.Float64,
        "shares": pl.Float64, "mkt_cap": pl.Float64, "ret": pl.Float64,
        "decile": pl.Int8,
    }
    if membership.height == 0:
        return pl.DataFrame(schema=empty_schema)

    years_needed = sorted({year for year in range(start_date.year - 1, end_date.year + 1)})
    files = []
    for year in years_needed:
        path = universe_panel.RAW_DATA_DIR / f"year={year}" / "part.parquet"
        if path.exists():
            files.append(str(path))
    if not files:
        return pl.DataFrame(schema=empty_schema)

    read_start = start_date - datetime.timedelta(days=10)
    daily = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlyprc", "shrout", "dlyret"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(read_start).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(end_date).cast(pl.Datetime("ns")))
        )
        .collect(engine="streaming")
    )

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
        pl.col("dlycaldt").dt.month().cast(pl.Int8).alias("_holding_month"),
    )
    daily = daily.join(
        membership, on=["permno", "_holding_year", "_holding_month"], how="inner"
    )

    daily = daily.filter(
        (pl.col("dlycaldt") >= pl.lit(start_date).cast(pl.Datetime("ns")))
        & (pl.col("dlycaldt") <= pl.lit(end_date).cast(pl.Datetime("ns")))
    )
    daily = daily.filter(pl.col("mkt_cap").is_not_null())

    return daily.select(
        pl.col("permno").cast(pl.Utf8).alias("id"),
        pl.col("dlycaldt").cast(pl.Date).alias("date"),
        pl.col("dlyprc").alias("price"),
        pl.col("shrout").cast(pl.Float64).alias("shares"),
        pl.col("mkt_cap"),
        pl.col("dlyret").alias("ret"),
        pl.col("decile"),
    )


def run_gate2_us(start_date: datetime.date, end_date: datetime.date) -> dict:
    """Runs Gate 2 for the US leg over [start_date, end_date]: builds
    the self-built per-decile daily VW index, compares against Ken
    French's own published daily VW decile returns, and reports the
    comparison per decile plus a breakpoint-computation diagnostic.

    Returns a dict with:
    - "comparison": date, decile, index_ret, ff_ret, diff -- an inner
      join per decile, so a date present in only one series for a given
      decile is silently absent here; check n_index_only_by_decile /
      n_benchmark_only_by_decile before trusting a decile's correlation.
    - "correlation_by_decile": {decile: pearson_r} for deciles 1-10.
    - "n_dates_by_decile", "n_index_only_by_decile",
      "n_benchmark_only_by_decile": {decile: count} -- same
      date-alignment-gap discipline as both Gate 1 legs.
    - "breakpoint_comparison": our computed NYSE breakpoints at the
      START DATE'S preceding June 30 vs. Ken French's own published
      breakpoints for the same month -- a second diagnostic isolating
      breakpoint computation from weighting/assignment if any decile's
      correlation comes back low.
    - "our_n_firms": count of NYSE-listed names actually used in the
      breakpoint computation (nyse_breakpoints_at()'s own denominator)
      for the formation date.
    - "ff_n_firms": Ken French's own published NYSE firm count for the
      same formation month (from load_nyse_breakpoints()'s "n_firms"
      column), or None if no matching row exists -- a firm-count
      mismatch here points at a universe-definition gap even when the
      breakpoint dollar values happen to look close.
    """
    panel = build_decile_daily_panel(start_date, end_date)
    our_index = gate1_index.build_vw_index(panel, extra_group_cols=["decile"])
    ff_returns = ken_french_loader.load_size_decile_daily_returns().rename({"ret": "ff_ret"})

    comparison = our_index.join(
        ff_returns, on=["date", "decile"], how="inner"
    ).with_columns((pl.col("index_ret") - pl.col("ff_ret")).alias("diff"))

    correlation_by_decile = {}
    n_dates_by_decile = {}
    n_index_only_by_decile = {}
    n_benchmark_only_by_decile = {}

    for decile in range(1, _N_DECILES + 1):
        our_dates = set(our_index.filter(pl.col("decile") == decile)["date"].to_list())
        ff_dates = set(ff_returns.filter(pl.col("decile") == decile)["date"].to_list())
        decile_comparison = comparison.filter(pl.col("decile") == decile)

        correlation_by_decile[decile] = decile_comparison.select(
            pl.corr("index_ret", "ff_ret")
        ).item()
        n_dates_by_decile[decile] = decile_comparison.height
        n_index_only_by_decile[decile] = len(our_dates - ff_dates)
        n_benchmark_only_by_decile[decile] = len(ff_dates - our_dates)

    formation_date = datetime.date(
        start_date.year if start_date.month > _FORMATION_MONTH else start_date.year - 1,
        _FORMATION_MONTH,
        _FORMATION_DAY,
    )
    our_breakpoints = nyse_breakpoints_at(formation_date)
    ff_breakpoints_row = ken_french_loader.load_nyse_breakpoints().filter(
        pl.col("month_end") == formation_date
    )
    percentile_cols = [f"p{i * 10}" for i in range(1, _N_DECILES)]
    ff_values = (
        [ff_breakpoints_row[col][0] for col in percentile_cols]
        if ff_breakpoints_row.height == 1
        else [None] * (_N_DECILES - 1)
    )
    breakpoint_comparison = pl.DataFrame(
        {
            "percentile": [i * 10 for i in range(1, _N_DECILES)],
            "our_breakpoint": our_breakpoints,
            "ff_breakpoint": ff_values,
        }
    )

    # our_n_firms recomputes the same NYSE-only, same-day-market-cap,
    # COMPANY-level (permco) aggregation nyse_breakpoints_at() uses
    # internally for its own quantile denominator (rather than changing
    # that function's return type just to expose a count already-simple
    # to recompute inline) -- counts distinct companies, matching Ken
    # French's own firm-count convention, not distinct share classes.
    our_n_firms = _nyse_company_caps_at(formation_date).height
    ff_n_firms = (
        ff_breakpoints_row["n_firms"][0] if ff_breakpoints_row.height == 1 else None
    )

    return {
        "comparison": comparison,
        "correlation_by_decile": correlation_by_decile,
        "n_dates_by_decile": n_dates_by_decile,
        "n_index_only_by_decile": n_index_only_by_decile,
        "n_benchmark_only_by_decile": n_benchmark_only_by_decile,
        "breakpoint_comparison": breakpoint_comparison,
        "our_n_firms": our_n_firms,
        "ff_n_firms": ff_n_firms,
    }
