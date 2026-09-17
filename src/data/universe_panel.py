"""
Point-in-time US universe membership, CRSP-primary full-history panel.
Reads the partitioned parquet produced by pull_universe_us_crsp.py.

(An earlier Compustat-based single-month prototype, src/data/universe.py,
and its one-shot pull scripts were part of the research repo but are not
included in this submission -- they were superseded by this CRSP-primary
panel and are not on the report's code path.)

Membership base rule (usincflg='Y', securitytype/securitysubtype/
sharetype) is already applied at pull time (Task 1) -- this module only
adds the point-in-time listing/delisting bound, the major-exchange
filter, and REIT exclusion. MLP/partnership/trust exclusion needs no
separate logic here: sharetype='UG' names never entered the pulled panel
(Task 1's base filter requires sharetype='NS'), so there is nothing to
filter out at this layer.

REIT exclusion uses issuertype, NOT icbindustry -- confirmed this
session (docs/superpowers/specs/2026-09-03-us-full-history-panel-crsp-
primary-design.md, finding 5) that icbindustry is a sector/exposure
classification (catches non-REIT real-estate-adjacent companies like
CBRE/Zillow/Realogy, misses mortgage REITs), while issuertype is the
correct legal-structure signal.

Two implementation details diverge from the task brief's illustrative
reference code, found necessary while implementing against the real
pulled panel (both documented in the task report):

1. Memory: the full panel is ~74.5M rows across 62 year-partitions and
   ~19 columns (docstring of pull_universe_us_crsp.py: 6-11GB estimated
   in pandas). An eager `pl.read_parquet()` of the whole directory (all
   columns) reliably OOMs in this environment. This module uses
   `pl.scan_parquet()` with hive partitioning, projects only the columns
   this module needs, and prunes to the year(s) surrounding `month_end`
   before collecting -- polars' streaming engine handles the rest.

2. Trading-day resolution, not a raw spell-bounds filter: the underlying
   panel is one row per (permno, dlycaldt) trading day, not one row per
   listing spell. A `universe_at()` that filters the full daily panel by
   securitybegdt/securityenddt alone (the brief's literal reference
   implementation) would return one row per trading day within each
   eligible permno's spell -- not the permno-keyed single-row-per-name
   output this module promises, and reading enough of the panel to do
   that is also what causes the OOM in (1). Naively "fixing" this with a
   per-permno as-of/backward-fill join (take the most recent row on or
   before month_end) is ALSO wrong: it can resurrect a permno whose
   listing spell is nominally still open (securityenddt far in the
   future) but which stopped trading for an extended period before
   month_end -- confirmed directly in this pull (permno 69550, Mylan
   Inc: securityenddt=2020-11-16, a real later delisting, but no
   dlycaldt row between 2015-02-27 and month_end=2015-06-30). This is
   the same class of trap CLAUDE.md documents for Compustat
   (comp.secd having a row for a date doesn't mean the security traded
   -- gate on cshtrd>0); here the CRSP-side analog is that a stale
   as-of join must not be trusted as "still trading now" just because
   the spell metadata hasn't closed out.

   The correct, leakage-safe resolution: find the actual last trading
   day on or before month_end (from the panel's own distinct dlycaldt
   values, restricted to the pruned year window), then take the exact
   snapshot on that resolved date. This exactly reproduces the
   live-verified 2015-06-30 counts (4,014 base rows, 188 REITs
   excluded) and correctly handles a month_end that isn't itself a
   trading day (e.g. 2007-06-30, a Saturday, resolves to 2007-06-29).
"""

import datetime
from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "universe.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "us_panel_crsp_full"

# Columns universe_at() actually needs from the panel. Projecting down to
# this set (instead of reading all ~19 pulled columns) is a large part of
# what keeps the scan within memory -- see module docstring point 1.
_PANEL_COLUMNS = [
    "permno",
    "permco",
    "dlycaldt",
    "securitynm",
    "issuertype",
    "primaryexch",
    "securitybegdt",
    "securityenddt",
]

# Columns market_cap_at() needs from the panel -- separate projection list
# (same memory-safety rationale as _PANEL_COLUMNS above, see module
# docstring point 1).
_MKT_CAP_COLUMNS = ["permno", "dlycaldt", "dlyprc", "shrout"]

# Real populated-path dtypes for _PANEL_COLUMNS, confirmed directly against
# data/raw/us_panel_crsp_full/year=2015/part.parquet's schema. Used to build
# a correctly-typed empty frame when no trading day resolves at all (e.g.
# month_end far before the panel's 1965 start) -- an all-Utf8 empty frame
# would fail loudly (SchemaError) on any downstream typed join, which is the
# safe direction, but the real dtypes are strictly better: they let a caller
# handle "no data" as an ordinary empty-but-typed result instead.
_EMPTY_PANEL_SCHEMA: dict[str, pl.DataType] = {
    "permno": pl.Int64,
    "permco": pl.Int64,
    "dlycaldt": pl.Datetime("ns"),
    "securitynm": pl.Utf8,
    "issuertype": pl.Utf8,
    "primaryexch": pl.Utf8,
    "securitybegdt": pl.Datetime("ns"),
    "securityenddt": pl.Datetime("ns"),
}

# How many calendar days _resolve_last_trading_day() may walk backward from
# month_end before treating the result as suspiciously stale rather than a
# genuine weekend/holiday gap. Real trading calendars never have a gap this
# long (the longest is ~4 calendar days over a long weekend); a gap bigger
# than this means month_end is past the edge of real panel coverage (e.g. a
# WRDS data-lag year whose partition file exists but is still empty) and
# silently resolving backward into it would reproduce the exact stale-data
# hazard this module already rejects mid-panel for individual permnos (see
# module docstring point 2, the Mylan case) -- just at the panel's upper
# boundary instead.
_MAX_TRADING_DAY_STALENESS_DAYS = 7

# CORRECTED 2026-09-08 (Gate 2 Task 6 investigation): docs/01_data_notes.md
# section 15 originally concluded shrout needed no multiplier, based on
# dlycap / (abs(dlyprc) * shrout) ~= 1.0 for sampled rows. That check only
# proves dlycap and dlyprc*shrout are INTERNALLY self-consistent with each
# other -- both are CRSP-computed quantities, so it cannot rule out both
# being wrong by the same factor together. A real-world anchor check
# (Apple and ExxonMobil, both permno/date pairs on 2015-06-30, checked
# against their actual public market caps) proves the stored values are
# uniformly ~1000x too small: Apple priced at 5,705,400 "shares" implies a
# $715.6 MILLION market cap, but Apple's real cap that day was ~$715
# BILLION. shrout in this CRSP-primary panel (crsp.wrds_dsfv2_query) is in
# THOUSANDS of shares, matching legacy crsp.dsf.shrout's convention after
# all -- not raw shares as originally concluded. SHROUT_UNITS_MULTIPLIER
# (matching market_cap.py's existing constant of the same name/value) is
# applied below. See docs/01_data_notes.md section 15's correction and
# docs/worklog.md's dated entry for the full derivation.
SHROUT_UNITS_MULTIPLIER = 1000


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["us_crsp_primary"]


def _apply_pit_bounds(reference: pl.DataFrame, month_end) -> pl.DataFrame:
    """Keep only permnos alive at month_end: securitybegdt <= month_end
    and (securityenddt is null or securityenddt >= month_end)."""
    return reference.filter(
        (pl.col("securitybegdt") <= month_end)
        & (pl.col("securityenddt").is_null() | (pl.col("securityenddt") >= month_end))
    )


def _apply_exchange_filter(reference: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Restrict to major exchanges (config us_major_exchanges_crsp),
    excluding inactive ('X') and non-primary/secondary ('R') listings."""
    return reference.filter(pl.col("primaryexch").is_in(cfg["us_major_exchanges_crsp"]))


def _apply_reit_exclusion(reference: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Drop REITs by issuertype (config reit_issuertype) -- NOT
    icbindustry. See module docstring."""
    return reference.filter(pl.col("issuertype") != cfg["reit_issuertype"])


def _to_date(month_end) -> datetime.date:
    """Normalize month_end to a concrete datetime.date. Accepts either a
    plain datetime.date/datetime.datetime, or a polars Expr built from a
    bare pl.date(y, m, d) call (as this module's own tests and docstring
    example do) -- pl.date(...) with literal ints returns an unevaluated
    Expr, not a date, so it must be evaluated with no column context
    before any .year/comparison access is valid."""
    if isinstance(month_end, pl.Expr):
        month_end = pl.select(month_end).item()
    if isinstance(month_end, datetime.datetime):
        return month_end.date()
    return month_end


def _year_partition_files(year: int) -> list[str]:
    """Existing year-partition parquet paths for `year` and `year - 1`
    (the prior year is needed so a month_end near Jan 1 can still resolve
    backward to the prior year's last trading day)."""
    files = []
    for y in (year - 1, year):
        path = RAW_DATA_DIR / f"year={y}" / "part.parquet"
        if path.exists():
            files.append(str(path))
    return files


class StaleTradingDayError(ValueError):
    """Raised when the resolved last trading day is more than
    _MAX_TRADING_DAY_STALENESS_DAYS calendar days before month_end. This
    means month_end is past the edge of real panel coverage (e.g. a
    WRDS data-lag year whose partition file exists but has 0 rows) --
    resolving backward anyway would silently return stale data, the same
    hazard this module already rejects for individual stale permnos (see
    module docstring point 2)."""


def _resolve_last_trading_day(month_end: datetime.date, files: list[str]) -> datetime.date | None:
    """The actual last trading day on or before month_end, resolved from
    the panel's own dlycaldt values (not assumed to equal month_end
    itself -- month_end may fall on a weekend/holiday). Returns None if
    no trading day on or before month_end exists in `files`.

    Raises StaleTradingDayError if the resolved trading day is more than
    _MAX_TRADING_DAY_STALENESS_DAYS calendar days before month_end --
    a real weekend/holiday gap never gets this large, so a gap this size
    means month_end has walked past the edge of real panel coverage (e.g.
    a WRDS data-lag year) rather than landing on an ordinary non-trading
    day."""
    if not files:
        return None

    month_end_dt = pl.lit(month_end).cast(pl.Datetime("ns"))
    lf = pl.scan_parquet(files).select("dlycaldt").filter(pl.col("dlycaldt") <= month_end_dt)
    last = lf.select(pl.col("dlycaldt").max()).collect(engine="streaming")
    value = last.item()
    if value is None:
        return None
    resolved = value.date()
    staleness = (month_end - resolved).days
    if staleness > _MAX_TRADING_DAY_STALENESS_DAYS:
        raise StaleTradingDayError(
            f"month_end={month_end} resolved back to {resolved}, "
            f"{staleness} calendar days earlier -- exceeds the "
            f"{_MAX_TRADING_DAY_STALENESS_DAYS}-day staleness bound for a "
            f"genuine weekend/holiday gap. This likely means month_end is "
            f"beyond real panel coverage (e.g. a WRDS data-lag year whose "
            f"partition file exists but is still empty), not an ordinary "
            f"non-trading day. Refusing to silently return stale data."
        )
    return resolved


def universe_at(month_end) -> pl.DataFrame:
    """Point-in-time eligible US universe at month_end: listed, not yet
    delisted, on a major exchange, not a REIT. MLPs/partnerships/trusts
    are already excluded upstream (Task 1's pull filter). month_end: a
    polars-comparable date (e.g. pl.date(...) or datetime.date).

    Returns one row per permno (permno, securitynm, plus the panel
    columns this module reads). month_end need not itself be a trading
    day -- it resolves backward to the actual last trading day on or
    before month_end (see module docstring point 2).
    """
    cfg = _load_config()
    month_end_date = _to_date(month_end)
    files = _year_partition_files(month_end_date.year)

    trading_day = _resolve_last_trading_day(month_end_date, files)
    if trading_day is None:
        return pl.DataFrame(schema=_EMPTY_PANEL_SCHEMA)

    snapshot = _snapshot_on(trading_day, files)

    bounded = _apply_pit_bounds(snapshot, month_end=trading_day)
    on_exchange = _apply_exchange_filter(bounded, cfg)
    return _apply_reit_exclusion(on_exchange, cfg)


def _snapshot_on(trading_day: datetime.date, files: list[str]) -> pl.DataFrame:
    """The single-day snapshot of _PANEL_COLUMNS on exactly trading_day.
    Factored out of universe_at() so market_cap_at() can resolve the same
    trading day once (via _resolve_last_trading_day) and reuse it here,
    rather than universe_at() and market_cap_at() each independently
    deciding what date month_end resolves to (see Critical review
    finding: they must never disagree)."""
    trading_day_dt = pl.lit(trading_day).cast(pl.Datetime("ns"))
    return (
        pl.scan_parquet(files)
        .select(_PANEL_COLUMNS)
        .filter(pl.col("dlycaldt") == trading_day_dt)
        .collect(engine="streaming")
    )


def _mkt_cap_from_panel(panel: pl.DataFrame, month_end) -> pl.DataFrame:
    """Given a permno/dlycaldt/dlyprc/shrout panel, compute mkt_cap at
    month_end using the prior trading day's price x shares (lagged
    weights): the last dlycaldt strictly before month_end, resolved from
    the panel's own date index -- not assumed to be month_end minus one
    calendar day, and not a second WRDS query. dlyprc is abs()'d: like
    legacy crsp.dsf.prc, a negative value flags a bid/ask-midpoint
    stand-in for a no-trade day (see market_cap.py, same convention).

    Also returns `price` (the same abs(dlyprc) used in the mkt_cap
    product, on the same lagged prior-trading-day row) -- added for the
    long-only backtester's price > $1 fringe filter (docs/
    04_handoff_lowbeta_longonly.md Sec 4.1), which needs price
    independently of mkt_cap and must use the identical lagged snapshot
    rather than a second, differently-timed resolution.

    shrout here is in THOUSANDS of shares, matching legacy crsp.dsf.shrout's
    convention -- corrected 2026-09-08 via a real-world market-cap anchor
    check (see SHROUT_UNITS_MULTIPLIER's module-level comment above); the
    original "no multiplier" conclusion only checked internal consistency
    against dlycap, which cannot catch both fields being wrong together.

    A permno's resolved prior-trading-day row can have a null dlyprc
    and/or null shrout (a real, observed situation -- e.g. permno 14093
    in the 2015 panel has null dlyprc on 75 of its 252 rows that year,
    including its last row before 2015-01-02). abs(null) * shrout is
    null, and that null must never silently pass through as a mkt_cap
    value: unlike a permno with no prior-day row at all (already
    correctly excluded -- nothing for it to match in the join below),
    a null price/shares here means the row IS present but its market
    cap is unknown, and it must be excluded the same way -- matching
    market_cap.py's build_us_market_cap(), which does
    dropna(subset=["prc", "shrout"]) before computing mkt_cap so a null
    price never becomes a null market-cap value (CLAUDE.md: a wrong
    number that looks right is the worst possible outcome).

    dlycaldt is normalized to real datetime.date values before comparison:
    a caller building a synthetic frame with bare `pl.date(y, m, d)`
    literals inside a data dict (as this module's own tests do -- see
    universe_at()'s _apply_pit_bounds test docstring for the same trap)
    gets a column of unevaluated Expr objects (Object dtype), not real
    dates -- a plain pl.Date cast fails outright on Object dtype (it isn't
    even silently wrong), so each value is resolved via _to_date() first.
    Real panel data read via pl.scan_parquet already has a proper Datetime
    dtype, so this normalization is a no-op there (each value is already
    a datetime.datetime) and only matters for synthetic test frames.
    """
    month_end_date = _to_date(month_end)
    if panel.schema["dlycaldt"] == pl.Object:
        panel = panel.with_columns(
            pl.Series(
                "dlycaldt", [_to_date(v) for v in panel["dlycaldt"]], dtype=pl.Date
            )
        )
    prior_day_rows = (
        panel.filter(pl.col("dlycaldt") < month_end_date)
        .sort("dlycaldt")
        .group_by("permno", maintain_order=True)
        .last()
    )
    # Drop permnos whose resolved prior-day row has a null price or null
    # shares outstanding BEFORE computing mkt_cap -- matches market_cap.py's
    # dropna(subset=["prc", "shrout"]) pattern, adapted to polars. Doing
    # this after resolving each permno's most-recent row (rather than
    # filtering the whole panel first) means a permno with a null row on
    # its actual most-recent trading day is excluded even if an EARLIER
    # row for that permno happened to have a valid price -- the resolution
    # must reflect the most-recent day, not silently fall back further.
    valid_prior_day_rows = prior_day_rows.filter(
        pl.col("dlyprc").is_not_null() & pl.col("shrout").is_not_null()
    )
    return valid_prior_day_rows.with_columns(
        (pl.col("dlyprc").abs() * pl.col("shrout") * SHROUT_UNITS_MULTIPLIER).alias(
            "mkt_cap"
        ),
        pl.col("dlyprc").abs().alias("price"),
    ).select(["permno", "mkt_cap", "price"])


def market_cap_at(month_end) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Market cap (prior trading day's price x shares) for every permno
    eligible per universe_at(month_end).

    Returns (market_cap_df, coverage_log_df):
    - market_cap_df: permno, mkt_cap, price for permnos with a resolvable,
      non-null prior-trading-day price/shares row.
    - coverage_log_df: permno, matched (bool) for every permno
      universe_at(month_end) returned -- an eligible permno with no
      resolvable prior-day market cap (no prior trading day at all, or a
      null price/shares on its most-recent row) shows up here as
      matched=False instead of silently vanishing from the join. Mirrors
      market_cap.py's build_us_market_cap() coverage-log pattern
      (CLAUDE.md: an unmatched name must look obviously missing
      downstream, not quietly absent from a sum).

    The "prior trading day" is resolved relative to the SAME trading day
    universe_at(month_end) snapshots on (via _resolve_last_trading_day),
    not the raw month_end argument -- when month_end itself isn't a
    trading day (e.g. a Saturday), filtering the price panel on the raw
    month_end would let both functions land on the same resolved
    snapshot day, producing a same-day "lagged" price that isn't actually
    lagged at all. Resolving once and sharing that date is what prevents
    this (see module's Critical review finding on this exact bug).

    Follows universe_at()'s memory-safe pattern (module docstring point
    1): scans only the year partitions surrounding month_end, projected
    to the columns this function needs, rather than reading the full
    ~74.5M-row panel eagerly.
    """
    eligible = universe_at(month_end)
    month_end_date = _to_date(month_end)
    files = _year_partition_files(month_end_date.year)

    # Resolve the SAME trading day universe_at() snapshotted on -- not a
    # second, independent resolution of "what date is this" against the
    # raw month_end. See docstring above and the module's Critical review
    # finding: two independent resolutions of month_end is exactly what
    # let market_cap_at() silently use a same-day, unlagged price.
    snapshot_day = _resolve_last_trading_day(month_end_date, files)

    if snapshot_day is None or not files:
        market_cap = pl.DataFrame(
            schema={"permno": pl.Int64, "mkt_cap": pl.Float64, "price": pl.Float64}
        )
    else:
        snapshot_day_dt = pl.lit(snapshot_day).cast(pl.Datetime("ns"))
        panel = (
            pl.scan_parquet(files)
            .select(_MKT_CAP_COLUMNS)
            .filter(pl.col("dlycaldt") < snapshot_day_dt)
            .collect(engine="streaming")
        )
        market_cap = _mkt_cap_from_panel(panel, snapshot_day)

    market_cap_df = eligible.join(market_cap, on="permno", how="inner")

    coverage_log_df = eligible.select("permno").with_columns(
        pl.col("permno").is_in(market_cap["permno"].implode()).alias("matched")
    )
    return market_cap_df, coverage_log_df
