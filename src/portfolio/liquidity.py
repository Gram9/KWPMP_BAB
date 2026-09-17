"""
Liquidity filter, spec 00_spec.md section 3: "a filter on estimation-window
quality, not universe membership: minimum count of non-zero-volume days
within the vol window." Thresholds in config/liquidity.yaml (CLAUDE.md: no
magic numbers in src/).

Two independent rules, both real observed failure modes
(docs/01_data_notes.md section 6, docs/03_roadmap.md's E3 section):

- nonzero_volume_gate(): Lehman printed a frozen $0.55 with cshtrd=0 for 15
  straight days in 2014 -- a stale quote has zero realized variance,
  understating sigma_i and pushing a defunct shell into the LOW-beta long
  leg. Composes with src.estimation.beta_fp's existing sigma_min_obs gate
  as a SECOND, INDEPENDENT count over the SAME sigma window -- same
  two-gate inner-join shape _estimate_beta_fp_core already uses for its
  sigma/rho gates. Deliberately not a pre-filter that drops zero-volume
  rows before counting: that would silently change what sigma_i is
  measured over and move Gate 3's already-pinned numbers.
- sub_penny_gate(): Circuit City traded GENUINELY (nonzero volume) at
  $0.0021-$0.0035 during wind-down, where a one-tick move is a ~40%
  "return" -- inflating sigma_i and pushing the name into the HIGH-beta
  short leg. Applied to the LAGGED close at the formation date (never the
  same-day close -- CLAUDE.md's core rule), local currency. A name may
  exit and later re-enter if its price recovers.

US only, matching src.estimation.beta_fp's own leg dispatch: the Canadian
stock-side beta path (per-permno joins keyed on a non-permno identity) is
not yet built (F2b/F2c), so there is no Canadian daily-volume path for this
module to read either.
"""

import datetime
from pathlib import Path

import polars as pl
import yaml

from src.data import universe_panel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "liquidity.yaml"


def _load_liquidity_config() -> dict:
    """The liquidity thresholds (min_nonzero_volume_days, min_price) from
    config/liquidity.yaml. No variant key -- unlike beta_estimator.yaml,
    this file is one flat block (the liquidity rule doesn't vary by
    estimator), so this loader takes no argument."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _daily_price_volume(
    start_date: datetime.date, end_date: datetime.date, *, leg: str
) -> pl.DataFrame:
    """Every US permno's raw daily dlyprc/dlyvol for every trading day in
    [start_date, end_date] -- same unfiltered-by-universe_at() contract as
    beta_fp._daily_log_returns() (spec decision 1), and the same
    every-year-in-range file-gathering fix that function's own docstring
    documents (a 2-element {start.year, end.year} set silently drops any
    year strictly between the endpoints).

    Returns permno, date, dlyprc (signed -- CRSP's negative-price bid/ask
    proxy convention, NOT abs()'d here since the sub-penny rule cares about
    genuine trade price magnitude and callers apply abs() themselves,
    matching gate_adapters.us_gate1_panel()'s own convention), dlyvol.
    Rows with a null dlyvol are kept (not dropped) -- a null is not the
    same claim as a genuine zero, and the caller's gate must be able to
    tell "no trading" (volume == 0) from "not observed" (volume is null)
    rather than have that distinction erased before it ever sees the data.
    """
    if leg != "us":
        raise NotImplementedError(
            f"_daily_price_volume(leg={leg!r}) -- the liquidity filter is "
            "US-only today, matching src.estimation.beta_fp's own leg "
            "dispatch: the Canadian stock-side beta path this filter would "
            "gate does not exist yet (F2b/F2c)."
        )

    years_needed = sorted(set(range(start_date.year, end_date.year + 1)))
    files = []
    for year in years_needed:
        files.extend(universe_panel._year_partition_files(year))
    files = sorted(set(files))
    if not files:
        return pl.DataFrame(
            schema={"permno": pl.Int64, "date": pl.Date, "dlyprc": pl.Float64, "dlyvol": pl.Float64}
        )

    daily = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlyprc", "dlyvol"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(start_date).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(end_date).cast(pl.Datetime("ns")))
        )
        .collect(engine="streaming")
    )
    return daily.select(
        pl.col("permno"),
        pl.col("dlycaldt").cast(pl.Date).alias("date"),
        pl.col("dlyprc"),
        pl.col("dlyvol"),
    )


def nonzero_volume_counts(
    start_date: datetime.date, end_date: datetime.date, *, leg: str
) -> pl.DataFrame:
    """Per-permno count of TRADED days (dlyvol > 0) in [start_date,
    end_date] -- the callers' job is to intersect this against their own
    sigma-window date range and gate on
    config/liquidity.yaml's min_nonzero_volume_days. A dlyvol of exactly 0
    ("listed, genuinely didn't trade" -- the CLAUDE.md cshtrd>0 rule's own
    rationale) and a null dlyvol ("not observed") are both excluded from
    this count identically -- neither is a traded day -- but this function
    counts TRADED days directly rather than counting-then-subtracting, so
    it cannot silently conflate the two failure modes the way a
    total-minus-nonzero derivation could.

    Returns permno, n_traded_days -- one row per permno with at least one
    row in range (a permno with zero traded days in range still gets a
    row, n_traded_days=0, rather than silently vanishing from the count --
    CLAUDE.md: an excluded name must look obviously excluded, not quietly
    absent).
    """
    prices = _daily_price_volume(start_date, end_date, leg=leg)
    return (
        prices.group_by("permno")
        .agg((pl.col("dlyvol") > 0).sum().alias("n_traded_days"))
        .sort("permno")
    )


def formation_date_lagged_price(
    permnos: list[int], formation_date: datetime.date, *, leg: str
) -> pl.DataFrame:
    """The LAGGED close (prior trading day's dlyprc, abs()'d -- CRSP's
    negative-price bid/ask proxy convention) for each of `permnos` as of
    `formation_date`, matching universe_panel._mkt_cap_from_panel's and
    gate_adapters.us_gate1_panel's own lag convention (shift(1) over
    permno on the FULL unfiltered history, not a same-day read --
    CLAUDE.md's core rule: no t+1 information at t, and symmetrically no
    same-day close standing in for "prior day's close").

    "Exactly" matches market_cap_at()'s pricing date ONLY for permnos
    market_cap_at() itself returns -- both resolve independently per
    permno here, rather than sharing one panel-wide snapshot day, so
    they can diverge (always backward, never a leak) on a permno whose
    own last observation on or before formation_date precedes the
    panel-wide last trading day (e.g. a delisting-descriptor-blanked
    CRSP v2 row). This is unreachable in estimate_beta_fp()'s actual
    call sequence: market_cap_at() already drops null-price/shares rows
    (verified 2026-09-12: a stale permno with such a row, e.g. 81510 at
    2015-06-30, is simply ABSENT from market_cap_at()'s output, not
    present with a mismatched date), and betas are inner-joined against
    market_cap_at()'s output BEFORE the liquidity gate runs -- so every
    permno actually reaching this function already shares mkt_cap's
    resolved snapshot day. Gate-verifier finding, 2026-09-12.

    Reads a window ending at formation_date and starting far enough back
    (60 calendar days -- generous over any realistic holiday cluster) that
    every permno's most recent prior trading day resolves. Returns
    permno, lagged_price for permnos with a resolvable prior-day price;
    a permno with no prior trading day in the window (e.g. a very recent
    IPO) is absent from the result -- callers must treat absence as
    unresolvable, not silently pass the sub-penny gate for lack of data.

    formation_date need NOT itself be a trading day (gate-verifier
    finding, 2026-09-12): the ORIGINAL exact-match filter
    (`date == formation_date`) matched zero rows -- silently excluding
    EVERY permno, not just illiquid ones -- whenever formation_date fell
    on a weekend/holiday. 37 of 132 calendar month-ends in 2005-2015 land
    on a weekend alone. Resolved per-permno to the last trading day ON OR
    BEFORE formation_date (matching universe_panel._resolve_last_trading_
    day's "actual last trading day on or before" convention, the same
    fix already applied there for the identical reason), then that
    resolved day's OWN prior-day shift(1) value is returned -- so a
    weekend/holiday formation_date is transparent to the caller.
    """
    if leg != "us":
        raise NotImplementedError(
            f"formation_date_lagged_price(leg={leg!r}) -- US only today, "
            "same as the rest of this module."
        )
    if not permnos:
        return pl.DataFrame(schema={"permno": pl.Int64, "lagged_price": pl.Float64})

    read_start = formation_date - datetime.timedelta(days=60)
    prices = _daily_price_volume(read_start, formation_date, leg=leg)
    prices = prices.filter(pl.col("permno").is_in(permnos))

    prices = prices.sort(["permno", "date"]).with_columns(
        pl.col("dlyprc").shift(1).over("permno").alias("_lagged_price")
    )
    on_or_before = prices.filter(pl.col("date") <= formation_date)
    resolved = (
        on_or_before.group_by("permno")
        .agg(pl.col("date").max().alias("_resolved_date"))
    )
    on_formation = resolved.join(prices, left_on=["permno", "_resolved_date"], right_on=["permno", "date"])
    return (
        on_formation.filter(pl.col("_lagged_price").is_not_null())
        .select(
            pl.col("permno"),
            pl.col("_lagged_price").abs().alias("lagged_price"),
        )
    )


def liquid_permnos(
    candidate_permnos: list[int],
    sigma_window_start: datetime.date,
    lookback_end: datetime.date,
    formation_date: datetime.date,
    *,
    leg: str,
) -> pl.DataFrame:
    """The composed liquidity gate: a candidate permno survives only if it
    clears BOTH independent rules --

      1. n_traded_days (volume > 0) within [sigma_window_start,
         lookback_end] >= config/liquidity.yaml's min_nonzero_volume_days.
      2. formation_date's lagged close >= config/liquidity.yaml's
         min_price.

    Failing either excludes the permno entirely -- inner join, never a
    partial/flagged row, matching _estimate_beta_fp_core's own
    both-gates-required convention for sigma/rho.

    Returns permno, n_traded_days, lagged_price for permnos that pass
    BOTH gates -- the two diagnostic columns are kept (not dropped) so
    callers can log the required spec section 6 diagnostic (count and
    characteristics of names excluded) without a second pass over the
    same data.
    """
    cfg = _load_liquidity_config()

    volume_counts = nonzero_volume_counts(sigma_window_start, lookback_end, leg=leg)
    volume_counts = volume_counts.filter(pl.col("permno").is_in(candidate_permnos))
    volume_ok = volume_counts.filter(
        pl.col("n_traded_days") >= cfg["min_nonzero_volume_days"]
    )

    prices = formation_date_lagged_price(candidate_permnos, formation_date, leg=leg)
    price_ok = prices.filter(pl.col("lagged_price") >= cfg["min_price"])

    return volume_ok.join(price_ok, on="permno", how="inner").select(
        ["permno", "n_traded_days", "lagged_price"]
    )
