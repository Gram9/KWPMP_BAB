"""
US market cap from CRSP price x shares, CCM-linked to the US universe.
Resolves spec open items #7/#11 -- see
docs/superpowers/specs/2026-09-02-us-crsp-market-cap-design.md and
docs/01_data_notes.md.

Uses the *prior trading day's* price, per spec's lagged-weight index-
construction rule (prior day's close x shares outstanding), matching the
raw pull's two-date fetch (pull script not included in this submission).

CRSP shrout is in thousands -- verified against several known large-cap
names (docs/01_data_notes.md section 13, "Multiplier spot-check").
crsp.dsf.prc is signed: a negative value flags a bid/ask-midpoint stand-in
for a no-trade day: use abs(prc) here; a real liquidity gate on trading
activity is separate, later work (Gate 2/3), not built here.

Any US universe gvkey with no valid CCM-linked CRSP row at as_of_date is
logged in the coverage output, never silently dropped -- an unmatched name
must look obviously missing downstream, not quietly absent from a sum
(CLAUDE.md: a wrong number that looks right is the worst possible outcome).
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

_CACHED_MONTH = pd.Timestamp("2015-06-30")

# CRSP shrout is reported in thousands of shares.
SHROUT_UNITS_MULTIPLIER = 1000


def filter_valid_ccm_links(links: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Keep only CCM links valid at as_of_date: linkdt <= as_of_date <=
    linkenddt (open-ended if linkenddt is null). A gvkey can have sequential
    PERMNO links across time -- never drop_duplicates() on gvkey here, the
    date bound is what selects the correct one.

    This mirrors the SQL-level bound already applied in
    the raw pull's query; kept here too as a standalone, directly
    testable function so the leakage property has a unit test independent
    of a live WRDS connection.

    No production code currently calls this function -- the raw pull applies
    the same date bound directly in SQL rather than routing through here. This function exists purely as a testable mirror of that SQL-level
    bound. The two are independent implementations of the same point-in-time
    rule and must be kept in sync manually if either changes; routing the
    pull through this function is future work, planned for once the pull
    extends beyond a single cached month to a full time series.
    """
    as_of_date = pd.Timestamp(as_of_date)
    valid_start = links["linkdt"] <= as_of_date
    valid_end = links["linkenddt"].isna() | (links["linkenddt"] >= as_of_date)
    return links.loc[valid_start & valid_end].copy()


def build_us_market_cap(as_of_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """US market cap (prior trading day's price x shares outstanding) for
    every gvkey in the cached US universe at as_of_date.

    Returns (market_cap_df, coverage_log_df):
    - market_cap_df: gvkey, permno, mkt_cap for gvkeys with a valid CCM-
      linked CRSP row. Rows with a null price on the prior trading day
      (no valid quote) are excluded here and surfaced as unmatched in
      coverage_log_df, rather than propagating a NaN market cap silently.
    - coverage_log_df: gvkey, matched (bool) for every gvkey in the US
      universe cache -- unmatched names are visible here, not dropped
      silently.
    """
    as_of_date = pd.Timestamp(as_of_date)
    if as_of_date.to_period("M").end_time.normalize() != _CACHED_MONTH:
        raise NotImplementedError(
            f"only the cached month {_CACHED_MONTH.date()} is available"
        )

    universe = pd.read_parquet(RAW_DATA_DIR / "us_universe_2015_06.parquet")
    raw = pd.read_parquet(RAW_DATA_DIR / "us_market_cap_2015_06.parquet")
    raw["date"] = pd.to_datetime(raw["date"])

    prior_day_rows = raw.loc[raw["date"] < as_of_date].copy()
    # A null prc (no valid quote on the prior trading day) must not become
    # a null mkt_cap that silently drops out of downstream sums -- exclude
    # it from market_cap_df and let it fall through to "unmatched" below.
    prior_day_rows = prior_day_rows.dropna(subset=["prc", "shrout"])
    prior_day_rows["mkt_cap"] = (
        prior_day_rows["prc"].abs() * prior_day_rows["shrout"] * SHROUT_UNITS_MULTIPLIER
    )
    market_cap_df = prior_day_rows[["gvkey", "permno", "mkt_cap"]].drop_duplicates(
        subset="gvkey"
    )

    all_gvkeys = universe[["gvkey"]].drop_duplicates()
    coverage_log_df = all_gvkeys.copy()
    coverage_log_df["matched"] = coverage_log_df["gvkey"].isin(market_cap_df["gvkey"])

    return market_cap_df, coverage_log_df
