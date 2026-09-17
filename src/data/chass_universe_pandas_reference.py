"""
Point-in-time Canadian universe membership, built on the CHASS monthly
panel loaded by src/data/chass_loader.py's load_monthly(). Parallel in
role to src/data/universe_panel.py (US CRSP-primary) -- membership at a
given month_end, domestic-only, fund/REIT/MLP-excluded.

CHASS provides no direct point-in-time membership field the way CRSP's
securitybegdt/securityenddt do. The authoritative source would be the
CHASS Ticker History table (DICTION.DAT: an explicit Delisting Date and
one-or-more Listing Dates per (symbol-Ticker, usage-Usage Number)) --
but that table is not accessible via this project's CHASS browser
install (the GUI does not expose a Ticker History / Dictionary option),
an access problem not resolved this session. universe_at() below is
therefore an INFERRED PROXY, empirically validated against real data
this session, not an authoritative field. It cannot perfectly
distinguish "genuinely delisted" from "our converted parquet's coverage
just ends" at the dataset's current edge (2025).

Row-presence membership design, validated this session:

1. PRIMARY SIGNAL: a security is a member in month t iff it has a
   monthly row at t. CHASS does not emit placeholder rows once a
   security is gone -- rows stop outright, they don't continue as
   zero-value rows the way daily no-trade days do (see
   chass_daily_trade_status's NO_TRADE case, which is a DIFFERENT thing:
   a listed security not trading on a given day still gets a daily row;
   a delisted security gets no monthly row at all, ever again).

   Verified: Pembina Pipeline Corporation (PPL, usage 2) -- confirmed
   real and currently listed -- has 339 monthly rows spanning
   1997-10-31 to 2025-12-31 with a maximum gap of 33 days between
   consecutive rows (one ordinary calendar month, no coverage gaps).
   Morgan Hydrocarbons Inc. (MHI, usage 0) has monthly rows through
   1996-10-31 and NONE after, in any later year -- matching its real
   1996 acquisition by Talisman Energy.

   REJECTED ALTERNATIVE: "last day price/return goes to zero or NaN" as
   a delisting proxy. Tested directly against real daily data this
   session and found unreliable in both directions:
   - False positive: American Express (AXP), a Foreign Firm per
     foreign_flag, shows 15 consecutive zero-price days on the TSX in
     late 1993 with zero relation to any real corporate delisting --
     just light/no Canadian trading for a US-primary-listed name.
   - False negative: many domestic names show the identical interleaved
     zero-price/real-trade pattern (e.g. NEW CACHE PETROLEUMS, LIONHEART
     ENERGY CORP) purely from ordinary illiquidity, with no delisting at
     all -- trading resumes normally.
   Domestic-only filtering (see chass_domestic_only in
   config/universe.yaml) removes the AXP-style false positive but does
   NOT fix the underlying problem: it cannot distinguish "real trading
   resumed later, outside a given sample window" (a censoring effect)
   from "genuinely stopped forever," nor "prolonged illiquidity" from
   "delisting." Row presence at MONTHLY grain sidesteps this: a merely
   illiquid security still gets a monthly row (its Total Volume/
   Transactions may be small, but the row exists), while a truly
   delisted one produces no row at all.

2. CORROBORATION (not a membership trigger, a confidence signal):
   `_terminal_row_flags()` checks whether a security's very last-ever
   monthly row has shares_out AND monthly return both NaN. Live-
   measured this session: true for 71.0% of names whose last row falls
   before 2015 (2,794 of 3,935) on shares_out, 93.4% (3,677 of 3,935) on
   return -- versus a 0/30 false-positive rate in a sample of names
   still trading as of 2025 (no mid-history NaN ever observed on that
   field for a name that keeps trading). Confirmed directly on Morgan
   Hydrocarbons: its terminal row (1996-10-31) has shares_out=NaN,
   return=NaN, while total_volume/transactions that same month are real
   nonzero values (51,545 / 472) -- consistent with a company mid-
   acquisition still trading but no longer having a well-defined share
   count.

3. RENAME SAFETY NET: `_check_cusip_continuity()` -- before treating a
   stopped (ticker, usage) as a genuine exit, check whether its CUSIP
   reappears under a DIFFERENT (ticker, usage) starting within ~60
   days. This guards against the real risk that row-presence membership
   otherwise can't see: a security that merely changed its ticker
   symbol (not delisted at all) would show its OLD ticker's rows
   stopping, since CHASS's Summary Information tables key trading data
   by whatever (ticker, usage) was current at the time -- ticker
   history reconstruction is DICTION.DAT's job, which this project
   cannot access (see module docstring intro).

   Tested this session against the 179 names whose terminal row lacked
   the NaN corroboration above (the group most likely to hide a rename,
   since a rename wouldn't produce reclassification-style NaN fields) --
   zero CUSIP-continuity matches found. On inspection, that group turned
   out to be genuine M&A delistings with clean trading right to the end
   (Bell Aliant -> BCE 2014, Bema Gold -> Goldcorp 2007, Burlington
   Resources -> ConocoPhillips 2005), not renames: an acquired company's
   CUSIP does not resurface because it's absorbed into a different legal
   entity with a different CUSIP. This suggests true CHASS
   ticker-renames-without-delisting may be rare in practice, but the
   check is retained as a cheap safety net (it only reads data already
   loaded) rather than assumed away.

NOT handled by this module: the CHASS delisting-return gap. A
security's return series simply stops at its last real trade -- there
is no field capturing the actual merger/liquidation consideration the
way CRSP's dlret does. See docs/01_data_notes.md section 6 ("CHASS --
no delisting return at all") for the full derivation; this is a
documented, accepted limitation, not something universe_at() attempts
to correct.
"""

import datetime

import pandas as pd
import yaml

from src.data.chass_loader_pandas_reference import (
    PROJECT_ROOT,
    SECURITY_ID_COLUMNS,
    apply_chass_fund_mlp_reit_exclusion,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "universe.yaml"

_CUSIP_CONTINUITY_WINDOW_DAYS = 60


def _load_domestic_only_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["chass_domestic_only"]


def _apply_domestic_only(monthly: pd.DataFrame) -> pd.DataFrame:
    """Restrict to foreign_flag-Foreign Flag == 'Domestic Firm' (config
    chass_domestic_only), symmetric with the US side's usincflg='Y'
    rule. Does not mutate input."""
    cfg = _load_domestic_only_config()
    return monthly.loc[
        monthly["foreign_flag-Foreign Flag"] == cfg["foreign_flag_value"]
    ].copy()


def universe_at(monthly: pd.DataFrame, month_end: datetime.date) -> pd.DataFrame:
    """Point-in-time eligible Canadian universe at month_end: has a
    monthly row at month_end, domestic, not a fund/REIT/MLP. See module
    docstring for the row-presence membership design and its validated
    accuracy/limitations.

    monthly: a frame as returned by chass_loader.load_monthly() (or the
    same shape -- callers pass it in rather than this function calling
    load_monthly() itself, so tests can supply small synthetic frames
    without touching the real parquet).

    month_end: a plain datetime.date. Unlike universe_panel.py's
    universe_at(), no separate trading-day resolution is needed here --
    CHASS's monthly trdate-Trade Date values are themselves already
    each month's actual last trading day (per the CHASS user's guide's
    Monthly Trading Data section), not a naive calendar month-end, so a
    caller passing a real month-end date (e.g. the last calendar day of
    June) will not match a June row unless it happens to equal that
    month's actual last trading day. Callers needing "the eligible
    universe for June" should resolve month_end to the same real
    trading date CHASS itself would report for that month before
    calling this function -- this function does no such resolution
    itself, matching its narrower contract (row presence at an exact
    date, not a calendar month).
    """
    # Fund/REIT/MLP exclusion MUST run before domestic-only filtering, not
    # after: apply_chass_fund_mlp_reit_exclusion's stale-CSV safety check
    # (chass_loader.py) compares config/chass_fund_classification.csv's
    # (symbol, usage) keys against monthly's -- and several of that CSV's
    # entries (e.g. CDI.A/CDI.B) are Foreign Firm rows. Applying domestic-
    # only first would drop those rows before the exclusion step ever
    # sees them, tripping that check's stale-entry alarm on every call
    # even though nothing is actually stale (confirmed live while writing
    # this module's tests). Order doesn't change the final membership
    # result either way (both are independent boolean filters) -- only
    # this order satisfies apply_chass_fund_mlp_reit_exclusion's
    # precondition that it sees close to the full monthly universe.
    fund_excluded = apply_chass_fund_mlp_reit_exclusion(monthly)
    domestic = _apply_domestic_only(fund_excluded)
    month_end_ts = pd.Timestamp(month_end)
    return domestic.loc[domestic["trdate-Trade Date"] == month_end_ts].copy()


def _terminal_row_flags(monthly: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol-Ticker, usage-Usage Number), whether its last-ever
    row in `monthly` shows the NaN corroboration pattern (shares_out AND
    return both NaN) -- a confidence signal for a genuine delisting, not
    a membership filter itself. See module docstring for the live-
    measured true/false-positive rates.

    Returns SECURITY_ID_COLUMNS plus terminal_nan_corroboration (bool).
    """
    sorted_monthly = monthly.sort_values(SECURITY_ID_COLUMNS + ["trdate-Trade Date"])
    last_rows = sorted_monthly.groupby(SECURITY_ID_COLUMNS, as_index=False).tail(1)

    result = last_rows[SECURITY_ID_COLUMNS].copy()
    result["terminal_nan_corroboration"] = (
        last_rows["shares_out-Monthly Shares outstanding (100s of shares)"].isna()
        & last_rows["return-Monthly Return"].isna()
    )
    return result.reset_index(drop=True)


def _check_cusip_continuity(
    monthly: pd.DataFrame, ticker: str, usage: int
) -> tuple[str, int] | None:
    """Given a (ticker, usage) whose rows have stopped, check whether
    its CUSIP reappears under a DIFFERENT (ticker, usage) whose first
    row starts within _CUSIP_CONTINUITY_WINDOW_DAYS of this one's last
    row. Returns the new (ticker, usage) if found, else None. See
    module docstring for why this exists and what was found testing it
    (zero matches in the 179-name suspect group this session -- that
    group turned out to be genuine delistings, not renames).

    A CUSIP shared by more than one OTHER (ticker, usage) at once (the
    kind of real CUSIP collision documented in chass_loader.py's module
    docstring, e.g. the TWE/TRANSWEST case) is not usable for this
    check -- returns None rather than guessing among several matches.
    """
    own_rows = monthly[
        (monthly["symbol-Ticker"] == ticker) & (monthly["usage-Usage Number"] == usage)
    ]
    if own_rows.empty:
        return None
    cusip = own_rows["cusip-CUSIP"].iloc[0]
    own_last_date = own_rows["trdate-Trade Date"].max()

    same_cusip = monthly[monthly["cusip-CUSIP"] == cusip]
    other_keys = (
        same_cusip.loc[
            ~((same_cusip["symbol-Ticker"] == ticker) & (same_cusip["usage-Usage Number"] == usage)),
            ["symbol-Ticker", "usage-Usage Number"],
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    other_keys = list(other_keys)
    if len(other_keys) != 1:
        return None

    other_ticker, other_usage = other_keys[0]
    other_first_date = monthly[
        (monthly["symbol-Ticker"] == other_ticker) & (monthly["usage-Usage Number"] == other_usage)
    ]["trdate-Trade Date"].min()

    gap_days = abs((other_first_date - own_last_date).days)
    if gap_days <= _CUSIP_CONTINUITY_WINDOW_DAYS:
        return (other_ticker, other_usage)
    return None


# Monthly shares_out is in 100s of shares (confirmed directly: Pembina
# Pipeline's shares_out-Monthly Shares outstanding shows 5,810,572 as of
# 2025-12-31; x100 = 581,057,200 shares, matching its real public share
# count -- x1000, the initially-guessed multiplier, would overstate by
# 10x). The column name states this literally ("100s of shares"); stated
# here explicitly since it's an easy order-of-magnitude mistake.
_SHARES_OUT_UNITS_MULTIPLIER = 100


def market_cap_at(
    monthly: pd.DataFrame, daily: pd.DataFrame, month_end: datetime.date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Market cap (prior trading day's close x current-month shares
    outstanding) for every name eligible per universe_at(monthly,
    month_end). Mirrors universe_panel.py's market_cap_at() contract on
    the US CRSP-primary leg.

    CHASS's monthly file carries no closing-price column at all
    (confirmed against both the converted parquet and the raw CSV header
    -- the Closing Price field the CHASS user's guide describes in its
    Monthly Trading Data section, section 3.5, was never included in
    this project's pull). Market cap therefore joins a price in from the
    DAILY file, unlike CRSP's market_cap_at() where price and shares
    both come from the same daily panel -- this is a genuine daily<->
    monthly join, not just a same-file lookup.

    Price lag: the prior trading day strictly before month_end, resolved
    from `daily`'s own trdate-Trade Date values per name (never assumed
    to be month_end minus one calendar day) -- same principle as
    universe_panel.py's _mkt_cap_from_panel, adapted to a per-name
    resolution since CHASS's daily coverage can have name-specific gaps
    a single shared market calendar wouldn't (though in practice this
    should align with the whole market's last trading day almost
    always). CHASS's daily closeprice has no negative-value proxy
    convention the way CRSP's dlyprc does (confirmed: zero negative
    closeprice rows in a full year, 2020, of daily data -- CHASS's own
    negative-flag convention, section 3.6, is specific to MONTHLY
    closing price, not daily), so no abs() is applied here.

    Shares-outstanding period: the monthly row's OWN shares_out value at
    month_end (e.g. June's market cap pairs June's shares_out with the
    prior trading day's close), not the prior month's. shares_out
    describes shares outstanding as of that row's own trdate-Trade Date
    (CHASS user's guide section 3.5: "total shares outstanding at month
    end"), so pairing it with a price one trading day earlier is a
    one-day mismatch at most, not a genuine lookahead the way using a
    FUTURE month's shares would be. Confirmed low-risk with the user:
    share-count changes are lumpy/infrequent (buybacks, issuances) and
    essentially never move within a single trading day the way price
    does, and market cap here is a portfolio-construction weighting
    input only, not a return-generating signal -- a one-day-stale share
    count cannot bias any return calculation.

    Returns (market_cap_df, coverage_log_df):
    - market_cap_df: SECURITY_ID_COLUMNS + mkt_cap, for names with a
      resolvable, non-null prior-trading-day close AND non-null current-
      month shares_out.
    - coverage_log_df: SECURITY_ID_COLUMNS + matched (bool) for every
      name universe_at(monthly, month_end) returned -- an eligible name
      with no resolvable market cap (no prior trading day at all, a null
      close, or null shares_out) shows up here as matched=False instead
      of silently vanishing from a downstream sum (CLAUDE.md: a wrong
      number that looks right is the worst possible outcome).
    """
    eligible = universe_at(monthly, month_end)
    month_end_ts = pd.Timestamp(month_end)

    prior_day_rows = (
        daily[daily["trdate-Trade Date"] < month_end_ts]
        .sort_values("trdate-Trade Date")
        .groupby(SECURITY_ID_COLUMNS, as_index=False)
        .tail(1)
    )

    merged = eligible[
        SECURITY_ID_COLUMNS + ["shares_out-Monthly Shares outstanding (100s of shares)"]
    ].merge(
        prior_day_rows[SECURITY_ID_COLUMNS + ["closeprice-Daily Closing price"]],
        on=SECURITY_ID_COLUMNS,
        how="left",
    )

    valid = merged.dropna(
        subset=[
            "closeprice-Daily Closing price",
            "shares_out-Monthly Shares outstanding (100s of shares)",
        ]
    ).copy()
    valid["mkt_cap"] = (
        valid["closeprice-Daily Closing price"]
        * valid["shares_out-Monthly Shares outstanding (100s of shares)"]
        * _SHARES_OUT_UNITS_MULTIPLIER
    )
    valid["price"] = valid["closeprice-Daily Closing price"]
    market_cap_df = valid[SECURITY_ID_COLUMNS + ["mkt_cap", "price"]].reset_index(drop=True)

    coverage_log_df = eligible[SECURITY_ID_COLUMNS].copy()
    matched_keys = set(
        zip(market_cap_df["symbol-Ticker"], market_cap_df["usage-Usage Number"])
    )
    coverage_log_df["matched"] = [
        (t, u) in matched_keys
        for t, u in zip(coverage_log_df["symbol-Ticker"], coverage_log_df["usage-Usage Number"])
    ]
    return market_cap_df, coverage_log_df.reset_index(drop=True)
