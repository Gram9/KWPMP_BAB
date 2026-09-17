"""
Loader for the CHASS (CFMRC) Canadian daily/monthly parquet files
(data/raw/CHASS_Data/parquet/, converted from the raw CSVs by
convert_chass_csv.py). Establishes the correct security identity key --
CUSIP is NOT reliable in this dataset, and using it as a key silently
merges unrelated securities.

Polars port of the original pandas implementation (kept, unmodified, at
src/data/chass_loader_pandas_reference.py for parity testing -- see
tests/unit/test_chass_polars_parity.py). All behavior below is intended to
be identical to that reference; any divergence is a bug, not a design
choice, unless documented as such.

CUSIP investigation (live-measured against the full 46-year daily file,
16,715,944 rows, this session):

- (cusip, date) has 106,519 duplicate rows across 9,336 groups.
- Two CUSIP values are outright placeholders shared by unrelated
  securities: "000000000" and "305915xxx" (literal "xxx" suffix --
  e.g. ticker FL "FALCONBRIDGE LTD." and FL.B "FALCONBRIDGE NICKEL"
  both carry cusip 305915xxx for the whole sample).
- The remaining ~6,590 groups are cases where two DIFFERENT securities
  genuinely share one real CUSIP. Confirmed example: ticker TWE usage 0
  ("TRANS-WESTERN EXPLORATION INC.") and TWE usage 1 ("TRANSWEST ENERGY
  INC.") both trade live -- distinct non-zero prices and volumes -- on
  the same dates for weeks in May-June 1983, both under CUSIP
  893921106. This is not a rename transition with one side going stale
  (checked the full timeline: the "old" name keeps trading with real
  volume for weeks after the "new" name starts, so it isn't a frozen
  record left over from a delisting).
- CHASS's own field documentation (CFMRC user guide) says exactly why:
  ticker symbols are reused by the TSX over time, so CFMRC assigns a
  "usage number" to each reuse specifically to keep security identity
  unique -- CUSIP plays no such role in their design. Verified directly:
  zero cases where a (symbol-Ticker, usage-Usage Number) pair maps to
  more than one CUSIP, in either the daily or monthly file. That pair is
  the correct identity key.

Dividend fan-out (the only thing left after switching to the ticker+
usage key): daily can carry two dividend-event rows for the same
(ticker, usage, date) -- e.g. "Cash dividend - Regular" and "Cash
dividend - Extra" declared on the same ex-date -- identical on every
trading field (open/close/high/low/volume/return) and differing only in
the dividend-specific columns. Confirmed by full row diff on multiple
sampled cases (e.g. BCC / CUSIP 087239109 / 1980-02-25). This is a
one-to-many join artifact from folding a dividend-events table onto the
daily trading table, not a security-identity collision, and dropping
dividend detail loses no price/return/volume information -- BAB doesn't
need per-event dividend detail, since dividends are already embedded in
return-Daily Return. Collapsing this way removes exactly 2,735 of the
16,715,944 raw daily rows (10 of which were already exact full-row
duplicates), live-measured this session -- see test_chass_loader.py for
the pinned regression counts.

The monthly file has zero (ticker, usage, date) duplicates already (no
per-event dividend fan-out at monthly grain) -- only the key change
applies there, no dedup logic needed.

CUSIP is kept as a plain column in both loaders' output (useful for
cross-reference/debugging against other sources) but must never be
treated as a key.

Also provides the ETF/mutual-fund/REIT/MLP exclusion required by the
BAB mandate (chass_fund_mlp_reit_classification /
apply_chass_fund_mlp_reit_exclusion) -- see config/universe.yaml's
chass_fund_mlp_reit_exclusion block for the full derivation and
config/chass_fund_classification.csv for the hand-classified lookup
table it depends on.
"""

from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "CHASS_Data" / "parquet"
CONFIG_PATH = PROJECT_ROOT / "config" / "universe.yaml"

MONTHLY_PATH = PARQUET_DIR / "monthly.parquet"
DAILY_DIR = PARQUET_DIR / "daily"

# The verified clean identity key -- see module docstring. Exported so
# callers never reach for cusip-CUSIP as a key out of habit.
SECURITY_ID_COLUMNS = ["symbol-Ticker", "usage-Usage Number"]

# Dropping these before dedup is what collapses the dividend fan-out (see
# module docstring) -- the fields left behind are exactly the trading
# fields, which are identical across a fan-out group by construction.
_DIVIDEND_COLUMNS = [
    "trdate-Ex-dividend date",
    "dividend-Dividend",
    "flag1-Dividend flag 1",
    "flag2-Dividend flag 2",
    "flag3-Dividend flag 3",
]

_KEY = SECURITY_ID_COLUMNS + ["trdate-Trade Date"]


def _assert_key_unique(df: pl.DataFrame, source: str) -> None:
    n_dupes = df.height - df.unique(subset=_KEY, keep="first").height
    if n_dupes > 0:
        raise AssertionError(
            f"{source}: (symbol-Ticker, usage-Usage Number, trdate-Trade "
            "Date) is not unique -- this invariant was verified live "
            "against the current parquet files; a re-pull or CSV update "
            "may have changed the data. Investigate before trusting any "
            "downstream count (CLAUDE.md: a wrong number that looks "
            "right is the worst possible outcome)."
        )


def load_monthly() -> pl.DataFrame:
    """CHASS monthly panel, keyed by (symbol-Ticker, usage-Usage Number,
    trdate-Trade Date). No dedup needed -- verified zero duplicates on
    this key in the raw file."""
    monthly = pl.read_parquet(MONTHLY_PATH)
    _assert_key_unique(monthly, "load_monthly")
    return monthly


def load_daily(years: list[int] | None = None) -> pl.DataFrame:
    """CHASS daily panel, keyed by (symbol-Ticker, usage-Usage Number,
    trdate-Trade Date), with the dividend-event fan-out collapsed (see
    module docstring). years: restrict to these calendar years; None
    loads the full history.

    Dedup is applied per year-partition file independently (each file
    is already a closed calendar year, so the fan-out never spans
    files) -- this keeps load_daily(years=[Y]) identical to the
    corresponding slice of load_daily(), which the test suite checks
    directly.
    """
    if years is None:
        paths = sorted(DAILY_DIR.glob("year=*/part.parquet"))
    else:
        paths = [DAILY_DIR / f"year={year}" / "part.parquet" for year in years]

    frames = []
    for path in paths:
        year_df = pl.read_parquet(path)
        # unique() on polars does not guarantee which duplicate survives
        # unless keep is specified; "first" reproduces pandas
        # drop_duplicates()'s default (keeps the first-seen row in
        # original file order), the same convention the pandas reference
        # relies on for the BCC-style fan-out collapse.
        deduped = year_df.drop(_DIVIDEND_COLUMNS).unique(keep="first", maintain_order=True)
        _assert_key_unique(deduped, f"load_daily({path})")
        frames.append(deduped)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical")


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["chass_fund_mlp_reit_exclusion"]


def chass_fund_mlp_reit_classification(monthly: pl.DataFrame) -> pl.DataFrame:
    """Per-row fund/REIT/MLP classification for a CHASS monthly frame
    (as returned by load_monthly()). See config/universe.yaml's
    chass_fund_mlp_reit_exclusion block for the full derivation.

    Returns monthly's SECURITY_ID_COLUMNS plus name-Name (for
    debugging/lookup convenience) and two added columns:
    - classification: "FUND_PRODUCT", "REAL_COMPANY", or "UNKNOWN" for
      rows in one of the three contaminated business-Business
      categories (MUTUAL FUNDS/INVESTMENT COMPANY/TRUST FUND, looked up
      from config/chass_fund_classification.csv); "CLEAN_EXCLUDE" for
      rows in one of the three clean categories (INVESTMENT FUND/
      INVESTMENT TRUST/LIMITED PARTNERSHIP); "NOT_FUND_RELATED"
      otherwise.
    - excluded: True for CLEAN_EXCLUDE and FUND_PRODUCT rows, False
      otherwise. UNKNOWN rows are NOT excluded -- see the config block
      for why: confidence too low to classify, left in the universe as
      an open item rather than silently resolved either direction.

    Raises if config/chass_fund_classification.csv contains a (symbol,
    usage_number) pair absent from `monthly` -- a stale entry from a
    prior version of the CSV would otherwise silently classify zero
    rows without anyone noticing (CLAUDE.md: a wrong number that looks
    right is the worst possible outcome).
    """
    cfg = _load_config()
    classification_csv = PROJECT_ROOT / cfg["classification_csv"]
    hand_classified = pl.read_csv(
        classification_csv, schema_overrides={"usage_number": pl.Int64}
    )

    monthly_keys = set(
        zip(
            monthly["symbol-Ticker"].to_list(),
            monthly["usage-Usage Number"].to_list(),
        )
    )
    csv_keys = set(
        zip(
            hand_classified["symbol"].to_list(),
            hand_classified["usage_number"].to_list(),
        )
    )
    stale_keys = csv_keys - monthly_keys
    if stale_keys:
        raise AssertionError(
            f"{classification_csv} contains {len(stale_keys)} (symbol, "
            "usage_number) pair(s) not present in the loaded monthly "
            f"frame: {sorted(stale_keys)[:5]}... -- this CSV was hand-"
            "curated against a specific pull of monthly.parquet; a "
            "re-pull or CSV edit may have gone stale. Investigate before "
            "trusting the exclusion (CLAUDE.md: a wrong number that "
            "looks right is the worst possible outcome)."
        )

    hand_classified_renamed = hand_classified.select(
        pl.col("symbol").alias("symbol-Ticker"),
        pl.col("usage_number").alias("usage-Usage Number"),
        pl.col("classification").alias("_hand_classification"),
    )

    result = monthly.select(SECURITY_ID_COLUMNS + ["name-Name", "business-Business"])
    result = result.join(
        hand_classified_renamed, on=SECURITY_ID_COLUMNS, how="left"
    )

    clean_categories = cfg["clean_exclude_categories"]
    contaminated_categories = cfg["contaminated_categories"]

    result = result.with_columns(
        pl.when(pl.col("business-Business").is_in(clean_categories))
        .then(pl.lit("CLEAN_EXCLUDE"))
        .when(pl.col("business-Business").is_in(contaminated_categories))
        .then(pl.col("_hand_classification"))
        .otherwise(pl.lit("NOT_FUND_RELATED"))
        .alias("classification")
    )
    result = result.with_columns(
        pl.col("classification").is_in(["CLEAN_EXCLUDE", "FUND_PRODUCT"]).alias("excluded")
    )
    return result.select(SECURITY_ID_COLUMNS + ["name-Name", "classification", "excluded"])


def apply_chass_fund_mlp_reit_exclusion(monthly: pl.DataFrame) -> pl.DataFrame:
    """Drop ETFs/mutual funds/REITs/MLPs from a CHASS monthly frame per
    the BAB mandate. Does not mutate input, matches
    apply_us_reit_mlp_exclusion's contract (the superseded Compustat-based
    universe prototype, not included in this submission).

    UNKNOWN-classified rows are kept (not excluded) -- see
    chass_fund_mlp_reit_classification's docstring."""
    classification = chass_fund_mlp_reit_classification(monthly)
    excluded_keys = classification.filter(pl.col("excluded")).select(SECURITY_ID_COLUMNS)
    return monthly.join(excluded_keys, on=SECURITY_ID_COLUMNS, how="anti")


def chass_daily_trade_status(daily: pl.DataFrame) -> pl.Series:
    """Per-row trading status for a CHASS daily frame (as returned by
    load_daily()), the direct analog of CLAUDE.md's cshtrd > 0 rule.

    volume-Daily Volume partitions cleanly into three non-overlapping
    states, verified this session against the full year=1980 partition
    (140,563 / 68,984 / 7,341 of 216,888 total rows, summing exactly with
    no overlap):

    - "TRADED": volume > 0 -- a real trading day, usable observation.
    - "NO_TRADE": volume == 0 -- the security was listed and genuinely
      did not trade that day. This is real information, not missing data
      -- confirmed closeprice is always populated on these rows (often
      0.00, sometimes a carried displayed price), while return is always
      NaN (no valid return can be computed over a day with no trade).
    - "MISSING": volume is NaN -- true missing data, CHASS's own -9
      sentinel. Confirmed directly against the raw CSV (not just the
      documentation) that this sentinel arrives as a literal BLANK field,
      not the digit string "-9" -- e.g. ticker BHG.B, 1980-12-01, a
      "Recapitalization" price-adjustment event, has every single trading
      field blank in the source CSV, matching the CHASS user's guide's
      rule that a reclassification zeroes out (in their terms, -9.0's)
      the return at both t and t-1. pandas already converts a blank CSV
      field to NaN correctly -- no bug in convert_chass_csv.py, nothing
      to fix there.

    Returns a plain Python list-backed pl.Series named "trade_status",
    aligned to daily's row order (not a column added to daily itself,
    matching the pandas reference's contract of returning a bare Series
    the caller indexes daily by).
    """
    volume = daily["volume-Daily Volume"]
    status = (
        pl.select(
            pl.when(volume.is_null())
            .then(pl.lit("MISSING"))
            .when(volume == 0)
            .then(pl.lit("NO_TRADE"))
            .otherwise(pl.lit("TRADED"))
            .alias("trade_status")
        )
    )["trade_status"]
    return status


def filter_chass_daily_traded(daily: pl.DataFrame) -> pl.DataFrame:
    """Keep only rows with a genuine trade (volume > 0). Does not mutate
    input. Drops both NO_TRADE and MISSING rows -- see
    chass_daily_trade_status's docstring for why they must not be
    conflated with a real trading observation, even though both are
    excluded here."""
    return daily.filter(pl.col("volume-Daily Volume") > 0)
