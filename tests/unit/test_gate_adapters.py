"""
Tests for the Phase B output-normalizing adapters (src/data/gate_adapters.py)
-- see the Gate 1 implementation plan
(C:\\Users\\spite\\.claude\\plans\\given-the-overall-goal-crystalline-hoare.md,
Phase B). Each adapter wraps its country's existing universe_at()/
market_cap_at() (US: universe_panel.py; Canada: chass_universe.py) plus a
direct daily-panel read for return, and normalizes OUTPUT ONLY to one
common schema: id (str), date, price, shares, mkt_cap, ret -- so Gate 1
and later estimation/portfolio code can be written once against this
schema regardless of country.

Design, confirmed with the user 2026-09-07: this is a date-range panel
builder, not a single-date snapshot -- membership is resolved once per
month via universe_at() (a name eligible at month-end m is held through
month m+1, the same formation-timing convention docs/00_spec.md section
10 describes for the real BAB portfolio), then daily price/return/shares
are read directly from each country's own daily panel for every day in
that membership window, with mkt_cap computed as LAGGED price x shares
each day (never same-day -- CLAUDE.md's core rule).

US return field is dlyret (total return, confirmed 2026-09-07 to
genuinely diverge from dlyretx on real ex-dividend dates -- e.g. AAPL
permno 14593, 2015-08-06, dlyret=0.002166 vs dlyretx=-0.00234), matching
vwretd (also total return) as the Gate 1 comparison target -- NOT dlyretx/
vwretx, a known Gate 1 failure mode already flagged in the plan.
"""

import datetime
from pathlib import Path

import polars as pl
import pytest

from src.data import gate_adapters
from src.data.universe_panel import RAW_DATA_DIR as US_RAW_DATA_DIR

pytestmark = pytest.mark.skipif(
    not US_RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)

_COMMON_COLUMNS = {"id", "date", "price", "shares", "mkt_cap", "ret"}


def test_us_panel_returns_common_schema():
    panel = gate_adapters.us_gate1_panel(
        start_date=datetime.date(2015, 6, 1), end_date=datetime.date(2015, 6, 30)
    )
    assert set(panel.columns) == _COMMON_COLUMNS
    assert panel.schema["id"] == pl.Utf8
    assert panel.schema["date"] == pl.Date or panel.schema["date"] == pl.Datetime("ns")


def test_us_panel_id_is_str_permno():
    panel = gate_adapters.us_gate1_panel(
        start_date=datetime.date(2015, 6, 1), end_date=datetime.date(2015, 6, 30)
    )
    ids = panel["id"].to_list()
    assert all(isinstance(i, str) for i in ids)
    # AAPL permno 14593 must be present as the string "14593", not the
    # int 14593 or any other representation.
    assert "14593" in ids


def test_us_panel_never_uses_same_day_or_future_price_for_mkt_cap():
    """The core leakage check for this adapter, mirroring
    universe_panel.py's own test_market_cap_never_uses_month_end_or_later_price:
    mkt_cap on any given date must be derived from a STRICTLY EARLIER
    day's price, never that day's own price."""
    panel = gate_adapters.us_gate1_panel(
        start_date=datetime.date(2015, 6, 1), end_date=datetime.date(2015, 6, 30)
    )
    aapl = panel.filter(pl.col("id") == "14593").sort("date")
    assert aapl.height > 1

    # mkt_cap on row i should equal price*shares from row i-1, not row i.
    for i in range(1, aapl.height):
        prior_price = aapl["price"][i - 1]
        prior_shares = aapl["shares"][i - 1]
        expected_mkt_cap = abs(prior_price) * prior_shares
        assert aapl["mkt_cap"][i] == pytest.approx(expected_mkt_cap, rel=1e-9), (
            f"row {i} mkt_cap does not match prior row's price*shares -- "
            "possible same-day/unlagged leakage"
        )


def test_us_panel_ret_matches_dlyret_not_dlyretx():
    """AAPL (permno 14593) 2015-08-06 is a confirmed ex-dividend date
    where dlyret (0.002166) and dlyretx (-0.00234) genuinely diverge --
    the adapter's ret column must match dlyret."""
    panel = gate_adapters.us_gate1_panel(
        start_date=datetime.date(2015, 8, 1), end_date=datetime.date(2015, 8, 31)
    )
    row = panel.filter(
        (pl.col("id") == "14593") & (pl.col("date") == datetime.date(2015, 8, 6))
    )
    assert row.height == 1
    assert row["ret"][0] == pytest.approx(0.002166, abs=1e-6)


def test_us_panel_excludes_names_not_in_universe_at_month():
    """A permno excluded by universe_at() must not appear in the panel at
    all, for any date in the range. June's trading days are governed by
    MAY's universe_at() snapshot (formation timing -- see module
    docstring), not June's own, so the reference set here is May's."""
    panel = gate_adapters.us_gate1_panel(
        start_date=datetime.date(2015, 6, 1), end_date=datetime.date(2015, 6, 30)
    )
    from src.data.universe_panel import universe_at

    eligible = universe_at(datetime.date(2015, 5, 31))
    eligible_ids = {str(p) for p in eligible["permno"].to_list()}
    panel_ids = set(panel["id"].to_list())
    assert panel_ids <= eligible_ids


def test_us_panel_mkt_cap_uses_immediately_prior_trading_day():
    """The adapter's mkt_cap on date D is D's own immediately-prior
    trading day's price x shares -- NOT the same thing as
    universe_panel.market_cap_at(D)'s value, which is anchored to D's
    OWN prior trading day, treated as a month-end snapshot lag. These
    answer different questions (market_cap_at(2015-06-30) uses
    2015-06-29's price, the lag for a 2015-06-30 SNAPSHOT; the adapter's
    2015-07-01 row uses 2015-06-30's price, the lag for 2015-07-01
    TRADING) and are only coincidentally comparable when the two
    anchor dates happen to be adjacent trading days -- confirmed directly
    2026-09-07 that they are NOT equal here (different anchor dates,
    different prices), which is correct, not a bug.

    This test instead confirms the adapter's own internal consistency:
    07-01's mkt_cap equals 06-30's own raw price x shares from the panel
    (with the corrected SHROUT_UNITS_MULTIPLIER, per
    docs/01_data_notes.md section 15's 2026-09-08 correction -- shrout is
    in thousands, not raw shares), directly recomputed here (not sourced
    from market_cap_at at all)."""
    from src.data.universe_panel import RAW_DATA_DIR, SHROUT_UNITS_MULTIPLIER

    prior_day_row = (
        pl.scan_parquet(str(RAW_DATA_DIR / "year=2015" / "part.parquet"))
        .select(["permno", "dlycaldt", "dlyprc", "shrout"])
        .filter(
            (pl.col("permno") == 14593)
            & (pl.col("dlycaldt") == pl.lit(datetime.date(2015, 6, 30)).cast(pl.Datetime("ns")))
        )
        .collect()
    )
    assert prior_day_row.height == 1
    expected = (
        abs(prior_day_row["dlyprc"][0])
        * prior_day_row["shrout"][0]
        * SHROUT_UNITS_MULTIPLIER
    )

    panel = gate_adapters.us_gate1_panel(
        start_date=datetime.date(2015, 7, 1), end_date=datetime.date(2015, 7, 1)
    )
    row = panel.filter(pl.col("id") == "14593")
    assert row.height == 1
    assert row["mkt_cap"][0] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# canada_gate1_panel
# ---------------------------------------------------------------------------

CHASS_RAW_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "CHASS_Data" / "parquet"
)

pytestmark_canada = pytest.mark.skipif(
    not CHASS_RAW_DATA_DIR.exists(),
    reason="data/raw/CHASS_Data/parquet/ not present",
)


@pytestmark_canada
def test_canada_panel_returns_common_schema():
    panel = gate_adapters.canada_gate1_panel(
        start_date=datetime.date(2015, 6, 1), end_date=datetime.date(2015, 6, 30)
    )
    assert set(panel.columns) == _COMMON_COLUMNS
    assert panel.schema["id"] == pl.Utf8


@pytestmark_canada
def test_canada_panel_id_is_symbol_underscore_usage():
    """Pembina Pipeline is ticker PPL, usage 2 (named ground-truth case
    used throughout chass_universe.py's own tests)."""
    panel = gate_adapters.canada_gate1_panel(
        start_date=datetime.date(2015, 6, 1), end_date=datetime.date(2015, 6, 30)
    )
    ids = panel["id"].to_list()
    assert all(isinstance(i, str) for i in ids)
    assert "PPL_2" in ids


@pytestmark_canada
def test_canada_panel_never_uses_same_day_or_future_price_for_mkt_cap():
    """Mirrors the US adapter's core leakage check: mkt_cap on any given
    date must be derived from a strictly earlier day's close, never that
    day's own close."""
    panel = gate_adapters.canada_gate1_panel(
        start_date=datetime.date(2015, 6, 1), end_date=datetime.date(2015, 6, 30)
    )
    ppl = panel.filter(pl.col("id") == "PPL_2").sort("date")
    assert ppl.height > 1

    for i in range(1, ppl.height):
        prior_price = ppl["price"][i - 1]
        this_shares = ppl["shares"][i]
        expected_mkt_cap = abs(prior_price) * this_shares
        assert ppl["mkt_cap"][i] == pytest.approx(expected_mkt_cap, rel=1e-9), (
            f"row {i} mkt_cap does not use prior day's price -- possible "
            "same-day/unlagged leakage"
        )


@pytestmark_canada
def test_canada_panel_ret_matches_daily_return_column():
    """Direct passthrough: the adapter's ret column must match CHASS
    daily's own return-Daily Return for the same (symbol, usage, date),
    not a recomputed value."""
    from src.data.chass_loader import load_daily

    daily = load_daily(years=[2015])
    reference = daily.filter(
        (pl.col("symbol-Ticker") == "PPL")
        & (pl.col("usage-Usage Number") == 2)
        & (pl.col("trdate-Trade Date") == datetime.date(2015, 6, 15))
    )
    assert reference.height == 1
    expected_ret = reference["return-Daily Return"][0]

    panel = gate_adapters.canada_gate1_panel(
        start_date=datetime.date(2015, 6, 1), end_date=datetime.date(2015, 6, 30)
    )
    row = panel.filter((pl.col("id") == "PPL_2") & (pl.col("date") == datetime.date(2015, 6, 15)))
    assert row.height == 1
    assert row["ret"][0] == pytest.approx(expected_ret, rel=1e-9)


@pytestmark_canada
def test_canada_panel_excludes_names_not_in_universe_at_month():
    """A name excluded by universe_at() (e.g. a fund/REIT/MLP, or a
    foreign firm) must not appear in the panel at all. June's trading
    days are governed by MAY's real CHASS trading-month-end snapshot
    (formation timing), not June's own."""
    from src.data.chass_loader import load_monthly
    from src.data.chass_universe import universe_at

    monthly = load_monthly()
    may_month_end = (
        monthly.filter(pl.col("trdate-Trade Date").dt.year() == 2015)
        .filter(pl.col("trdate-Trade Date").dt.month() == 5)
        .select("trdate-Trade Date")
        .unique()["trdate-Trade Date"][0]
        .date()
    )
    eligible = universe_at(monthly, may_month_end)
    eligible_ids = {f"{t}_{u}" for t, u in zip(
        eligible["symbol-Ticker"].to_list(), eligible["usage-Usage Number"].to_list()
    )}

    panel = gate_adapters.canada_gate1_panel(
        start_date=datetime.date(2015, 6, 1), end_date=datetime.date(2015, 6, 30)
    )
    panel_ids = set(panel["id"].to_list())
    assert panel_ids <= eligible_ids


@pytestmark_canada
def test_canada_panel_covers_every_year_in_multi_year_range():
    """Found while validating Gate 4's chunked market-index builder
    (2026-09-13): canada_gate1_panel's `years_needed =
    sorted({start_date.year, end_date.year})` only loads the START and
    END years' CHASS daily files -- the SAME bug class already found and
    fixed in beta_fp._daily_log_returns (see that function's own
    docstring: years_needed must cover EVERY calendar year in range, not
    just the two endpoints, or a middle year is silently dropped with no
    error). Every existing test of this function only ever used a
    single-month span within one calendar year, so {start.year, end.year}
    was always a single-element set and this was invisible.

    Direct discriminator: a 1995-08-31..2000-08-31 call (5 full years)
    must return roughly 5 years' worth of unique trading dates (~1200+),
    not one year's worth (~252) -- confirmed against the pre-fix code
    that the bug produces exactly 252 unique dates, silently dropping
    1996-1999 entirely."""
    panel = gate_adapters.canada_gate1_panel(
        start_date=datetime.date(1995, 8, 31), end_date=datetime.date(2000, 8, 31)
    )
    unique_dates = panel["date"].unique()
    assert unique_dates.len() > 1000, (
        f"only {unique_dates.len()} unique dates across a 5-year window "
        "(1995-08-31..2000-08-31) -- years_needed is silently dropping "
        "middle years (same bug class as beta_fp._daily_log_returns's "
        "own fixed regression)"
    )
    years_present = {d.year for d in unique_dates.to_list()}
    assert years_present == {1995, 1996, 1997, 1998, 1999, 2000}, (
        f"years present: {sorted(years_present)} -- expected all six "
        "years 1995 through 2000 to have at least one trading date"
    )


def _us_gate1_panel_reference_unchunked(
    start_date: datetime.date, end_date: datetime.date
) -> pl.DataFrame:
    """A verbatim reproduction of us_gate1_panel's PRE-chunking
    implementation (single .collect() + .shift(1).over("permno") over
    the WHOLE requested range in one pass, no year-by-year chunking) --
    kept here, not in src/, purely as this test's own ground truth. This
    is what us_gate1_panel actually computed before 2026-09-13's
    chunking change; if the two ever disagree, the chunked version is
    wrong, not this reference."""
    from src.data import universe_panel

    membership = gate_adapters._us_gate1_membership(start_date, end_date)
    if membership.height == 0:
        return pl.DataFrame(schema=gate_adapters._empty_schema())

    month_ends = gate_adapters._month_ends_in_range(start_date, end_date)
    years_needed = sorted({d.year for d in month_ends} | {start_date.year, end_date.year})
    files = []
    for year in years_needed:
        path = universe_panel.RAW_DATA_DIR / f"year={year}" / "part.parquet"
        if path.exists():
            files.append(str(path))
    if not files:
        return pl.DataFrame(schema=gate_adapters._empty_schema())

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
        pl.col("dlycaldt").dt.month().alias("_holding_month"),
    )
    daily = daily.join(
        membership, on=["permno", "_holding_year", "_holding_month"], how="inner"
    )
    daily = daily.filter(
        (pl.col("dlycaldt") >= pl.lit(start_date).cast(pl.Datetime("ns")))
        & (pl.col("dlycaldt") <= pl.lit(end_date).cast(pl.Datetime("ns")))
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
    return result.select(gate_adapters._COMMON_SCHEMA_ORDER)


def test_us_gate1_panel_chunked_matches_reference_implementation():
    """us_gate1_panel now processes its daily join/shift logic in yearly
    chunks internally (2026-09-13, Gate 4: the pre-chunking version
    crashed with a Rust memory allocation failure on a 56-year, 72.9M-row
    span). The boundary carry (_LAG_CARRY_DAYS=10 calendar days,
    re-applied at EVERY year boundary rather than only the original
    single start_date) is the one thing chunking could get wrong -- a
    too-short carry would silently drop a real prior-trading-day price
    at a year boundary, handing one name a null mkt_cap on the first
    trading day of a year, which would then be filtered out rather than
    raising any error.

    A single-year span cannot exercise this (there is only one chunk,
    so no boundary is ever crossed) -- this test spans THREE full
    calendar years (2015-2017) specifically to cross two year
    boundaries, and asserts the chunked and unchunked implementations
    are bit-identical row-for-row over that span, not merely close."""
    start_date, end_date = datetime.date(2015, 1, 1), datetime.date(2017, 12, 31)

    chunked = gate_adapters.us_gate1_panel(start_date, end_date)
    reference = _us_gate1_panel_reference_unchunked(start_date, end_date)

    assert chunked.height > 0, "test setup: expected a non-empty real panel"
    assert chunked.height == reference.height, (
        f"chunked panel has {chunked.height} rows, unchunked reference has "
        f"{reference.height} -- the boundary carry is dropping or "
        "duplicating rows at a year boundary"
    )

    chunked_sorted = chunked.sort(["id", "date"])
    reference_sorted = reference.sort(["id", "date"])
    from polars.testing import assert_frame_equal

    assert_frame_equal(chunked_sorted, reference_sorted, check_row_order=True)


def test_us_gate1_panel_no_null_mkt_cap_on_first_trading_day_of_year():
    """Direct check on the failure mode a too-short boundary carry would
    cause: a name eligible on the first trading day of a year (e.g.
    2016-01-04, after the 2015-01-01 holiday) must resolve a real,
    non-null LAGGED mkt_cap from late-December trading -- not silently
    lose that day because the chunk carry didn't reach back far enough.
    This is a narrower, cheaper check than the full bit-identical
    comparison above; both matter, since this one would catch a
    regression even if a future change also happened to alter row COUNT
    in a way that coincidentally matched (this test checks a specific
    date's coverage, not aggregate row counts)."""
    panel = gate_adapters.us_gate1_panel(
        datetime.date(2015, 1, 1), datetime.date(2016, 12, 31)
    )
    first_trading_days_2016 = (
        panel.filter(pl.col("date").dt.year() == 2016)
        .select(pl.col("date").min())
        .item()
    )
    assert first_trading_days_2016 is not None
    first_day_rows = panel.filter(pl.col("date") == first_trading_days_2016)
    assert first_day_rows.height > 0, (
        f"no rows at all for {first_trading_days_2016}, the first trading "
        "day of 2016 in this panel -- the year-boundary chunk carry is "
        "not reaching back far enough"
    )
    assert first_day_rows["mkt_cap"].null_count() == 0, (
        f"{first_day_rows['mkt_cap'].null_count()} of "
        f"{first_day_rows.height} names on {first_trading_days_2016} have "
        "a null mkt_cap -- the year-boundary carry is too short to "
        "resolve a real prior-trading-day price across the year-end"
    )
