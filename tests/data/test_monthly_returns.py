"""Tests for src/data/monthly_returns.py -- Gate 4
(docs/02_validation_gates.md), the per-permno monthly arithmetic
compounding of daily CRSP returns feeding src.portfolio.rebalance's
monthly_rets input.

Delisting rows are compounded exactly where CRSP stamped them -- see
src/data/monthly_returns.py's module docstring for why an earlier
"fold_back" mode (reassigning an orphaned delisting row to an earlier
month) was built, found wrong, and removed: two real cases (permno
22518, 2008; permno 87899, 1995) are structurally identical at the daily-
panel level -- "last real trade, delisting row exactly one trading day
later, crossing a calendar-month boundary" -- with OPPOSITE correct
answers, distinguishable only by a portfolio's own formation calendar,
which this module cannot see. Both permnos are used as this file's
real-data anchors, per feedback_brainstorm_verify_fixtures: a cited known
case must be reachable through the actual pipeline, not merely real.
"""

import datetime

import polars as pl
import pytest

from src.data import monthly_returns
from src.data.universe_panel import RAW_DATA_DIR, _year_partition_files

pytestmark = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)


def test_schema():
    result = monthly_returns.monthly_arithmetic_returns(
        datetime.date(2008, 1, 1), datetime.date(2008, 12, 31), leg="us",
    )
    assert set(result.columns) == {"permno", "month", "ret", "n_days", "has_delisting"}
    assert result.schema["permno"].is_integer()
    assert str(result.schema["month"]) == "Date"
    assert result.schema["ret"].is_float()
    assert result.schema["has_delisting"] == pl.Boolean


def test_raises_on_canada_leg():
    """Mirrors beta_fp._daily_log_returns(leg='canada') -- the Canadian
    stock-side path is not built; this must raise before any file is
    touched, never silently substitute the US panel."""
    with pytest.raises(NotImplementedError):
        monthly_returns.monthly_arithmetic_returns(
            datetime.date(2015, 1, 1), datetime.date(2015, 1, 31), leg="canada",
        )


def test_raises_on_unknown_leg():
    with pytest.raises(KeyError):
        monthly_returns.monthly_arithmetic_returns(
            datetime.date(2015, 1, 1), datetime.date(2015, 1, 31), leg="mars",
        )


def test_month_with_zero_real_observations_produces_no_row():
    """A permno with no trading days in some calendar month (delisted
    before it, or not yet listed) must be ABSENT from that month, never
    present with ret=0.0 -- a fabricated zero is exactly the bug F2c's
    truncation test caught in rebalance.py. Permno 22518's last real
    trade is 2008-08-29; its ONLY row after that is the 2008-09-02
    delisting row (has_delisting=True, compounded into September, exactly
    as CRSP stamped it) -- no row for permno 22518 exists in October or
    later."""
    result = monthly_returns.monthly_arithmetic_returns(
        datetime.date(2008, 1, 1), datetime.date(2008, 12, 31), leg="us",
    )
    permno_rows = result.filter(pl.col("permno") == 22518)
    months = sorted(permno_rows["month"].to_list())
    assert months, "permno 22518 must have at least its September row"
    assert months[-1] == datetime.date(2008, 9, 30), (
        f"permno 22518's last row is {months[-1]}, expected 2008-09-30 "
        "(its delisting row, compounded as-stamped) -- a later calendar "
        "month was fabricated with no real trading days"
    )


def test_delisting_row_compounds_in_its_own_stamped_calendar_month():
    """Permno 22518: last real trade 2008-08-29, delisting row dated
    2008-09-02 (dlyret=-0.35). September's return must be EXACTLY -0.35
    (the delisting row is the only row that month) and August must be
    UNCHANGED by it -- the exact contrast that made fold_back wrong: the
    August formation that would otherwise claim this permno never gets a
    September return, so run_backtest's missing-coverage skip (not this
    module) is what correctly excludes the loss from a formation that
    never held it. Expected August value computed independently from
    the raw panel: compounding the 21 real August trading days gives
    -0.33355522165176915 -- anchored outside this test file, not
    self-referential (CLAUDE.md: every gate needs an absolute check)."""
    result = monthly_returns.monthly_arithmetic_returns(
        datetime.date(2008, 8, 1), datetime.date(2008, 9, 30), leg="us",
    )
    august_row = result.filter(
        (pl.col("permno") == 22518) & (pl.col("month") == datetime.date(2008, 8, 31))
    )
    assert august_row.height == 1
    assert august_row["ret"][0] == pytest.approx(-0.33355522165176915, abs=1e-9)
    assert bool(august_row["has_delisting"][0]) is False

    september_row = result.filter(
        (pl.col("permno") == 22518) & (pl.col("month") == datetime.date(2008, 9, 30))
    )
    assert september_row.height == 1, (
        "the delisting row must stay in September, exactly as CRSP dated it"
    )
    assert september_row["ret"][0] == pytest.approx(-0.35, abs=1e-9)
    assert bool(september_row["has_delisting"][0]) is True


def test_delisting_row_that_is_a_real_held_period_return_flows_through():
    """Permno 87899: last real trade 1995-01-31 (itself a calendar
    month-end), delisting row 1995-02-01 (+0.03) -- ONE trading day
    later, crossing into February. Structurally identical to permno
    22518's case above (last trade, then a delisting row one trading day
    later, in the following month), but here February IS a real held
    month: a portfolio formed on 1995-01-31 holds this position into
    February and genuinely earns the +3% delisting return on 1995-02-01.
    This module must compound it into February UNCHANGED -- the whole
    reason fold_back (which would have moved this into January, a month
    before the position was ever held) was wrong."""
    result = monthly_returns.monthly_arithmetic_returns(
        datetime.date(1995, 1, 1), datetime.date(1995, 2, 28), leg="us",
    )
    february_row = result.filter(
        (pl.col("permno") == 87899) & (pl.col("month") == datetime.date(1995, 2, 28))
    )
    assert february_row.height == 1, (
        "permno 87899 must have a real February row -- its delisting "
        "return is a genuine held-period return, not an orphan to be "
        "folded back into January"
    )
    assert february_row["ret"][0] == pytest.approx(0.03, abs=1e-9)
    assert bool(february_row["has_delisting"][0]) is True

    january_row = result.filter(
        (pl.col("permno") == 87899) & (pl.col("month") == datetime.date(1995, 1, 31))
    )
    assert january_row.height == 1
    assert bool(january_row["has_delisting"][0]) is False, (
        "January (the formation month) must NOT carry the delisting "
        "return -- that would charge the loss to a month before the "
        "position was ever held"
    )


def test_covers_every_year_in_multi_year_range():
    """Mirrors _daily_log_returns's own regression test (F2b): years_needed
    must span every calendar year from start.year through end.year
    inclusive, not just the two endpoints -- {start.year, end.year} alone
    would silently drop a middle year (_year_partition_files(year) only
    pulls `year` and `year - 1`)."""
    result = monthly_returns.monthly_arithmetic_returns(
        datetime.date(2015, 1, 1), datetime.date(2018, 12, 31), leg="us",
    )
    months_present = {m.month for m in result["month"].to_list() if m.year == 2016}
    assert len(months_present) == 12, (
        f"2016 (a middle year in a 2015-2018 range) has {len(months_present)} "
        "distinct months present, expected 12 -- a middle year is being "
        "silently dropped, the exact bug fixed in beta_fp._daily_log_returns"
    )


def test_2008_orphan_delisting_row_count_is_pinned():
    """Structural regression pin on the phenomenon this module's
    docstring is built around: 45 of 396 2008 delisting rows with a
    prior trade land in a LATER calendar month than the permno's last
    real trade (measured against the raw panel via _year_partition_files,
    docs/02_validation_gates.md Gate 4). This module does nothing special
    with these rows any more (they compound as-stamped, like every other
    row) -- this pin exists so a future session re-measures before
    reaching for a fold_back-style transformation again, rather than
    rediscovering the same wrong fix from scratch."""
    files = _year_partition_files(2008)
    df = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlyret", "dlydelflg"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(datetime.date(2008, 1, 1)).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(datetime.date(2008, 12, 31)).cast(pl.Datetime("ns")))
        )
        .collect(engine="streaming")
        .with_columns(pl.col("dlycaldt").cast(pl.Date).alias("date"))
    )
    delist_rows = df.filter(pl.col("dlydelflg") == "Y").select(["permno", "date", "dlyret"])
    non_delist = df.filter(pl.col("dlydelflg") != "Y").group_by("permno").agg(
        pl.col("date").max().alias("last_trade")
    )
    joined = delist_rows.join(non_delist, on="permno", how="left").filter(
        pl.col("last_trade").is_not_null()
    )
    joined = joined.with_columns(
        [
            (pl.col("date").dt.year() * 12 + pl.col("date").dt.month()).alias("dm"),
            (pl.col("last_trade").dt.year() * 12 + pl.col("last_trade").dt.month()).alias("tm"),
        ]
    )
    assert joined.height == 396, (
        f"396 2008 delisting rows with a prior trade expected, got "
        f"{joined.height} -- the raw panel has changed since this was measured"
    )
    orphans = joined.filter(pl.col("dm") > pl.col("tm"))
    assert orphans.height == 45, (
        f"45 orphaned (later-calendar-month) delisting rows expected in "
        f"2008, got {orphans.height} -- if this moved, re-measure before "
        "trusting the frequency claim in this module's docstring"
    )
