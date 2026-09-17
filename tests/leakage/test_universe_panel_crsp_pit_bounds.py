"""
Leakage tests for point-in-time US universe membership (CRSP-primary,
src/data/universe_panel.py). Proves the securitybegdt/securityenddt bound
and the prior-trading-day market-cap resolution cannot let t+1
information influence a decision made at t (CLAUDE.md's core rule) --
independent of a live WRDS connection, using synthetic fixtures.

One test (test_market_cap_at_lags_relative_to_resolved_snapshot_not_raw_
month_end) is an exception: it uses the real pulled panel, since the bug
it guards against (market_cap_at() resolving a non-trading month_end
differently than universe_at()) can only be reproduced against a real
trading calendar -- a synthetic frame can't encode "which calendar dates
are/aren't real trading days". It is skipped if the pull isn't present.
"""

import datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.universe_panel import _apply_pit_bounds, _mkt_cap_from_panel

_RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "us_panel_crsp_full"


def _pull_available() -> bool:
    return _RAW_DATA_DIR.exists() and any(_RAW_DATA_DIR.glob("year=*/part.parquet"))

# Note: pl.date(y, m, d) with literal ints returns an unevaluated polars
# Expr, not a datetime.date -- using it as a raw value inside a dict passed
# to pl.DataFrame() produces an Object-dtype column holding Expr instances,
# not real Date values (it fails outright, not silently, with a polars
# SchemaError on comparison -- confirmed while writing this file).
# datetime.date(...) is the correct way to build literal date *column data*;
# pl.date(...) is for building expressions/scalars (e.g. the month_end
# argument below), matching the convention already established in
# tests/unit/test_universe_panel_us.py and documented in
# src/data/universe_panel.py's market_cap_at()/_mkt_cap_from_panel()
# docstrings.


def test_name_delisted_the_day_after_month_end_still_included():
    """The tight boundary case: securityenddt == month_end + 1 day must
    still be included -- the name was alive AT month_end."""
    synthetic = pl.DataFrame({
        "permno": [10001],
        "securitynm": ["EDGE CASE CO"],
        "issuertype": ["CORP"],
        "primaryexch": ["N"],
        "securitybegdt": [datetime.date(2000, 1, 1)],
        "securityenddt": [datetime.date(2015, 7, 1)],
    })
    result = _apply_pit_bounds(synthetic, month_end=pl.date(2015, 6, 30))
    assert len(result) == 1


def test_name_delisted_exactly_on_month_end_still_included():
    """securityenddt == month_end itself: still alive at month_end
    (boundary is inclusive on the delisting side)."""
    synthetic = pl.DataFrame({
        "permno": [10001],
        "securitynm": ["SAME DAY CO"],
        "issuertype": ["CORP"],
        "primaryexch": ["N"],
        "securitybegdt": [datetime.date(2000, 1, 1)],
        "securityenddt": [datetime.date(2015, 6, 30)],
    })
    result = _apply_pit_bounds(synthetic, month_end=pl.date(2015, 6, 30))
    assert len(result) == 1


def test_name_listed_the_day_after_month_end_excluded():
    """The tight boundary case on the listing side: securitybegdt ==
    month_end + 1 day must be excluded -- did not exist yet at month_end."""
    synthetic = pl.DataFrame({
        "permno": [10001],
        "securitynm": ["TOO NEW CO"],
        "issuertype": ["CORP"],
        "primaryexch": ["N"],
        "securitybegdt": [datetime.date(2015, 7, 1)],
        "securityenddt": [None],
    })
    result = _apply_pit_bounds(synthetic, month_end=pl.date(2015, 6, 30))
    assert len(result) == 0


def test_market_cap_never_uses_month_end_or_later_price():
    """The core leakage case for market_cap_at: a price row dated AT or
    AFTER month_end must never be selected, even if it's the only row in
    the panel."""
    synthetic = pl.DataFrame({
        "permno": [10001],
        "dlycaldt": [datetime.date(2015, 6, 30)],
        "dlyprc": [99.0],
        "shrout": [1000.0],
    })
    result = _mkt_cap_from_panel(synthetic, month_end=pl.date(2015, 6, 30))
    assert len(result) == 0, (
        "a panel with only a same-day-or-later price row must produce no "
        "market cap row, not fall back to the leaking same-day price"
    )


@pytest.mark.skipif(
    not _pull_available(),
    reason="data/raw/us_panel_crsp_full/ not present -- run "
    "pull_universe_us_crsp.py first (Task 1)",
)
def test_market_cap_at_lags_relative_to_resolved_snapshot_not_raw_month_end():
    """Critical regression (final whole-branch review): on a non-trading
    month_end (2007-06-30, a Saturday), universe_at() resolves to the
    actual last trading day (2007-06-29) before taking its snapshot. If
    market_cap_at() independently filtered the price panel on the RAW
    month_end (dlycaldt < 2007-06-30) instead of that same resolved
    snapshot day, it would land on the exact same trading day
    (2007-06-29) as its "prior" price -- i.e. no lag at all, silently,
    on every non-trading month_end (~29% of calendar month-ends). This
    test asserts the market-cap price date actually used is STRICTLY
    EARLIER than universe_at()'s resolved snapshot date for the same
    month_end. Uses the real pulled panel (data/raw/us_panel_crsp_full/)
    rather than a synthetic fixture, since this bug only shows up against
    the real trading calendar (a synthetic frame can't reproduce "which
    calendar dates are/aren't real trading days")."""
    from src.data.universe_panel import (
        _year_partition_files,
        market_cap_at,
        universe_at,
    )

    month_end = datetime.date(2007, 6, 30)
    assert month_end.weekday() == 5, "sanity check: 2007-06-30 must be a Saturday"

    universe = universe_at(month_end)
    assert len(universe) > 0, "sanity check: universe_at(2007-06-30) must be non-empty"
    resolved_snapshot_day = universe["dlycaldt"].dt.date().max()

    market_cap_df, _coverage_log = market_cap_at(month_end)
    assert len(market_cap_df) > 0, "sanity check: market_cap_at(2007-06-30) must be non-empty"

    files = _year_partition_files(month_end.year)
    priced_permnos = market_cap_df["permno"]
    price_rows = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlyprc", "shrout"])
        .filter(pl.col("permno").is_in(priced_permnos.implode()))
        .filter(pl.col("dlycaldt") < pl.lit(resolved_snapshot_day).cast(pl.Datetime("ns")))
        .collect(engine="streaming")
    )
    actual_price_date = (
        price_rows.sort("dlycaldt")
        .group_by("permno", maintain_order=True)
        .last()["dlycaldt"]
        .dt.date()
        .max()
    )

    assert actual_price_date < resolved_snapshot_day, (
        f"market_cap_at(2007-06-30)'s prior-day price date "
        f"({actual_price_date}) must be STRICTLY earlier than "
        f"universe_at(2007-06-30)'s resolved snapshot date "
        f"({resolved_snapshot_day}) -- if they're equal, market_cap_at() "
        f"is filtering on the raw month_end instead of the resolved "
        f"snapshot day, and is silently using a same-day, unlagged price."
    )
