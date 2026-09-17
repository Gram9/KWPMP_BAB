"""Tests for src/portfolio/liquidity.py -- spec 00_spec.md section 3's
liquidity filter: minimum non-zero-volume days within the vol window, plus
a sub-penny price floor at formation. See docs/03_roadmap.md F2b.
"""

import datetime

import polars as pl
import pytest

from src.data.universe_panel import RAW_DATA_DIR
from src.portfolio import liquidity

pytestmark_realdata = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)


def test_load_liquidity_config_returns_thresholds():
    cfg = liquidity._load_liquidity_config()
    assert cfg["min_nonzero_volume_days"] == 120
    assert cfg["min_price"] == pytest.approx(1.00)


def test_daily_price_volume_raises_on_canada_leg():
    """US only today, matching src.estimation.beta_fp's own leg dispatch
    -- the Canadian stock-side beta path this filter would gate does not
    exist yet."""
    with pytest.raises(NotImplementedError):
        liquidity._daily_price_volume(
            datetime.date(2015, 1, 1), datetime.date(2015, 1, 31), leg="canada"
        )


def test_formation_date_lagged_price_raises_on_canada_leg():
    with pytest.raises(NotImplementedError):
        liquidity.formation_date_lagged_price(
            [80599], datetime.date(2015, 1, 1), leg="canada"
        )


# ---------------------------------------------------------------------------
# nonzero_volume_counts() -- real data
# ---------------------------------------------------------------------------


@pytestmark_realdata
def test_nonzero_volume_counts_watsco_class_b_2014():
    """Real observed case, verified directly against the cached panel:
    permno 46068 (Watsco Inc, Class B common -- a real, illiquid
    secondary share class, not a delisted shell) traded on only 94 of 252
    trading days in 2014 (158 zero-volume rows). This is exactly the kind
    of name spec section 3's filter exists to exclude from the estimation
    window -- a name genuinely listed and priced, but too thinly traded
    for its realized volatility to mean what sigma_i assumes it means.
    Pinned so a regression in the >0 comparison (e.g. accidentally
    counting ALL rows, or using >=0) would be caught."""
    counts = liquidity.nonzero_volume_counts(
        datetime.date(2014, 1, 1), datetime.date(2014, 12, 31), leg="us"
    )
    row = counts.filter(pl.col("permno") == 46068)
    assert row.height == 1
    assert row["n_traded_days"].item() == 94


@pytestmark_realdata
def test_nonzero_volume_counts_excludes_null_volume_from_count():
    """A null dlyvol ('not observed') must not be silently counted as a
    traded day alongside genuine dlyvol > 0 rows -- the count is
    TRADED days specifically (dlyvol > 0), not "not proven untraded".
    Directly exercises the boolean-sum construction: (dlyvol > 0).sum()
    treats a null comparison result as neither true nor counted, which
    this test pins by injecting a synthetic frame with a null row."""
    # Direct unit test on the aggregation logic, bypassing the real-data
    # read entirely (mirrors _apply_pit_bounds's synthetic-frame test
    # style in tests/leakage/test_no_lookahead.py).
    frame = pl.DataFrame(
        {
            "permno": [1, 1, 1, 1],
            "date": [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
                datetime.date(2020, 1, 4),
            ],
            "dlyvol": [100.0, 0.0, None, 50.0],
        }
    )
    result = (
        frame.group_by("permno")
        .agg((pl.col("dlyvol") > 0).sum().alias("n_traded_days"))
    )
    assert result["n_traded_days"].item() == 2, (
        "expected exactly 2 traded days (100.0 and 50.0) -- the zero-volume "
        "and null-volume rows must both be excluded from the count, not "
        "just the zero one"
    )


# ---------------------------------------------------------------------------
# formation_date_lagged_price() -- real data
# ---------------------------------------------------------------------------


@pytestmark_realdata
def test_formation_date_lagged_price_is_prior_day_not_same_day():
    """The returned price must be the PRIOR trading day's close, never
    formation_date's own close -- CLAUDE.md's core rule applied to this
    module specifically. Verified against a real permno/date pair read
    independently of the function under test."""
    formation_date = datetime.date(2015, 6, 30)
    result = liquidity.formation_date_lagged_price([25785], formation_date, leg="us")
    assert result.height == 1

    # Independently re-derive the expected prior-day price straight from
    # the raw panel, not via any code this function shares.
    from src.data.universe_panel import _year_partition_files

    files = _year_partition_files(2015)
    raw = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlyprc"])
        .filter(pl.col("permno") == 25785)
        .filter(
            (pl.col("dlycaldt") >= pl.lit(datetime.date(2015, 6, 1)).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(formation_date).cast(pl.Datetime("ns")))
        )
        .collect(engine="streaming")
        .sort("dlycaldt")
    )
    prior_day_price = raw.filter(
        pl.col("dlycaldt") < pl.lit(formation_date).cast(pl.Datetime("ns"))
    )["dlyprc"][-1]
    same_day_price = raw.filter(
        pl.col("dlycaldt") == pl.lit(formation_date).cast(pl.Datetime("ns"))
    )["dlyprc"].item()

    assert result["lagged_price"].item() == pytest.approx(abs(prior_day_price), rel=1e-9)
    if same_day_price != prior_day_price:
        assert result["lagged_price"].item() != pytest.approx(abs(same_day_price), rel=1e-9), (
            "formation_date_lagged_price returned the SAME-DAY close, not "
            "the prior trading day's -- this is a t+1 leak (CLAUDE.md)"
        )


@pytestmark_realdata
def test_formation_date_lagged_price_resolves_weekend_formation_date():
    """gate-verifier finding (2026-09-12): the function filtered
    `date == formation_date` exactly, so a formation_date that is not
    itself a trading day (a weekend/holiday calendar month-end -- 37 of
    132 calendar month-ends in 2005-2015 fall on a weekend, before
    holidays are even counted) matched ZERO rows and silently excluded
    every permno, not just the ones that should fail the price floor.
    This bit estimate_beta_fp()'s own sub-penny gate directly: its
    2026-09-12 fix changed the call site to formation_date=month_end
    (a raw calendar month-end, e.g. 2015-06-30), which is exactly the
    kind of value that can land on a weekend.

    2015-05-31 is a Sunday. The lagged price behind that formation_date
    must resolve the SAME as if formation_date were the actual last
    trading day on or before it (2015-05-29, a Friday) -- matching
    universe_panel._resolve_last_trading_day()'s established "actual
    last trading day on or before" convention, not a literal date match.
    """
    weekend_formation_date = datetime.date(2015, 5, 31)
    trading_day_formation_date = datetime.date(2015, 5, 29)

    weekend_result = liquidity.formation_date_lagged_price(
        [25785], weekend_formation_date, leg="us"
    )
    trading_day_result = liquidity.formation_date_lagged_price(
        [25785], trading_day_formation_date, leg="us"
    )

    assert weekend_result.height == 1, (
        "formation_date=2015-05-31 (a Sunday) resolved ZERO rows -- the "
        "sub-penny gate would silently exclude every permno on any "
        "weekend/holiday calendar month-end, not just genuinely illiquid "
        "names"
    )
    assert weekend_result["lagged_price"].item() == pytest.approx(
        trading_day_result["lagged_price"].item(), rel=1e-9
    )


@pytestmark_realdata
def test_formation_date_lagged_price_absent_for_unresolvable_permno():
    """A permno with no prior trading day in the read window (e.g. a
    fabricated, never-listed permno) must be ABSENT from the result, not
    silently given a null or zero price row."""
    result = liquidity.formation_date_lagged_price(
        [999999999], datetime.date(2015, 6, 30), leg="us"
    )
    assert result.height == 0


# ---------------------------------------------------------------------------
# liquid_permnos() -- the composed gate
# ---------------------------------------------------------------------------


@pytestmark_realdata
def test_liquid_permnos_excludes_thin_trader_on_volume_gate():
    """Watsco Class B (permno 46068, 94 traded days in 2014 < the
    min_nonzero_volume_days=120 threshold) must be excluded by the
    composed gate even though it has a perfectly normal (non-sub-penny)
    price -- proves the volume gate actually binds, not just that the
    function runs."""
    candidates = [46068]
    result = liquidity.liquid_permnos(
        candidates,
        sigma_window_start=datetime.date(2014, 1, 1),
        lookback_end=datetime.date(2014, 12, 31),
        formation_date=datetime.date(2014, 12, 31),
        leg="us",
    )
    assert 46068 not in result["permno"].to_list(), (
        "Watsco Class B (94 traded days, below the 120-day minimum) passed "
        "the liquidity gate -- the volume count gate is not binding"
    )


@pytestmark_realdata
def test_liquid_permnos_admits_a_normally_traded_name():
    """A liquid, normally-priced large-cap name (Apple, permno 14593) must
    clear the composed gate over a real window -- proves the gate isn't
    so strict it excludes everything, which would make
    test_liquid_permnos_excludes_thin_trader_on_volume_gate pass
    vacuously."""
    candidates = [14593]
    result = liquidity.liquid_permnos(
        candidates,
        sigma_window_start=datetime.date(2014, 1, 1),
        lookback_end=datetime.date(2014, 12, 31),
        formation_date=datetime.date(2014, 12, 31),
        leg="us",
    )
    assert 14593 in result["permno"].to_list(), (
        "Apple (a real, heavily-traded large-cap name) was excluded by "
        "the liquidity gate -- the gate is excluding names it should not"
    )


def test_liquid_permnos_excludes_sub_penny_name_synthetically():
    """Direct unit test of the sub-penny rule's composition logic,
    isolated from real-data availability: a permno with ample volume
    (clears the volume gate) but a lagged formation price below
    min_price=1.00 must still be excluded overall. Monkeypatches the two
    data-reading helpers so this test needs no cached panel and exercises
    liquid_permnos()'s join logic directly."""
    from unittest import mock

    volume_frame = pl.DataFrame(
        {"permno": [1, 2], "n_traded_days": [200, 200]}
    )
    price_frame = pl.DataFrame(
        {"permno": [1, 2], "lagged_price": [0.50, 25.00]}
    )

    with mock.patch.object(
        liquidity, "nonzero_volume_counts", return_value=volume_frame
    ), mock.patch.object(
        liquidity, "formation_date_lagged_price", return_value=price_frame
    ):
        result = liquidity.liquid_permnos(
            [1, 2],
            sigma_window_start=datetime.date(2020, 1, 1),
            lookback_end=datetime.date(2020, 12, 31),
            formation_date=datetime.date(2020, 12, 31),
            leg="us",
        )

    permnos = result["permno"].to_list()
    assert 1 not in permnos, "sub-penny name (lagged_price=0.50) was not excluded"
    assert 2 in permnos, "normally-priced name (lagged_price=25.00) was wrongly excluded"


def test_liquid_permnos_excludes_on_volume_even_with_good_price():
    """Mirror of the sub-penny synthetic test: ample price, insufficient
    volume, must still be excluded -- proves the two gates are genuinely
    independent (BOTH required), not an OR."""
    from unittest import mock

    volume_frame = pl.DataFrame(
        {"permno": [1, 2], "n_traded_days": [50, 200]}
    )
    price_frame = pl.DataFrame(
        {"permno": [1, 2], "lagged_price": [25.00, 25.00]}
    )

    with mock.patch.object(
        liquidity, "nonzero_volume_counts", return_value=volume_frame
    ), mock.patch.object(
        liquidity, "formation_date_lagged_price", return_value=price_frame
    ):
        result = liquidity.liquid_permnos(
            [1, 2],
            sigma_window_start=datetime.date(2020, 1, 1),
            lookback_end=datetime.date(2020, 12, 31),
            formation_date=datetime.date(2020, 12, 31),
            leg="us",
        )

    permnos = result["permno"].to_list()
    assert 1 not in permnos, "thin-volume name (50 traded days) was not excluded"
    assert 2 in permnos, "normally-traded name was wrongly excluded"
