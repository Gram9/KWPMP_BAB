"""Tests for src/estimation/beta_fp.py -- Gate 3
(docs/superpowers/specs/2026-09-08-gate3-beta-estimator-design.md):
the Frazzini-Pedersen (2014) beta estimator, validated via the four
sub-tests in docs/02_validation_gates.md's Gate 3 section.
"""

import datetime

import polars as pl
import pytest

from src.data.chass_loader import MONTHLY_PATH
from src.data.universe_panel import RAW_DATA_DIR
from src.estimation import beta_fp

pytestmark_realdata = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)
# Matches tests/gates/test_gate1_index.py's own pytestmark_chass convention
# -- applied per-test, not module-wide, since most of this file needs only
# the US panel.
pytestmark_chass = pytest.mark.skipif(
    not MONTHLY_PATH.exists(),
    reason="data/raw/CHASS_Data/parquet/monthly.parquet not present",
)


def test_load_beta_config_returns_fp_spec_block():
    cfg = beta_fp._load_beta_config("fp_spec")
    assert cfg["sigma_window_days"] == 252
    assert cfg["sigma_min_obs"] == 120
    assert cfg["rho_window_days"] == 1260
    assert cfg["rho_min_obs"] == 750
    assert cfg["rho_overlap_days"] == 3
    assert cfg["shrinkage_weight"] == pytest.approx(0.6)
    assert cfg["shrinkage_target"] == pytest.approx(1.0)


def test_load_beta_config_defaults_to_fp_spec():
    default_cfg = beta_fp._load_beta_config()
    explicit_cfg = beta_fp._load_beta_config("fp_spec")
    assert default_cfg == explicit_cfg


def test_load_beta_config_raises_on_unknown_variant():
    with pytest.raises(KeyError):
        beta_fp._load_beta_config("not_a_real_variant")


@pytestmark_realdata
def test_daily_log_returns_has_expected_schema():
    result = beta_fp._daily_log_returns(
        datetime.date(2015, 1, 1), datetime.date(2015, 1, 31), leg="us"
    )
    assert set(result.columns) == {"permno", "date", "log_ret"}
    assert result.schema["permno"] == pl.Int64
    assert result.schema["date"] == pl.Date
    assert result.schema["log_ret"] == pl.Float64
    assert result.height > 0


@pytestmark_realdata
def test_daily_log_returns_covers_every_year_in_multi_year_range():
    """Regression test: a range spanning 3+ calendar years must have
    EVERY year in between represented in the output, not just the
    start/end years. years_needed used to be built from
    {start_date.year, end_date.year} alone -- a 2-element set that never
    enumerates years strictly between the endpoints -- so any year
    strictly inside a multi-year range was silently dropped (e.g.
    2015-01-01..2018-12-31 only ever read 2015 and 2018's partitions,
    since _year_partition_files(year) only pulls in `year` and
    `year - 1`, leaving 2016 completely missing with no error). This
    matters a great deal for Task 3's ~5-year rho_window_days=1260
    rolling window, which spans 3+ calendar years for essentially every
    stock."""
    result = beta_fp._daily_log_returns(
        datetime.date(2015, 1, 1), datetime.date(2018, 12, 31), leg="us"
    )
    years_present = sorted(result["date"].dt.year().unique().to_list())
    assert years_present == [2015, 2016, 2017, 2018], (
        f"expected every year 2015-2018 represented, got {years_present} "
        "-- a year strictly between start_date.year and end_date.year "
        "was silently dropped from the file-gathering logic"
    )


@pytestmark_realdata
def test_daily_log_returns_is_log_not_arithmetic():
    """log_ret must be ln(1 + dlyret), not dlyret itself -- these
    genuinely diverge for any nonzero return, so this test would catch
    a copy-paste that forgot the log transform."""
    result = beta_fp._daily_log_returns(
        datetime.date(2015, 1, 1), datetime.date(2015, 1, 31), leg="us"
    )
    # A real return of exactly 0.0 gives log_ret == 0.0 either way, so
    # find a row with a real nonzero return to distinguish the two.
    nonzero = result.filter(pl.col("log_ret") != 0.0)
    assert nonzero.height > 0
    # For small returns, ln(1+r) ~= r, so this alone wouldn't catch a
    # missing log transform -- instead check the exact relationship
    # holds by re-deriving arithmetic return from log_ret and comparing
    # against a fresh independent read of dlyret for the same rows.
    import math

    from src.data.universe_panel import _year_partition_files

    files = _year_partition_files(2015)
    raw = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlyret"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(datetime.date(2015, 1, 1)).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(datetime.date(2015, 1, 31)).cast(pl.Datetime("ns")))
        )
        .collect(engine="streaming")
    )
    sample = nonzero.head(5)
    for row in sample.iter_rows(named=True):
        raw_row = raw.filter(
            (pl.col("permno") == row["permno"])
            & (pl.col("dlycaldt") == pl.lit(row["date"]).cast(pl.Datetime("ns")))
        )
        assert raw_row.height == 1
        expected_log_ret = math.log(1.0 + raw_row["dlyret"][0])
        assert row["log_ret"] == pytest.approx(expected_log_ret, rel=1e-9)


@pytestmark_realdata
def test_daily_log_returns_not_filtered_by_universe_at():
    """A permno excluded from universe_at() for a given month (e.g. not
    on a major exchange, or a REIT) must STILL appear in
    _daily_log_returns()'s output for that month if it has a real
    dlyret row -- this function reads raw trading history, not
    universe-filtered history (spec decision 1)."""
    from src.data import universe_panel

    # Pick a permno present in the raw panel but NOT in universe_at()
    # for the same month -- confirms the raw reader doesn't silently
    # apply universe_at()'s exchange/REIT filters.
    month_start = datetime.date(2015, 6, 1)
    month_end = datetime.date(2015, 6, 30)
    files = universe_panel._year_partition_files(2015)
    raw_permnos = set(
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(month_start).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(month_end).cast(pl.Datetime("ns")))
        )
        .select("permno")
        .unique()
        .collect(engine="streaming")["permno"]
        .to_list()
    )
    universe_permnos = set(
        universe_panel.universe_at(month_end)["permno"].to_list()
    )
    excluded_but_traded = raw_permnos - universe_permnos
    assert len(excluded_but_traded) > 0, (
        "test setup assumption failed -- expected at least one permno "
        "excluded from universe_at() but present in the raw panel for "
        "2015-06; if this fails, the raw panel/universe composition "
        "changed and this test needs a different month"
    )
    sample_permno = next(iter(excluded_but_traded))

    result = beta_fp._daily_log_returns(month_start, month_end, leg="us")
    assert sample_permno in result["permno"].to_list(), (
        f"permno {sample_permno} is excluded from universe_at() for "
        "2015-06 but has real trading history -- _daily_log_returns() "
        "must still include it (raw history, not universe-filtered)"
    )


@pytestmark_realdata
def test_market_log_return_series_has_expected_schema():
    result = beta_fp._market_log_return_series(
        datetime.date(2015, 1, 1), datetime.date(2015, 12, 31),
        leg="us", market_index="vw_uncapped",
    )
    assert set(result.columns) == {"date", "log_ret"}
    assert result.schema["date"] == pl.Date
    assert result.schema["log_ret"] == pl.Float64
    assert result.height > 200  # ~252 trading days in a year, generously bounded


@pytestmark_realdata
@pytest.mark.parametrize(
    "method", ["vw_uncapped", "vw_capped_10pct", "vw_msci_like"]
)
def test_market_log_return_series_is_log_of_build_index(method):
    """log_ret must be ln(1 + index_ret) from
    src.market_index.build.build_index(panel, method) -- the function
    F2b wires _market_log_return_series() to call -- not re-derived some
    other way, and not hardcoded to build_vw_index()/vw_uncapped
    regardless of the `market_index` argument (F2b, docs/03_roadmap.md:
    F2a built vw_capped_10pct/vw_msci_like but this module ignored them
    until now). Parametrized over all three variants so this test would
    fail if the wiring silently ignored `market_index` and always
    delegated to the uncapped path -- a single-method version of this
    test could pass under that exact bug."""
    from src.data import gate_adapters
    from src.market_index import build as market_index_build

    start, end = datetime.date(2015, 1, 1), datetime.date(2015, 3, 31)
    result = beta_fp._market_log_return_series(
        start, end, leg="us", market_index=method
    )

    panel = gate_adapters.us_gate1_panel(start, end)
    index = market_index_build.build_index(panel, method)
    expected = index.with_columns(
        (pl.col("index_ret") + 1.0).log().alias("log_ret")
    ).select(["date", "log_ret"])

    joined = result.join(expected, on="date", how="inner", suffix="_expected")
    assert joined.height == result.height  # every date in result also in expected
    for row in joined.iter_rows(named=True):
        assert row["log_ret"] == pytest.approx(row["log_ret_expected"], rel=1e-9)


@pytestmark_realdata
def test_market_log_return_series_chunked_bit_identical_across_year_boundary():
    """market_index_build.build_index_chunked (2026-09-13, Gate 4:
    _LEG_PANELS[leg](start, end) + build_index() unchunked crashed with a
    Rust memory allocation failure at Gate 4's 56-year scale -- see that
    function's own docstring) is the ONE thing that could shift every
    beta in the project by a tiny, membership-decision-invisible amount
    if its year-boundary carry were wrong. Gate 3's own pinned numbers
    (excluded-name counts, shrinkage values) are NOT sufficient to catch
    this -- this project's own Gate 3 record shows the estimator's
    existing tests are structurally blind to small common-mode/scale/
    misalignment errors (a +50bp/day bias, a 1.5x scale error, and a
    1-day misalignment all passed every existing assertion), so a small
    boundary-carry defect could easily move every beta without ever
    flipping which names land in which leg.

    Direct equivalence on the SERIES ITSELF, not downstream betas: a
    reference reproduction of the pre-chunking logic
    (_LEG_PANELS[leg](start, end) + build_index(), the literal call this
    function used before 2026-09-13) compared bit-for-bit against the
    now-chunked _market_log_return_series, over 2015-2017 -- a real span
    crossing two year boundaries, small enough to compute both ways.
    Exact equality (not pytest.approx), and same row count / same date
    set (an inner join alone could hide a dropped boundary row)."""
    from src.data import gate_adapters
    from src.market_index import build as market_index_build

    start, end = datetime.date(2015, 1, 1), datetime.date(2017, 12, 31)

    chunked = beta_fp._market_log_return_series(
        start, end, leg="us", market_index="vw_uncapped"
    )

    reference_panel = gate_adapters.us_gate1_panel(start, end)
    reference_index = market_index_build.build_index(reference_panel, "vw_uncapped")
    reference = reference_index.with_columns(
        (pl.col("index_ret") + 1.0).log().alias("log_ret")
    ).select(["date", "log_ret"])

    assert chunked.height > 0, "test setup: expected a non-empty real series"
    assert chunked.height == reference.height, (
        f"chunked series has {chunked.height} rows, unchunked reference has "
        f"{reference.height} -- a year boundary is dropping or duplicating "
        "a date"
    )
    assert set(chunked["date"].to_list()) == set(reference["date"].to_list()), (
        "chunked and unchunked date sets differ -- an inner-join comparison "
        "alone would hide this"
    )

    joined = chunked.sort("date").join(
        reference.sort("date"), on="date", how="inner", suffix="_reference"
    )
    max_abs_diff = (
        (joined["log_ret"] - joined["log_ret_reference"]).abs().max()
    )
    assert max_abs_diff == 0.0, (
        f"max|chunked - unchunked| = {max_abs_diff}, expected EXACTLY 0.0 -- "
        "the chunked and unchunked market series must be bit-identical, "
        "not merely close, since this series feeds every beta estimated "
        "against it"
    )


# ---------------------------------------------------------------------------
# F2b: market_index wiring -- proof the parameter is actually READ, not just
# accepted. These are the tests the plan requires before this wiring can be
# trusted: correlation/equality-to-tolerance CANNOT see this defect class
# (this project has retracted two gates for exactly that reason), so every
# assertion here is an ABSOLUTE, non-ratio statistic.
# ---------------------------------------------------------------------------


@pytestmark_chass
def test_canada_sigma_m_differs_between_capped_and_uncapped_index():
    """THE primary proof the wiring works, and the one most likely to
    catch a regression back to the F2a-era hardcoded
    gate1_index.build_vw_index() call. Nortel reached 27.94% of the
    self-built Canadian index on 2000-07-27 (F2a, tests/market_index/
    test_build.py) -- capped to exactly 10%, per config/market_index.yaml.
    Capping a name with far higher volatility than the redistributed
    pool must move the market's own realized volatility (sigma_m), not
    just its composition.

    sigma_m is the right target, not the beta cross-section or a
    correlation between the two index series: sigma_m is a LEVEL (an
    absolute standard deviation in log-return units) that enters every
    single beta as a denominator, and it is exactly what a correlation
    statistic cannot see (rho is scale-invariant, sigma_m is not).

    Measured directly against this implementation, sigma window ending
    2000-08-30 (month_end=2000-08-31, the first month-end after Nortel's
    peak date enters the trailing 252-day window):
        sigma_m(vw_uncapped)     = 0.014273921104199888
        sigma_m(vw_capped_10pct) = 0.012874614225219776
        abs diff                = 0.0013993068789801121 (9.80% relative)
        log_ret[2000-07-27]: uncapped=-0.014554271373932426, capped=-0.0034933104149740563
    RE-PINNED 2026-09-13: the original 2026-09-12 pins
    (sigma_m_uncapped=0.01378518082127453,
    sigma_m_capped=0.012443924215418192) were computed against a
    canada_gate1_panel() call that silently dropped 1996-1999 from this
    window's daily data -- that function's own years_needed only loaded
    the START and END years (a bug, since fixed, of the exact same class
    beta_fp._daily_log_returns's own docstring already documents: "middle
    year silently dropped, not just the endpoints"). Confirmed the
    per-day log_ret on 2000-07-27 itself is UNCHANGED by the fix (that
    single day never depended on the missing years); only the aggregate
    252-day sigma_m shifted once the trailing window drew from the real,
    complete panel instead of a data-starved one. These are the pinned
    anchors below. If this test is reverted to hardcode vw_uncapped for
    both calls (the F2a-era bug), both sigma_m values collapse to the
    SAME number and every assertion below fails.
    """
    start, end = datetime.date(1995, 8, 31), datetime.date(2000, 8, 31)
    month_end = datetime.date(2000, 8, 31)
    cfg = beta_fp._load_beta_config()

    uncapped = beta_fp._market_log_return_series(
        start, end, leg="canada", market_index="vw_uncapped"
    )
    capped = beta_fp._market_log_return_series(
        start, end, leg="canada", market_index="vw_capped_10pct"
    )

    def _sigma_m(series: pl.DataFrame) -> float:
        sigma_start, _rho_start, lookback_end = beta_fp._resolve_window_starts(
            month_end, series, cfg
        )
        window = series.filter(
            (pl.col("date") >= sigma_start) & (pl.col("date") <= lookback_end)
        )
        return window["log_ret"].std()

    sigma_m_uncapped = _sigma_m(uncapped)
    sigma_m_capped = _sigma_m(capped)

    assert sigma_m_uncapped == pytest.approx(0.014273921104199888, rel=1e-6)
    assert sigma_m_capped == pytest.approx(0.012874614225219776, rel=1e-6)
    assert abs(sigma_m_uncapped - sigma_m_capped) > 1e-4, (
        "sigma_m is (nearly) identical between capped and uncapped Canadian "
        "indices -- the market_index parameter is not reaching build_index(), "
        "e.g. _market_log_return_series() is still hardcoded to the uncapped "
        "path regardless of what `market_index` was passed"
    )

    date_2000_07_27 = datetime.date(2000, 7, 27)
    log_ret_uncapped = uncapped.filter(pl.col("date") == date_2000_07_27)[
        "log_ret"
    ].item()
    log_ret_capped = capped.filter(pl.col("date") == date_2000_07_27)[
        "log_ret"
    ].item()
    assert log_ret_uncapped == pytest.approx(-0.014554271373932426, rel=1e-6)
    assert log_ret_capped == pytest.approx(-0.0034933104149740563, rel=1e-6)
    assert abs(log_ret_uncapped - log_ret_capped) > 1e-6, (
        "2000-07-27's index return is unchanged between capped and uncapped "
        "-- Nortel's cap is not being applied on this leg"
    )


@pytestmark_realdata
def test_us_capped_vs_uncapped_is_a_negative_control_not_evidence():
    """NOT proof the wiring works -- the opposite. No US name in this
    project's sample approaches the 10% single-name cap (confirmed:
    measured bit-identical, diff=0.0, on the 2015-06-30 formation
    window), so a US capped-vs-uncapped comparison has ZERO
    discriminating power for this defect class and must never be cited
    as evidence the wiring is correct. Kept explicitly, labelled, so a
    future reader doesn't mistake US equality for a passing proof --
    see test_canada_sigma_m_differs_between_capped_and_uncapped_index
    and test_us_msci_like_sigma_m_differs_from_uncapped for the actual
    proofs (Canada leg-and-index, US index-only respectively)."""
    start, end = datetime.date(2010, 6, 1), datetime.date(2015, 6, 30)
    month_end = datetime.date(2015, 6, 30)
    cfg = beta_fp._load_beta_config()

    uncapped = beta_fp._market_log_return_series(
        start, end, leg="us", market_index="vw_uncapped"
    )
    capped = beta_fp._market_log_return_series(
        start, end, leg="us", market_index="vw_capped_10pct"
    )

    def _sigma_m(series: pl.DataFrame) -> float:
        sigma_start, _rho_start, lookback_end = beta_fp._resolve_window_starts(
            month_end, series, cfg
        )
        window = series.filter(
            (pl.col("date") >= sigma_start) & (pl.col("date") <= lookback_end)
        )
        return window["log_ret"].std()

    assert _sigma_m(uncapped) == pytest.approx(_sigma_m(capped), abs=1e-12), (
        "US capped and uncapped sigma_m diverged -- unexpected given no US "
        "name in this sample approaches the 10% cap; investigate before "
        "treating this as a passing control"
    )


@pytestmark_realdata
def test_us_msci_like_sigma_m_differs_from_uncapped():
    """The real US-side proof the market_index parameter is read: unlike
    the 10% cap (test_us_capped_vs_uncapped_is_a_negative_control_not_evidence),
    the MSCI-like 85%-coverage cutoff (config/market_index.yaml) excludes
    names on EVERY panel regardless of concentration, so it must move
    sigma_m on the US leg too, not just Canada's.

    Measured directly, 2026-09-12, sigma window ending 2015-06-29:
        sigma_m(vw_uncapped)  = 0.007698962492110666
        sigma_m(vw_msci_like) = 0.007621973520639246
        abs diff              = 7.698897147141986e-05
    Smaller than the Canada capped-vs-uncapped gap (as expected -- no
    single US name is anywhere near dominant), but genuinely nonzero and
    reproducible, which is what distinguishes "the parameter is read" from
    "the parameter is ignored"."""
    start, end = datetime.date(2010, 6, 1), datetime.date(2015, 6, 30)
    month_end = datetime.date(2015, 6, 30)
    cfg = beta_fp._load_beta_config()

    uncapped = beta_fp._market_log_return_series(
        start, end, leg="us", market_index="vw_uncapped"
    )
    msci = beta_fp._market_log_return_series(
        start, end, leg="us", market_index="vw_msci_like"
    )

    def _sigma_m(series: pl.DataFrame) -> float:
        sigma_start, _rho_start, lookback_end = beta_fp._resolve_window_starts(
            month_end, series, cfg
        )
        window = series.filter(
            (pl.col("date") >= sigma_start) & (pl.col("date") <= lookback_end)
        )
        return window["log_ret"].std()

    sigma_m_uncapped = _sigma_m(uncapped)
    sigma_m_msci = _sigma_m(msci)

    assert sigma_m_uncapped == pytest.approx(0.007698962492110666, rel=1e-6)
    assert sigma_m_msci == pytest.approx(0.007621973520639246, rel=1e-6)
    assert abs(sigma_m_uncapped - sigma_m_msci) > 1e-6, (
        "sigma_m is identical between vw_uncapped and vw_msci_like on the "
        "US leg -- the market_index parameter is not reaching build_index()"
    )


def test_daily_log_returns_raises_on_canada_leg():
    """The Canadian stock-side beta path is explicitly NOT built (F2b,
    docs/03_roadmap.md) -- _daily_log_returns(leg='canada') must raise
    NotImplementedError naming the gap, never silently return an empty
    or US-substituted frame. No real data needed -- this must raise
    before any file is touched."""
    with pytest.raises(NotImplementedError):
        beta_fp._daily_log_returns(
            datetime.date(2015, 1, 1), datetime.date(2015, 1, 31), leg="canada"
        )


def test_daily_log_returns_raises_on_unknown_leg():
    with pytest.raises(KeyError):
        beta_fp._daily_log_returns(
            datetime.date(2015, 1, 1), datetime.date(2015, 1, 31), leg="mars"
        )


def test_market_log_return_series_raises_on_unknown_leg():
    with pytest.raises(KeyError):
        beta_fp._market_log_return_series(
            datetime.date(2015, 1, 1), datetime.date(2015, 1, 31),
            leg="mars", market_index="vw_uncapped",
        )


# ---------------------------------------------------------------------------
# D-1: estimate_beta_fp() -- the core estimator
# ---------------------------------------------------------------------------


def test_estimate_beta_fp_core_excludes_data_on_or_after_month_end():
    """Direct pin on the single most load-bearing line in this module
    (leakage-auditor finding, 2026-09-10): every window boundary in
    _estimate_beta_fp_core is guarded only indirectly (via
    test_betas_are_point_in_time, reached through the pandas adapter and
    a real-data fixture). This test asserts the strict-boundary property
    directly: contaminating every market/stock observation ON OR AFTER
    month_end with extreme values must not change the output AT ALL,
    while contaminating month_end - 1 day (the last legitimately
    included day) MUST change it. Mutating any `< month_end` boundary in
    the source to `<=` should fail this test immediately."""
    import numpy as np

    rng = np.random.default_rng(55)
    n_days = 1300
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    market_log_ret = rng.normal(0.0, 0.01, n_days)
    true_beta = 1.2
    stock_log_ret = true_beta * market_log_ret + rng.normal(0.0, 0.01, n_days)

    market_returns = pl.DataFrame({"date": dates, "log_ret": market_log_ret})
    daily_returns = pl.DataFrame(
        {"permno": [55555] * n_days, "date": dates, "log_ret": stock_log_ret}
    )
    month_end = dates[-1] + datetime.timedelta(days=1)

    baseline = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)

    # Contaminate month_end itself and every date after it (there are
    # none in this fixture beyond month_end's own implied boundary, so
    # extend the panel with extra future rows carrying extreme values).
    future_dates = [month_end + datetime.timedelta(days=i) for i in range(5)]
    contaminated_market = pl.concat(
        [
            market_returns,
            pl.DataFrame({"date": future_dates, "log_ret": [5.0] * 5}),
        ]
    )
    contaminated_stock = pl.concat(
        [
            daily_returns,
            pl.DataFrame(
                {"permno": [55555] * 5, "date": future_dates, "log_ret": [-5.0] * 5}
            ),
        ]
    )
    result_future_contaminated = beta_fp._estimate_beta_fp_core(
        month_end, contaminated_stock, contaminated_market
    )
    assert result_future_contaminated.equals(baseline), (
        "contaminating data ON OR AFTER month_end changed the output -- "
        "a lookahead boundary is not strictly excluding month_end"
    )

    # Sanity check the test itself isn't vacuous: contaminating the LAST
    # legitimately-included day (month_end - 1 day) MUST change the
    # result, proving this fixture is actually sensitive to its inputs.
    corrupted_market = market_returns.with_columns(
        pl.when(pl.col("date") == dates[-1])
        .then(5.0)
        .otherwise(pl.col("log_ret"))
        .alias("log_ret")
    )
    result_last_day_corrupted = beta_fp._estimate_beta_fp_core(
        month_end, daily_returns, corrupted_market
    )
    assert not result_last_day_corrupted.equals(baseline), (
        "contaminating month_end - 1 day (the last legitimately included "
        "day) did NOT change the output -- this test fixture isn't "
        "actually sensitive to its own inputs, so the equality check "
        "above proves nothing"
    )


def test_estimate_beta_fp_recovers_known_synthetic_beta_exactly():
    """One stock, zero idiosyncratic noise, known beta against a
    synthetic market series -- beta_raw must recover it near-exactly
    (rho == 1.0 and sigma_i/sigma_m == true_beta exactly when there is no
    noise, so beta_raw == true_beta to floating-point precision). Fast,
    deterministic sanity check of the window-slicing/overlap/shrinkage
    mechanics before Test A's fuller noisy synthetic harness (D-3)."""
    import numpy as np

    rng = np.random.default_rng(42)
    n_days = 1300  # clears both sigma (120d) and rho (750d) minimums
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    market_log_ret = rng.normal(0.0, 0.01, n_days)
    true_beta = 1.5
    stock_log_ret = true_beta * market_log_ret  # zero idio noise

    market_returns = pl.DataFrame({"date": dates, "log_ret": market_log_ret})
    daily_returns = pl.DataFrame(
        {"permno": [99999] * n_days, "date": dates, "log_ret": stock_log_ret}
    )

    month_end = dates[-1] + datetime.timedelta(days=1)
    result = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)

    row = result.filter(pl.col("permno") == 99999)
    assert row.height == 1
    assert row["beta_raw"][0] == pytest.approx(true_beta, rel=1e-6)
    cfg = beta_fp._load_beta_config()
    expected_shrunk = (
        cfg["shrinkage_weight"] * true_beta
        + (1 - cfg["shrinkage_weight"]) * cfg["shrinkage_target"]
    )
    assert row["beta_shrunk"][0] == pytest.approx(expected_shrunk, rel=1e-6)


def test_estimate_beta_fp_excludes_permno_with_insufficient_sigma_history():
    """A permno with enough RHO history (full rho window populated, well
    over rho_min_obs=750) but fewer than sigma_min_obs=120 observations
    WITHIN the trailing sigma window specifically must be excluded
    entirely -- never given a partial or null beta. sigma and rho
    minimums are independent gates (spec: 'not one combined check'):
    this permno passes rho's gate and must still be rejected on sigma
    alone, which only holds if the two gates are checked independently
    rather than one combined "has enough total history" check."""
    import numpy as np

    rng = np.random.default_rng(7)
    n_days = 1300
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    market_log_ret = rng.normal(0.0, 0.01, n_days)
    market_returns = pl.DataFrame({"date": dates, "log_ret": market_log_ret})

    # Full 1300-day history (>> rho_min_obs=750, comfortably clears the
    # rho gate), but only every 10th day within the trailing ~252-day
    # sigma window is populated -- roughly 25 obs there, below
    # sigma_min_obs=120. Older history (outside the sigma window, inside
    # the rho window) stays fully populated so rho's own count is
    # unaffected by this sparsification.
    sigma_window_cutoff = n_days - 252
    keep_mask = np.array(
        [True if i < sigma_window_cutoff else (i % 10 == 0) for i in range(n_days)]
    )
    sparse_dates = [d for d, keep in zip(dates, keep_mask) if keep]
    sparse_log_ret = (1.0 * market_log_ret)[keep_mask]
    daily_returns = pl.DataFrame(
        {
            "permno": [88888] * len(sparse_dates),
            "date": sparse_dates,
            "log_ret": sparse_log_ret,
        }
    )

    month_end = dates[-1] + datetime.timedelta(days=1)
    result = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)
    assert 88888 not in result["permno"].to_list()


def test_estimate_beta_fp_shrinkage_uses_configured_weight_and_target():
    """beta_shrunk = shrinkage_weight * beta_raw + (1 - shrinkage_weight)
    * shrinkage_target exactly, using the real fp_spec config values
    (0.6 / 1.0) -- not hardcoded 0.6/0.4 anywhere in the implementation."""
    import numpy as np

    rng = np.random.default_rng(3)
    n_days = 1300
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    market_log_ret = rng.normal(0.0, 0.01, n_days)
    true_beta = 2.0
    stock_log_ret = true_beta * market_log_ret

    market_returns = pl.DataFrame({"date": dates, "log_ret": market_log_ret})
    daily_returns = pl.DataFrame(
        {"permno": [77777] * n_days, "date": dates, "log_ret": stock_log_ret}
    )
    month_end = dates[-1] + datetime.timedelta(days=1)
    result = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)

    row = result.filter(pl.col("permno") == 77777)
    beta_raw = row["beta_raw"][0]
    beta_shrunk = row["beta_shrunk"][0]
    assert beta_shrunk == pytest.approx(0.6 * beta_raw + 0.4 * 1.0, rel=1e-9)


@pytestmark_realdata
def test_estimate_beta_fp_2015_06_30_real_data_smoke():
    """Real-data smoke test: full estimator for the 2015-06-30 formation
    date returns a non-trivial, correctly-typed result with no null
    values in any returned column -- a permno either gets a full row or
    is excluded, never a partial null."""
    daily_returns = beta_fp._daily_log_returns(
        datetime.date(2010, 6, 1), datetime.date(2015, 6, 30), leg="us"
    )
    market_returns = beta_fp._market_log_return_series(
        datetime.date(2010, 6, 1), datetime.date(2015, 6, 30),
        leg="us", market_index="vw_uncapped",
    )
    result = beta_fp.estimate_beta_fp(
        datetime.date(2015, 6, 30), daily_returns, market_returns
    )
    assert result.height > 100  # a real, non-trivial cross-section
    assert set(result.columns) == {
        "permno", "mkt_cap", "sigma_i", "sigma_m", "rho", "beta_raw", "beta_shrunk"
    }
    # NaN as well as null (gate-verifier, second round, 2026-09-10): the
    # original version asserted null_count() == 0 alone, and NaN is NOT
    # null in polars -- so a NaN-poisoned output passed. A zero-variance
    # name (halted or flat security) clears both min-obs gates on count,
    # then yields sigma_i == 0 and a NaN rho from pl.corr, giving
    # beta_shrunk = NaN. Real 2015-06-30 output measures 0 NaN, so this
    # was safe by data rather than by design -- exactly the standing
    # _real_obs()'s own docstring calls unacceptable. The estimator now
    # excludes such names; this asserts the property directly.
    for col in ["mkt_cap", "sigma_i", "sigma_m", "rho", "beta_raw", "beta_shrunk"]:
        assert result[col].null_count() == 0, f"{col} has null values in the output"
        assert not result[col].is_nan().any(), f"{col} has NaN values in the output"


@pytestmark_realdata
def test_estimate_beta_fp_liquidity_gate_excludes_real_names():
    """F2b (docs/03_roadmap.md): proof the spec section 3 liquidity filter
    is actually wired into estimate_beta_fp()'s output, not just present
    in src/portfolio/liquidity.py as an unreferenced module. Measures the
    exclusion directly: build betas through universe_at()/market_cap_at()
    ONLY (bypassing the liquidity gate, reproducing estimate_beta_fp()'s
    own join sequence up to that point), then compare against the real
    estimate_beta_fp() output.

    Measured directly against this implementation, 2026-09-12,
    month_end=2015-06-30: pre-liquidity height=3071, post-liquidity
    height=2950, excluded=121. An absolute count, not a ratio -- if the
    liquidity gate is ever silently disconnected (e.g. estimate_beta_fp()
    stops calling liquidity.liquid_permnos()), pre and post heights
    collapse to the SAME number and this fails immediately.

    Re-measured 2026-09-12 (was 116, off by 5): leakage-auditor found
    estimate_beta_fp() was passing formation_date=lookback_end into
    liquidity.liquid_permnos(), which formation_date_lagged_price()
    then shifts back ANOTHER day -- pricing the sub-penny gate at
    lookback_end - 1 instead of lookback_end, one day earlier than
    mkt_cap for the same permno. Fixed to formation_date=month_end (see
    test_estimate_beta_fp_sub_penny_gate_prices_same_day_as_mkt_cap,
    which pins the corrected date directly). 121 is the exclusion count
    under the CORRECT convention; the 5-name difference is real names
    that only cross the sub-penny floor on one of the two adjacent
    trading days.
    """
    from src.data import universe_panel

    month_end = datetime.date(2015, 6, 30)
    daily_returns = beta_fp._daily_log_returns(
        datetime.date(2010, 6, 1), month_end, leg="us"
    )
    market_returns = beta_fp._market_log_return_series(
        datetime.date(2010, 6, 1), month_end, leg="us", market_index="vw_uncapped"
    )

    core_betas = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)
    eligible = universe_panel.universe_at(month_end)
    mkt_cap_df, _coverage = universe_panel.market_cap_at(month_end)
    pre_liquidity = core_betas.join(
        eligible.select("permno"), on="permno", how="inner"
    ).join(mkt_cap_df, on="permno", how="inner")

    post_liquidity = beta_fp.estimate_beta_fp(month_end, daily_returns, market_returns)

    assert pre_liquidity.height == 3071
    assert post_liquidity.height == 2950
    excluded = pre_liquidity.height - post_liquidity.height
    assert excluded == 121, (
        f"expected exactly 121 names excluded by the liquidity gate at "
        f"2015-06-30, got {excluded} -- either the gate's thresholds "
        "changed (check config/liquidity.yaml) or the gate is not "
        "actually being applied inside estimate_beta_fp()"
    )
    # Every post-liquidity permno must ALSO be in pre-liquidity -- the
    # gate must only REMOVE names, never introduce one that wasn't
    # already universe/market-cap eligible.
    assert set(post_liquidity["permno"].to_list()) <= set(
        pre_liquidity["permno"].to_list()
    )


@pytestmark_realdata
def test_estimate_beta_fp_sub_penny_gate_prices_same_day_as_mkt_cap():
    """leakage-auditor finding (2026-09-12): estimate_beta_fp()'s call
    into liquidity.liquid_permnos() passed formation_date=lookback_end.
    formation_date_lagged_price() internally does shift(1) then filters
    on date == formation_date, so that call actually read the close from
    lookback_end - 1 -- one trading day EARLIER than mkt_cap, which
    universe_panel.market_cap_at() prices at lookback_end itself (the
    trading day strictly before month_end).

    Not a look-ahead leak (look-behind is the safe direction), but a
    correctness bug: two prices anchored to different dates land in the
    SAME output row, and liquidity.py's own docstring claims this
    matches market_cap_at()'s convention "exactly". Pinned directly
    against liquidity.formation_date_lagged_price() itself (not just
    estimate_beta_fp()'s aggregate exclusion count, which cannot
    distinguish the two date conventions): the price the sub-penny gate
    actually used for a sample permno must equal the CORRECT lagged
    price -- formation_date_lagged_price([permno], month_end, ...),
    i.e. the close strictly before month_end, which is lookback_end's
    close -- not formation_date_lagged_price([permno], lookback_end,
    ...), which is lookback_end's own close shifted back ANOTHER day.
    Both are independently computable from the same function; a
    regression back to the off-by-one would make this assert the wrong
    one of the two.
    """
    from src.portfolio import liquidity

    month_end = datetime.date(2015, 6, 30)
    daily_returns = beta_fp._daily_log_returns(
        datetime.date(2010, 6, 1), month_end, leg="us"
    )
    market_returns = beta_fp._market_log_return_series(
        datetime.date(2010, 6, 1), month_end, leg="us", market_index="vw_uncapped"
    )
    betas = beta_fp.estimate_beta_fp(month_end, daily_returns, market_returns)
    assert betas.height > 0
    sample_permnos = betas["permno"].to_list()[:5]

    cfg = beta_fp._load_beta_config()
    _sigma_start, _rho_start, lookback_end = beta_fp._resolve_window_starts(
        month_end, market_returns, cfg
    )
    assert lookback_end < month_end

    correct_price = liquidity.formation_date_lagged_price(
        sample_permnos, month_end, leg="us"
    ).sort("permno")
    off_by_one_price = liquidity.formation_date_lagged_price(
        sample_permnos, lookback_end, leg="us"
    ).sort("permno")

    # Sanity: the two candidate conventions must actually differ on real
    # data, or this test would pass vacuously regardless of which one
    # the production code uses.
    assert not correct_price["lagged_price"].equals(off_by_one_price["lagged_price"]), (
        "correct_price and off_by_one_price are identical on this sample "
        "-- pick different sample permnos/month_end so the two "
        "conventions are actually distinguishable"
    )

    # Behavioral pin, not a source-text match: intercept the exact
    # formation_date estimate_beta_fp() passes into
    # liquidity.liquid_permnos() and assert it resolves (via
    # formation_date_lagged_price's own shift(1)) to lookback_end's
    # close -- i.e. the CORRECT convention -- not lookback_end - 1.
    import inspect

    captured = {}
    real_liquid_permnos = liquidity.liquid_permnos
    real_sig = inspect.signature(real_liquid_permnos)

    def _spy(*args, **kwargs):
        # Bind against the real signature rather than reading kwargs
        # alone (gate-verifier finding, 2026-09-12): formation_date is
        # positional-or-keyword on liquid_permnos, so a caller passing it
        # positionally would make kwargs.get("formation_date") silently
        # read None -- a false positive that fails even on CORRECT code.
        # Binding resolves the argument regardless of calling convention.
        bound = real_sig.bind(*args, **kwargs)
        captured["formation_date"] = bound.arguments.get("formation_date")
        return real_liquid_permnos(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(beta_fp.liquidity, "liquid_permnos", _spy)
        beta_fp.estimate_beta_fp(month_end, daily_returns, market_returns)
    finally:
        monkeypatch.undo()

    assert captured["formation_date"] == month_end, (
        "estimate_beta_fp() passed formation_date="
        f"{captured['formation_date']} into liquidity.liquid_permnos() -- "
        f"expected month_end={month_end}, so the internal shift(1) lands "
        f"on lookback_end={lookback_end} and matches mkt_cap's own "
        "pricing date. Passing lookback_end directly (the off-by-one) "
        "shifts the sub-penny gate's price one trading day earlier than "
        "mkt_cap for the same permno."
    )


@pytestmark_realdata
def test_estimate_beta_fp_diagnostics_reports_exclusions():
    """Spec section 6's required diagnostic: the count of names excluded
    by insufficient sigma/rho history must be visible, not silent -- AND
    must reconcile against what the estimator actually excludes.

    gate-verifier finding (2026-09-10): the original version of this test
    asserted only `n_excluded_sigma >= 0` and `n_excluded_rho >= 0`. Both
    are non-negative BY CONSTRUCTION (they are `len()` of a set
    difference), so the test could not fail under any defect whatsoever.
    Proven, not theorised: injecting halved windows inside the diagnostic
    produced `n_excluded_rho = 5294` against a 3071-name cross-section --
    an exclusion count larger than the entire candidate universe, i.e.
    physically impossible -- and the test stayed green. This is the same
    failure mode as the retracted `our_n_firms`/`ff_n_firms` gate (see
    docs/02_validation_gates.md Gate 2): a real number, printed, that the
    assertion was structurally blind to.

    The reconciliation below ties the diagnostic's counts to the REAL
    cross-section, computed here independently of the diagnostic itself:

      candidates = distinct permnos with any observation < month_end
      core_out   = rows _estimate_beta_fp_core actually returns
      dropped    = candidates - core_out

    Because sigma and rho are INDEPENDENT gates (a permno may fail either
    or both), the two counts must bracket `dropped` two-sidedly:

      max(n_sigma, n_rho) <= dropped <= n_sigma + n_rho

    The lower bound holds because every permno failing a gate is dropped;
    the upper bound because a permno failing BOTH gates is counted twice
    in the sum but dropped once. An INFLATED count breaks the upper
    bound (or the universe-size bound above it); a DEFLATED count breaks
    the lower bound. Measured at 2015-06-30: candidates=5294,
    core_out=3309, dropped=1985, n_excluded_sigma=1328,
    n_excluded_rho=1706 -- so max=1706 <= 1985 <= 3034. All five figures
    measured against the current implementation (2026-09-10), not carried
    over from an earlier run.
    """
    month_end = datetime.date(2015, 6, 30)
    daily_returns = beta_fp._daily_log_returns(
        datetime.date(2010, 6, 1), month_end, leg="us"
    )
    market_returns = beta_fp._market_log_return_series(
        datetime.date(2010, 6, 1), month_end,
        leg="us", market_index="vw_uncapped",
    )
    diag = beta_fp.estimate_beta_fp_diagnostics(
        month_end, daily_returns, market_returns
    )
    assert "n_excluded_sigma" in diag
    assert "n_excluded_rho" in diag
    assert isinstance(diag["n_excluded_sigma"], int)
    assert isinstance(diag["n_excluded_rho"], int)

    n_sigma = diag["n_excluded_sigma"]
    n_rho = diag["n_excluded_rho"]

    # Independently recomputed here -- deliberately NOT read back from the
    # diagnostic, so the reconciliation has an external reference.
    candidates = (
        daily_returns.filter(pl.col("date") < month_end)["permno"].n_unique()
    )
    core_out = beta_fp._estimate_beta_fp_core(
        month_end, daily_returns, market_returns
    ).height
    dropped = candidates - core_out

    print(
        f"\nGate 3 diagnostics reconciliation (2015-06-30): "
        f"candidates={candidates}, core_out={core_out}, dropped={dropped}, "
        f"n_excluded_sigma={n_sigma}, n_excluded_rho={n_rho}"
    )

    # An exclusion count can never exceed the candidate universe. This is
    # the assertion the halved-window injection violated (5294 > 3071).
    assert 0 <= n_sigma <= candidates, (
        f"n_excluded_sigma={n_sigma} is outside [0, {candidates}] -- an "
        f"exclusion count cannot exceed the number of candidate permnos; "
        f"the diagnostic's window boundaries have diverged from the "
        f"estimator's"
    )
    assert 0 <= n_rho <= candidates, (
        f"n_excluded_rho={n_rho} is outside [0, {candidates}] -- an "
        f"exclusion count cannot exceed the number of candidate permnos; "
        f"the diagnostic's window boundaries have diverged from the "
        f"estimator's"
    )

    # Real exclusions exist at this date (1328 / 1706) -- a diagnostic
    # that silently reports zero is broken, not clean.
    assert n_sigma > 0 and n_rho > 0, (
        f"expected real exclusions at 2015-06-30 (measured 1328 sigma / "
        f"1706 rho), got sigma={n_sigma} rho={n_rho} -- a silently-zero "
        f"diagnostic reports nothing while appearing healthy"
    )

    # The two-sided bracket: catches gross inflation and deflation.
    assert max(n_sigma, n_rho) <= dropped, (
        f"max(n_excluded_sigma={n_sigma}, n_excluded_rho={n_rho}) exceeds "
        f"the {dropped} permnos the estimator actually dropped "
        f"(candidates={candidates} - core_out={core_out}) -- the "
        f"diagnostic is over-reporting exclusions relative to the real "
        f"cross-section"
    )
    assert dropped <= n_sigma + n_rho, (
        f"the estimator dropped {dropped} permnos "
        f"(candidates={candidates} - core_out={core_out}) but the "
        f"diagnostic accounts for at most {n_sigma + n_rho} "
        f"(sigma={n_sigma} + rho={n_rho}) -- exclusions are happening "
        f"that the diagnostic is not reporting"
    )

    # THE BRACKET ALONE IS NOT ENOUGH (gate-verifier, second round,
    # 2026-09-10). Both max(a, b) and a + b are SYMMETRIC in their
    # arguments, so transposing the sigma and rho labels satisfies the
    # bracket exactly as well as the correct order does -- verified
    # directly: with sigma=1328/rho=1706 swapped to sigma=1706/rho=1328
    # the bracket still passes. That is an algebraic invariance, not a
    # loose tolerance, so no threshold repairs it. It is also precisely
    # the our_n_firms/ff_n_firms failure class this project already
    # retracted a gate over. Five further defects also passed the
    # bracket alone (halved rho_min_obs, halved sigma_min_obs, doubled
    # windows, and two compensating mis-set minimums).
    #
    # Two additions close it:
    #   1. n_excluded_either (the UNION -- permnos failing either gate)
    #      equals what the estimator drops EXACTLY, not within a
    #      bracket: 1985 == 1985, set symmetric difference 0. An exact
    #      identity has no admissible region to hide in.
    #   2. The two counts pinned individually, which breaks the
    #      symmetry a swap exploits.
    n_either = diag["n_excluded_either"]
    assert n_either == dropped, (
        f"n_excluded_either={n_either} does not equal the {dropped} "
        f"permnos the estimator actually dropped (candidates="
        f"{candidates} - core_out={core_out}). These must match EXACTLY "
        f"-- the union of the two exclusion sets IS the set of dropped "
        f"permnos, so any difference means the diagnostic's gates no "
        f"longer reproduce the estimator's own"
    )
    # Pinned at the measured values (2026-09-10, current implementation),
    # deliberately NOT as inequalities: an exact pin is what fails under
    # a label transposition. If the panel legitimately changes these,
    # re-derive and update them -- do not widen them into a range.
    assert (n_sigma, n_rho) == (1328, 1706), (
        f"expected the measured exclusion counts (sigma=1328, rho=1706) "
        f"at 2015-06-30, got sigma={n_sigma}, rho={n_rho}. If the panel "
        f"changed legitimately, re-derive these pins; a SWAP of the two "
        f"(sigma=1706, rho=1328) satisfies every inequality above and is "
        f"caught only here"
    )


def test_resolve_window_starts_is_the_single_boundary_definition():
    """Blocker 2 (2026-09-10): _estimate_beta_fp_core and
    estimate_beta_fp_diagnostics MUST resolve their window boundaries
    through the same code path.

    Commit d8aacfe's message claimed the diagnostic "shares the same
    trading-day boundary resolution logic" as the core estimator. That
    claim was FALSE -- they were two separately-written copies
    (beta_fp.py:196-205 and :363-373) which had ALREADY diverged twice:
    the short-panel fallback branch differed, and core clipped stock data
    at `<= lookback_end` while the diagnostic clipped at `< month_end`
    with no lookback_end concept at all.

    This test pins the property that makes divergence impossible rather
    than merely unlikely: both functions call _resolve_window_starts, so
    a single change to that helper moves BOTH. Verified by injection --
    perturbing the helper alone changes the core estimator's output rows
    AND the diagnostic's counts together (a two-copy implementation moves
    only one).
    """
    import numpy as np

    rng = np.random.default_rng(909)
    n_days = 1300
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    market_returns = pl.DataFrame(
        {"date": dates, "log_ret": rng.normal(0.0, 0.01, n_days)}
    )
    month_end = dates[-1] + datetime.timedelta(days=1)
    cfg = beta_fp._load_beta_config()

    sigma_start, rho_start, lookback_end = beta_fp._resolve_window_starts(
        month_end, market_returns, cfg
    )

    # The boundary is resolved from the market's own TRADING-day calendar:
    # the Nth date back from lookback_end, not month_end minus N calendar
    # days. With one row per day in this fixture those differ by a known
    # amount, so this pins the trading-day semantics directly.
    assert lookback_end == dates[-1], (
        "lookback_end must be the last market date strictly before "
        f"month_end, got {lookback_end}"
    )
    assert sigma_start == dates[-cfg["sigma_window_days"]]
    assert rho_start == dates[-cfg["rho_window_days"]]

    # Short-panel fallback: fewer market dates than the window length must
    # clamp to the earliest available date, not raise or wrap negatively.
    short_market = market_returns.head(50)
    s_start, r_start, s_lookback = beta_fp._resolve_window_starts(
        month_end, short_market, cfg
    )
    assert s_start == dates[0] and r_start == dates[0]
    assert s_lookback == dates[49]


def test_resolve_window_starts_formation_lag_days_shifts_boundary_back():
    """F2b: `formation_lag_days` (config/runs.yaml, enum {0, 1}) adds a
    further BACKWARD skip on top of the t-1 month-end rule -- spec Section
    10's skip-one-day robustness variant, so the closing prices ending the
    estimation window do not also produce the first held return.

    Default (0) must be a byte-identical no-op -- every existing call site
    and every other test in this file calls _resolve_window_starts with 3
    positional args and must keep passing unmodified. lag=1 must shift the
    window back exactly one TRADING day (from the market's own calendar,
    consistent with this function's existing trading-day semantics), not
    one calendar day and not shorten the window by one observation --
    sigma/rho starts index back from the SHORTENED date list, so the whole
    window slides, it does not shrink.
    """
    rng_dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(1300)]
    market_returns = pl.DataFrame({"date": rng_dates, "log_ret": [0.0] * len(rng_dates)})
    month_end = rng_dates[-1] + datetime.timedelta(days=1)
    cfg = beta_fp._load_beta_config()

    # Default is unchanged -- positional call, no new arg.
    default_sigma, default_rho, default_lookback = beta_fp._resolve_window_starts(
        month_end, market_returns, cfg
    )
    assert default_lookback == rng_dates[-1]

    # formation_lag_days=0 explicitly must equal the positional-call default.
    lag0_sigma, lag0_rho, lag0_lookback = beta_fp._resolve_window_starts(
        month_end, market_returns, cfg, formation_lag_days=0
    )
    assert (lag0_sigma, lag0_rho, lag0_lookback) == (default_sigma, default_rho, default_lookback)

    # lag=1: lookback_end moves back exactly one trading day; sigma/rho
    # starts shift by the same one-day amount (window SLIDES, not shrinks).
    lag1_sigma, lag1_rho, lag1_lookback = beta_fp._resolve_window_starts(
        month_end, market_returns, cfg, formation_lag_days=1
    )
    assert lag1_lookback == rng_dates[-2]
    assert lag1_sigma == rng_dates[-1 - cfg["sigma_window_days"]]
    assert lag1_rho == rng_dates[-1 - cfg["rho_window_days"]]

    # Enum enforcement -- the schema must not be ABLE to express a leak.
    # A negative lag would move the boundary FORWARD; not in {0, 1} at all
    # is a load-time validation error per config/runs.yaml's own comment.
    for bad in (-1, 2, 3):
        with pytest.raises(ValueError):
            beta_fp._resolve_window_starts(
                month_end, market_returns, cfg, formation_lag_days=bad
            )


def test_diagnostics_and_core_agree_when_market_series_ends_early():
    """Blocker 2's demonstrated divergence, as a regression test.

    With the market series ending 3 days BEFORE the stock series, the old
    two-copy implementation diverged concretely: the core estimator
    clipped stock observations at `<= lookback_end` (excluding the 3
    trailing stock-only days) while the diagnostic clipped at
    `< month_end` (admitting them), so the diagnostic reported
    n_excluded_sigma: 0 for a name the estimator was in fact excluding.

    A shared boundary helper makes the two agree by construction.

    NOTE on fixture design -- this test was initially written with a
    stock that had a full 1300-day history and it PASSED against the
    known-divergent implementation, for a reason worth recording. The
    divergence was genuinely present (measured: core admitted 252
    sigma-window observations, the diagnostic 255, exactly the 3-day
    gap), but a name at 252-vs-255 observations clears
    sigma_min_obs=120 either way, so no include/exclude decision flipped,
    `dropped` was 0, both counts were 0, and the reconciliation bracket
    was satisfied. The bracket can only see a divergence that CHANGES a
    membership decision. Fixed by sitting the observation count exactly
    ON the threshold, so the 3 disputed days are decisive: the stock has
    sigma_min_obs - 1 observations inside the true (market-clipped)
    window and sigma_min_obs + 2 if the 3 stock-only trailing days are
    wrongly admitted. The core estimator must exclude it; a diagnostic
    using the wider boundary would report 0 sigma exclusions for a name
    that IS excluded, breaking the lower bound.
    """
    import numpy as np

    rng = np.random.default_rng(313)
    cfg = beta_fp._load_beta_config()
    n_days = 1300
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    market_log_ret = rng.normal(0.0, 0.01, n_days)

    # Market ends 3 days early; the stock keeps trading past it.
    market_returns = pl.DataFrame(
        {"date": dates[:-3], "log_ret": market_log_ret[:-3]}
    )
    month_end = dates[-1] + datetime.timedelta(days=1)

    # Resolve the true (market-clipped) sigma window, then give the stock
    # exactly sigma_min_obs - 1 observations inside it, plus the 3
    # stock-only trailing days the divergent diagnostic would admit.
    market_dates = sorted(d for d in market_returns["date"].to_list() if d < month_end)
    lookback_end = market_dates[-1]
    sigma_start = market_dates[-cfg["sigma_window_days"]]
    in_window = [d for d in dates if sigma_start <= d <= lookback_end]
    trailing_only = [d for d in dates if d > lookback_end and d < month_end]
    assert len(trailing_only) == 3, "fixture expects exactly 3 stock-only days"

    kept_in_window = in_window[: cfg["sigma_min_obs"] - 1]
    # Older history outside the sigma window, so rho's own gate is not
    # what drives the exclusion here.
    older = [d for d in dates if d < sigma_start]
    stock_dates = older + kept_in_window + trailing_only
    stock_log_ret = rng.normal(0.0, 0.01, len(stock_dates))

    daily_returns = pl.DataFrame(
        {
            "permno": [42424] * len(stock_dates),
            "date": stock_dates,
            "log_ret": stock_log_ret,
        }
    )

    core = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)
    diag = beta_fp.estimate_beta_fp_diagnostics(
        month_end, daily_returns, market_returns
    )

    candidates = (
        daily_returns.filter(pl.col("date") < month_end)["permno"].n_unique()
    )
    dropped = candidates - core.height
    n_sigma, n_rho = diag["n_excluded_sigma"], diag["n_excluded_rho"]

    # The stock has sigma_min_obs - 1 real observations in the true
    # window: the estimator must exclude it.
    assert dropped == 1, (
        f"fixture assumption failed -- expected the estimator to exclude "
        f"the name on sigma (it has {cfg['sigma_min_obs'] - 1} "
        f"observations inside the market-clipped window, below "
        f"sigma_min_obs={cfg['sigma_min_obs']}), got dropped={dropped}"
    )
    # A diagnostic clipping at `< month_end` instead of `<= lookback_end`
    # sees sigma_min_obs + 2 observations and reports 0 exclusions.
    assert max(n_sigma, n_rho) <= dropped <= n_sigma + n_rho, (
        f"diagnostic counts (sigma={n_sigma}, rho={n_rho}) do not "
        f"reconcile with the {dropped} permnos the estimator dropped "
        f"(candidates={candidates}, core_out={core.height}) -- the "
        f"diagnostic's window boundaries have diverged from the "
        f"estimator's for a market series ending before the stock series"
    )


def test_overlapping_returns_do_not_bridge_a_trading_gap():
    """leakage-auditor finding (2026-09-10): rolling_sum counts ROWS, not
    trading days, so after a gap in a name's series it sums the resume-day
    return with the last pre-gap returns -- fabricating a "3-day return"
    that actually spans the whole absence, which is then correlated
    against a genuine 3-day market return.

    NOT a lookahead defect: every summand is in the past. It is a
    measurement error, and it is concentrated in exactly the
    halted/illiquid names that populate BAB's low-beta long leg.
    Measured on real 2015-06-30 data before the fix: 88 of 5279 names had
    internal missing trading days, 363 overlapping observations were
    bridged, max |rho| error 0.0421 (permno 88421: 0.1376 -> 0.1797),
    mean 0.0001. On the worst name sigma_i/sigma_m = 6.04, so that rho
    error implies a beta_shrunk error of 0.1525 -- 42% of Test C's entire
    cross-sectional SD (0.3617), on one name. No name's rho_min_obs
    membership changed (delta 0).

    A name absent for a stretch must therefore produce NO overlapping
    return spanning that stretch: the window has to cover
    rho_overlap_days consecutive MARKET trading days, not merely
    rho_overlap_days rows of that name's own series.
    """
    cfg = beta_fp._load_beta_config()
    k = cfg["rho_overlap_days"]

    # 10 consecutive market trading days; the stock misses days 3-6, so
    # its rows jump from index 2 straight to index 7.
    market_dates = [datetime.date(2020, 1, 6) + datetime.timedelta(days=i) for i in range(10)]
    market = pl.DataFrame(
        {"date": market_dates, "log_ret": [0.001] * len(market_dates)}
    )
    present_idx = [0, 1, 2, 7, 8, 9]
    stock = pl.DataFrame(
        {
            "permno": [1] * len(present_idx),
            "date": [market_dates[i] for i in present_idx],
            "log_ret": [0.10, 0.20, 0.30, 0.01, 0.02, 0.03],
        }
    )

    out = beta_fp._overlapping_log_returns_by_permno(stock, k, market_returns=market)
    by_date = dict(zip(out["date"].to_list(), out["log_ret"].to_list()))

    # The last pre-gap date has a genuine 3-consecutive-day window.
    assert by_date[market_dates[2]] == pytest.approx(0.60, abs=1e-9), (
        "the pre-gap window (0.10+0.20+0.30) should be a real overlapping "
        f"return, got {by_date[market_dates[2]]}"
    )
    # The resume date's row-based window would be 0.20+0.30+0.01 = 0.51,
    # spanning the entire absence. It must be null instead.
    assert by_date[market_dates[7]] is None, (
        f"the resume-day overlapping return is "
        f"{by_date[market_dates[7]]} -- rolling_sum bridged the gap "
        f"(0.20+0.30+0.01=0.51 spans days 1-7 of the market calendar, "
        f"not {k} consecutive trading days). It must be null"
    )
    # Once k consecutive present days have accumulated post-gap, real
    # overlapping returns resume.
    assert by_date[market_dates[9]] == pytest.approx(0.06, abs=1e-9), (
        "after the gap, a window of k genuinely consecutive trading days "
        f"(0.01+0.02+0.03) should be valid again, got {by_date[market_dates[9]]}"
    )


def test_estimator_gates_are_nan_aware():
    """Blocker 2 hardening (b), 2026-09-10: is_not_null() does NOT filter
    NaN in polars, and .count() counts NaN as a valid observation. A name
    with 50 real observations and 202 NaNs would therefore clear
    sigma_min_obs=120 on a count of 252, producing a beta from 50 points
    while appearing to satisfy the minimum.

    Production is currently safe by luck of ordering, not by design:
    _daily_log_returns() filters nulls BEFORE the log transform and the
    real 2015-06-30 panel measures 0 NaN / 0 inf. But the estimator must
    not depend on its caller for that. This feeds NaN directly into the
    core estimator and asserts the min-obs gate counts only REAL
    observations -- the name has too few genuine points and must be
    excluded entirely, never given a beta.
    """
    import numpy as np

    rng = np.random.default_rng(4242)
    n_days = 1300
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    market_log_ret = rng.normal(0.0, 0.01, n_days)
    market_returns = pl.DataFrame({"date": dates, "log_ret": market_log_ret})

    # A clean control name that must survive, and a NaN-poisoned name that
    # must not: only every 30th observation inside the sigma window is
    # real (~9 real points, far below sigma_min_obs=120), the rest NaN.
    clean = 1.0 * market_log_ret
    poisoned = (1.0 * market_log_ret).copy()
    sigma_cutoff = n_days - 252
    for i in range(sigma_cutoff, n_days):
        if i % 30 != 0:
            poisoned[i] = np.nan

    # A FLAT name: full observation count, zero variance. It clears both
    # min-obs gates on count, so it is not a min-obs case at all -- it
    # exercises the OUTPUT side (gate-verifier, second round,
    # 2026-09-10). pl.corr on a constant series returns NaN, so before
    # the fix this name came back with sigma_i=0.0, rho=NaN,
    # beta_shrunk=NaN, and the smoke test's null_count()==0 could not
    # see it because NaN is not null in polars. Must be excluded: a beta
    # is not estimable from a series with no variance.
    flat = np.zeros(n_days)

    daily_returns = pl.concat(
        [
            pl.DataFrame({"permno": [1111] * n_days, "date": dates, "log_ret": clean}),
            pl.DataFrame({"permno": [2222] * n_days, "date": dates, "log_ret": poisoned}),
            pl.DataFrame({"permno": [3333] * n_days, "date": dates, "log_ret": flat}),
        ]
    )
    month_end = dates[-1] + datetime.timedelta(days=1)
    result = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)
    permnos = result["permno"].to_list()

    assert 1111 in permnos, (
        "the clean control name was excluded -- this fixture is not "
        "actually exercising the estimator"
    )
    assert 2222 not in permnos, (
        "a name with ~9 real observations inside the sigma window (the "
        "rest NaN) was admitted -- the min-obs gate is counting NaN as a "
        "valid observation, so sigma_min_obs is not enforced"
    )
    assert 3333 not in permnos, (
        "a zero-variance (flat) name was admitted -- it clears both "
        "min-obs gates on observation count, but pl.corr returns NaN on "
        "a constant series, so it comes back with a NaN beta rather than "
        "being excluded"
    )
    # No NaN may reach the output for any admitted name.
    for col in ("sigma_i", "rho", "beta_raw", "beta_shrunk"):
        assert not result[col].is_nan().any(), f"{col} contains NaN in the output"


def test_market_returns_duplicate_dates_do_not_shorten_the_window():
    """Blocker 2 hardening (a), 2026-09-10: the entire trading-day window
    fix rests on market_returns having exactly one row per trading day,
    and nothing asserted it. A duplicated date would silently shorten the
    window -- taking the Nth row back rather than the Nth trading day
    back, so a window nominally 252 trading days long would span fewer.

    Real data satisfies uniqueness today (measured 1280 rows / 1280
    unique dates at 2015-06-30), which is exactly why this could regress
    unnoticed. _resolve_window_starts deduplicates, so a duplicated feed
    resolves the SAME boundary as a clean one.
    """
    import numpy as np

    rng = np.random.default_rng(77)
    n_days = 1300
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    market_returns = pl.DataFrame(
        {"date": dates, "log_ret": rng.normal(0.0, 0.01, n_days)}
    )
    month_end = dates[-1] + datetime.timedelta(days=1)
    cfg = beta_fp._load_beta_config()

    clean = beta_fp._resolve_window_starts(month_end, market_returns, cfg)

    # Duplicate 40 dates inside the sigma window. Row COUNT changes; the
    # set of real trading days does not, so the boundary must not move.
    dupes = market_returns.tail(40)
    duplicated = pl.concat([market_returns, dupes]).sort("date")
    assert duplicated.height == market_returns.height + 40
    with_dupes = beta_fp._resolve_window_starts(month_end, duplicated, cfg)

    assert with_dupes == clean, (
        f"duplicated market dates moved the resolved window boundary "
        f"from {clean} to {with_dupes} -- the window is being measured in "
        f"ROWS rather than distinct trading days, so a duplicated feed "
        f"silently shortens it"
    )


# ---------------------------------------------------------------------------
# D-2: decompose_beta_variance() -- Test D's diagnostic
# ---------------------------------------------------------------------------


def test_decompose_beta_variance_sigma_driven_case():
    """beta_raw constructed as an EXACT linear function of sigma_i alone
    (rho held constant across all rows, zero cross-sectional variance,
    explains nothing) -- r2_sigma must be ~1.0. This is the direction
    spec section 8 actually expects in real data (rho is estimated over
    5y and only sigma over 1y, so dispersion should be sigma-dominated)."""
    import numpy as np

    n = 100
    sigma_i = np.linspace(0.01, 0.05, n)
    rho_constant = np.full(n, 0.5)
    beta_raw = 2.0 * sigma_i

    betas = pl.DataFrame(
        {
            "permno": list(range(n)),
            "mkt_cap": [1.0] * n,
            "sigma_i": sigma_i,
            "sigma_m": [0.02] * n,
            "rho": rho_constant,
            "beta_raw": beta_raw,
            "beta_shrunk": beta_raw,
        }
    )
    result = beta_fp.decompose_beta_variance(betas)
    assert set(result.keys()) == {"r2_sigma", "r2_rho"}
    assert result["r2_sigma"] == pytest.approx(1.0, abs=1e-6)


def test_decompose_beta_variance_rho_driven_case():
    """Mirror of the sigma-driven case: beta_raw constructed as an EXACT
    linear function of rho alone (sigma_i held constant, explains
    nothing) -- r2_rho must be ~1.0. Covers the opposite direction so
    the function isn't only ever tested against a sigma-favoring input,
    which would not catch the two regressors being silently swapped."""
    import numpy as np

    n = 100
    rho = np.linspace(0.1, 0.9, n)
    sigma_constant = np.full(n, 0.02)
    beta_raw = 3.0 * rho

    betas = pl.DataFrame(
        {
            "permno": list(range(n)),
            "mkt_cap": [1.0] * n,
            "sigma_i": sigma_constant,
            "sigma_m": [0.02] * n,
            "rho": rho,
            "beta_raw": beta_raw,
            "beta_shrunk": beta_raw,
        }
    )
    result = beta_fp.decompose_beta_variance(betas)
    assert set(result.keys()) == {"r2_sigma", "r2_rho"}
    assert result["r2_rho"] == pytest.approx(1.0, abs=1e-6)


@pytestmark_realdata
def test_estimate_beta_fp_real_data_cross_sectional_sanity_2015_06_30():
    """Gate 3 Test C: real-data sanity at formation date 2015-06-30.

    Four assertions, in ascending order of how much they actually
    constrain (all bounds and figures below measured against the current
    implementation, 2026-09-10; n=3071):

      1. VW mean beta_shrunk in (0.7, 1.3) -- measured 1.0536. A COARSE
         JUDGEMENT BOUND, not a calibrated one, and specifically NOT
         scale defense: see the long comment at the assertion for what it
         does and does not catch. Mechanically expected near 1 because
         the VW index is the value-weighted average of its own
         constituents, but the realized figure sits at 1.0536, which is
         further from 1.0 than some real defects land.
      2. Cross-sectional SD in (0.15, 0.55) -- measured 0.3617, against
         FP's reported US 0.32. Loose by design (this project's universe
         is a deliberate subset of FP's -- same framing as Gates 1/2's
         correlation-not-exact-equality precedent), and permutation-
         invariant by construction.
      3. Named-anchor ordering -- Apple > Southern Co, Apple > ConEd.
         External truth, no tolerance, 3 names.
      4. Full-cross-section rank correlation against an independently
         computed OLS beta, > 0.70 -- measured 0.8507. Covers the 3068
         names assertions 1-3 cannot see.

    Assertions 1 and 2 are distributional and therefore blind to the
    permno<->beta correspondence; 3 and 4 exist because of that. Uses the
    self-built VW index as both the estimation AND evaluation benchmark
    (spec section 7 -- never vwretd/SPX/external).
    """
    import numpy as np

    daily_returns = beta_fp._daily_log_returns(
        datetime.date(2010, 6, 1), datetime.date(2015, 6, 30), leg="us"
    )
    market_returns = beta_fp._market_log_return_series(
        datetime.date(2010, 6, 1), datetime.date(2015, 6, 30),
        leg="us", market_index="vw_uncapped",
    )
    betas = beta_fp.estimate_beta_fp(
        datetime.date(2015, 6, 30), daily_returns, market_returns
    )
    assert betas.height > 100, "expected a real, non-trivial cross-section"

    total_mkt_cap = betas["mkt_cap"].sum()
    vw_mean_beta = (betas["mkt_cap"] * betas["beta_shrunk"]).sum() / total_mkt_cap
    sd_beta = betas["beta_shrunk"].std()

    print(
        f"\nGate 3 Test C (2015-06-30): n={betas.height}, "
        f"vw_mean_beta={vw_mean_beta:.4f}, sd_beta={sd_beta:.4f}"
    )
    # VW-MEAN IS A WEAK CHECK, and the band below is a JUDGEMENT CALL --
    # not a derived value (Blocker 4, 2026-09-10). Recording this
    # honestly because an earlier version of this comment implied
    # 0.7-1.3 had been derived from something. It was not.
    #
    # What was actually found: the legitimate baseline measures 1.0536
    # under the corrected trading-day windows, which is FURTHER from 1.0
    # than an inverted-weighting defect measures (1.0471). That is not a
    # derivation of any particular width -- it means no tolerance can
    # both accept the real baseline and reject that defect, i.e. VW-mean
    # CANNOT DISCRIMINATE that defect class at all. 0.7-1.3 is a round
    # judgement call about what counts as "obviously broken", chosen
    # after that finding, and it should not be read as calibrated.
    #
    # All values below measured against the CURRENT implementation
    # (2026-09-10, formation 2015-06-30, legitimate baseline 1.0536):
    #   CATCHES: wrong shrink target (0.6536), dropped sigma_m (0.4050),
    #            1.25x scale (1.3170)
    #   MISSES:  inverted weighting (1.0471), equal weighting (1.1216),
    #            double shrinkage (1.0321), 0.80x scale (0.8429)
    # It tolerates a uniform scale error from 0.6644x to 1.2339x
    # (+23% / -34%), so it must NOT be counted as scale defense.
    # (The dropped-sigma_m figure is 0.4050 measured through beta_shrunk,
    # where the 0.4 shrinkage floor dominates once beta_raw collapses to
    # ~0.01; an earlier note citing 0.0084 was measuring un-shrunk
    # beta_raw. Same verdict either way -- comfortably caught.) The rank-correlation check further
    # down is what covers the cross-section; the named anchors are what
    # cover identity.
    assert 0.7 < vw_mean_beta < 1.3, (
        f"value-weighted mean beta_shrunk should be roughly 1.0 by "
        f"construction (coarse judgement bound only, NOT calibrated -- "
        f"see the rank-correlation and named-anchor checks below for the "
        f"real defenses), got {vw_mean_beta:.4f}"
    )
    # Also a loose band, and permutation-INVARIANT by construction: SD is
    # a property of the multiset of betas, so no reordering of the
    # permno<->beta correspondence can move it (measured EXACTLY 0.3617
    # under every scramble tested; 0.3618 under a full rank reversal,
    # differing only because the reversal maps onto ties slightly
    # differently). It bounds dispersion against FP's reported US 0.32
    # and nothing else.
    assert 0.15 < sd_beta < 0.55, (
        f"cross-sectional SD of beta_shrunk should be in a loose band "
        f"around FP's reported 0.32, got {sd_beta:.4f}"
    )

    # gate-verifier finding (2026-09-10): both statistics above are
    # (near-)permutation-invariant -- shuffling the permno<->beta
    # correspondence, including a full RANK REVERSAL (which would invert
    # BAB's long/short legs -- the single most consequential defect
    # possible here), leaves vw_mean_beta and sd_beta inside their bands.
    #
    # Evidence RE-DERIVED against current code (Blocker 4, 2026-09-10 --
    # the figures previously cited here, "baseline 0.9869, sd 0.3191",
    # were measured BEFORE the calendar-vs-trading-day window fix and
    # were never updated; the conclusion survived re-derivation but the
    # numbers did not). Legitimate baseline is vw_mean 1.0536 / sd
    # 0.3617. Under injection: scrambling all 3068 non-anchor names gives
    # vw_mean 1.1026 / sd 0.3617; a full rank reversal with the 3 anchors
    # restored gives vw_mean 1.1344 / sd 0.3618; scrambling the
    # below-median-cap half gives vw_mean 1.0519 / sd 0.3617. Every one
    # of those passes both bands above. SD is invariant by construction
    # (a permutation cannot change a multiset's standard deviation) --
    # exactly, not approximately.
    #
    # Two checks close this. First, a named-anchor IDENTITY check: real
    # companies whose relative beta ordering is a fact about the world,
    # not about this dataset, so it is destroyed by any permutation
    # touching them. Apple (permno 14593, megacap tech, current real
    # beta_shrunk=0.9364) must estimate a HIGHER beta than regulated
    # utilities Southern Co (18411, current real 0.6711) and
    # Consolidated Edison (11404, current real 0.7122) on the same date
    # -- two independent pairs, not one, so the check isn't fragile to
    # either single name's estimate moving slightly with a future data
    # refresh. Second, the full-cross-section rank correlation added
    # below, which covers the other 3068 names the anchors cannot.
    beta_by_permno = dict(
        zip(betas["permno"].to_list(), betas["beta_shrunk"].to_list())
    )
    apple, southern_co, con_ed = 14593, 18411, 11404
    for anchor_permno in (apple, southern_co, con_ed):
        assert anchor_permno in beta_by_permno, (
            f"expected permno {anchor_permno} in the 2015-06-30 cross-section "
            f"for the named-anchor ordering check -- if this fails, the "
            f"underlying panel composition has changed and this test needs "
            f"different anchors, not a removed check"
        )
    assert beta_by_permno[apple] > beta_by_permno[southern_co], (
        f"Apple (beta={beta_by_permno[apple]:.4f}) should have a HIGHER beta "
        f"than Southern Co (beta={beta_by_permno[southern_co]:.4f}, a "
        f"regulated utility) -- this is a fact about the world that a "
        f"permno-to-beta join/permutation bug would destroy"
    )
    assert beta_by_permno[apple] > beta_by_permno[con_ed], (
        f"Apple (beta={beta_by_permno[apple]:.4f}) should have a HIGHER beta "
        f"than Consolidated Edison (beta={beta_by_permno[con_ed]:.4f}, a "
        f"regulated utility) -- this is a fact about the world that a "
        f"permno-to-beta join/permutation bug would destroy"
    )

    # ------------------------------------------------------------------
    # Full-cross-section ordering statistic (Blocker 3, 2026-09-10).
    #
    # The named anchors above are clean and untunable but cover 3 of 3071
    # names. Measured against the current implementation, ALL of these
    # pass every assertion above: scrambling all 3068 non-anchor names;
    # a full rank REVERSAL with just the 3 anchors restored; scrambling
    # the below-median-cap half (1536 names); rank-reversing the
    # bottom-quartile cap (768 names). A complete inversion of BAB's
    # long/short legs across 3071 names passes Test C as long as 3
    # values are put back. The SD assertion in particular is
    # mathematically invariant, not approximately so: permuting a set of
    # numbers cannot change its standard deviation, so sd_beta is
    # EXACTLY 0.3617 under every permutation above.
    #
    # Closing that requires a statistic over the WHOLE cross-section.
    # Compute a second, INDEPENDENT beta per permno here in the test --
    # plain OLS, cov(stock, market)/var(market), never via the estimator
    # under test -- and rank-correlate the two beta vectors. If each beta
    # is attached to the right company, both methods rank Apple high and
    # ConEd low; under any permutation of that correspondence the two
    # rankings decouple and the statistic collapses.
    #
    # The comparison beta uses RHO's own window (rho_window_days=1260,
    # 3-day overlapping returns) rather than the 1-year daily window.
    # Both were measured; the window-matched comparator scores
    # 0.8076-0.8676 across four formation dates versus 0.6952-0.7344 for
    # the 1-year one. The gap is explained by window mismatch exactly as
    # FP's split-window rationale predicts (a high-0.9s correlation would
    # be evidence the split-window construction was NOT working), and
    # the matched comparator leaves more headroom above the injections
    # while isolating the permno<->beta correspondence rather than
    # partly re-measuring the window choice.
    #
    # RANK correlation, not a regression fit, because shrinkage is
    # rank-preserving by construction (bab-methodology: that is why FP
    # keep it) and BAB itself only ever consumes the ranking
    # (z = cross-sectional rank of beta). So a rank statistic tests the
    # estimator at precisely the granularity the strategy uses, without
    # being contaminated by the level compression shrinkage legitimately
    # applies.
    #
    # WHAT THIS CANNOT CATCH, stated explicitly: rank correlation is
    # location- and scale-invariant -- the exact blindness this project
    # has twice been burned by. A units error, a constant bias, or any
    # defect corrupting both beta vectors identically (a bad market
    # series, say) moves them together and leaves this statistic high.
    # It is the CORRESPONDENCE check only. The named anchors remain the
    # external-truth check; the VW-mean band remains the (weak, see
    # above) level check.
    #
    # Threshold 0.70 is a FLOOR with a measured margin, not a derived
    # constant: it sits 0.108 below the worst of four measured formation
    # dates (2011-06-30 0.8676, 2012-12-31 0.8076, 2014-03-31 0.8358,
    # 2015-06-30 0.8507) and 0.332 above the strongest surviving
    # injection. Injection values, all measured against current code at
    # 2015-06-30: scramble all non-anchors 0.0002; full reversal with
    # anchors restored -0.8475; scramble below-median-cap half 0.3115;
    # reverse bottom-quartile cap 0.3676. Any value in (0.3676, 0.8076)
    # would discriminate; 0.70 is chosen inside that band to favour not
    # failing on correct code as the panel drifts.
    cfg = beta_fp._load_beta_config()
    _sigma_start, rho_start, lookback_end = beta_fp._resolve_window_starts(
        datetime.date(2015, 6, 30), market_returns, cfg
    )
    market_overlap = (
        market_returns.filter(
            (pl.col("date") >= rho_start) & (pl.col("date") <= lookback_end)
        )
        .sort("date")
        .with_columns(
            pl.col("log_ret")
            .rolling_sum(window_size=cfg["rho_overlap_days"])
            .alias("m_ret")
        )
        .select(["date", "m_ret"])
        .drop_nulls()
    )
    stock_overlap = (
        daily_returns.filter(
            (pl.col("date") >= rho_start) & (pl.col("date") <= lookback_end)
        )
        .sort(["permno", "date"])
        .with_columns(
            pl.col("log_ret")
            .rolling_sum(window_size=cfg["rho_overlap_days"])
            .over("permno")
            .alias("s_ret")
        )
        .select(["permno", "date", "s_ret"])
        .drop_nulls()
    )
    indep_ols = (
        stock_overlap.join(market_overlap, on="date", how="inner")
        .group_by("permno")
        .agg(
            pl.cov("s_ret", "m_ret").alias("cov_sm"),
            pl.col("m_ret").var().alias("var_m"),
            pl.len().alias("n_obs"),
        )
        .filter(pl.col("n_obs") >= cfg["rho_min_obs"])
        .with_columns((pl.col("cov_sm") / pl.col("var_m")).alias("ols_beta"))
        .select(["permno", "ols_beta"])
    )

    matched = betas.select(["permno", "beta_shrunk"]).join(
        indep_ols, on="permno", how="inner"
    )
    assert matched.height > 0.9 * betas.height, (
        f"only {matched.height} of {betas.height} permnos got an "
        f"independent comparison beta -- the comparator is not covering "
        f"the cross-section, so the rank statistic below would be "
        f"measured on an unrepresentative subset"
    )

    fp_beta = matched["beta_shrunk"].to_numpy()
    ols_beta = matched["ols_beta"].to_numpy()
    rank_fp = matched["beta_shrunk"].rank().to_numpy()
    rank_ols = matched["ols_beta"].rank().to_numpy()
    spearman = float(np.corrcoef(rank_fp, rank_ols)[0, 1])

    # Level relationship: recorded, NOT gated (same precedent as Test D's
    # decompose_beta_variance, spec section 8 -- informational). Expected
    # well below 1.0 because shrinkage compresses beta toward the target;
    # measured slope 0.5197-0.6406 across the four dates above. FP give
    # no reference value for this quantity, so there is nothing to assert
    # against and inventing a threshold would be exactly the
    # round-number-presented-as-derived mistake this session is fixing.
    slope, _intercept = np.polyfit(ols_beta, fp_beta, 1)
    level_r2 = float(np.corrcoef(ols_beta, fp_beta)[0, 1]) ** 2
    print(
        f"Gate 3 Test C rank check (2015-06-30): n={matched.height}, "
        f"spearman={spearman:.4f} | level diagnostic (not gated): "
        f"slope={slope:.4f}, r2={level_r2:.4f}"
    )

    assert spearman > 0.70, (
        f"rank correlation between beta_shrunk and an independently "
        f"computed OLS beta over the same 1260-day/3-day-overlap window "
        f"is only {spearman:.4f} (threshold 0.70; legitimate baseline "
        f"measures 0.8076-0.8676 across four formation dates). A value "
        f"this low means each beta is not reliably attached to the right "
        f"permno -- the distributional assertions above and the 3 named "
        f"anchors cannot see that, since permuting betas among names "
        f"leaves the VW mean and SD unchanged"
    )

    # UPPER bound, and it is not symmetry for its own sake
    # (gate-verifier, second round, 2026-09-10). The comparator is NOT
    # fully independent of the estimator: both are built from the same
    # daily_returns/market_returns through the same
    # _resolve_window_starts, and they share the rho factor BIT-FOR-BIT
    # (verified: max|rho_estimator - rho_comparator| = 0.000e+00,
    # spearman(rho, rho) = 1.000000). Ranks therefore reduce to
    # rank(rho_i * sigma_i^1y) vs rank(rho_i * sigma_i^5y3d) -- the two
    # vectors differ ONLY in which volatility estimate multiplies a
    # shared correlation.
    #
    # The consequence is directional, which is why a floor alone is not
    # evidence: collapsing the two windows into one makes those two
    # volatility estimates the same object and drives the statistic UP,
    # not down. Measured -- correct config 0.8507; sigma_window
    # 252->1260 gives 0.9847; 252->630 gives 0.9205. The correct
    # configuration scores LOWER than the defect. Since window collapse
    # destroys FP's split-window construction (the thing Gate 3 exists
    # to validate), it must fail, and only an upper bound can catch it.
    # 0.95 sits 0.099 above the legitimate maximum (0.8676) and 0.035
    # below the collapse case (0.9847).
    assert spearman < 0.95, (
        f"rank correlation is {spearman:.4f}, ABOVE the 0.95 ceiling. "
        f"The comparator shares this estimator's rho factor exactly and "
        f"differs only in its volatility window, so a near-perfect score "
        f"means the sigma and rho windows have collapsed toward each "
        f"other -- destroying FP's split-window construction (correct "
        f"config measures 0.8076-0.8676; sigma_window 252->1260 measures "
        f"0.9847). This is the one defect class a floor alone cannot see, "
        f"because the statistic RISES under it"
    )

    # Rank correlation is dominated by the bulk of the cross-section, and
    # BAB trades only the tails -- so a permutation confined to a narrow
    # rank band keeps every squared rank displacement small while
    # destroying the legs (gate-verifier, second round). Measured at
    # 2015-06-30: reversing ranks inside the top and bottom 20% scores
    # spearman 0.8140 and inside the 30% tails 0.7377 -- BOTH above the
    # 0.70 floor -- while bottom-decile membership collapses from 83.4%
    # to 16.0% and 0.3%. Scrambling within 3 equal beta groups likewise
    # passes at 0.7488 with 27.4% bottom-decile overlap.
    #
    # So assert membership in the leg BAB actually forms, not just global
    # ordering. BOTTOM decile only, and the reason is measured: the
    # bottom-decile overlap between the estimator and the comparator runs
    # 76.6%-85.6% across four formation dates, but the TOP-decile overlap
    # runs only 43.5%-56.0% legitimately -- high-beta names are where the
    # 1-year and 5-year volatility windows disagree most, so a top-decile
    # threshold tight enough to catch anything would fail on correct
    # code. Floor at 0.65: 0.116 below the worst legitimate date (76.6%)
    # and 0.49 above the 20%-tail-reversal case (16.0%).
    n_decile = matched.height // 10
    fp_order = np.argsort(fp_beta)
    ols_order = np.argsort(ols_beta)
    bottom_overlap = len(
        set(fp_order[:n_decile].tolist()) & set(ols_order[:n_decile].tolist())
    ) / n_decile
    print(
        f"Gate 3 Test C bottom-decile membership overlap: "
        f"{bottom_overlap:.1%} (n_decile={n_decile})"
    )
    assert bottom_overlap > 0.65, (
        f"only {bottom_overlap:.1%} of the estimator's bottom beta decile "
        f"is also in the independent comparator's bottom decile "
        f"(threshold 65%; legitimate baseline 76.6%-85.6% across four "
        f"formation dates). The bottom decile IS BAB's long leg, so a "
        f"disagreement this large means the leg is being formed from the "
        f"wrong names -- a tail-local permutation that leaves the "
        f"whole-cross-section rank correlation above its floor "
        f"(measured: 20%-tail reversal scores spearman 0.8140 but only "
        f"16.0% bottom-decile overlap)"
    )
    # KNOWN REMAINING BLIND SPOT, recorded rather than hidden: reversing
    # ranks inside the top and bottom 10% only scores spearman 0.8452
    # AND 83.4% bottom-decile overlap -- indistinguishable from correct
    # code by every statistic in this test. A permutation that small
    # moves names within, not across, decile boundaries. Catching it
    # needs a genuinely independent beta (a different data source, or a
    # published third-party value), which Gate 3 does not yet have. See
    # docs/02_validation_gates.md Gate 3 for this as an explicit open
    # item.


@pytestmark_realdata
def test_decompose_beta_variance_real_data_2015_06_30():
    """Test D: run the diagnostic on the real 2015-06-30 cross-section.
    No assertion on the values themselves (informational, not a gate
    criterion per spec section 8) -- confirms the function runs against
    real estimate_beta_fp output and returns both figures in [0, 1]."""
    daily_returns = beta_fp._daily_log_returns(
        datetime.date(2010, 6, 1), datetime.date(2015, 6, 30), leg="us"
    )
    market_returns = beta_fp._market_log_return_series(
        datetime.date(2010, 6, 1), datetime.date(2015, 6, 30),
        leg="us", market_index="vw_uncapped",
    )
    betas = beta_fp.estimate_beta_fp(
        datetime.date(2015, 6, 30), daily_returns, market_returns
    )
    result = beta_fp.decompose_beta_variance(betas)
    print(f"\nGate 3 Test D (2015-06-30): r2_sigma={result['r2_sigma']:.4f}, "
          f"r2_rho={result['r2_rho']:.4f}")
    assert 0.0 <= result["r2_sigma"] <= 1.0
    assert 0.0 <= result["r2_rho"] <= 1.0


# ---------------------------------------------------------------------------
# D-3: Gate 3 Test A (synthetic recovery) and Test B (non-synchronous trading)
# ---------------------------------------------------------------------------


def _generate_synthetic_market_and_stocks(
    n_stocks: int,
    n_days: int,
    seed: int,
    idio_std: float = 0.012,
    start_date: datetime.date = datetime.date(2005, 1, 1),
):
    """Shared synthetic-data generator for Test A and Test B: one market
    factor (i.i.d. daily log returns) plus n_stocks synthetic names, each
    with its own known true beta drawn from a realistic range and real
    idiosyncratic noise (not zero -- exercises the estimator's
    noise-handling, not just window-slicing mechanics).

    Returns (dates, market_log_ret: np.ndarray, true_betas: np.ndarray,
    stock_log_ret: np.ndarray of shape (n_days, n_stocks)).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = [start_date + datetime.timedelta(days=i) for i in range(n_days)]
    market_log_ret = rng.normal(0.0, 0.01, n_days)
    true_betas = rng.uniform(0.3, 2.0, n_stocks)
    idio = rng.normal(0.0, idio_std, (n_days, n_stocks))
    stock_log_ret = np.outer(market_log_ret, true_betas) + idio
    return dates, market_log_ret, true_betas, stock_log_ret


def test_estimate_beta_fp_recovers_synthetic_betas_with_noise():
    """Gate 3 Test A: the single most valuable test in this project
    (per docs/03_roadmap.md) -- the only check capable of catching an
    alignment/off-by-one bug that real data hides completely, because on
    real data every estimator produces plausible-looking output. 500
    synthetic stocks, each own known beta, WITH idiosyncratic noise (not
    the zero-noise sanity check above). Recovered beta_raw must
    correlate strongly with the true, known betas."""
    import numpy as np

    n_stocks, n_days = 500, 1300  # clears both sigma(120d) and rho(750d) minimums
    dates, market_log_ret, true_betas, stock_log_ret = _generate_synthetic_market_and_stocks(
        n_stocks, n_days, seed=100
    )

    market_returns = pl.DataFrame({"date": dates, "log_ret": market_log_ret})
    daily_returns = pl.concat(
        [
            pl.DataFrame(
                {
                    "permno": [10000 + i] * n_days,
                    "date": dates,
                    "log_ret": stock_log_ret[:, i],
                }
            )
            for i in range(n_stocks)
        ]
    )

    month_end = dates[-1] + datetime.timedelta(days=1)
    result = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)

    assert result.height == n_stocks, "every synthetic name should clear both minimums"
    result = result.sort("permno")
    recovered = result["beta_raw"].to_numpy()

    corr = np.corrcoef(recovered, true_betas)[0, 1]
    assert corr > 0.9, f"estimator recovers true betas at only corr={corr:.3f}"
    mean_abs_error = np.mean(np.abs(recovered - true_betas))
    assert mean_abs_error < 0.12, (
        f"mean abs error {mean_abs_error:.3f} too large for idio_std=0.012 "
        f"noise over n_days={n_days}"
    )
    # Correlation and MAE alone are both compatible with a uniform SCALE
    # error (gate-verifier finding, 2026-09-10): a mismatched overlap
    # window between the stock and market rolling-sum paths in
    # _estimate_beta_fp_core produced betas a uniform ~22% too low
    # (mean ratio 0.7831) while corr stayed 0.9881 and MAE stayed inside
    # the old 0.25 threshold. Ratio-of-means is NOT scale-invariant and
    # anchors the recovered betas' overall LEVEL against the true one.
    mean_ratio = np.mean(recovered / true_betas)
    assert mean_ratio == pytest.approx(1.0, abs=0.05), (
        f"recovered betas are systematically {mean_ratio:.3f}x true betas -- "
        f"a uniform scale error (e.g. a stock/market overlap-window "
        f"mismatch) would show here even though corr/MAE alone can miss it"
    )


def test_estimate_beta_fp_nonsynchronous_trading_correction():
    """Gate 3 Test B: FP's stated rationale for the 3-day overlapping rho
    construction is non-synchronous trading -- a name that doesn't trade
    every day has its daily return correlation biased DOWNWARD against a
    market series that DOES move every day, because on the stale day the
    stock's "return" is artificially 0 while the market moved. The
    overlapping multi-day window partially absorbs a stale day into a
    window that also captures the eventual catch-up move.

    Demonstrated here, not assumed: build a minimal 1-day-OLS beta
    (test-only baseline, not a first-class estimator) for a subset of
    names with artificial stale pricing (repeated log_ret=0.0 on ~30% of
    days, uncorrelated with the true return-generating process), and
    assert (1) naive 1-day-OLS beta is biased downward vs. true beta for
    those names, (2) estimate_beta_fp's rho-based estimate is closer to
    true beta than the naive OLS estimate, for those same names.
    """
    import numpy as np

    n_stocks, n_days = 50, 1300
    dates, market_log_ret, true_betas, stock_log_ret = _generate_synthetic_market_and_stocks(
        n_stocks, n_days, seed=200, idio_std=0.005  # low idio noise so the
        # staleness effect dominates rather than being swamped by noise
    )

    # Stale-price a subset of names: on ~30% of days (deterministic,
    # every 3rd day so it's reproducible), that name's price does not
    # move (log_ret forced to 0.0) and its TRUE return is DEFERRED --
    # carried forward via log-additivity into the next day's observed
    # return, not discarded. This is what non-synchronous trading
    # actually looks like: the stock doesn't trade, then "catches up"
    # on its next print. (An earlier version of this test discarded the
    # stale day's return entirely rather than deferring it -- that
    # corrupts sigma_i identically to how it corrupts the naive OLS
    # covariance, since both are built from the same 1-day series, so
    # neither the OLS baseline nor FP's own sigma_i/sigma_m ratio could
    # be distinguished; only a DEFERRED-return injection isolates the
    # effect the overlapping-window construction actually corrects for.)
    stale_permnos = list(range(10))  # first 10 of 50 names
    stale_log_ret = stock_log_ret.copy()
    stale_day_mask = np.array([i % 3 == 0 for i in range(n_days)])
    for stock_idx in stale_permnos:
        carried = 0.0
        for t in range(n_days):
            if stale_day_mask[t] and t < n_days - 1:
                carried += stale_log_ret[t, stock_idx]
                stale_log_ret[t, stock_idx] = 0.0
            else:
                stale_log_ret[t, stock_idx] += carried
                carried = 0.0

    market_returns = pl.DataFrame({"date": dates, "log_ret": market_log_ret})
    daily_returns = pl.concat(
        [
            pl.DataFrame(
                {
                    "permno": [20000 + i] * n_days,
                    "date": dates,
                    "log_ret": stale_log_ret[:, i],
                }
            )
            for i in range(n_stocks)
        ]
    )

    month_end = dates[-1] + datetime.timedelta(days=1)
    fp_result = beta_fp._estimate_beta_fp_core(month_end, daily_returns, market_returns)
    fp_result = fp_result.sort("permno")
    assert fp_result.height == n_stocks

    # Minimal 1-day-OLS baseline, test-only: beta = cov(stock, market) / var(market)
    # over the same trailing sigma_window_days window as sigma_i uses.
    cfg = beta_fp._load_beta_config()
    sigma_window_start = month_end - datetime.timedelta(days=cfg["sigma_window_days"])
    window_mask = np.array(
        [d >= sigma_window_start and d < month_end for d in dates]
    )
    market_window = market_log_ret[window_mask]
    ols_betas = np.array(
        [
            np.cov(stale_log_ret[window_mask, i], market_window)[0, 1] / np.var(market_window)
            for i in range(n_stocks)
        ]
    )

    stale_idx = np.array(stale_permnos)
    nonstale_idx = np.array([i for i in range(n_stocks) if i not in stale_permnos])

    ols_stale_bias = np.mean(ols_betas[stale_idx] - true_betas[stale_idx])
    ols_nonstale_bias = np.mean(ols_betas[nonstale_idx] - true_betas[nonstale_idx])
    assert ols_stale_bias < -0.05, (
        f"naive OLS beta should be biased DOWNWARD for stale-priced names, "
        f"got mean bias {ols_stale_bias:.3f}"
    )
    assert ols_stale_bias < ols_nonstale_bias, (
        "staleness-induced downward bias should be much worse for the "
        "stale subset than for normally-trading names"
    )

    fp_beta_raw = fp_result["beta_raw"].to_numpy()
    fp_stale_error = np.mean(np.abs(fp_beta_raw[stale_idx] - true_betas[stale_idx]))
    ols_stale_error = np.mean(np.abs(ols_betas[stale_idx] - true_betas[stale_idx]))
    assert fp_stale_error < ols_stale_error, (
        f"FP's overlap-corrected estimate (mean abs error {fp_stale_error:.3f}) "
        f"should be closer to true beta than naive 1-day OLS "
        f"(mean abs error {ols_stale_error:.3f}) for stale-priced names"
    )

    # gate-verifier finding (2026-09-10): the comparison above changes
    # TWO things at once vs. the OLS baseline -- the overlap width (3d
    # vs 1d) AND the estimation window length (rho_window_days=1260 vs
    # sigma_window_days=252). FP could win purely because it uses 5x
    # more history, with the overlap contributing nothing -- confirmed
    # this really happens: setting rho_overlap_days=1 (overlap OFF,
    # window UNCHANGED at 1260 days) still beats the OLS baseline
    # (mae 0.40 vs 0.43), so the assertion above alone cannot attribute
    # the win to the overlap mechanism specifically. Isolate it: hold
    # the rho window fixed, vary ONLY the overlap, via a config
    # override (monkeypatch on _load_beta_config, not a real yaml
    # change -- this is test-local isolation, not a new estimator
    # variant).
    import copy

    from src.estimation import beta_fp as beta_fp_module

    real_load_beta_config = beta_fp_module._load_beta_config
    no_overlap_cfg = copy.deepcopy(real_load_beta_config("fp_spec"))
    no_overlap_cfg["rho_overlap_days"] = 1

    original = beta_fp_module._load_beta_config
    try:
        beta_fp_module._load_beta_config = lambda variant="fp_spec": no_overlap_cfg
        fp_no_overlap_result = beta_fp_module._estimate_beta_fp_core(
            month_end, daily_returns, market_returns
        ).sort("permno")
    finally:
        beta_fp_module._load_beta_config = original

    fp_no_overlap_beta_raw = fp_no_overlap_result["beta_raw"].to_numpy()
    fp_no_overlap_stale_error = np.mean(
        np.abs(fp_no_overlap_beta_raw[stale_idx] - true_betas[stale_idx])
    )
    assert fp_stale_error < fp_no_overlap_stale_error, (
        f"with the rho WINDOW held fixed at 1260 days and only the "
        f"OVERLAP varied, the 3-day overlap (mae {fp_stale_error:.3f}) "
        f"should beat no overlap / 1-day (mae {fp_no_overlap_stale_error:.3f}) "
        f"for stale-priced names -- this isolates the overlap mechanism's "
        f"own contribution from the window-length confound in the OLS "
        f"comparison above"
    )
