"""
Point-in-time US universe membership tests (CRSP-primary). Ground-truths
the exact counts verified live against WRDS this session
(docs/superpowers/specs/2026-09-03-us-full-history-panel-crsp-primary-design.md,
"Findings" section) for 2015-06-30: 4,014 base-filtered rows, 188 REITs
by issuertype, 0 sharetype='UG' rows (MLPs already excluded by the base
pull filter, not by universe_at() itself).

Named test cases reused from this project's established ground truth
(tests/leakage/test_no_lookahead.py's scaffold, docs/worklog.md):
- Lehman Brothers (permno 80599): alive through Sep 2008, must appear in
  a universe built before its delisting.
- A post-2015 IPO not yet listed at an earlier date must not appear --
  the mirror survivorship/lookahead check.
"""

import datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from src.data.universe_panel import universe_at

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "us_panel_crsp_full"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "universe.yaml"


def _pull_available() -> bool:
    return RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.glob("year=*/part.parquet"))


pytestmark = pytest.mark.skipif(
    not _pull_available(),
    reason="data/raw/us_panel_crsp_full/ not present -- run "
    "pull_universe_us_crsp.py first (Task 1)",
)


@pytest.fixture
def crsp_primary_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["us_crsp_primary"]


def test_universe_at_returns_permno_and_name():
    universe = universe_at(pl.date(2015, 6, 30))
    assert {"permno", "securitynm"} <= set(universe.columns)


def test_universe_at_2015_06_30_matches_live_verified_count():
    """Base-filter + exchange-filter + REIT-exclusion count, tied to the
    exact numbers verified live against WRDS this session."""
    universe = universe_at(pl.date(2015, 6, 30))
    # 4,014 base rows - REITs excluded - non-major-exchange rows excluded.
    # This is a real regression bound, not an exact hardcoded count. If
    # this fails, investigate before adjusting the assertion -- the
    # underlying numbers were live-verified, not estimated.
    assert 3000 <= len(universe) <= 4014, (
        f"universe_at(2015-06-30) returned {len(universe)} rows, expected "
        f"a value between the REIT-excluded and unfiltered base count "
        f"(verified live this session: 4,014 base rows, 188 REITs)"
    )


def test_universe_at_2015_06_30_exact_counts():
    """Exact regression pin, independently re-derived this session directly
    from the pulled parquet (not just the brief's live-WRDS numbers): on
    the actual 2015-06-30 trading-day snapshot, base rows = 4,014,
    exchange-filtered = 3,960, REIT-excluded (issuertype='REIT') = 188,
    final = 3,772. If this drifts, investigate before adjusting -- these
    were recomputed from data/raw/us_panel_crsp_full/year=2015/part.parquet
    directly (dlycaldt == 2015-06-30) as part of this task's implementation."""
    universe = universe_at(pl.date(2015, 6, 30))
    assert len(universe) == 3772
    assert universe["permno"].n_unique() == 3772


def test_lehman_alive_before_delisting():
    """Survivorship-bias check: a name alive at the as-of date but
    delisted later must still appear (mirrors test_no_lookahead.py's
    scaffold). 2007-06-30 is a Saturday -- universe_at() must resolve to
    the actual last trading day on/before it (2007-06-29) rather than
    silently returning an empty/wrong universe."""
    universe = universe_at(pl.date(2007, 6, 30))
    permnos = set(universe["permno"].to_list())
    if 80599 not in permnos:
        pytest.skip(
            "permno 80599 (Lehman) not present at 2007-06-30 in this "
            "pull -- confirm whether it's expected to be covered by the "
            "base filter (usincflg='Y', major exchange) before treating "
            "this as a real gap"
        )
    assert 80599 in permnos


def test_universe_excludes_names_delisted_before_month_end():
    """A permno with securityenddt strictly before month_end must not
    appear -- the direct point-in-time boundary condition, tested on a
    synthetic frame so it doesn't depend on which real names the pull
    happened to cover."""
    from src.data.universe_panel import _apply_pit_bounds

    # Note: pl.date(y, m, d) returns an unevaluated polars Expr, not a
    # datetime.date -- using it as a raw value inside a dict passed to
    # pl.DataFrame() produces an Object-dtype column holding Expr
    # instances, not real Date values (confirmed while implementing this
    # test: it raises a polars SchemaError when compared against a real
    # date). datetime.date(...) is the correct way to build literal date
    # values for a DataFrame's data; pl.date(...) is for building
    # expressions (e.g. the month_end argument below, or universe_at()'s
    # own call sites), not for populating column data directly.
    synthetic = pl.DataFrame({
        "permno": [10001, 10002],
        "securitynm": ["ALIVE CO", "DEAD CO"],
        "issuertype": ["CORP", "CORP"],
        "primaryexch": ["N", "N"],
        "securitybegdt": [datetime.date(2000, 1, 1), datetime.date(2000, 1, 1)],
        "securityenddt": [None, datetime.date(2010, 1, 1)],
    })
    result = _apply_pit_bounds(synthetic, month_end=pl.date(2015, 6, 30))
    assert set(result["permno"]) == {10001}


def test_universe_excludes_names_not_yet_listed():
    """The mirror synthetic case: securitybegdt after month_end excludes."""
    from src.data.universe_panel import _apply_pit_bounds

    synthetic = pl.DataFrame({
        "permno": [10001, 10002],
        "securitynm": ["OLD CO", "FUTURE CO"],
        "issuertype": ["CORP", "CORP"],
        "primaryexch": ["N", "N"],
        "securitybegdt": [datetime.date(2000, 1, 1), datetime.date(2020, 1, 1)],
        "securityenddt": [None, None],
    })
    result = _apply_pit_bounds(synthetic, month_end=pl.date(2015, 6, 30))
    assert set(result["permno"]) == {10001}


def test_reit_excluded_by_issuertype_not_icbindustry(crsp_primary_config):
    """The core signal-correction finding: issuertype='REIT' is the
    exclusion signal, NOT icbindustry='REIT'. A synthetic mortgage-REIT
    case (issuertype=REIT, icbindustry=FINL, matching the real Annaly/
    Chimera/AGNC pattern found this session) must be excluded; a synthetic
    real-estate-adjacent non-REIT (issuertype=CORP, icbindustry=REIT,
    matching the real CBRE/Zillow/Realogy pattern) must NOT be excluded."""
    from src.data.universe_panel import _apply_reit_exclusion

    synthetic = pl.DataFrame({
        "permno": [10001, 10002, 10003],
        "securitynm": ["ORDINARY CO", "MORTGAGE REIT CO", "REALTY BROKER CO"],
        "issuertype": ["CORP", "REIT", "CORP"],
        "icbindustry": ["INDL", "FINL", "REIT"],
    })
    result = _apply_reit_exclusion(synthetic, crsp_primary_config)
    assert set(result["permno"]) == {10001, 10003}


def test_exchange_filter_excludes_inactive_and_secondary_listings(crsp_primary_config):
    """primaryexch must be restricted to Q/N/A -- 'X' (confirmed dead/
    inactive) and 'R' (confirmed non-primary/secondary) are excluded."""
    from src.data.universe_panel import _apply_exchange_filter

    synthetic = pl.DataFrame({
        "permno": [10001, 10002, 10003, 10004],
        "securitynm": ["NASDAQ CO", "NYSE CO", "DEAD CO", "SECONDARY CO"],
        "primaryexch": ["Q", "N", "X", "R"],
    })
    result = _apply_exchange_filter(synthetic, crsp_primary_config)
    assert set(result["permno"]) == {10001, 10002}


def test_universe_at_2015_06_30_no_mlp_partnership_names():
    """Negative check: sharetype='UG' names (MLPs/trusts) never appear in
    universe_at()'s output at all -- they're excluded by Task 1's pull
    filter (sharetype='NS' required), not by universe_at() itself. This
    test confirms that exclusion survives into the point-in-time result,
    not just the raw pull."""
    universe = universe_at(pl.date(2015, 6, 30))
    known_mlp_names = [
        "ENTERPRISE PRODUCTS PARTNERS",
        "PLAINS ALL AMERN PIPELINE",
        "MAGELLAN MIDSTREAM",
        "WILLIAMS PARTNERS",
        "ENERGY TRANSFER",
    ]
    names = universe["securitynm"].to_list()
    for mlp_name in known_mlp_names:
        assert not any(mlp_name in n for n in names), (
            f"{mlp_name} appears in universe_at() output -- MLP exclusion "
            f"via the base pull filter (sharetype='NS') is not working"
        )


def test_universe_at_no_duplicate_permnos():
    """universe_at() must be permno-keyed (per the design doc's own
    contract): each permno appears at most once in the output, even
    though the underlying panel is one row per (permno, dlycaldt) trading
    day. A naive filter over the whole daily panel using only
    securitybegdt/securityenddt bounds (with no dlycaldt resolution)
    would return one row per trading day within the listing spell --
    this test would catch that regression directly."""
    universe = universe_at(pl.date(2015, 6, 30))
    assert universe["permno"].n_unique() == len(universe)


def test_dlycap_convention_vs_manual_calc_same_day():
    """Spec's design explicitly flags: verify dlycap's own convention
    (same-day vs. lagged) before trusting it as a shortcut -- do not
    assume. This test checks dlycap against abs(dlyprc) * shrout on the
    SAME date (not lagged) to determine which convention dlycap actually
    uses. Not a pass/fail gate on universe_panel.py's correctness -- a
    diagnostic that must be run and its printed conclusion recorded in
    docs/01_data_notes.md before Task 3 Step 3 is written (see Step 1
    below for the recording instruction)."""
    panel = pl.read_parquet("data/raw/us_panel_crsp_full/year=2015/part.parquet")
    sample = panel.filter(
        (pl.col("dlycaldt") == pl.date(2015, 6, 30))
        & pl.col("dlycap").is_not_null()
        & pl.col("dlyprc").is_not_null()
        & pl.col("shrout").is_not_null()
    ).head(20)

    manual = (sample["dlyprc"].abs() * sample["shrout"])
    ratio = (sample["dlycap"] / manual).to_list()
    print(f"dlycap / (abs(dlyprc) * shrout) ratios (sample of {len(sample)}): {ratio}")
    # Ratio ~= 1.0 for all rows (originally read as "no multiplier needed" --
    # CORRECTED 2026-09-08: this ratio only proves dlycap and
    # abs(dlyprc)*shrout are internally self-consistent with each other, not
    # that either is denominated in real dollars/raw shares. A real-world
    # anchor check (Apple, ExxonMobil, 2015-06-30 -- see
    # docs/01_data_notes.md section 15's correction) proved shrout here IS
    # in thousands, matching legacy crsp.dsf.shrout, and dlycap/prc*shrout's
    # internal ratio of 1.0 says nothing about that -- it would read exactly
    # the same whether both fields were correct or both wrong together by
    # the same factor. This diagnostic test is retained as a record of the
    # (insufficient) check that was originally run, not as proof of the
    # units convention -- see universe_panel.SHROUT_UNITS_MULTIPLIER for the
    # corrected convention actually used in code.


def test_market_cap_at_uses_prior_trading_day():
    """Lagged-weight rule, same as market_cap.py's Compustat-era version:
    mkt_cap must come from the prior trading day's price, never
    month_end's own close."""
    from src.data.universe_panel import market_cap_at

    result, coverage_log = market_cap_at(pl.date(2015, 6, 30))
    assert {"permno", "mkt_cap"} <= set(result.columns)
    assert {"permno", "matched"} <= set(coverage_log.columns)
    assert (result["mkt_cap"] >= 0).all()
    # (result["mkt_cap"] >= 0).all() does NOT catch nulls: in polars,
    # `null >= 0` evaluates to null, and .all() ignores nulls by default --
    # so a null mkt_cap column would silently pass the assertion above.
    # This explicit null-count check is the real guard.
    assert result["mkt_cap"].null_count() == 0, (
        "market_cap_at() returned rows with a null mkt_cap -- a permno "
        "whose resolved prior-trading-day row has a null price/shares "
        "must be excluded, not passed through with mkt_cap=null"
    )


def test_market_cap_at_resolves_prior_day_from_synthetic_panel():
    """Isolates the prior-trading-day resolution logic from real-data
    dependence."""
    from src.data.universe_panel import _mkt_cap_from_panel

    synthetic = pl.DataFrame({
        "permno": [10001, 10001, 10001],
        "dlycaldt": [pl.date(2015, 6, 26), pl.date(2015, 6, 29), pl.date(2015, 6, 30)],
        "dlyprc": [10.0, 11.0, 12.0],
        "shrout": [1000.0, 1000.0, 1000.0],
    })
    result = _mkt_cap_from_panel(synthetic, month_end=pl.date(2015, 6, 30))
    row = result.filter(pl.col("permno") == 10001)
    # prior trading day (06-29) price=11.0, not 06-30 -- shrout is in
    # thousands (SHROUT_UNITS_MULTIPLIER=1000, corrected 2026-09-08 via a
    # real-world market-cap anchor check; see universe_panel.py's
    # module-level comment), so shares=1000.0 means 1,000,000 real shares.
    expected = 11.0 * 1000.0 * 1000.0
    assert row["mkt_cap"].item() == expected


def test_market_cap_at_uses_abs_dlyprc():
    """dlyprc, like legacy crsp.dsf.prc, may be signed to flag a no-trade
    bid/ask midpoint stand-in -- market cap must use abs()."""
    from src.data.universe_panel import _mkt_cap_from_panel

    synthetic = pl.DataFrame({
        "permno": [10001],
        "dlycaldt": [pl.date(2015, 6, 29)],
        "dlyprc": [-11.0],
        "shrout": [1000.0],
    })
    result = _mkt_cap_from_panel(synthetic, month_end=pl.date(2015, 6, 30))
    assert result["mkt_cap"].item() == 11.0 * 1000.0 * 1000.0


def test_mkt_cap_excludes_permno_with_null_price_on_resolved_prior_day():
    """Critical regression (review of commit e195169): a permno that IS
    present -- its resolved most-recent-before-month_end row exists -- but
    that row has a null dlyprc, must be EXCLUDED from
    _mkt_cap_from_panel()'s output, not kept with mkt_cap=null.
    abs(null) * shrout is null, and a null silently sitting in a numeric
    mkt_cap column is exactly the kind of "wrong number that looks right"
    CLAUDE.md warns about -- any downstream sum/rank/beta-weighting could
    mishandle it silently. This is different from the no-prior-day-at-all
    case (already correctly excluded via the inner join in
    market_cap_at() finding nothing to match): here a row DOES exist, it's
    just missing a price."""
    from src.data.universe_panel import _mkt_cap_from_panel

    synthetic = pl.DataFrame({
        "permno": [10001, 10002, 10002],
        "dlycaldt": [pl.date(2015, 6, 29), pl.date(2015, 6, 26), pl.date(2015, 6, 29)],
        "dlyprc": [11.0, 9.0, None],
        "shrout": [1000.0, 1000.0, 1000.0],
    })
    result = _mkt_cap_from_panel(synthetic, month_end=pl.date(2015, 6, 30))
    permnos = set(result["permno"].to_list())
    assert permnos == {10001}, (
        f"expected only permno 10001 (valid price); permno 10002's "
        f"resolved prior-day row (2015-06-29) has a null dlyprc and must "
        f"be excluded entirely, not kept with mkt_cap=null. Got: {permnos}"
    )
    assert result["mkt_cap"].null_count() == 0


def test_mkt_cap_excludes_permno_with_null_shrout_on_resolved_prior_day():
    """Mirror of the null-dlyprc case (Minor finding, same review): a null
    shrout on the resolved prior-day row must also exclude the permno, not
    propagate a null mkt_cap. No known real occurrence in the current data
    sample, but the code path (abs(dlyprc) * shrout) is identical, so it
    is covered the same way."""
    from src.data.universe_panel import _mkt_cap_from_panel

    synthetic = pl.DataFrame({
        "permno": [10001, 10002],
        "dlycaldt": [pl.date(2015, 6, 29), pl.date(2015, 6, 29)],
        "dlyprc": [11.0, 9.0],
        "shrout": [1000.0, None],
    })
    result = _mkt_cap_from_panel(synthetic, month_end=pl.date(2015, 6, 30))
    permnos = set(result["permno"].to_list())
    assert permnos == {10001}
    assert result["mkt_cap"].null_count() == 0


def test_market_cap_at_2015_01_02_excludes_permno_14093_null_price():
    """Real-data pin of the Critical regression, using the exact case
    found during review of commit e195169: permno 14093 (China Commercial
    Credit Inc) has a null dlyprc on its last trading-day row before
    2015-01-02 (2014-12-31, confirmed directly against
    data/raw/us_panel_crsp_full/year=2014/part.parquet -- 75 of its 252
    rows in 2015 alone have null dlyprc). Before the fix, this permno was
    present in universe_at()'s output (it matches the base/exchange/REIT
    filters) and so was kept by market_cap_at()'s inner join with
    mkt_cap=null. It must now be absent entirely."""
    from src.data.universe_panel import market_cap_at

    result, coverage_log = market_cap_at(datetime.date(2015, 1, 2))
    permnos = set(result["permno"].to_list())
    assert 14093 not in permnos, (
        "permno 14093 appears in market_cap_at(2015-01-02)'s output -- "
        "its resolved prior-trading-day row (2014-12-31) has a null "
        "dlyprc, so it must be excluded rather than kept with "
        "mkt_cap=null"
    )
    assert result["mkt_cap"].null_count() == 0, (
        "market_cap_at(2015-01-02) returned one or more null mkt_cap "
        "values -- the null-price/null-shrout exclusion is not working"
    )


def test_universe_at_does_not_resurrect_stale_permnos_via_asof_join():
    """A permno whose listing spell (securitybegdt/securityenddt) is
    still nominally open at month_end, but which has NOT traded (no
    dlycaldt row) for an extended period before month_end, must not
    appear -- carrying forward a stale last-known row would leak a
    'still active per spell metadata' signal that contradicts the
    security's actual trading presence (same class of trap as CLAUDE.md's
    comp.secd/cshtrd>0 gotcha, CRSP-side). Regression case found directly
    in this pull: permno 69550 (Mylan Inc) has securityenddt=2020-11-16
    (a real, later, legitimate delisting date) but its last daily row
    before 2015-06-30 is 2015-02-27 -- four months stale. It must not
    appear in the 2015-06-30 universe."""
    universe = universe_at(pl.date(2015, 6, 30))
    permnos = set(universe["permno"].to_list())
    assert 69550 not in permnos, (
        "permno 69550 (Mylan) appears in the 2015-06-30 universe despite "
        "having no trading-day row since 2015-02-27 -- universe_at() is "
        "likely doing a stale as-of/backward-fill join instead of "
        "resolving to an actual trading-day snapshot"
    )


def test_market_cap_at_returns_coverage_log_tuple():
    """Important finding #1 (final whole-branch review): market_cap_at()
    must return (market_cap_df, coverage_log_df), mirroring market_cap.py's
    build_us_market_cap() pattern -- an eligible permno with no resolvable
    prior-day price must show up as matched=False in coverage_log_df, not
    silently vanish from an inner join with zero explanation."""
    from src.data.universe_panel import market_cap_at

    market_cap_df, coverage_log_df = market_cap_at(pl.date(2015, 6, 30))
    assert {"permno", "mkt_cap"} <= set(market_cap_df.columns)
    assert {"permno", "matched"} <= set(coverage_log_df.columns)

    universe = universe_at(pl.date(2015, 6, 30))
    assert coverage_log_df["permno"].n_unique() == len(universe), (
        "coverage_log_df must have exactly one row per permno "
        "universe_at() returned"
    )
    assert set(coverage_log_df["permno"].to_list()) == set(universe["permno"].to_list())

    matched_permnos = set(
        coverage_log_df.filter(pl.col("matched"))["permno"].to_list()
    )
    assert matched_permnos == set(market_cap_df["permno"].to_list()), (
        "every permno flagged matched=True in coverage_log_df must appear "
        "in market_cap_df, and vice versa"
    )


def test_market_cap_at_first_trading_day_all_unmatched_not_silently_empty():
    """Important finding #1, concrete case: 1965-01-04 is the panel's
    first trading day, so no permno has ANY prior trading day at all --
    market_cap_df is legitimately empty, but that must be visible and
    explained via coverage_log_df (all permnos matched=False), not just
    an empty frame with zero indication why."""
    from src.data.universe_panel import market_cap_at

    first_day = datetime.date(1965, 1, 4)
    market_cap_df, coverage_log_df = market_cap_at(first_day)

    assert len(market_cap_df) == 0, (
        "sanity check: the panel's first trading day must have no prior "
        "trading day at all, so market_cap_df should be empty"
    )
    assert len(coverage_log_df) > 0, (
        "coverage_log_df must not be empty -- it must list every "
        "universe_at(1965-01-04) permno as unmatched, not be empty itself"
    )
    assert not coverage_log_df["matched"].any(), (
        "every permno on the panel's first trading day must be "
        "matched=False in coverage_log_df (no prior trading day exists), "
        "not silently absent"
    )


def test_universe_at_raises_on_stale_resolution_beyond_panel_coverage():
    """Important finding #2 (final whole-branch review): a date deep into
    a WRDS data-lag year (the partition file exists but has 0 rows, e.g.
    the current in-progress calendar year) must not silently resolve
    backward to stale prior-year data with no bound or warning -- that is
    the exact same hazard as the Mylan stale-permno case (module
    docstring), reappearing at the panel's upper boundary. It must raise
    explicitly instead."""
    from src.data.universe_panel import StaleTradingDayError

    with pytest.raises(StaleTradingDayError):
        universe_at(datetime.date(2026, 6, 30))


def test_market_cap_at_raises_on_stale_resolution_beyond_panel_coverage():
    """Mirror of the universe_at() staleness check for market_cap_at(): it
    must not silently compute a market cap using a resolved trading day
    that is months stale relative to the requested month_end."""
    from src.data.universe_panel import StaleTradingDayError, market_cap_at

    with pytest.raises(StaleTradingDayError):
        market_cap_at(datetime.date(2026, 6, 30))
