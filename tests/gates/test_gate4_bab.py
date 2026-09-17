"""
Tests for src/gates/gate4_bab.py -- Gate 4 (docs/02_validation_gates.md):
"THE GATE THAT MATTERS." Runs the US BAB strategy monthly under FP's exact
specification and correlates the result against AQR's published US BAB
factor -- the project's first genuine external anchor.

Correlation alone is NOT enough (CLAUDE.md: correlation is location- and
scale-invariant). Three further, absolute checks are required alongside
it -- mean monthly return, Sharpe, and realized market loading -- each
derived from values MEASURED on a real run over docs/02_validation_gates.md's
Gate 4 section, not round numbers. See that section for where each band
comes from and the full sample-window reasoning.

Sample window: 1970-01-31 through 2025-12-31. The FP-spec rho window is
1260 trading days (~5 years); before ~1970 beta_fp._resolve_window_starts
falls back to the earliest available market date (its own docstring),
giving a truncated lookback and a thin, unstable cross-section for the
first ~60 formation dates of the 1965-max sample. Asserting from 1970
measures the estimator, not the burn-in artifact.
"""

import datetime

import pytest

from src.data.universe_panel import RAW_DATA_DIR
from src.gates import gate4_bab

pytestmark = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)

_FORMATION_START = datetime.date(1970, 1, 1)
_FORMATION_END = datetime.date(2025, 12, 31)


def test_calendar_month_ends_excludes_partial_trailing_month():
    """_calendar_month_ends must never include a month whose own
    month-end falls after `end` -- a partial window at the tail must be
    excluded, not silently rounded up to a month-end beyond the caller's
    own range (that would let the loop form a portfolio on a date the
    caller never asked for)."""
    result = gate4_bab._calendar_month_ends(
        datetime.date(2015, 1, 15), datetime.date(2015, 4, 10)
    )
    assert result == [
        datetime.date(2015, 1, 31),
        datetime.date(2015, 2, 28),
        datetime.date(2015, 3, 31),
    ]


def test_build_bab_series_returns_backtest_result_with_expected_columns():
    """Fast path: a small 2-year window, real data, checking STRUCTURE
    (not the full-sample thresholds below) so this runs quickly as part
    of the ordinary suite."""
    result = gate4_bab.build_bab_series(
        datetime.date(1995, 1, 1), datetime.date(1996, 12, 31)
    )
    assert set(result.returns.columns) == {"month", "ret"}
    assert result.returns.height > 0, (
        "a real 2-year US window must produce at least one held-month "
        "return -- zero rows means every formation date skipped, which "
        "is itself a finding worth investigating before anything else"
    )


def test_characterize_skips_counts_only_missing_coverage_reason():
    """characterize_skips must NOT count thin-universe/thin-leg skips --
    those are a data-coverage-at-formation phenomenon, not the
    delisting-adjacent missing-return-coverage phenomenon this
    diagnostic exists to quantify. Constructed with a synthetic
    BacktestResult carrying one of each skip reason."""
    import polars as pl

    from src.portfolio.rebalance import BacktestResult

    result = BacktestResult(
        returns=pl.DataFrame(schema={"month": pl.Date, "ret": pl.Float64}),
        positions=pl.DataFrame(),
        diagnostics=pl.DataFrame(),
        skips={
            datetime.date(1995, 1, 31): "total universe too thin: 5 names < min_names_total=10",
            datetime.date(1995, 2, 28): "leg too thin: 1 long / 3 short names < min_names_per_leg=3",
            datetime.date(1995, 3, 31): (
                "held month 1995-04-30 missing return data for 1 long-leg / "
                "0 short-leg names -- refusing to silently renormalize or "
                "fabricate a return over an incomplete leg"
            ),
        },
    )
    characterization = gate4_bab.characterize_skips(result)
    assert characterization["n_skips_missing_coverage"] == 1
    assert characterization["by_decade"] == {1990: 1}


def test_characterize_skips_empty_when_no_skips():
    import polars as pl

    from src.portfolio.rebalance import BacktestResult

    result = BacktestResult(
        returns=pl.DataFrame(schema={"month": pl.Date, "ret": pl.Float64}),
        positions=pl.DataFrame(),
        diagnostics=pl.DataFrame(),
        skips={},
    )
    characterization = gate4_bab.characterize_skips(result)
    assert characterization["n_skips_missing_coverage"] == 0
    assert characterization["by_decade"] == {}


# ---------------------------------------------------------------------------
# THE ACTUAL GATE -- full 1970-2025 US sample vs. AQR. Slow (~25-30 minutes,
# measured): loads ~35M+ daily rows, estimates a beta cross-section at each
# of ~672 formation dates, and joins against AQR's own published factor.
# Deliberately NOT cached (src.gates.gate4_bab.build_bab_series's own
# docstring: a cache could silently serve a pre-injection value during a
# failure-injection demonstration). Run directly, not via `python run.py
# gates`, when validating a real Gate 4 change -- see docs/02_validation_
# gates.md's Gate 4 section for the injection-demonstration log.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gate4_us_1970_2025_correlation_exceeds_point_nine():
    """The spec-given threshold (docs/00_spec.md, docs/02_validation_
    gates.md): correlation below ~0.9 means something is wrong. NOT
    tightened to the achieved value the way Gate 1's 0.995 was -- unlike
    Gate 1 (self-built index vs. CRSP's own vwretd, same underlying
    panel), this compares against a different vendor's entirely
    independent universe and estimator, so re-run variance from a
    legitimate future change (a universe refinement, a delisting-merge
    fix) is real and unknown. Catches: timing errors, a broken beta
    sort, wrong-sign leg assignment -- anything that destroys the
    monotone relationship between our series and AQR's."""
    result = gate4_bab.run_gate4_us(_FORMATION_START, _FORMATION_END)
    # gate-verifier (2026-09-13): run_gate4_us's own docstring instructs
    # "check n_ours_only/n_aqr_only before trusting correlation" (the same
    # join-asymmetry guard Gate 1/2 needed after their retractions) but
    # nothing previously asserted on it. n_aqr_only is legitimately large
    # (AQR's series starts 1930, decades before ours can start) and is
    # NOT asserted zero. n_ours_only must be exactly 0 -- every month our
    # pipeline produces a return for must be matched by AQR somewhere in
    # its much longer history; a nonzero value would mean our series
    # extends past AQR's own coverage (e.g. a date-alignment bug), which
    # correlation over an INNER join can silently ignore.
    assert result["n_ours_only"] == 0, (
        f"n_ours_only={result['n_ours_only']} -- our pipeline produced a "
        "held-month return AQR's published series never covers; the "
        "inner join silently drops these months before correlation is "
        "computed, so this would not otherwise be visible"
    )
    assert result["correlation"] > 0.9, (
        f"Gate 4 correlation {result['correlation']} is below the 0.9 "
        "credibility threshold -- see docs/02_validation_gates.md's "
        "debug order: (1) universe definition, (2) delisting returns, "
        "(3) estimator windows/minimums, (4) weighting scheme, "
        "(5) leg scaling. Check realized market loading FIRST -- if it "
        "isn't near zero, the estimator is the problem, not the "
        "portfolio construction."
    )


@pytest.mark.slow
def test_gate4_us_1970_2025_mean_return_within_measured_band():
    """The required ABSOLUTE, non-scale-invariant anchor (CLAUDE.md:
    correlation cannot detect a units bug, a constant bias, or a scale
    error). Band derivation and the achieved mean/gap values are
    recorded in docs/02_validation_gates.md's Gate 4 section -- filled
    in from this test's own measured run, not chosen a priori."""
    result = gate4_bab.run_gate4_us(_FORMATION_START, _FORMATION_END)
    # See docs/02_validation_gates.md Gate 4 for the measured mean_ours,
    # mean_aqr, and the derived band (max(2*SE_diff, 1.5*measured_gap)).
    assert abs(result["mean_ours"] - result["mean_aqr"]) < gate4_bab.mean_return_band(result)


@pytest.mark.slow
def test_gate4_us_1970_2025_sharpe_within_measured_band():
    """Catches a leg-scaling/leverage error (e.g. beta_long/beta_high
    swapped in asymmetric_inverse_beta) that correlation would barely
    move but that changes the risk/return ratio materially."""
    result = gate4_bab.run_gate4_us(_FORMATION_START, _FORMATION_END)
    assert abs(result["sharpe_ours"] - result["sharpe_aqr"]) < gate4_bab.sharpe_band(result)


@pytest.mark.slow
def test_gate4_us_1970_2025_market_loading_near_zero():
    """FP report -0.06 realized US market loading (bab-methodology
    skill). A large |loading| means the ex-ante beta neutralization
    failed -- checked FIRST on any gate failure per the documented debug
    order, since it points at the estimator rather than portfolio
    construction. market_loading_band bounds the MAGNITUDE of our own
    loading (catches a gross break, e.g. long-only)."""
    result = gate4_bab.run_gate4_us(_FORMATION_START, _FORMATION_END)
    assert abs(result["market_loading"]) < gate4_bab.market_loading_band(result)


@pytest.mark.slow
def test_gate4_us_1970_2025_market_loading_matches_aqr_own_loading():
    """The retired assertion here was |market_loading_t| < 2.0 -- tested
    directly against docs/02_validation_gates.md's Gate 4 section: this
    fails on AQR's OWN published US BAB factor (t=-2.165 against the
    same market series) and on FP's own published -0.06 evaluated at our
    SE (t=-2.17). It measures sample size (n=599 gives real statistical
    power to a tiny loading), not defect presence -- rejecting the
    benchmark itself is proof it was never a correctness bar.

    Replaced with a comparison against the EXTERNAL ANCHOR itself: is our
    loading significantly different from AQR's own realized loading on
    the identical market series? Measured: diff -0.01267, t -1.258 --
    indistinguishable. This is the actual purpose of an external-anchor
    gate (docs/02_validation_gates.md: 'Gate 4's AQR comparison is the
    project's first genuine external anchor')."""
    result = gate4_bab.run_gate4_us(_FORMATION_START, _FORMATION_END)
    assert abs(result["market_loading_diff"]) < gate4_bab.loading_diff_band(result)
    assert abs(result["market_loading_diff_t"]) < gate4_bab.LOADING_DIFF_T_BOUND, (
        f"market_loading_diff_t={result['market_loading_diff_t']} -- our "
        "realized market loading differs significantly from AQR's own "
        "published factor's loading on the identical market series, "
        "not merely from zero"
    )

    # gate-verifier (2026-09-13): market_loading_aqr is the ONE genuinely
    # external, real-world anchor in this whole function's output -- it is
    # AQR's OWN published series regressed on our market index, checkable
    # against FP's published -0.06 (bab-methodology skill) independent of
    # anything our pipeline computes. Never previously asserted. Bounds
    # are wide (this is a sanity check on the shared market/RF inputs
    # every other assertion in this file depends on, not a tight fit) --
    # measured -0.0645, comfortably inside; a value outside this range
    # would mean our market series or RF construction is wrong regardless
    # of whether the BAB pipeline itself is correct.
    assert -0.15 < result["market_loading_aqr"] < 0.0, (
        f"market_loading_aqr={result['market_loading_aqr']} -- AQR's own "
        "published US BAB factor should show a mildly negative realized "
        "loading against OUR market series (FP report -0.06 published); "
        "a value outside this range points at our market index or RF "
        "construction, not at the BAB pipeline itself"
    )


@pytest.mark.slow
def test_gate4_us_1970_2025_n_dates_and_skip_count_are_pinned():
    """A silently shortened return series (e.g. a truncated
    monthly_rets fetch) would raise every OTHER statistic above by
    removing the hardest months -- pin the exact date count and skip
    count so that failure mode is caught directly, not just inferred."""
    result = gate4_bab.run_gate4_us(_FORMATION_START, _FORMATION_END)
    # See docs/02_validation_gates.md Gate 4 for the measured n_dates
    # and total skip count, pinned here once measured.
    assert result["n_dates"] == gate4_bab.EXPECTED_N_DATES_1970_2025
    assert len(result["skips"]) == gate4_bab.EXPECTED_N_SKIPS_1970_2025

    # gate-verifier / leakage-auditor (2026-09-13) found the original
    # version of this check compared two CONFIG CONSTANTS against a pure
    # function of two dates -- true by arithmetic (599+73==672) forever,
    # regardless of what the pipeline actually does; it could not have
    # caught the two pins themselves drifting apart from reality. Also:
    # gate4_bab.build_bab_series's beta loop silently drops a formation
    # date if estimate_beta_fp returns an EMPTY cross-section (betas.height
    # == 0) -- such a date reaches neither run_backtest's returns NOR its
    # skips dict, a genuine "vanishes without being recorded" gap.
    #
    # Fixed: assert the invariant against the LIVE run's own counts, not
    # the frozen constants, and account for the one place n_dates and
    # len(skips) are NOT directly comparable -- n_dates is POST the AQR
    # inner join (comparison.height), while skips is measured over
    # build_bab_series's pre-join candidate set. n_ours_only (months our
    # pipeline produced that AQR's series doesn't cover) is exactly the
    # gap between the two -- already returned, never previously asserted.
    n_candidate_formation_dates = len(
        gate4_bab._calendar_month_ends(_FORMATION_START, _FORMATION_END)
    )
    n_pre_join_returns = result["n_dates"] + result["n_ours_only"]
    assert n_pre_join_returns + len(result["skips"]) == n_candidate_formation_dates, (
        f"n_dates({result['n_dates']}) + n_ours_only({result['n_ours_only']}) + "
        f"len(skips)({len(result['skips'])}) != "
        f"n_candidate_formation_dates({n_candidate_formation_dates}) -- a "
        "formation date vanished without producing either a held return "
        "or a recorded skip reason"
    )


# ---------------------------------------------------------------------------
# Band-logic unit tests -- fast, no pipeline run.
#
# These bands are FROZEN (config/gate4.yaml), evaluated once against the
# real 1970-2025 baseline, deliberately NOT recomputed from whatever
# `result` a caller passes in. Found this session: a live-recomputed
# "1.5 * measured_gap" is tautologically self-satisfying -- a x100 scale
# injection inflates its own gap term by the same x100, so the band widens
# to match the very defect it exists to catch (verified directly: it
# PASSED under that injection). The tests below assert the frozen-value
# CONTRACT itself: the returned band must not move when `result`'s
# defect-sensitive fields change, and must match config/gate4.yaml exactly.
# ---------------------------------------------------------------------------


def test_mean_return_band_is_frozen_not_recomputed_from_result():
    """The band must be identical for a clean result and a result with a
    massive injected gap -- if it moved, the assertion it feeds would be
    unable to fail (the exact defect this test guards against)."""
    clean = {"mean_ours": 0.006, "mean_aqr": 0.007}
    injected = {"mean_ours": 0.60, "mean_aqr": 0.007}  # x100-style blowup
    assert gate4_bab.mean_return_band(clean) == gate4_bab.mean_return_band(injected)
    assert gate4_bab.mean_return_band(clean) == pytest.approx(0.00114784)


def test_sharpe_band_is_frozen_not_recomputed_from_result():
    clean = {"sharpe_ours": 0.70, "sharpe_aqr": 0.73}
    injected = {"sharpe_ours": -0.90, "sharpe_aqr": 0.73}  # beta-swap-style blowup
    assert gate4_bab.sharpe_band(clean) == gate4_bab.sharpe_band(injected)
    assert gate4_bab.sharpe_band(clean) == pytest.approx(0.15)


def test_market_loading_band_is_frozen_not_recomputed_from_result():
    clean = {"market_loading": -0.077}
    injected = {"market_loading": -1.70}  # beta-swap-style blowup
    assert gate4_bab.market_loading_band(clean) == gate4_bab.market_loading_band(injected)
    assert gate4_bab.market_loading_band(clean) == pytest.approx(0.11577)


def test_loading_diff_band_is_frozen_not_recomputed_from_result():
    clean = {"market_loading_diff": -0.013, "market_loading_diff_se": 0.010}
    injected = {"market_loading_diff": -1.64, "market_loading_diff_se": 0.042}
    assert gate4_bab.loading_diff_band(clean) == gate4_bab.loading_diff_band(injected)
    assert gate4_bab.loading_diff_band(clean) == pytest.approx(0.020132)


def test_expected_pins_are_ints_from_config():
    """Guards against a YAML parse turning these into strings/floats,
    which would make an == comparison against a real int count silently
    always False rather than a loud type error."""
    assert isinstance(gate4_bab.EXPECTED_N_DATES_1970_2025, int)
    assert isinstance(gate4_bab.EXPECTED_N_SKIPS_1970_2025, int)
