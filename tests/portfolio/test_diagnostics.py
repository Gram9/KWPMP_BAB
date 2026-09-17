"""Tests for src/portfolio/diagnostics.py -- spec 00_spec.md section 8 /
schema decision D8: six diagnostics, EVERY run, non-optional. See
config/portfolio.yaml's diagnostics.always_emitted block and
docs/03_roadmap.md F2b/F3.
"""

import polars as pl
import pytest

from src.portfolio import diagnostics, legs, weights


def _sample_betas() -> pl.DataFrame:
    betas = pl.DataFrame(
        {
            "permno": [1, 2, 3, 4, 5],
            "beta_shrunk": [0.4, 0.8, 1.0, 1.3, 2.1],
            "mkt_cap": [500.0, 300.0, 100.0, 200.0, 400.0],
        }
    )
    return weights.rank_deviation_from_median(betas)


def test_compute_diagnostics_emits_all_six_keys():
    """D8: 'Diagnostics are fixed and cannot be switched off.' Direct
    check that all six named diagnostics are present in the output,
    every call, with no parameter that could omit one."""
    weighted = _sample_betas()
    result = diagnostics.compute_diagnostics(
        weighted, realized_market_loading=0.01, r_long=0.02, r_short=0.05,
        dollars_long=1.4, dollars_short=0.7,
    )
    expected_keys = {
        "realized_market_loading", "dollars_long", "dollars_short",
        "ex_ante_beta_spread", "name_count", "weight_fraction_smallest_size_decile",
    }
    assert expected_keys <= set(result.keys()), (
        f"missing diagnostics: {expected_keys - set(result.keys())}"
    )


def test_compute_diagnostics_has_no_toggle_parameter():
    """D8's structural guarantee, verified directly on the function
    signature: 'the emitting code must not read a toggle.' There must be
    no parameter named anything like enable/disable/include/toggle/
    switch that could suppress one of the six required diagnostics."""
    import inspect

    sig = inspect.signature(diagnostics.compute_diagnostics)
    suspicious_terms = ("enable", "disable", "include", "toggle", "switch", "skip")
    for param_name in sig.parameters:
        lowered = param_name.lower()
        assert not any(term in lowered for term in suspicious_terms), (
            f"compute_diagnostics has a parameter {param_name!r} that "
            "looks like a diagnostic on/off switch -- D8 requires "
            "diagnostics be unconditional, not configurable"
        )


def test_name_count_matches_input_height():
    weighted = _sample_betas()
    result = diagnostics.compute_diagnostics(
        weighted, realized_market_loading=0.0, r_long=0.0, r_short=0.0,
        dollars_long=1.0, dollars_short=1.0,
    )
    assert result["name_count"] == 5


def test_ex_ante_beta_spread_formula():
    """(beta_H - beta_L)/(beta_H * beta_L), per config/portfolio.yaml's
    own comment on ex_ante_beta_spread. Direct formula pin using
    legs.leg_ex_ante_beta() on the same weighted frame -- not
    re-derived independently, so a change to leg_ex_ante_beta()'s
    convention automatically keeps this in sync."""
    weighted = _sample_betas()
    result = diagnostics.compute_diagnostics(
        weighted, realized_market_loading=0.0, r_long=0.0, r_short=0.0,
        dollars_long=1.0, dollars_short=1.0,
    )
    beta_l = legs.leg_ex_ante_beta(weighted, weight_col="weight_long", beta_col="beta_shrunk")
    beta_h = legs.leg_ex_ante_beta(weighted, weight_col="weight_short", beta_col="beta_shrunk")
    expected = (beta_h - beta_l) / (beta_h * beta_l)
    assert result["ex_ante_beta_spread"] == pytest.approx(expected, rel=1e-9)


def test_weight_fraction_smallest_size_decile_requires_full_cross_section():
    """N-M&V's finding (bab-methodology skill: 'FP's BAB commits ~$1.05
    per $1 invested to stocks in the bottom 1% of market cap') is a
    property of the FULL universe's size distribution, not just the
    names selected into the portfolio -- the decile breakpoints must be
    computed over ALL candidate names at formation, matching CLAUDE.md's
    'no filtering on names with N observations over the full sample'
    spirit (the decile cutoff itself must be point-in-time and
    cross-sectional, never derived from the post-selection subset)."""
    import inspect

    sig = inspect.signature(diagnostics.compute_diagnostics)
    assert "full_universe_mkt_cap" in sig.parameters, (
        "compute_diagnostics has no way to receive the FULL universe's "
        "market-cap cross-section -- weight_fraction_smallest_size_decile "
        "cannot be computed correctly from the selected-names frame alone"
    )


def test_weight_fraction_smallest_size_decile_uses_full_universe_not_selected_subset():
    """Direct numeric check that the decile cutoff comes from
    full_universe_mkt_cap, NOT from weighted_betas' own mkt_cap column --
    the exact vacuity trap this diagnostic's docstring warns about (a bug
    that computes the cutoff over the SELECTED names would, on many
    fixtures, coincidentally select the same names and pass anyway).

    Constructed so the two cutoffs are FAR apart and select DIFFERENT
    names: the full universe has 90 tiny names (mkt_cap 1-90) and 10
    huge ones (mkt_cap 10,000-10,090), so its true 10th percentile sits
    around 9-10 -- deep in the tiny-name cluster. The SELECTED names are
    only 4 of the huge ones (mkt_cap 10,000-10,030), whose OWN 10th
    percentile would sit around 10,003 -- entirely within the selected
    set itself, wrongly counting ~all of it as "smallest decile." Using
    the correct (full-universe) cutoff, none of the selected huge names
    are anywhere near the true bottom decile, so the answer must be
    exactly 0.0."""
    tiny_names = pl.DataFrame(
        {"permno": list(range(1, 91)), "mkt_cap": [float(i) for i in range(1, 91)]}
    )
    huge_names = pl.DataFrame(
        {
            "permno": list(range(1000, 1010)),
            "mkt_cap": [10000.0 + i for i in range(10)],
        }
    )
    full_universe = pl.concat([tiny_names, huge_names])

    weighted = pl.DataFrame(
        {
            "permno": [1000, 1001, 1008, 1009],
            "beta_shrunk": [0.3, 0.5, 1.5, 1.8],
            "mkt_cap": [10000.0, 10001.0, 10008.0, 10009.0],
            "weight_long": [0.6, 0.4, 0.0, 0.0],
            "weight_short": [0.0, 0.0, 0.4, 0.6],
        }
    )
    result = diagnostics.compute_diagnostics(
        weighted, realized_market_loading=0.0, r_long=0.0, r_short=0.0,
        dollars_long=1.0, dollars_short=1.0, full_universe_mkt_cap=full_universe,
    )
    assert result["weight_fraction_smallest_size_decile"] == pytest.approx(0.0, abs=1e-9), (
        "none of the selected names (mkt_cap ~10,000) are anywhere near "
        "the FULL universe's true bottom decile (~90 names with mkt_cap "
        "1-90) -- a nonzero result here means the decile cutoff was "
        "computed from the selected subset, not the full universe"
    )


# ---------------------------------------------------------------------------
# annualized_sharpe / full_sample_market_loading -- Gate 4
# (docs/02_validation_gates.md). Both are FULL-SAMPLE statistics computed
# post-loop (see this module's own docstring on why realized_market_loading
# is inherently a t+1, full-sample-only quantity), not per-month diagnostics.
# ---------------------------------------------------------------------------


def test_annualized_sharpe_hand_computed():
    """excess = [0.01, -0.02, 0.03, 0.005, -0.01]. mean=0.003,
    std(ddof=1)=0.019235384061671346, annualized Sharpe =
    mean/std*sqrt(12) = 0.5402702026688977 -- computed independently in
    numpy, not re-derived from the function under test."""
    excess = [0.01, -0.02, 0.03, 0.005, -0.01]
    result = diagnostics.annualized_sharpe(excess)
    assert result == pytest.approx(0.5402702026688977, rel=1e-9)


def test_annualized_sharpe_periods_per_year_default_is_12():
    """periods_per_year defaults to 12 (monthly) -- every existing
    caller (gate4_bab, this test file) relies on this default being
    preserved unchanged when the parameter was added."""
    import inspect

    sig = inspect.signature(diagnostics.annualized_sharpe)
    assert sig.parameters["periods_per_year"].default == 12


def test_annualized_sharpe_quarterly_periods_per_year_differs_from_monthly():
    """Same fixture as test_annualized_sharpe_hand_computed, but with
    periods_per_year=4 (quarterly) -- must give a DIFFERENT result
    (sqrt(4) vs sqrt(12), a real ~1.73x difference), computed
    independently in numpy: 0.31192514694602175. Guards against
    docs/04_handoff_lowbeta_longonly.md's quarterly backtest silently
    reusing the monthly default and misannualizing by sqrt(3)."""
    excess = [0.01, -0.02, 0.03, 0.005, -0.01]
    monthly_result = diagnostics.annualized_sharpe(excess)
    quarterly_result = diagnostics.annualized_sharpe(excess, periods_per_year=4)
    assert quarterly_result == pytest.approx(0.31192514694602175, rel=1e-9)
    assert quarterly_result != pytest.approx(monthly_result, rel=1e-3)


def test_annualized_sharpe_takes_no_risk_free_argument():
    """run_backtest's BAB return is ALREADY an excess return --
    legs.asymmetric_inverse_beta subtracts r_f internally. A Sharpe
    helper that also accepted an rf argument would invite a silent
    double-subtraction (a 40-60% Sharpe error). Structural check:
    the signature must not accept anything named like a risk-free rate."""
    import inspect

    sig = inspect.signature(diagnostics.annualized_sharpe)
    suspicious_terms = ("rf", "risk_free", "riskfree")
    for param_name in sig.parameters:
        lowered = param_name.lower()
        assert not any(term in lowered for term in suspicious_terms), (
            f"annualized_sharpe has a parameter {param_name!r} that looks "
            "like a risk-free rate -- the input is already an excess "
            "return (legs.asymmetric_inverse_beta subtracts r_f), so "
            "accepting rf here would risk a silent double-subtraction"
        )


def test_full_sample_market_loading_recovers_planted_beta():
    """Synthetic OLS fixture with a KNOWN planted alpha=0.01, beta=0.5
    plus small deterministic residuals -- NOT drawn from beta itself, so
    the check is a genuine recovery test, not circular. Expected
    coefficients computed independently via numpy.linalg.lstsq plus the
    standard OLS variance formula (not re-derived from the function
    under test): alpha_hat=0.009809523809523808,
    beta_hat=0.5380952380952382, t_beta=35.152662451361394,
    alpha_t_stat=27.325345136036606 (n=8, dof=6). The residuals are
    deliberately small and non-symmetric so beta_hat differs visibly from
    the planted 0.5 -- this is expected OLS sampling behaviour on n=8,
    not a bug in the fixture."""
    market_excess = [0.02, -0.01, 0.03, 0.00, -0.02, 0.04, 0.01, -0.03]
    bab_rets = [0.021, 0.004, 0.027, 0.01, -0.002, 0.031, 0.014, -0.005]
    result = diagnostics.full_sample_market_loading(bab_rets, market_excess)
    assert result["alpha"] == pytest.approx(0.009809523809523808, rel=1e-6)
    assert result["beta"] == pytest.approx(0.5380952380952382, rel=1e-6)
    assert result["t_stat"] == pytest.approx(35.152662451361394, rel=1e-5)
    assert result["alpha_t_stat"] == pytest.approx(27.325345136036606, rel=1e-5)
    assert result["n"] == 8


def test_full_sample_market_loading_near_zero_beta_gives_small_t_stat():
    """A BAB series with NO relationship to the market (beta exactly 0,
    pure noise) must produce a small |t_stat| -- the direct case Gate 4's
    assertion (d) checks. Constructed so bab_rets is literally independent
    of market_excess (a fixed alternating series uncorrelated with the
    market draw), not merely close to it. Also carries a near-zero
    alpha_hat by construction (bab_rets is centered near 0), so
    alpha_t_stat is checked here too, independently computed via numpy
    (alpha_t_stat=-0.035225622259844824) -- distinct from t_stat
    (0.4543627379421989), proving the two statistics are not accidentally
    aliased to the same value in this function."""
    market_excess = [0.02, -0.01, 0.03, 0.00, -0.02, 0.04, 0.01, -0.03]
    bab_rets = [0.005, -0.004, 0.006, -0.005, 0.004, -0.006, 0.005, -0.004]
    result = diagnostics.full_sample_market_loading(bab_rets, market_excess)
    assert abs(result["t_stat"]) < 2.0, (
        f"t_stat={result['t_stat']} -- expected a small t-stat for a "
        "BAB series constructed to be independent of the market"
    )
    assert result["alpha_t_stat"] == pytest.approx(-0.035225622259844824, rel=1e-5)
    assert result["t_stat"] != pytest.approx(result["alpha_t_stat"], rel=1e-3), (
        "t_stat and alpha_t_stat came out equal -- they are different "
        "statistics (beta's vs alpha's own SE) and must not be aliased"
    )


def test_full_sample_market_loading_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        diagnostics.full_sample_market_loading([0.01, 0.02], [0.01, 0.02, 0.03])


def test_full_sample_market_loading_raises_on_too_few_observations():
    """n=2 gives dof=0 -- the residual variance (and therefore the
    standard error and t-stat) is undefined, not merely noisy. Must
    raise rather than silently return inf/NaN (CLAUDE.md: a wrong number
    that looks right is the worst possible outcome)."""
    with pytest.raises(ValueError):
        diagnostics.full_sample_market_loading([0.01, 0.02], [0.01, 0.02])
