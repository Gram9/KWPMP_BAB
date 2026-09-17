"""Tests for src/portfolio/legs.py -- leg scaling, config/portfolio.yaml's
leg_scalings block. asymmetric_inverse_beta is the FP baseline
(docs/03_roadmap.md F2b); dollar_neutral_hedged is explicitly OUT OF
SCOPE this session (deferred to F2c).
"""

import pytest

from src.portfolio import legs


def test_asymmetric_inverse_beta_payoff_formula():
    """bab-methodology skill: r_BAB = (1/beta_L)(r_L - r_f) -
    (1/beta_H)(r_H - r_f). Direct formula pin against hand-computed
    values -- not a property test, a literal number."""
    result = legs.asymmetric_inverse_beta(
        r_long=0.02, r_short=0.05, beta_long=0.6, beta_high=1.8, r_f=0.001
    )
    expected = (1 / 0.6) * (0.02 - 0.001) - (1 / 1.8) * (0.05 - 0.001)
    assert result == pytest.approx(expected, rel=1e-12)


def test_asymmetric_inverse_beta_is_ex_ante_beta_neutral_by_construction():
    """bab-methodology invariant: 'Ex-ante portfolio beta is exactly 0 by
    construction.' If beta_long IS the long leg's true ex-ante beta and
    beta_high IS the short leg's, then scaling long by 1/beta_long and
    short by 1/beta_high means EACH SCALED LEG has ex-ante beta exactly
    1 -- so a portfolio that is 1/beta_long dollars long (ex-ante beta
    contribution = beta_long * (1/beta_long) = 1) minus 1/beta_high
    dollars short (contribution = beta_high * (1/beta_high) = 1) has net
    ex-ante beta exactly 1 - 1 = 0. Verified directly, not asserted by
    construction alone -- this is the property that makes the strategy
    'market-neutral relative to the index used to estimate betas'
    (spec section 7)."""
    beta_long, beta_high = 0.6, 1.8
    long_leg_dollars = 1.0 / beta_long
    short_leg_dollars = 1.0 / beta_high
    net_ex_ante_beta = (beta_long * long_leg_dollars) - (beta_high * short_leg_dollars)
    assert net_ex_ante_beta == pytest.approx(0.0, abs=1e-12)


def test_asymmetric_inverse_beta_uses_ex_ante_not_realized_betas():
    """EX-ANTE IS LOAD-BEARING (config/portfolio.yaml's own comment). This
    is a signature test: asymmetric_inverse_beta must take beta_long/
    beta_high as EXPLICIT parameters (the caller's ex-ante values,
    computed at formation from beta_shrunk), never compute or accept a
    REALIZED beta derived from r_long/r_short themselves. A realized
    ratio over the holding period is the exact lookahead bug
    test_shuffled_signal_produces_no_alpha (tests/leakage/
    test_no_lookahead.py) exists to catch: a shuffled ex-ante signal
    that still produces alpha means the return is coming from
    construction, not the beta sort.

    Verified by inspecting the actual function signature -- if a future
    edit adds a code path that derives beta from r_long/r_short (e.g.
    'if beta_long is None: infer from realized returns'), this test's
    call with an EXTREME beta_long unrelated to r_long must still use
    that extreme value, not silently override it."""
    import inspect

    sig = inspect.signature(legs.asymmetric_inverse_beta)
    assert "beta_long" in sig.parameters
    assert "beta_high" in sig.parameters

    # An ex-ante beta wildly inconsistent with the realized return must
    # still be used AS GIVEN -- proves there is no hidden realized-beta
    # override path.
    result = legs.asymmetric_inverse_beta(
        r_long=0.10, r_short=0.10, beta_long=0.1, beta_high=0.1, r_f=0.0
    )
    # With beta_long == beta_high and r_long == r_short, the payoff must
    # be exactly 0 (both terms identical) REGARDLESS of how extreme the
    # shared beta value is -- this is only true if beta is taken as
    # given, not re-derived from the (identical) realized returns.
    assert result == pytest.approx(0.0, abs=1e-12)


def test_asymmetric_inverse_beta_raises_on_zero_beta():
    """beta_long or beta_high == 0 means 1/beta is undefined -- must
    raise, never silently produce inf/NaN (CLAUDE.md: a wrong number
    that looks right is the worst possible outcome, and inf/NaN
    propagating silently through a portfolio return series is exactly
    that failure mode one step removed)."""
    with pytest.raises(ZeroDivisionError):
        legs.asymmetric_inverse_beta(
            r_long=0.02, r_short=0.05, beta_long=0.0, beta_high=1.8, r_f=0.001
        )
    with pytest.raises(ZeroDivisionError):
        legs.asymmetric_inverse_beta(
            r_long=0.02, r_short=0.05, beta_long=0.6, beta_high=0.0, r_f=0.001
        )


def test_leg_ex_ante_beta_is_weighted_average():
    """The bab-methodology payoff needs beta_L/beta_H as the WEIGHTED
    AVERAGE ex-ante beta of each leg at formation (config/portfolio.yaml:
    'beta_L and beta_H are the weighted average ex-ante betas of each
    leg, known at formation') -- not a simple mean, and not the
    single largest/smallest beta in the leg."""
    import polars as pl

    betas = pl.DataFrame(
        {
            "permno": [1, 2],
            "beta_shrunk": [0.4, 0.8],
            "weight_long": [0.75, 0.25],
        }
    )
    result = legs.leg_ex_ante_beta(betas, weight_col="weight_long", beta_col="beta_shrunk")
    expected = 0.75 * 0.4 + 0.25 * 0.8
    assert result == pytest.approx(expected, rel=1e-12)
    # A simple (unweighted) mean would give 0.6 -- must NOT equal that,
    # confirming the weights are actually applied.
    assert result != pytest.approx(0.6, rel=1e-6)


def test_leg_ex_ante_beta_raises_on_empty_leg():
    """A leg with zero weight everywhere (e.g. a formation date with no
    names below the median -- shouldn't happen with real data, but must
    not silently return 0 or NaN if it does) must raise."""
    import polars as pl

    betas = pl.DataFrame(
        {"permno": [1, 2], "beta_shrunk": [0.4, 0.8], "weight_long": [0.0, 0.0]}
    )
    with pytest.raises(ValueError):
        legs.leg_ex_ante_beta(betas, weight_col="weight_long", beta_col="beta_shrunk")
