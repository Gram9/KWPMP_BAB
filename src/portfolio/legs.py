"""
Leg scaling, config/portfolio.yaml's leg_scalings block --
asymmetric_inverse_beta is FP's own construction (bab-methodology skill),
implemented here per the FP baseline scope of docs/03_roadmap.md F2b.
dollar_neutral_hedged is declared in config/portfolio.yaml but explicitly
OUT OF SCOPE for this module -- deferred to F2c, since it needs the
market_index hedge wiring (D10) this phase does not build.

r_BAB = (1/beta_L)(r_L - r_f) - (1/beta_H)(r_H - r_f)

beta_L, beta_H are the WEIGHTED AVERAGE ex-ante betas of the long/short
legs, known at formation (config/portfolio.yaml's own comment: "EX-ANTE
IS LOAD-BEARING"). Using realized betas here is the lookahead bug
tests/leakage/test_no_lookahead.py's test_shuffled_signal_produces_no_alpha
exists to catch -- a shuffled ex-ante signal that still produces alpha
means the return is coming from construction, not the beta sort.
"""

import polars as pl


def asymmetric_inverse_beta(
    r_long: float, r_short: float, beta_long: float, beta_high: float, r_f: float
) -> float:
    """FP's asymmetric leg-scaling payoff for ONE holding-period return.
    beta_long/beta_high are the EX-ANTE (formation-date) weighted-average
    betas of the long/short legs -- see leg_ex_ante_beta() below for how
    a caller derives them from a beta cross-section. r_long/r_short are
    the REALIZED holding-period returns of each leg (rank- or
    value-weighted portfolio returns), r_f the risk-free rate over the
    same period.

    This function takes beta_long/beta_high as plain floats, not derived
    from r_long/r_short in any way -- there is no code path here that
    could substitute a realized beta for the caller's ex-ante one.
    Raises ZeroDivisionError if either beta is exactly 0 (1/beta
    undefined) rather than silently producing inf/NaN, which would
    propagate through a return series undetected (CLAUDE.md: a wrong
    number that looks right is the worst possible outcome).
    """
    if beta_long == 0.0:
        raise ZeroDivisionError("beta_long is 0.0 -- 1/beta_long is undefined")
    if beta_high == 0.0:
        raise ZeroDivisionError("beta_high is 0.0 -- 1/beta_high is undefined")

    return (1.0 / beta_long) * (r_long - r_f) - (1.0 / beta_high) * (r_short - r_f)


def leg_ex_ante_beta(betas: pl.DataFrame, *, weight_col: str, beta_col: str) -> float:
    """The weighted-average ex-ante beta of one leg at formation:
    sum(weight_i * beta_i) / sum(weight_i) over names with weight_i > 0
    in `weight_col`. Both weight_col and beta_col are column names in
    `betas` (e.g. weight_col="weight_long", beta_col="beta_shrunk" --
    weights.rank_deviation_from_median()'s output columns), so the SAME
    formation-date frame that produced the portfolio weights also
    produces the beta used to scale that leg, keeping both derived from
    one consistent ex-ante snapshot.

    Raises ValueError if every weight is 0 (an empty/degenerate leg) --
    the weighted average is undefined (0/0), and this must not silently
    resolve to 0.0 or NaN.
    """
    leg = betas.filter(pl.col(weight_col) > 0.0)
    total_weight = leg[weight_col].sum()
    if leg.height == 0 or total_weight == 0.0:
        raise ValueError(
            f"leg_ex_ante_beta: no names with positive {weight_col!r} -- "
            "cannot compute a weighted-average beta for an empty leg"
        )
    weighted_sum = (leg[weight_col] * leg[beta_col]).sum()
    return float(weighted_sum) / float(total_weight)
