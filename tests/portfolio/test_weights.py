"""Tests for src/portfolio/weights.py -- rank weighting (FP) and value
weighting, per config/portfolio.yaml's weighting_schemes block and the
bab-methodology skill. See docs/03_roadmap.md F2b.
"""


import polars as pl
import pytest

from src.portfolio import weights

# ---------------------------------------------------------------------------
# rank_deviation_from_median -- FP's rank weighting
# ---------------------------------------------------------------------------


def test_rank_weights_median_split_is_automatic():
    """The median-rank name (or names straddling the median in an even-n
    cross-section) gets weight exactly 0 in BOTH legs -- z == zbar means
    (z - zbar) and (zbar - z) are both non-positive, so positive-part
    clipping zeroes it in each leg. This is the bab-methodology skill's
    own stated property: "the median split is automatic."""
    betas = pl.DataFrame({"permno": [1, 2, 3], "beta_shrunk": [0.5, 1.0, 1.5]})
    result = weights.rank_deviation_from_median(betas)

    median_row = result.filter(pl.col("permno") == 2)
    assert median_row.height == 1
    assert median_row["weight_long"].item() == pytest.approx(0.0)
    assert median_row["weight_short"].item() == pytest.approx(0.0)


def test_rank_weights_each_leg_sums_to_one():
    """bab-methodology: 'Each leg's weights sum to 1 by construction.'
    Direct, absolute assertion -- not a ratio, and not vacuously true for
    a degenerate cross-section (n=5, non-trivial split)."""
    betas = pl.DataFrame(
        {"permno": [1, 2, 3, 4, 5], "beta_shrunk": [0.4, 0.8, 1.0, 1.3, 2.1]}
    )
    result = weights.rank_deviation_from_median(betas)

    total_long = result["weight_long"].sum()
    total_short = result["weight_short"].sum()
    assert total_long == pytest.approx(1.0, abs=1e-9)
    assert total_short == pytest.approx(1.0, abs=1e-9)


def test_rank_weights_below_median_beta_goes_long_above_goes_short():
    """The long leg (low beta, BAB's 'betting against beta' long side) is
    every name with beta RANK below the median; the short leg is every
    name above. Direct membership check, not just a sum-to-1 sanity
    check -- proves the two legs partition on the correct SIDE."""
    betas = pl.DataFrame(
        {"permno": [1, 2, 3, 4, 5], "beta_shrunk": [0.4, 0.8, 1.0, 1.3, 2.1]}
    )
    result = weights.rank_deviation_from_median(betas)

    below_median = result.filter(pl.col("permno").is_in([1, 2]))
    above_median = result.filter(pl.col("permno").is_in([4, 5]))

    assert (below_median["weight_long"] > 0).all()
    assert (below_median["weight_short"] == 0).all()
    assert (above_median["weight_short"] > 0).all()
    assert (above_median["weight_long"] == 0).all()


def test_rank_weights_higher_beta_gets_higher_short_weight():
    """Within the short leg, weight is proportional to rank DEVIATION
    from the median, not just membership -- the highest-beta name must
    get strictly more short weight than a name just above the median."""
    betas = pl.DataFrame(
        {"permno": [1, 2, 3, 4, 5], "beta_shrunk": [0.4, 0.8, 1.0, 1.3, 5.0]}
    )
    result = weights.rank_deviation_from_median(betas).sort("permno")

    w4 = result.filter(pl.col("permno") == 4)["weight_short"].item()
    w5 = result.filter(pl.col("permno") == 5)["weight_short"].item()
    assert w5 > w4, (
        "the highest-beta name (permno 5) must carry more short weight "
        "than a name closer to the median (permno 4) -- weight must "
        "scale with RANK deviation, not just leg membership"
    )


def test_rank_weights_uses_rank_not_raw_beta_magnitude():
    """FP's construction is RANK-based, not beta-magnitude-based -- an
    extreme outlier beta must not distort the weight distribution beyond
    what its RANK implies. Two cross-sections with identical RANK
    ORDERING but wildly different raw beta magnitudes must produce
    IDENTICAL weights. This is the assertion that would catch an
    implementation that weights on (beta - median_beta) instead of
    (rank - median_rank)."""
    moderate = pl.DataFrame(
        {"permno": [1, 2, 3, 4, 5], "beta_shrunk": [0.4, 0.8, 1.0, 1.3, 2.1]}
    )
    extreme = pl.DataFrame(
        {"permno": [1, 2, 3, 4, 5], "beta_shrunk": [0.4, 0.8, 1.0, 1.3, 500.0]}
    )
    result_moderate = weights.rank_deviation_from_median(moderate).sort("permno")
    result_extreme = weights.rank_deviation_from_median(extreme).sort("permno")

    for col in ["weight_long", "weight_short"]:
        assert result_moderate[col].to_list() == pytest.approx(
            result_extreme[col].to_list(), abs=1e-9
        ), (
            f"{col} changed when only the MAGNITUDE (not rank order) of "
            "the highest beta changed -- rank weighting must depend on "
            "rank, not raw beta value"
        )


def test_rank_weights_raises_on_missing_beta_column():
    with pytest.raises(KeyError):
        weights.rank_deviation_from_median(pl.DataFrame({"permno": [1, 2, 3]}))


# Value weighting (config/portfolio.yaml's `value` weighting_scheme,
# kind: market_cap) is explicitly OUT OF SCOPE for this session -- deferred
# to F2c per the plan agreed with the user (docs/03_roadmap.md F2b). Not
# tested here; do not add weights.market_cap() calls until that phase.
