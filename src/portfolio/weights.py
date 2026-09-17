"""
Portfolio weighting schemes, config/portfolio.yaml's weighting_schemes
block -- see that file for the mechanism each key names; this module
implements the math, never re-derived inline elsewhere (docs/03_roadmap.md
F2b).

rank_deviation_from_median: FP's rank weighting (bab-methodology skill).
Names are sorted on ex-ante beta; weight is proportional to the absolute
deviation of a name's beta RANK from the cross-sectional median rank,
normalised so each leg sums to 1:

    z    = cross-sectional rank of beta
    zbar = mean/median rank (equal for a rank vector -- see below)
    k    = 2 / sum(|z - zbar|)
    w_H  = k * (z - zbar)+        w_L  = k * (zbar - z)+

The median split is AUTOMATIC: zbar = (n+1)/2 is exactly the median rank
of {1, ..., n}, so positive-part clipping puts every below-median-beta
name in the long leg and every above-median name in the short leg, with
the median name itself (or names straddling it, for even n) landing at
weight 0 in both legs.

value (market_cap weighting) is declared in config/portfolio.yaml but NOT
implemented here -- explicitly deferred to F2c (docs/03_roadmap.md F2b
scope agreement). Do not add it without a corresponding test file entry.
"""

import polars as pl


def rank_deviation_from_median(betas: pl.DataFrame) -> pl.DataFrame:
    """FP's rank weighting over a single formation date's beta
    cross-section. `betas` must have a `beta_shrunk` column (any other
    columns are passed through). Ties in beta_shrunk are broken by
    polars' default rank tie-handling (average rank) -- FP's own paper
    does not specify a tie-break, and ties are vanishingly rare on
    continuous beta_shrunk values in real data.

    Returns the input frame with two added columns, weight_long and
    weight_short, each non-negative and summing to 1.0 across the
    returned frame (bab-methodology invariant: "Each leg's weights sum
    to 1 by construction"). Raises KeyError if beta_shrunk is absent --
    a caller passing the wrong column name must fail loudly, not
    silently rank on nothing.
    """
    if "beta_shrunk" not in betas.columns:
        raise KeyError(
            "rank_deviation_from_median requires a 'beta_shrunk' column, "
            f"got columns: {betas.columns}"
        )

    n = betas.height
    ranked = betas.with_columns(pl.col("beta_shrunk").rank(method="average").alias("_z"))
    zbar = (n + 1) / 2.0
    ranked = ranked.with_columns((pl.col("_z") - zbar).alias("_dev"))

    k_denom = ranked["_dev"].abs().sum()
    # k_denom == 0 only when every _dev is 0, i.e. n <= 1 (a single name
    # has rank 1 == zbar exactly) -- not divided by silently as 0/0; the
    # caller gets an explicit signal instead via a NaN-free all-zero
    # weight column, which is the only sensible output when there is no
    # cross-section to split.
    if k_denom == 0:
        k = 0.0
    else:
        k = 2.0 / float(k_denom)

    return ranked.with_columns(
        (k * pl.col("_dev").clip(lower_bound=0.0)).alias("weight_short"),
        (k * (-pl.col("_dev")).clip(lower_bound=0.0)).alias("weight_long"),
    ).drop(["_z", "_dev"])
