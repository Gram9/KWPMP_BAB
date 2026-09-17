"""Beta-bucket / security-market-line (SML) analysis,
docs/04_handoff_lowbeta_longonly.md Sec 4.3b and docs/05_report_spec.md
Sec 4 ("the chart the whole report rests on").

Distinct from src/portfolio/longonly.py (the 20+20 top-N portfolio
selection) -- this module buckets the FULL fringe-filtered cross-section
into many portfolios at once, to see whether return rises with beta the
way CAPM predicts, or is flat/inverted the way the BAB literature claims.
Genuinely different analysis, not a backtest variant, per report spec
Sec 4's own framing.

BUCKET EDGES (user-confirmed): 5%-wide bands BELOW the cross-sectional
median beta (10 bands covering the bottom half of the distribution --
each is a "vigintile" of the FULL distribution, i.e. a 1/20th slice, but
there are only 10 of them since only the bottom half is sliced this
finely), 10%-wide bands (deciles) AT OR ABOVE the median (5 bands
covering the top half) -- 10 + 5 = 15 buckets total. Computed cross-
sectionally AT EACH FORMATION DATE, never pooled over the full panel
(CLAUDE.md) -- the median and every bucket edge can move quarter to
quarter as the universe's beta distribution shifts.

The per-formation-date aggregation loop (fringe filter -> group
assignment -> equal-weighted quarterly return -> full-sample stats) is
shared with src.analysis.target_beta via
src.analysis._grouped_quarterly_analysis -- extracted this session once
a second caller (target-beta portfolios) needed byte-identical
aggregation behavior. This module now only defines HOW names get
assigned to the 15 percentile buckets (assign_buckets); everything else
lives in the shared core.
"""

import polars as pl

from src.analysis._grouped_quarterly_analysis import (
    GroupedQuarterlyResult,
    run_grouped_quarterly_analysis,
)

# BetaBucketResult is an alias, not a new type -- this session's
# extraction moved the dataclass to the shared core module, but every
# existing caller/test imports BetaBucketResult from here, so the name
# is kept as an alias rather than requiring a repo-wide rename.
BetaBucketResult = GroupedQuarterlyResult

# 10 fine (5%-wide) buckets below the median, 5 decile (10%-wide)
# buckets at or above it -- 15 total, user-confirmed this session (and
# re-confirmed after an arithmetic slip in the original proposal: "20
# buckets at 5% each below the median" is impossible, since 20x5%=100%
# cannot fit in half the distribution -- 10x5%=50% is what actually
# covers the bottom half). Buckets are labeled by their lower edge:
# "V01".."V10" for the 10 fine bands below median, "D06".."D10" for the
# 5 deciles at/above it (D-numbering starts at 6 since deciles 1-5 are
# exactly the span the V-bands cover at finer resolution).


def _bucket_edges() -> list[float]:
    """The 15 lower-percentile edges bounding this module's 15 buckets:
    10 edges (0.00, 0.05, ..., 0.45) at 5% resolution BELOW the median
    (0.00-0.50), then 5 edges (0.50, 0.60, ..., 0.90) at 10% resolution
    AT OR ABOVE the median (0.50-1.00). Built explicitly rather than
    hand-listed, so the 10+5=15 count is a computed invariant, not a
    typed literal that could silently drift."""
    below = [i / 20 for i in range(10)]  # 10 edges: 0.00 .. 0.45
    at_or_above = [0.50 + i / 10 for i in range(5)]  # 5 edges: 0.50 .. 0.90
    edges = below + at_or_above
    assert len(edges) == 15, f"expected 15 lower edges (10 fine + 5 decile), got {len(edges)}"
    return edges


_N_BUCKETS = len(_bucket_edges())  # 15 -- derived, not retyped, so it can't drift from the edges above

# Minimum cross-section size for bucketing to be meaningful at a
# formation date: at least one name per bucket (CLAUDE.md: no magic
# numbers in src/ -- derived from _N_BUCKETS, not a bare literal).
_MIN_NAMES_FOR_BUCKETING = _N_BUCKETS


def _bucket_width(lower_edge: float) -> float:
    """5% (vigintile) if lower_edge < 0.50, else 10% (decile)."""
    return 0.05 if lower_edge < 0.50 else 0.10


def assign_buckets(cross_section: pl.DataFrame, *, beta_col: str = "beta") -> pl.DataFrame:
    """Assigns each row in `cross_section` to one of 15 buckets by its
    cross-sectional percentile rank on `beta_col`, computed WITHIN this
    call (never pooled across formation dates -- CLAUDE.md). Percentile
    rank uses average-rank-based ties (polars' default 'average' method)
    divided by row count, so it is stable under duplicate beta values.

    Returns cross_section with an added `bucket` column (string, e.g.
    "V01".."V10" for the 10 fine 5%-wide bands below median, "D06".."D10"
    for the 5 deciles at/above median -- V-prefix numbers are 1-indexed
    by 5%-band, D-prefix numbers are 1-indexed by 10%-band starting at
    the 6th decile since the first 5 deciles are covered at finer
    resolution by the V-bands).

    Exactly one row per input row -- unlike target_beta.py's
    assign_target_beta_groups, this partitions the universe exclusively
    (every name belongs to exactly one bucket).
    """
    n = cross_section.height
    if n == 0:
        return cross_section.with_columns(pl.lit(None, dtype=pl.Utf8).alias("bucket"))

    # (rank - 1) / n, NOT rank / n: rank() is 1-indexed, so plain rank/n
    # never produces exactly 0.0 for the lowest-ranked name -- with
    # n=20, rank 1 gives percentile 1/20=0.05, which lands EXACTLY on
    # the V01/V02 boundary and gets misclassified into V02 (confirmed by
    # a direct check this session: a clean 1..20 beta fixture put rank-1
    # in V02, not V01). Subtracting 1 first maps rank 1 -> percentile
    # 0.0 (the true start of the lowest bucket) and rank n -> (n-1)/n
    # (strictly < 1.0, the true top of the highest bucket).
    ranked = cross_section.with_columns(
        ((pl.col(beta_col).rank(method="average") - 1) / n).alias("_pctile")
    )

    edges = _bucket_edges()

    def _label(pctile: float) -> str:
        # Find the bucket whose [lower_edge, lower_edge+width) contains
        # pctile -- edges are sorted ascending, so the last edge <=
        # pctile (with a ceiling at 1.0 for the top bucket) is correct.
        chosen = edges[0]
        for e in edges:
            if pctile >= e:
                chosen = e
            else:
                break
        if chosen < 0.50:
            idx = round(chosen / 0.05) + 1  # 1-indexed vigintile
            return f"V{idx:02d}"
        idx = round(chosen / 0.10) + 1  # 1-indexed decile (6..10)
        return f"D{idx:02d}"

    labels = [_label(p) for p in ranked["_pctile"].to_list()]
    return ranked.drop("_pctile").with_columns(pl.Series("bucket", labels, dtype=pl.Utf8))


def _assign_fn_adapter(cross_section: pl.DataFrame, beta_col: str) -> pl.DataFrame:
    """Adapts assign_buckets' keyword-only beta_col to the shared core's
    positional assign_fn(cross_section, beta_col) contract."""
    return assign_buckets(cross_section, beta_col=beta_col)


def run_beta_bucket_analysis(
    betas_by_date,
    fringe_by_date,
    quarterly_monthly_rets: pl.DataFrame,
    market_monthly: pl.DataFrame,
    *,
    mkt_cap_floor: float | None,
    price_floor: float | None,
    id_col: str = "id",
    beta_col: str = "beta",
) -> BetaBucketResult:
    """Forms 15 beta buckets at every formation date, holds each one
    quarter forward equal-weighted, then aggregates full-sample per
    bucket -- see src.analysis._grouped_quarterly_analysis.
    run_grouped_quarterly_analysis for the full field-by-field docstring
    (this function is now a thin wrapper naming assign_buckets as the
    grouping rule and _MIN_NAMES_FOR_BUCKETING as the minimum
    cross-section size, extracted to the shared core this session).

    mkt_cap_floor/price_floor: the SAME fringe filter as the 20+20
    backtest (report spec Sec 4's "Bucketing decision": filtered is
    headline). Pass BOTH as None for the UNFILTERED robustness variant
    (Sec 4.4) -- closer to FP's own Table III construction, which used
    no size screen.
    """
    return run_grouped_quarterly_analysis(
        betas_by_date,
        fringe_by_date,
        quarterly_monthly_rets,
        market_monthly,
        mkt_cap_floor=mkt_cap_floor,
        price_floor=price_floor,
        id_col=id_col,
        beta_col=beta_col,
        assign_fn=_assign_fn_adapter,
        min_names_for_assignment=_MIN_NAMES_FOR_BUCKETING,
    )
