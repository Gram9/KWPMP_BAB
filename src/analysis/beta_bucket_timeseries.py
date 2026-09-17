"""Time-stability view of the beta-bucket analysis
(src.analysis.beta_buckets), docs/04_handoff_lowbeta_longonly.md Sec 10's
carried-forward finding: the full-sample SML average can hide real
instability (US long-only low-beta effect measured negative in the
1990s and 2020s, positive in the 2000s/1970s). A full-sample bucket mean
answers "did this work on average over 60 years" -- this module answers
"would this have worked in any given stretch," the more actionable
question for a fund manager deciding whether to run the tilt now.

Consumes BetaBucketResult.quarterly (src.analysis.beta_buckets) directly
-- no new backtest, no new WRDS/CHASS call. That frame is already
per-(bucket, held-quarter); this module only re-groups it by a coarser
time period.
"""

import datetime

import polars as pl

# Named period lengths in months -- CLAUDE.md: no magic numbers in src/.
# 5y is the default: with ~231 US quarters (~58 years) / 4 quarters-per-
# year, a 5-year window gives ~20 quarters per cell, enough for a real
# mean without an uneven first/last bin the way calendar decades would
# (a sample starting 1965 split into calendar decades gives a 5-year
# first bin and however-many-are-left last bin).
_PERIOD_MONTHS = {
    "1y": 12,
    "3y": 36,
    "5y": 60,
    "decade": 120,
}


def _period_label(quarter: datetime.date, period_months: int, anchor: datetime.date) -> str:
    """The period-of(quarter) label, as a "<start>-<end>" year range,
    computed from whole-period offsets since `anchor` (the earliest
    quarter in the series) -- so period boundaries are a fixed grid from
    the sample's own start, not calendar-year-aligned (which would bias
    the first/last bin width depending on where the sample happens to
    start)."""
    months_since_anchor = (quarter.year - anchor.year) * 12 + (quarter.month - anchor.month)
    period_index = months_since_anchor // period_months
    period_start_months = period_index * period_months
    period_start_year = anchor.year + (anchor.month - 1 + period_start_months) // 12
    period_end_year = anchor.year + (anchor.month - 1 + period_start_months + period_months - 1) // 12
    if period_start_year == period_end_year:
        return str(period_start_year)
    return f"{period_start_year}-{period_end_year}"


def bucket_returns_by_period(
    quarterly: pl.DataFrame,
    *,
    period: str = "5y",
) -> pl.DataFrame:
    """Re-groups BetaBucketResult.quarterly (bucket, quarter, ret, beta,
    n) into (bucket, period) cells: annualized mean return (mean
    quarterly ret * 4), n_quarters actually present in that cell (can be
    less than the nominal period length at the series' own start/end, or
    if some formation dates were skipped upstream -- e.g. too few names
    to bucket meaningfully that quarter), and mean realized beta.

    period: one of "1y", "3y", "5y" (default), "decade" -- a named
    window length, not a magic number (CLAUDE.md). Coarser periods give
    more quarters per cell (a more reliable mean) at the cost of fewer,
    chunkier cells to compare; finer periods show more granular
    instability at the cost of noisier per-cell means.

    Period boundaries are a fixed grid anchored to the EARLIEST quarter
    in `quarterly` (not calendar-year-aligned) -- see _period_label's
    own docstring for why. This is purely a re-aggregation of already-
    formed, already-point-in-time-correct bucket returns; it introduces
    no new formation-date logic and cannot leak information across
    quarters, since each quarter's return was already fixed by
    beta_buckets.run_beta_bucket_analysis before this function ever
    sees it -- grouping already-computed, already-dated observations
    into coarser labels cannot move a later quarter's data into an
    earlier period or vice versa.

    Returns: bucket, period, ann_ret, n_quarters, mean_realized_beta.
    Empty input returns an empty, correctly-typed frame.
    """
    if period not in _PERIOD_MONTHS:
        raise KeyError(
            f"period must be one of {sorted(_PERIOD_MONTHS)}, got {period!r}"
        )
    period_months = _PERIOD_MONTHS[period]

    empty_schema = {
        "bucket": pl.Utf8,
        "period": pl.Utf8,
        "ann_ret": pl.Float64,
        "n_quarters": pl.Int64,
        "mean_realized_beta": pl.Float64,
    }
    if quarterly.height == 0:
        return pl.DataFrame(schema=empty_schema)

    anchor_value = quarterly["quarter"].min()
    assert isinstance(anchor_value, datetime.date)  # guaranteed by height > 0 above
    anchor = anchor_value

    labeled = quarterly.with_columns(
        pl.col("quarter")
        .map_elements(
            lambda q: _period_label(q, period_months, anchor), return_dtype=pl.Utf8
        )
        .alias("period")
    )

    grouped = (
        labeled.group_by(["bucket", "period"])
        .agg(
            pl.col("ret").mean().alias("_mean_qtr_ret"),
            pl.len().alias("n_quarters"),
            pl.col("beta").mean().alias("mean_realized_beta"),
            pl.col("quarter").min().alias("_period_start"),
        )
        .with_columns((pl.col("_mean_qtr_ret") * 4).alias("ann_ret"))
        .select(["bucket", "period", "ann_ret", "n_quarters", "mean_realized_beta", "_period_start"])
        .sort(["bucket", "_period_start"])
        .drop("_period_start")
    )
    return grouped
