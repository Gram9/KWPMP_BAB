"""Downturn/expansion regime analysis -- docs/06_handoff_section6.md
item C, report Sec 6.4 (the director's explicit ask).

DEFINITIONS ARE PRE-REGISTERED. docs/07_regime_preregistration.md was
written and committed BEFORE any regime statistic was computed, and
states that it may not be revised afterwards. The constants below are a
TRANSCRIPTION of that document, not an independent choice. If the two
ever disagree, THE DOCUMENT WINS and this module is wrong.

That is enforced, not merely asserted: tests/analysis/test_regimes.py::
test_crisis_windows_match_the_preregistration_document parses the
document's own table at test time and fails on any drift. This project
has retracted two gates whose statistics were chosen after seeing the
data; a transcription nobody checks would leave exactly that door open.

NO LOOK-AHEAD, AND WHY THE FULL-SAMPLE PERCENTILE IS STILL ACCEPTABLE.
CLAUDE.md's governing rule is that information from t+1 must never
influence a DECISION made at t. Definition 2's breakpoint is a
full-sample statistic, so it could not have been known in real time --
but it forms no position and drives no trade. It labels quarters after
the fact, for description only. label_threshold_regimes accepts only a
return frame: no formation dates, no cross-section, no weights, so it is
structurally incapable of feeding a portfolio. The report must state
that this split is descriptive and not tradeable; the pre-registration
document requires it.
"""

import datetime
from typing import Literal

import numpy as np
import polars as pl

from src.analysis import quarterly_compounding

# ---------------------------------------------------------------------------
# Definition 1 -- named historical crises.
#
# TRANSCRIBED from docs/07_regime_preregistration.md's Definition 1
# table. These windows are externally recognisable and were named before
# this analysis existed; they were NOT selected from our own return
# series. Do not edit without editing the document -- the test above
# parses it and will fail.
# ---------------------------------------------------------------------------
CRISIS_WINDOWS: tuple[tuple[str, datetime.date, datetime.date], ...] = (
    ("oil_shock_1973_74", datetime.date(1973, 1, 1), datetime.date(1974, 12, 31)),
    ("black_monday_1987", datetime.date(1987, 10, 1), datetime.date(1987, 10, 31)),
    ("dotcom_2000_02", datetime.date(2000, 3, 1), datetime.date(2002, 10, 31)),
    ("gfc_2008_09", datetime.date(2008, 9, 1), datetime.date(2009, 3, 31)),
    ("covid_2020", datetime.date(2020, 3, 1), datetime.date(2020, 3, 31)),
)

# ---------------------------------------------------------------------------
# Definition 2 -- the objective threshold, likewise transcribed.
# ---------------------------------------------------------------------------
DOWNTURN_PERCENTILE = 0.10
# polars' default; pinned so it cannot drift. Typed as the Literal the
# polars overload expects, rather than a bare str -- a plain str fails
# type checking and would otherwise invite a `# type: ignore` that
# suppresses real errors at this call site too.
DOWNTURN_INTERPOLATION: Literal[
    "nearest", "higher", "lower", "midpoint", "linear"
] = "linear"

DOWNTURN_LABEL = "downturn"
EXPANSION_LABEL = "expansion"

# Rendered for a crisis window that lies entirely outside a leg's
# sample. The Canadian leg starts 1989-03-31, so the 1973 and 1987
# windows have zero Canadian quarters -- a STRUCTURAL ABSENCE, not a
# finding of "no effect". The pre-registration document requires the
# exhibit show it as such, never as a blank or a zero.
OUT_OF_SAMPLE_LABEL = "n/a -- outside sample"


def _quarter_start(quarter_end: datetime.date) -> datetime.date:
    """The first day of the quarter ending at `quarter_end`."""
    start_month = quarter_end.month - 2
    year = quarter_end.year
    if start_month <= 0:
        start_month += 12
        year -= 1
    return datetime.date(year, start_month, 1)


def _matches_window(
    quarter_end: datetime.date, lo: datetime.date, hi: datetime.date
) -> bool:
    """The pre-registered quarter-assignment rule, verbatim in substance:

    a held quarter belongs to a crisis regime if the quarter's own END
    DATE falls inside the window, OR the window is fully contained
    within the quarter.

    The second clause is load-bearing, not a nicety. Three of the five
    windows (1987-10, 2020-03, and the 2008-09..2009-03 boundary months)
    are SHORTER than one quarter, so an end-date-only rule matches zero
    quarters for them and those crises silently vanish from the exhibit.
    """
    if lo <= quarter_end <= hi:
        return True
    return _quarter_start(quarter_end) <= lo and hi <= quarter_end


def label_crisis_regimes(quarters: list[datetime.date]) -> pl.DataFrame:
    """Label each quarter with the crisis window(s) it belongs to.

    Returns [quarter, regime] with ONE ROW PER (quarter, matching
    crisis). A quarter matching no crisis appears once with regime=None,
    so a caller can see the full sample rather than only the labelled
    subset. The pre-registered windows do not overlap, so in practice no
    quarter yields more than one row today -- the shape simply does not
    assume that.

    This is a labelling pass over already-computed returns. The crisis
    dates are historical fact known to any reader and nothing is
    re-estimated, so there is no look-ahead in the CLAUDE.md sense.
    """
    rows: list[dict] = []
    for quarter_end in quarters:
        matched = [
            label for label, lo, hi in CRISIS_WINDOWS if _matches_window(quarter_end, lo, hi)
        ]
        if matched:
            rows.extend({"quarter": quarter_end, "regime": label} for label in matched)
        else:
            rows.append({"quarter": quarter_end, "regime": None})

    if not rows:
        return pl.DataFrame(schema={"quarter": pl.Date, "regime": pl.Utf8})
    return pl.DataFrame(rows, schema={"quarter": pl.Date, "regime": pl.Utf8})


def crisis_coverage(quarters: list[datetime.date]) -> pl.DataFrame:
    """Which crisis windows this sample can actually speak to.

    Returns [regime, n_quarters, in_sample] for EVERY window in
    CRISIS_WINDOWS, including those matching zero quarters.

    The zero-match rows are the point of this function. Canada's sample
    starts 1989-03-31, so its 1973 and 1987 rows are structurally empty;
    reporting them as n_quarters=0 alongside real results would invite a
    reader to treat "no quarters" as "no effect". in_sample=False lets
    the exhibit print OUT_OF_SAMPLE_LABEL instead.
    """
    labelled = label_crisis_regimes(quarters)
    counts = dict(
        labelled.filter(pl.col("regime").is_not_null())
        .group_by("regime")
        .agg(pl.len().alias("n"))
        .iter_rows()
    )

    return pl.DataFrame(
        [
            {
                "regime": label,
                "n_quarters": counts.get(label, 0),
                "in_sample": counts.get(label, 0) > 0,
            }
            for label, _, _ in CRISIS_WINDOWS
        ],
        schema={"regime": pl.Utf8, "n_quarters": pl.Int64, "in_sample": pl.Boolean},
    )


def label_threshold_regimes(
    market_excess: pl.DataFrame,
    *,
    value_col: str,
) -> pl.DataFrame:
    """Definition 2: a quarter is a DOWNTURN if the MARKET's quarterly
    excess return is at or below the 10th percentile of the market's own
    quarterly excess returns over the full sample.

    market_excess: [quarter, <value_col>] -- the MARKET's excess return
    series, not the portfolio's. Splitting on the portfolio would define
    the regime by the very series under evaluation and make the strategy
    look defensive by construction.

    Returns [quarter, <value_col>, regime, threshold]; `threshold` is
    repeated on every row so the exhibit can quote the breakpoint
    without recomputing it.

    Ties go to downturn (the pre-registered rule is at-or-below).

    NOT TRADEABLE, and the report must say so. The breakpoint uses the
    whole sample, so it could not have been known in real time. It is
    admissible here only because it labels quarters after the fact and
    forms no position -- this function takes no formation dates, no
    cross-section and no weights, so it cannot feed a portfolio even by
    mistake.
    """
    if value_col not in market_excess.columns:
        raise ValueError(
            f"label_threshold_regimes: value_col={value_col!r} is not a column "
            f"of the market frame (has {market_excess.columns})"
        )
    if market_excess.height == 0:
        raise ValueError(
            "label_threshold_regimes: empty market series -- a percentile over "
            "no observations is undefined"
        )

    raw_threshold = market_excess[value_col].quantile(
        DOWNTURN_PERCENTILE, interpolation=DOWNTURN_INTERPOLATION
    )
    if raw_threshold is None:
        # polars returns None when every value is null. Raising beats
        # comparing against None, which would silently label every
        # quarter an expansion and report a clean-looking regime split
        # over a column containing no data at all.
        raise ValueError(
            f"label_threshold_regimes: the {DOWNTURN_PERCENTILE:.0%} percentile of "
            f"{value_col!r} is undefined -- the column is entirely null over "
            f"{market_excess.height} row(s)"
        )
    threshold = float(raw_threshold)

    return (
        market_excess.select(pl.col("quarter"), pl.col(value_col))
        .with_columns(
            pl.when(pl.col(value_col) <= threshold)
            .then(pl.lit(DOWNTURN_LABEL))
            .otherwise(pl.lit(EXPANSION_LABEL))
            .alias("regime"),
            pl.lit(threshold).alias("threshold"),
        )
        .sort("quarter")
    )


def regime_comparison(
    port_excess: pl.DataFrame,
    market_excess: pl.DataFrame,
    regime_labels: pl.DataFrame,
    *,
    port_col: str,
    market_col: str,
    label: str,
) -> pl.DataFrame:
    """Per-regime portfolio-vs-market comparison, reporting exactly the
    statistics pre-registered in docs/07_regime_preregistration.md:
    n_quarters, portfolio mean excess, market mean excess, the
    difference and its t-statistic, and the hit rate.

    regime_labels: [quarter, regime]. Rows with a null regime are
    IGNORED rather than pooled into an unnamed group -- crisis labelling
    leaves every non-crisis quarter null, and those must not silently
    become a residual "regime".

    diff_t_stat is mean(d)/(std(d, ddof=1)/sqrt(n)) and is None when
    n < 2, where a sample standard deviation does not exist. With only
    five crisis episodes this WILL occur on real data; emitting inf or
    0.0 there would put a number in front of a reader that reads as a
    finding. The driver prints n beside every t-statistic for the same
    reason.

    hit_rate counts STRICTLY port > mkt, matching the document's own
    wording; an exact tie is not a win.
    """
    # Reject nulls and duplicated quarters before any mean or t-stat --
    # a duplicated quarter is double-counted by every statistic below
    # and fans out the joins (gate-verifier finding 2026-09-15).
    quarterly_compounding.assert_clean_quarterly_series(
        port_excess, value_col=port_col, label=f"{label}:portfolio"
    )
    quarterly_compounding.assert_clean_quarterly_series(
        market_excess, value_col=market_col, label=f"{label}:market"
    )

    joined = (
        port_excess.select(pl.col("quarter"), pl.col(port_col).alias("_port"))
        .join(
            market_excess.select(pl.col("quarter"), pl.col(market_col).alias("_mkt")),
            on="quarter",
            how="inner",
        )
        .join(
            regime_labels.filter(pl.col("regime").is_not_null()),
            on="quarter",
            how="inner",
        )
        .sort("quarter")
    )

    rows: list[dict] = []
    for regime in sorted(joined["regime"].unique().to_list()):
        subset = joined.filter(pl.col("regime") == regime)
        port = np.asarray(subset["_port"].to_numpy(), dtype=float)
        mkt = np.asarray(subset["_mkt"].to_numpy(), dtype=float)
        diff = port - mkt
        n = diff.shape[0]

        if n >= 2:
            se = float(diff.std(ddof=1) / np.sqrt(n))
            t_stat = float(diff.mean() / se) if se > 0 else None
        else:
            se = None
            t_stat = None

        rows.append(
            {
                "label": label,
                "regime": regime,
                "n_quarters": n,
                "port_mean_excess": float(port.mean()),
                "mkt_mean_excess": float(mkt.mean()),
                "diff_mean": float(diff.mean()),
                "diff_se": se,
                "diff_t_stat": t_stat,
                "hit_rate": float((port > mkt).mean()),
            }
        )

    schema = {
        "label": pl.Utf8,
        "regime": pl.Utf8,
        "n_quarters": pl.Int64,
        "port_mean_excess": pl.Float64,
        "mkt_mean_excess": pl.Float64,
        "diff_mean": pl.Float64,
        "diff_se": pl.Float64,
        "diff_t_stat": pl.Float64,
        "hit_rate": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)
