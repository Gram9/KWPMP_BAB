"""
Gate 1 (docs/02_validation_gates.md): rebuild a value-weighted market
return from our own universe/return data and compare against an
externally-published benchmark. Validates the return merge, delisting
handling, market-cap calculation, and date alignment simultaneously --
"one test, four failure modes covered."

US leg: build_vw_index() over src/data/gate_adapters.py's
us_gate1_panel() output, compared against CRSP's own vwretd from the
same underlying panel (crsp.wrds_dsfv2_query / the local
us_panel_crsp_full parquet). Per the validation-gates doc, expect a
match to rounding -- run_gate1_us() reports correlation rather than
asserting exact equality, since the self-built universe (REIT/MLP-
excluded, major-exchange-only, per config/universe.yaml's
us_crsp_primary block) is a deliberate subset of vwretd's own universe
construction, not a byte-identical reproduction.

Known failure modes to check first if correlation is low (same order
docs/02_validation_gates.md's Gate 4 debug list gives): unlagged
weights, retx used instead of ret, delisting returns excluded on one
side, free-float instead of total shares. gate_adapters.us_gate1_panel()
already guards the first three (lagged mkt_cap, dlyret not dlyretx,
shares from shrout which is total shares) -- a low correlation here most
likely means a genuine universe-definition or weighting difference
against vwretd's own construction, not a repeat of an already-fixed bug.
"""

import datetime

import polars as pl

from src.data import chass_loader, gate_adapters, universe_panel


def build_vw_index(
    panel: pl.DataFrame, extra_group_cols: list[str] | None = None
) -> pl.DataFrame:
    """Value-weighted daily index return from a common-schema panel
    (id, date, price, shares, mkt_cap, ret -- see gate_adapters.py).

    index_ret[date] = sum(mkt_cap_i * ret_i) / sum(mkt_cap_i) across all
    names present on that date with a non-null mkt_cap AND a non-null
    ret -- both must be non-null for a row to contribute. A null mkt_cap
    (no resolvable lagged price -- gate_adapters.py already excludes
    these from its own output, but this function re-guards independently
    rather than assuming its caller's contract) or a null ret (e.g. a
    name's first day in the panel, with nothing to compute a return
    against) is excluded from both the numerator and denominator, never
    treated as a zero weight or a zero return that would silently
    distort the index -- a null ret with a non-null mkt_cap is
    particularly dangerous, since polars' .sum() skips the null in the
    numerator while a naive mkt_cap-only filter would still count that
    name's full weight in the denominator, silently handing it an
    effective 0% return instead of excluding it outright.

    extra_group_cols: additional columns to group by alongside date --
    e.g. ["decile"] for Gate 2's per-decile VW index, where the weighted
    average must be computed independently within each decile rather
    than pooling mkt_cap across deciles into one index. None (default)
    preserves Gate 1's original single-index behavior, grouping by date
    alone -- existing callers (run_gate1_us, run_gate1_canada) are
    unaffected.

    Returns one row per date (plus one row per extra_group_cols
    combination, if given): date [, *extra_group_cols], index_ret.
    """
    group_cols = ["date"] + (extra_group_cols or [])
    weighted = panel.filter(pl.col("mkt_cap").is_not_null() & pl.col("ret").is_not_null())
    return (
        weighted.group_by(group_cols)
        .agg(
            (pl.col("mkt_cap") * pl.col("ret")).sum().alias("_weighted_sum"),
            pl.col("mkt_cap").sum().alias("_total_mkt_cap"),
        )
        .with_columns(
            (pl.col("_weighted_sum") / pl.col("_total_mkt_cap")).alias("index_ret")
        )
        .select(group_cols + ["index_ret"])
        .sort(group_cols)
    )


def _vwretd_series(start_date: datetime.date, end_date: datetime.date) -> pl.DataFrame:
    """CRSP's own vwretd, one row per date, from the same underlying
    panel gate_adapters.us_gate1_panel() reads -- the actual external
    benchmark this gate compares against."""
    years_needed = sorted({start_date.year, end_date.year})
    files = []
    for year in years_needed:
        path = universe_panel.RAW_DATA_DIR / f"year={year}" / "part.parquet"
        if path.exists():
            files.append(str(path))
    if not files:
        return pl.DataFrame(schema={"date": pl.Date, "vwretd": pl.Float64})

    vwretd = (
        pl.scan_parquet(files)
        .select(["dlycaldt", "vwretd"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(start_date).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(end_date).cast(pl.Datetime("ns")))
        )
        .unique()
        .collect(engine="streaming")
    )
    return vwretd.select(
        pl.col("dlycaldt").cast(pl.Date).alias("date"), pl.col("vwretd")
    ).sort("date")


def run_gate1_us(start_date: datetime.date, end_date: datetime.date) -> dict:
    """Runs Gate 1 for the US leg over [start_date, end_date]: builds the
    self-built VW index, pulls vwretd for the same dates, and reports the
    comparison.

    Returns a dict with:
    - "comparison": date, index_ret, vwretd, diff -- an inner join, so
      any date present in only one series is silently absent here rather
      than raising; see the "n_index_only"/"n_vwretd_only" counts below
      to check whether that happened before trusting the correlation.
    - "correlation": Pearson correlation between index_ret and vwretd
      over the joined dates.
    - "n_dates": number of dates in the comparison.
    - "n_index_only" / "n_vwretd_only": dates present in one series but
      not the other -- a date-alignment red flag if nonzero (CLAUDE.md:
      a wrong number that looks right is the worst possible outcome --
      an inner join can silently hide a real gap unless this is checked).
    """
    panel = gate_adapters.us_gate1_panel(start_date, end_date)
    index = build_vw_index(panel)
    vwretd = _vwretd_series(start_date, end_date)

    index_dates = set(index["date"].to_list())
    vwretd_dates = set(vwretd["date"].to_list())

    comparison = index.join(vwretd, on="date", how="inner").with_columns(
        (pl.col("index_ret") - pl.col("vwretd")).alias("diff")
    )

    correlation = comparison.select(pl.corr("index_ret", "vwretd")).item()

    return {
        "comparison": comparison,
        "correlation": correlation,
        "n_dates": comparison.height,
        "n_index_only": len(index_dates - vwretd_dates),
        "n_vwretd_only": len(vwretd_dates - index_dates),
    }


def compound_daily_index_to_monthly(daily_index: pl.DataFrame) -> pl.DataFrame:
    """Chain-links a daily VW index (date, index_ret -- build_vw_index()'s
    own output) up to one compounded return per calendar month:
    prod(1 + index_ret) - 1 over each month's trading days.

    month_end is the LAST DATE ACTUALLY PRESENT in that month's data, not
    a naive calendar month-end (e.g. 2015-01-30, CHASS's own real last
    trading day of January 2015, not 2015-01-31) -- matches the real
    trading-date convention gate_adapters.canada_gate1_panel() already
    uses, so this frame's month_end joins directly against CHASS's own
    ind7 series keyed on the same real dates.

    Compounding happens on the already-value-weighted DAILY index, not on
    per-name returns compounded first and then weighted -- weighting
    then compounding and compounding then weighting are different
    calculations, and only the former is correct (per the Gate 1 Canada
    handoff doc).
    """
    return (
        daily_index.sort("date")
        .with_columns(
            pl.col("date").dt.year().alias("_year"),
            pl.col("date").dt.month().alias("_month"),
        )
        .group_by(["_year", "_month"])
        .agg(
            pl.col("date").max().alias("month_end"),
            (pl.col("index_ret") + 1.0).product().alias("_compounded"),
        )
        .with_columns((pl.col("_compounded") - 1.0).alias("monthly_ret"))
        .select(["month_end", "monthly_ret"])
        .sort("month_end")
    )


def _tsx_composite_tr_monthly(
    start_date: datetime.date, end_date: datetime.date
) -> pl.DataFrame:
    """CHASS's own S&P/TSX Composite Monthly Total Return Index
    (ind7), converted from a LEVEL series to month-over-month pct
    change -- the actual external benchmark this leg of the gate
    compares against. ind7 exists only at monthly grain (confirmed
    2026-09-07: the daily file's ind1 is a price index, not total
    return, so it cannot stand in here -- see the Gate 1 Canada handoff
    doc)."""
    col = "ind7-S&P/TSX Composite Monthly Total Return Index"
    levels = (
        chass_loader.load_monthly()
        .select(["trdate-Trade Date", col])
        .unique()
        .sort("trdate-Trade Date")
        .with_columns(pl.col("trdate-Trade Date").cast(pl.Date).alias("month_end"))
    )
    levels = levels.with_columns(
        (pl.col(col) / pl.col(col).shift(1) - 1.0).alias("benchmark_ret")
    )
    return levels.filter(
        (pl.col("month_end") >= start_date) & (pl.col("month_end") <= end_date)
    ).select(["month_end", "benchmark_ret"])


def run_gate1_canada(start_date: datetime.date, end_date: datetime.date) -> dict:
    """Runs Gate 1 for the Canada leg over [start_date, end_date]: builds
    the self-built daily VW index from canada_gate1_panel(), compounds it
    to monthly, and compares against CHASS's own S&P/TSX Composite
    Monthly Total Return Index (ind7, converted to pct-change).

    Unlike run_gate1_us(), this comparison happens at MONTHLY grain --
    CHASS carries no daily total-return benchmark (see module-level
    reasoning in _tsx_composite_tr_monthly and the Gate 1 Canada handoff
    doc). Returns the same shape as run_gate1_us(), with the vwretd-
    specific keys renamed since there's no vwretd on this side:

    - "comparison": month_end, monthly_ret, benchmark_ret, diff -- an
      inner join, so a month present in only one series is silently
      absent here; check n_index_only/n_benchmark_only before trusting
      the correlation.
    - "correlation": Pearson correlation between monthly_ret and
      benchmark_ret over the joined months.
    - "n_dates": number of months in the comparison.
    - "n_index_only" / "n_benchmark_only": months present in one series
      but not the other.
    """
    # Widen the daily panel's read window by one month on each side so
    # the first/last in-range month's compounding isn't starved of
    # trading days at the boundary, then compound and filter back to the
    # requested range -- mirrors us_gate1_panel's own boundary-widening
    # convention for the same reason (a lag/window needs data just
    # outside the nominal range to resolve its first/last real value).
    panel_start = start_date.replace(day=1) - datetime.timedelta(days=1)
    panel_start = panel_start.replace(day=1)
    panel = gate_adapters.canada_gate1_panel(panel_start, end_date)
    daily_index = build_vw_index(panel)
    monthly_index = compound_daily_index_to_monthly(daily_index)
    monthly_index = monthly_index.filter(
        (pl.col("month_end") >= start_date) & (pl.col("month_end") <= end_date)
    )

    benchmark = _tsx_composite_tr_monthly(start_date, end_date)

    index_months = set(monthly_index["month_end"].to_list())
    benchmark_months = set(benchmark["month_end"].to_list())

    comparison = monthly_index.join(benchmark, on="month_end", how="inner").with_columns(
        (pl.col("monthly_ret") - pl.col("benchmark_ret")).alias("diff")
    )

    correlation = comparison.select(pl.corr("monthly_ret", "benchmark_ret")).item()

    return {
        "comparison": comparison,
        "correlation": correlation,
        "n_dates": comparison.height,
        "n_index_only": len(index_months - benchmark_months),
        "n_benchmark_only": len(benchmark_months - index_months),
    }
