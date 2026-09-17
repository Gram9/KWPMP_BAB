"""
Gate 3 (docs/02_validation_gates.md, design:
docs/superpowers/specs/2026-09-08-gate3-beta-estimator-design.md):
the Frazzini-Pedersen (2014) beta estimator --

    beta_hat_i = rho_i,m * (sigma_i / sigma_m)

-- assembled from components estimated on DIFFERENT frequencies and
windows (not a single OLS regression): sigma_i/sigma_m from 1-day log
returns over a shorter window, rho_i,m from overlapping multi-day log
returns over a longer window (the non-synchronous-trading correction
FP's own paper justifies -- see Test B in tests/estimation/test_beta_fp.py).
Shrunk toward a fixed target (Vasicek 1973, fixed weight, not
estimated): beta = shrinkage_weight * beta_hat + (1 - shrinkage_weight)
* shrinkage_target.

Every window length, minimum-observation count, overlap width, and
shrinkage parameter is read from config/beta_estimator.yaml (CLAUDE.md:
no magic numbers in src/) via _load_beta_config() -- never hardcoded
here, so a future estimator variant (weekly, 6-month, yearly windows)
is a new config block, not new code.

Market benchmark: the project's OWN self-built market index, never
vwretd/SPX/any external index -- spec section 7's explicit requirement
("estimation and evaluation benchmarks must be the same object, or a
non-zero realized loading can't be distinguished from a bug"). WHICH
index variant (uncapped / capped / MSCI-like) and WHICH leg (us /
canada) are both REQUIRED, keyword-only arguments on
_daily_log_returns() and _market_log_return_series() -- see those
functions' docstrings and F2b (docs/03_roadmap.md). Deliberately no
default: a silent default to vw_uncapped is exactly the bug this
wiring closes (F2a built config/market_index.yaml's three variants but
left this module hardcoded to the uncapped one regardless of what a
run's market_index: key named).
"""

import datetime
from pathlib import Path

import polars as pl
import yaml

from src.data import gate_adapters, universe_panel
from src.market_index import build as market_index_build
from src.portfolio import liquidity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "beta_estimator.yaml"

# Parallel-sibling dispatch, matching this codebase's existing convention
# (gate_adapters.us_gate1_panel/canada_gate1_panel, no shared registry) --
# not a "full unification" of the two legs (see gate_adapters.py's own
# docstring on why that was considered and rejected). Both functions
# already return the SAME common schema (id, date, price, shares, mkt_cap,
# ret), so this dict only needs to pick which one to call; the leg's
# internal shape stays wherever it already lives.
_LEG_PANELS = {
    "us": gate_adapters.us_gate1_panel,
    "canada": gate_adapters.canada_gate1_panel,
}


def _load_beta_config(variant: str = "fp_spec") -> dict:
    """The named estimator-variant block from config/beta_estimator.yaml
    -- window lengths, minimum-observation counts, overlap width, and
    shrinkage weight/target. Raises KeyError (not silently returning an
    empty dict) if `variant` isn't a block in the file -- a typo'd
    variant name must fail loudly, not silently estimate with an
    unintended (or missing) configuration."""
    with open(CONFIG_PATH) as f:
        all_variants = yaml.safe_load(f)
    return all_variants[variant]


def _daily_log_returns(
    start_date: datetime.date, end_date: datetime.date, *, leg: str
) -> pl.DataFrame:
    """Every name's raw daily log return for every trading day in
    [start_date, end_date] -- NOT filtered by universe_at() (spec
    decision 1: a name's beta-relevant return history must not have
    holes punched in it by month-by-month universe-eligibility filters
    unrelated to whether it actually traded).

    leg: "us" or "canada", REQUIRED keyword-only, no default -- F2b
    (docs/03_roadmap.md). "us" reads directly from the CRSP
    year-partition parquet files, mirroring gate_adapters.us_gate1_panel()'s
    own file-listing pattern but without that function's monthly-
    membership join. "canada" is not yet built: the Canadian stock-side
    beta path needs a non-permno identity key (CHASS ids are
    f"{symbol}_{usage}" strings, not Int64) that _estimate_beta_fp_core's
    joins don't yet support -- deferred to a later phase rather than
    faked here. Explicit now (raising by name) rather than silently
    reachable, so a future Canadian wiring pass is a single new branch
    here, not a second pass over every call site that already had to
    state `leg=` for this change.

    log_ret = ln(1 + dlyret) -- dlyret is total return (confirmed
    2026-09-07 in gate_adapters.py's own investigation: dlyret genuinely
    diverges from dlyretx on real ex-dividend dates), matching the same
    return-field convention Gate 1/2 already use.

    FIXED (post-commit review): years_needed must cover EVERY calendar
    year from start_date.year through end_date.year inclusive, not just
    the two endpoint years -- {start_date.year, end_date.year} silently
    dropped any year strictly in between (e.g. 2015-01-01..2018-12-31
    only anchored on {2015, 2018}, and since
    universe_panel._year_partition_files(year) only pulls in `year` and
    `year - 1`, 2016 was never read at all, with no error). This matters
    a great deal here: Task 3's rho_window_days=1260 (~5 years) rolling
    window spans 3+ calendar years for essentially every stock, so the
    old logic would have silently punched a hole in the middle of the
    rho-estimation history -- a wrong number that looks plausible, which
    is exactly the failure mode CLAUDE.md calls out as worst-case.
    gate_adapters.us_gate1_panel() avoids this because its year list
    comes from {d.year for d in month_ends} (every month in the range)
    unioned with the endpoints, not the endpoints alone.
    """
    if leg == "canada":
        raise NotImplementedError(
            "_daily_log_returns(leg='canada') -- the Canadian stock-side "
            "beta path is not built (docs/03_roadmap.md F2b/F2c: needs a "
            "non-permno identity key through _estimate_beta_fp_core's "
            "joins). The market-index leg (_market_log_return_series) "
            "works for Canada today; this per-name path does not."
        )
    if leg != "us":
        raise KeyError(f"leg must be 'us' or 'canada', got {leg!r}")

    years_needed = sorted(set(range(start_date.year, end_date.year + 1)))
    files = []
    for year in years_needed:
        files.extend(universe_panel._year_partition_files(year))
    files = sorted(set(files))
    if not files:
        return pl.DataFrame(
            schema={"permno": pl.Int64, "date": pl.Date, "log_ret": pl.Float64}
        )

    daily = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlyret"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(start_date).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(end_date).cast(pl.Datetime("ns")))
        )
        .filter(pl.col("dlyret").is_not_null())
        .collect(engine="streaming")
    )
    return daily.select(
        pl.col("permno"),
        pl.col("dlycaldt").cast(pl.Date).alias("date"),
        (pl.col("dlyret") + 1.0).log().alias("log_ret"),
    )


def _market_log_return_series(
    start_date: datetime.date, end_date: datetime.date, *, leg: str, market_index: str
) -> pl.DataFrame:
    """The project's own self-built market index (spec section 7 --
    NEVER vwretd/SPX/any external index, since "estimation and
    evaluation benchmarks must be the same object"), log-transformed.

    leg and market_index are BOTH REQUIRED keyword-only arguments, no
    default (F2b, docs/03_roadmap.md). A default of market_index=
    "vw_uncapped" would silently reproduce the exact bug this wiring
    closes: F2a built config/market_index.yaml's three variants
    (uncapped / capped 10% / MSCI-like) behind one entry point,
    src.market_index.build.build_index(panel, method), but until this
    function reads `market_index` a Canadian run naming
    market_index: vw_capped_10pct (config/runs.yaml's fp_baseline_ca)
    would still have its betas estimated against the uncapped index --
    spec section 7 / schema decision D10 require the estimation and
    evaluation benchmark be the SAME object, and a silent mismatch here
    is exactly the defect class D10 exists to forbid.

    leg selects the panel adapter (_LEG_PANELS: "us" ->
    gate_adapters.us_gate1_panel, "canada" -> gate_adapters.canada_gate1_panel
    -- both already return the common schema build_index() consumes, so
    NO identity-key work is needed here even though the Canadian
    stock-side path in _daily_log_returns() above is not yet built).
    _LEG_PANELS itself is used only for the leg-validation check just
    below; the actual panel build goes through
    market_index_build.build_index_chunked (not a direct
    _LEG_PANELS[leg](...) call -- see that function's docstring for why:
    a direct call over a multi-decade span materializes the whole span's
    per-name panel before ever collapsing it to the per-date index this
    function needs). market_index selects the weighting rule applied via
    build_index() -- the VW index is itself universe-filtered by design
    (that IS what makes it the project's market proxy), unlike
    _daily_log_returns()'s deliberately-unfiltered per-name history.
    """
    if leg not in _LEG_PANELS:
        raise KeyError(
            f"leg must be one of {sorted(_LEG_PANELS)}, got {leg!r}"
        )
    # market_index_build.build_index_chunked, NOT _LEG_PANELS[leg](...) +
    # build_index() directly -- a direct call materializes the WHOLE
    # requested span's per-name panel (e.g. ~71M rows for Gate 4's 56-year
    # US sample) before ever collapsing it to the ~14,000-row per-date
    # index this function actually returns, which crashed with a Rust
    # memory allocation failure (Gate 4, docs/02_validation_gates.md,
    # 2026-09-13 -- no caller before Gate 4 ever requested more than a
    # single year here). build_index_chunked computes the same index one
    # calendar year at a time, keeping only the small per-date result;
    # see its own docstring for the bit-identical proof against the
    # direct/unchunked path.
    index = market_index_build.build_index_chunked(leg, start_date, end_date, market_index)
    return index.select(
        pl.col("date"),
        (pl.col("index_ret") + 1.0).log().alias("log_ret"),
    )


def _resolve_window_starts(
    month_end: datetime.date,
    market_returns: pl.DataFrame,
    cfg: dict,
    *,
    formation_lag_days: int = 0,
) -> tuple[datetime.date | None, datetime.date | None, datetime.date | None]:
    """THE single definition of this module's window boundaries. Returns
    (sigma_window_start, rho_window_start, lookback_end), or
    (None, None, None) if no market date precedes month_end.

    Window boundaries are TRADING days, not calendar days (gate-verifier
    finding, 2026-09-10): bab-methodology's spec describes
    sigma_window_days=252 / rho_window_days=1260 as "1 year" / "5 years"
    of trading-day counts (252 trading days/year is the standard
    convention), but the original implementation subtracted them as
    calendar days -- month_end - timedelta(days=252) only spans ~173
    actual trading days (weekends/holidays), making sigma_min_obs=120 /
    rho_min_obs=750 far more binding than the spec intended (measured:
    1528 of 3772 names excluded on rho at 2015-06-30 purely from this
    miscalibration). Resolved from the MARKET's own trading-day calendar:
    "the last N trading days" is the Nth date back from lookback_end in
    the sorted list of distinct market dates. Both legs (the market's own
    sigma_m/rho, and every permno's sigma_i/rho) share this boundary,
    matching FP's own "sigma from 1yr, rho from 5yr" framing -- the
    boundary is a property of calendar time as observed by the market,
    not a per-permno construct.

    Formation timing (CLAUDE.md's core rule): lookback_end is the last
    market date STRICTLY BEFORE month_end, so no row dated on or after
    month_end can enter either window. Callers must clip stock data at
    `<= lookback_end`, never merely at `< month_end` -- see below.

    EXTRACTED 2026-09-10 (Blocker 2). This logic previously existed as
    two separately-written copies, in _estimate_beta_fp_core and in
    estimate_beta_fp_diagnostics. Commit d8aacfe's message claimed the
    diagnostic "shares the same trading-day boundary resolution logic";
    that claim was FALSE, and the copies had already diverged twice:
      (1) the short-panel fallback branch differed between them, and
      (2) core clipped stock data at `<= lookback_end` while the
          diagnostic clipped at `< month_end` with no lookback_end
          concept at all -- so with a market series ending before the
          stock series, the diagnostic admitted observations the
          estimator excluded (measured: 255 vs 252 sigma-window
          observations for a 3-day gap) and reported exclusion counts
          that did not match what the estimator actually excluded.
    Both callers now resolve boundaries HERE, so the two cannot drift
    apart again: a change to this function moves both, which
    test_resolve_window_starts_is_the_single_boundary_definition and
    test_diagnostics_and_core_agree_when_market_series_ends_early pin.

    Market dates are DEDUPLICATED (sorted(set(...))) before indexing. The
    whole trading-day approach rests on one row per trading day; taking
    the Nth ROW back rather than the Nth distinct trading day back would
    silently shorten the window if a date were ever duplicated. Real data
    satisfies uniqueness today (1280 rows / 1280 distinct dates at
    2015-06-30), which is precisely why a regression here would be
    invisible.

    formation_lag_days (config/runs.yaml, enum {0, 1}): a further BACKWARD
    skip applied AFTER the t-1 month-end rule above -- spec Section 10's
    skip-one-day robustness variant, so the closing prices ending the
    estimation window do not also produce the first held return. Default
    0 is a no-op (every existing positional call site is unaffected).
    Dropping the last N dates before indexing SLIDES the whole window back
    by N trading days rather than shrinking it by N observations. Bounded
    to {0, 1}: the schema must not be ABLE to express a leak, and a
    negative value would move the boundary FORWARD, which is exactly what
    CLAUDE.md's rule forbids.
    """
    if formation_lag_days not in (0, 1):
        raise ValueError(
            "formation_lag_days must be 0 or 1, got "
            f"{formation_lag_days!r} -- a negative value would move the "
            "formation boundary forward, which is not permitted"
        )

    market_dates = sorted(
        {d for d in market_returns["date"].to_list() if d < month_end}
    )
    if formation_lag_days:
        market_dates = market_dates[:-formation_lag_days]
    if not market_dates:
        return None, None, None

    lookback_end = market_dates[-1]
    sigma_window_start = (
        market_dates[-cfg["sigma_window_days"]]
        if len(market_dates) >= cfg["sigma_window_days"]
        else market_dates[0]
    )
    rho_window_start = (
        market_dates[-cfg["rho_window_days"]]
        if len(market_dates) >= cfg["rho_window_days"]
        else market_dates[0]
    )
    return sigma_window_start, rho_window_start, lookback_end


def _real_obs(column: str = "log_ret") -> pl.Expr:
    """A polars predicate for "this observation is genuinely present".

    Blocker 2 hardening (b), 2026-09-10: `is_not_null()` does NOT filter
    NaN in polars, and `.count()` counts NaN as a valid observation. A
    name with 50 real observations and 202 NaNs would therefore clear
    sigma_min_obs=120 on a count of 252 -- producing a beta from 50
    points while appearing to satisfy the minimum. Production is
    currently safe by ordering rather than by design (_daily_log_returns
    filters nulls before the log transform, and the real 2015-06-30 panel
    measures 0 NaN / 0 inf), but the estimator must not depend on its
    caller for that. Every min-obs gate and every correlation/std input
    in this module filters through here.
    """
    return pl.col(column).is_not_null() & pl.col(column).is_not_nan()


def _overlapping_log_returns(series: pl.DataFrame, overlap_days: int) -> pl.DataFrame:
    """series: [date, log_ret]. Returns [date, log_ret] where log_ret is
    now the rolling sum of the trailing `overlap_days` daily log returns
    ending at (and including) that date -- log returns are additive, so
    an N-day overlapping return is just a rolling sum. The first
    (overlap_days - 1) rows have no full window yet and get a null
    rolling sum (polars' rolling_sum default); callers filter these out
    via a non-null count against rho_min_obs."""
    return series.sort("date").with_columns(
        pl.col("log_ret").rolling_sum(window_size=overlap_days).alias("log_ret")
    )


def _overlapping_log_returns_by_permno(
    frame: pl.DataFrame,
    overlap_days: int,
    alias: str = "log_ret",
    market_returns: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-permno equivalent of _overlapping_log_returns: the rolling sum
    of the trailing `overlap_days` daily log returns, computed WITHIN each
    permno via .over("permno") so one name's window never bleeds into the
    next name's rows.

    CONSOLIDATED 2026-09-10 (Blocker 2, the third duplicated copy). This
    construction previously appeared twice -- in _estimate_beta_fp_core
    and in estimate_beta_fp_diagnostics -- which an earlier review round
    flagged as a live divergence risk (both must read the same
    cfg["rho_overlap_days"] for their counts to be comparable) and which
    was never consolidated. Both now call here.

    Kept as a grouped polars expression rather than calling
    _overlapping_log_returns in a per-group Python loop: the loop would
    share marginally more code but is far slower on a real-size panel
    (thousands of permnos x ~1260 rows each), and Blocker 2's own
    instruction is not to trade real performance for code sharing. The
    duplication that mattered -- two independent call sites free to drift
    apart -- is gone; what remains in common with the market-series
    helper is two lines of polars with deliberately different grouping
    semantics.

    GAP GUARD (leakage-auditor finding, 2026-09-10). `rolling_sum` counts
    ROWS, not trading days. For a name with a hole in its series (a halt,
    a suspension, a delisting-and-relisting), the window at the resume
    date therefore summed the resume-day return together with the last
    pre-gap returns -- fabricating an "overlapping return" that actually
    spanned the entire absence, which was then correlated against a
    genuine `overlap_days`-day market return. Not a lookahead defect
    (every summand is in the past) but a real measurement error, and one
    concentrated in exactly the halted/illiquid names that populate BAB's
    low-beta long leg. Measured on real 2015-06-30 data before this
    guard: 88 of 5279 names had internal missing trading days, 363
    overlapping observations were bridged, max |rho| error 0.0421 (permno
    88421: 0.1376 -> 0.1797), mean 0.0001 -- and because the worst name
    ran sigma_i/sigma_m = 6.04, that rho error implies a beta_shrunk
    error of 0.1525, 42% of Test C's whole cross-sectional SD, on one
    name. No name's rho_min_obs membership changed (delta 0).

    When `market_returns` is supplied, each date is indexed into the
    market's own trading-day calendar and the overlapping return is
    nulled unless the window spans exactly `overlap_days` consecutive
    market trading days. Callers filter nulls before counting against
    rho_min_obs, so a bridged window is excluded rather than
    silently mismeasured. Still a grouped polars expression -- no
    per-group Python loop.
    """
    out = frame.sort(["permno", "date"]).with_columns(
        pl.col("log_ret")
        .rolling_sum(window_size=overlap_days)
        .over("permno")
        .alias(alias)
    )
    if market_returns is None or overlap_days <= 1:
        return out

    # Position of each date in the market's trading-day calendar. A valid
    # overlap_days window ends `overlap_days - 1` market days after it
    # starts; anything wider spans a gap in this name's own series.
    calendar = {
        d: i for i, d in enumerate(sorted(set(market_returns["date"].to_list())))
    }
    return (
        out.with_columns(
            pl.col("date").replace_strict(calendar, default=None).alias("_cal_idx")
        )
        .with_columns(
            (
                pl.col("_cal_idx") - pl.col("_cal_idx").shift(overlap_days - 1)
            )
            .over("permno")
            .alias("_cal_span")
        )
        .with_columns(
            pl.when(pl.col("_cal_span") == overlap_days - 1)
            .then(pl.col(alias))
            .otherwise(None)
            .alias(alias)
        )
        .drop(["_cal_idx", "_cal_span"])
    )


def _estimate_beta_fp_core(
    month_end: datetime.date,
    daily_returns: pl.DataFrame,
    market_returns: pl.DataFrame,
    variant: str = "fp_spec",
) -> pl.DataFrame:
    """The FP beta math ONLY -- sigma_i, sigma_m, rho, beta_raw,
    beta_shrunk for every permno in `daily_returns` with sufficient
    history, with NO universe_at()/market_cap_at() output filtering
    (unlike estimate_beta_fp, which wraps this and adds that filtering).
    Factored out so callers that need per-date betas for SYNTHETIC
    securities (e.g. the leakage adapter's estimate_betas(), which has no
    real permnos to look up in universe_at()) can reuse the exact same
    formation-timing-correct math without a real-panel dependency.

    Formation timing (CLAUDE.md's core rule): every lookback window ends
    STRICTLY BEFORE month_end -- resolved from market_returns' own dates
    by _resolve_window_starts (never assumed to be month_end minus one
    calendar day), so no row dated on or after month_end ever enters
    either window. estimate_beta_fp_diagnostics resolves its boundaries
    through that same helper, so the estimator and its exclusion
    diagnostic cannot drift apart.
    """
    cfg = _load_beta_config(variant)

    empty_schema = {
        "permno": pl.Int64, "sigma_i": pl.Float64, "sigma_m": pl.Float64,
        "rho": pl.Float64, "beta_raw": pl.Float64, "beta_shrunk": pl.Float64,
    }
    # Window boundaries come from the ONE shared definition
    # (_resolve_window_starts) -- trading days, deduplicated, resolved
    # from the market's own calendar. estimate_beta_fp_diagnostics calls
    # the same helper, so the two cannot diverge (Blocker 2, 2026-09-10).
    sigma_window_start, rho_window_start, lookback_end = _resolve_window_starts(
        month_end, market_returns, cfg
    )
    if lookback_end is None:
        return pl.DataFrame(schema=empty_schema)

    # sigma_m: market volatility over the trailing sigma window, strictly
    # before month_end.
    market_sigma_window = market_returns.filter(
        (pl.col("date") >= sigma_window_start)
        & (pl.col("date") <= lookback_end)
        & _real_obs()
    )
    sigma_m = market_sigma_window["log_ret"].std()
    if sigma_m is None or sigma_m == 0.0:
        return pl.DataFrame(schema=empty_schema)

    # rho inputs: the market's own overlapping-return series, same window.
    market_rho_window = market_returns.filter(
        (pl.col("date") >= rho_window_start)
        & (pl.col("date") <= lookback_end)
        & _real_obs()
    )
    market_overlap = _overlapping_log_returns(market_rho_window, cfg["rho_overlap_days"])

    # Per-permno sigma_i, independent min-obs gate. NaN-aware: the count
    # gating sigma_min_obs must count only genuinely-present
    # observations, never NaN (Blocker 2 hardening (b) -- see _real_obs).
    sigma_data = daily_returns.filter(
        (pl.col("date") >= sigma_window_start)
        & (pl.col("date") <= lookback_end)
        & (pl.col("date") < month_end)
        & _real_obs()
    )
    sigma_by_permno = (
        sigma_data.group_by("permno")
        .agg(
            pl.col("log_ret").std().alias("sigma_i"),
            pl.len().alias("n_obs_sigma"),
        )
        .filter(pl.col("n_obs_sigma") >= cfg["sigma_min_obs"])
        .select(["permno", "sigma_i"])
    )

    # Per-permno rho, independent min-obs gate -- via a join against the
    # market's own overlapping series on date.
    rho_data = daily_returns.filter(
        (pl.col("date") >= rho_window_start)
        & (pl.col("date") <= lookback_end)
        & (pl.col("date") < month_end)
        & _real_obs()
    )
    # Same overlap construction as the market series above, via the
    # shared _overlapping_log_returns_by_permno helper -- both paths read
    # the SAME cfg["rho_overlap_days"] and the construction now exists in
    # one place (Blocker 2 consolidation, 2026-09-10; previously a second
    # hand-written rolling_sum call here and a third in the diagnostic).
    stock_overlap = _overlapping_log_returns_by_permno(
        rho_data, cfg["rho_overlap_days"], market_returns=market_rho_window
    )
    joined_overlap = stock_overlap.join(
        market_overlap.select(pl.col("date"), pl.col("log_ret").alias("market_log_ret")),
        on="date",
        how="inner",
    ).filter(_real_obs() & _real_obs("market_log_ret"))

    rho_by_permno = (
        joined_overlap.group_by("permno")
        .agg(
            pl.corr("log_ret", "market_log_ret").alias("rho"),
            pl.len().alias("n_obs_rho"),
        )
        .filter(pl.col("n_obs_rho") >= cfg["rho_min_obs"])
        .select(["permno", "rho"])
    )

    # A permno needs BOTH gates to pass -- inner join, never a partial row.
    betas = sigma_by_permno.join(rho_by_permno, on="permno", how="inner")
    # A name that never moved (a halted or flat security) clears BOTH
    # min-obs gates on observation count, then yields sigma_i == 0 and a
    # NaN rho from pl.corr -- producing a NaN beta rather than an
    # exclusion (gate-verifier, second round, 2026-09-10). That is the
    # same null-vs-NaN confusion _real_obs() fixes on the INPUT side,
    # reproduced one layer down on the output side: the smoke test's
    # null_count() == 0 does not catch it, because NaN is not null in
    # polars. Real 2015-06-30 output measures 0 NaN, so production was
    # safe by data rather than by design. Exclude such names outright --
    # a beta is not estimable from a series with no variance, and a
    # partial/NaN row violates this function's own contract.
    betas = betas.filter(
        _real_obs("sigma_i")
        & _real_obs("rho")
        & (pl.col("sigma_i") > 0.0)
    )
    if betas.height == 0:
        return pl.DataFrame(schema=empty_schema)

    betas = betas.with_columns(
        pl.lit(sigma_m).alias("sigma_m"),
        ((pl.col("rho") * pl.col("sigma_i")) / pl.lit(sigma_m)).alias("beta_raw"),
    )
    betas = betas.with_columns(
        (
            cfg["shrinkage_weight"] * pl.col("beta_raw")
            + (1 - cfg["shrinkage_weight"]) * cfg["shrinkage_target"]
        ).alias("beta_shrunk")
    )
    return betas.select(
        ["permno", "sigma_i", "sigma_m", "rho", "beta_raw", "beta_shrunk"]
    )


def estimate_beta_fp(
    month_end: datetime.date,
    daily_returns: pl.DataFrame,
    market_returns: pl.DataFrame,
    variant: str = "fp_spec",
) -> pl.DataFrame:
    """The Frazzini-Pedersen (2014) beta estimator for every permno with
    sufficient history in `daily_returns`, filtered to
    universe_panel.universe_at(month_end)-eligible permnos with a
    resolvable universe_panel.market_cap_at(month_end) value (spec
    decision 1: universe_at() gates OUTPUT, never lookback history), AND
    passing the spec section 3 liquidity filter (F2b, docs/03_roadmap.md):
    minimum non-zero-volume days within the sigma window, plus a
    sub-penny price floor at the formation date's lagged close --
    src.portfolio.liquidity.liquid_permnos().

    daily_returns, market_returns: precomputed via
    _daily_log_returns()/_market_log_return_series() by the CALLER, over
    whatever date range covers this (and any other) month_end being
    estimated -- not re-fetched here.

    US only, same as universe_at()/market_cap_at() and
    src.portfolio.liquidity (both hardcoded to the US panel today --
    F2b/F2c). The liquidity gate is applied HERE, not inside
    _estimate_beta_fp_core, deliberately: the core function is also used
    by the leakage adapter's estimate_betas() on SYNTHETIC securities
    with no real permnos to look up volume/price for (see the core's own
    docstring) -- liquidity is a real-panel-only concern, exactly like
    universe_at()/market_cap_at() just above it.

    Returns one row per included permno: permno, mkt_cap (LAGGED,
    universe_panel.market_cap_at(month_end)'s existing convention),
    sigma_i, sigma_m, rho, beta_raw, beta_shrunk. A permno is included
    only if it has sufficient history for BOTH sigma and rho
    independently, AND resolves a non-null market_cap_at() value, AND is
    universe_at(month_end)-eligible, AND clears BOTH liquidity gates --
    any one of these failing excludes the permno entirely, never
    producing a partial/null row.
    """
    betas = _estimate_beta_fp_core(month_end, daily_returns, market_returns, variant)
    if betas.height == 0:
        return betas.with_columns(pl.lit(None, dtype=pl.Float64).alias("mkt_cap")).select(
            ["permno", "mkt_cap", "sigma_i", "sigma_m", "rho", "beta_raw", "beta_shrunk"]
        )

    eligible = universe_panel.universe_at(month_end)
    mkt_cap_df, _coverage = universe_panel.market_cap_at(month_end)

    betas = betas.join(eligible.select("permno"), on="permno", how="inner")
    betas = betas.join(mkt_cap_df, on="permno", how="inner")

    if betas.height > 0:
        cfg = _load_beta_config(variant)
        sigma_window_start, _rho_window_start, lookback_end = _resolve_window_starts(
            month_end, market_returns, cfg
        )
        # betas.height > 0 already implies _estimate_beta_fp_core resolved
        # a non-None lookback_end (it returns empty otherwise) -- this
        # guard is for the type checker and defensive correctness, not a
        # path expected to trigger in practice.
        if sigma_window_start is not None and lookback_end is not None:
            liquid = liquidity.liquid_permnos(
                betas["permno"].to_list(),
                sigma_window_start=sigma_window_start,
                lookback_end=lookback_end,
                # month_end, not lookback_end -- formation_date_lagged_price()
                # internally shifts back one trading day from formation_date,
                # so passing lookback_end here would price the sub-penny gate
                # a day EARLIER than mkt_cap (leakage-auditor finding,
                # 2026-09-12; see test_estimate_beta_fp_sub_penny_gate_prices_
                # same_day_as_mkt_cap). month_end's lag IS lookback_end,
                # matching universe_panel.market_cap_at()'s pricing date
                # exactly, as this module's own docstring claims.
                formation_date=month_end,
                leg="us",
            )
            betas = betas.join(liquid.select("permno"), on="permno", how="inner")

    return betas.select(
        ["permno", "mkt_cap", "sigma_i", "sigma_m", "rho", "beta_raw", "beta_shrunk"]
    )


def estimate_beta_fp_diagnostics(
    month_end: datetime.date,
    daily_returns: pl.DataFrame,
    market_returns: pl.DataFrame,
    variant: str = "fp_spec",
) -> dict:
    """Spec section 6's required diagnostic: the count of permnos
    excluded from estimate_beta_fp()'s output by insufficient sigma
    history and insufficient rho history, counted independently (a
    permno can fail either or both). Recent IPOs are systematically
    high-beta/high-vol, so this exclusion mechanically thins one tail --
    the count must be visible, not silent.

    market_returns is REQUIRED (was daily_returns-only before
    2026-09-10): window boundaries are trading days, not calendar days,
    and resolving "the last N trading days" needs the market's own
    trading calendar.

    Boundary resolution is DELEGATED to _resolve_window_starts, shared
    with _estimate_beta_fp_core. Commit d8aacfe's message claimed this
    function already "shares the same trading-day boundary resolution
    logic" as the core estimator -- that claim was false: it was a
    separately-written second copy, and the two had already diverged in
    the short-panel fallback branch and in the stock-data clip
    (`< month_end` here vs `<= lookback_end` in core). Corrected
    2026-09-10 (Blocker 2); the counts this function reports are
    reconciled against the estimator's real output by
    test_estimate_beta_fp_diagnostics_reports_exclusions.
    """
    cfg = _load_beta_config(variant)
    # The SAME shared boundary definition _estimate_beta_fp_core uses --
    # including its `<= lookback_end` clip on stock data, which this
    # function previously lacked entirely (Blocker 2, 2026-09-10). With
    # a market series ending before the stock series, the old
    # `< month_end` clip admitted observations the estimator excluded
    # (255 vs 252 for a 3-day gap) and reported exclusion counts that did
    # not tie to the estimator's real output.
    sigma_window_start, rho_window_start, lookback_end = _resolve_window_starts(
        month_end, market_returns, cfg
    )
    if lookback_end is None:
        return {"n_excluded_sigma": 0, "n_excluded_rho": 0}

    all_permnos = set(
        daily_returns.filter(pl.col("date") < month_end)["permno"].unique().to_list()
    )

    sigma_data = daily_returns.filter(
        (pl.col("date") >= sigma_window_start)
        & (pl.col("date") <= lookback_end)
        & (pl.col("date") < month_end)
        & _real_obs()
    )
    sigma_ok = set(
        sigma_data.group_by("permno")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= cfg["sigma_min_obs"])["permno"]
        .to_list()
    )

    rho_data = daily_returns.filter(
        (pl.col("date") >= rho_window_start)
        & (pl.col("date") <= lookback_end)
        & (pl.col("date") < month_end)
        & _real_obs()
    )
    rho_counts = (
        _overlapping_log_returns_by_permno(
            rho_data,
            cfg["rho_overlap_days"],
            alias="_overlap_ret",
            # Same calendar the estimator uses, so the diagnostic's
            # counts continue to reconcile exactly against its output.
            market_returns=market_returns.filter(
                (pl.col("date") >= rho_window_start)
                & (pl.col("date") <= lookback_end)
                & _real_obs()
            ),
        )
        .filter(_real_obs("_overlap_ret"))
        .group_by("permno")
        .agg(pl.len().alias("n"))
    )
    rho_ok = set(rho_counts.filter(pl.col("n") >= cfg["rho_min_obs"])["permno"].to_list())

    return {
        "n_excluded_sigma": len(all_permnos - sigma_ok),
        "n_excluded_rho": len(all_permnos - rho_ok),
        # n_excluded_either closes an algebraic hole in the two-sided
        # bracket the reconciliation test used to rely on alone
        # (gate-verifier, second round, 2026-09-10): max(a, b) and a + b
        # are both SYMMETRIC in their arguments, so transposing the sigma
        # and rho labels satisfied the bracket exactly as well as the
        # correct order did -- undetectable by any threshold choice,
        # since it is an invariance rather than a tolerance. This count
        # is the union (a permno failing EITHER gate), which equals the
        # number the estimator actually drops EXACTLY, not within a
        # bracket: measured 1985 == 1985 at 2015-06-30, set symmetric
        # difference 0. Asserting exact equality on this, plus the two
        # individual counts pinned, is what makes a label swap fail.
        "n_excluded_either": len(all_permnos - (sigma_ok & rho_ok)),
    }


def decompose_beta_variance(betas: pl.DataFrame) -> dict:
    """Spec section 8's required diagnostic: how much of beta_raw's
    cross-sectional variance is explained by sigma_i alone vs. rho
    alone, via two SEPARATE univariate R^2 fits (never one multivariate
    fit on both). Because rho is estimated over a 5y window and sigma
    only over 1y, real-data dispersion is expected to be sigma-
    dominated -- informational only, no pass/fail threshold; the R^2
    values are recorded in docs, not gated here.
    """
    import numpy as np

    def _r_squared(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) == 0.0:
            return 0.0
        correlation = np.corrcoef(x, y)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return float(correlation**2)

    sigma_i = betas["sigma_i"].to_numpy()
    rho = betas["rho"].to_numpy()
    beta_raw = betas["beta_raw"].to_numpy()

    return {
        "r2_sigma": _r_squared(sigma_i, beta_raw),
        "r2_rho": _r_squared(rho, beta_raw),
    }
