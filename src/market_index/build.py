"""
Market index construction, Phase F2a (docs/03_roadmap.md). Every run in
config/runs.yaml names a `market_index:` variant, resolved here, defined in
config/market_index.yaml (CLAUDE.md: no magic numbers in src/ -- the 10%
cap, its iteration guard, and the 85% coverage target all live there).

Consumes the SAME common-schema panel every downstream caller already uses
(src/data/gate_adapters.py: id, date, mkt_cap, ret -- mkt_cap is already
LAGGED, prior trading day's close x shares). Nothing here re-derives
mkt_cap or re-lags anything; capping/coverage cutoffs operate on weights
computed FROM that already-lagged mkt_cap, cross-sectionally at each date
-- never pooled over the full panel (CLAUDE.md).

Three variants, spec 00_spec.md section 7:

- vw_uncapped: a thin pass-through to gate1_index.build_vw_index(), which
  is Gate-1-validated. Wired, not rewritten.
- vw_capped_10pct: water-filling 10% single-name cap. Required for Canada
  -- Nortel reached ~28% of the self-built Canadian index on 2000-07-27
  (verified directly against the panel in tests/market_index/test_build.py),
  contaminating every Canadian beta estimated on a 5y correlation window
  from ~1997-2006.
- vw_msci_like: 85% cumulative-coverage large-cap proxy on total market
  cap (this project has no free-float data anywhere -- spec section 7's
  "total shares not free float" convention applies here too).

Spec section 7's structural requirement -- "estimation and evaluation
benchmarks must be the same object" -- is met by giving all three variants
ONE entry point (build_index()), so portfolio code (F2b) calls the same
function regardless of which index a run names.
"""

import datetime
from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "market_index.yaml"


def _load_index_config(method: str, overrides: dict | None = None) -> dict:
    """The named index block from config/market_index.yaml, with optional
    key overrides (used by tests to prove values are read from config, not
    hardcoded -- never used by production callers, which should edit the
    YAML instead of passing overrides). Raises KeyError on an unknown
    method name -- a typo must fail loudly, matching
    beta_fp._load_beta_config()'s convention."""
    with open(CONFIG_PATH) as f:
        all_indices = yaml.safe_load(f)["indices"]
    if method not in all_indices:
        raise KeyError(
            f"'{method}' is not a defined index in {CONFIG_PATH} -- "
            f"known indices: {sorted(all_indices)}"
        )
    config = dict(all_indices[method])
    if overrides:
        config.update(overrides)
    return config


def build_index(
    panel: pl.DataFrame, method: str, overrides: dict | None = None
) -> pl.DataFrame:
    """Builds the named market index variant from a common-schema panel
    (id, date, mkt_cap, ret). Returns one row per date: date, index_ret --
    the same shape gate1_index.build_vw_index() returns, so callers don't
    branch on which variant they asked for.
    """
    config = _load_index_config(method, overrides)
    kind = config["kind"]

    if kind == "uncapped":
        from src.gates import gate1_index

        return gate1_index.build_vw_index(panel)

    if kind == "capped":
        weights = _capped_weights(
            panel, cap=config["cap"], max_iterations=config["max_iterations"]
        )
    elif kind == "coverage_cutoff":
        weights = _msci_like_weights(
            panel, target_cumulative_weight=config["target_cumulative_weight"]
        )
    else:
        raise KeyError(f"unknown index kind '{kind}' for method '{method}'")

    return (
        weights.join(panel.select(["id", "date", "ret"]), on=["id", "date"], how="left")
        .group_by("date")
        .agg((pl.col("weight") * pl.col("ret")).sum().alias("index_ret"))
        .sort("date")
    )


def build_index_chunked(
    leg: str, start_date: datetime.date, end_date: datetime.date, method: str,
    overrides: dict | None = None,
) -> pl.DataFrame:
    """Builds the named market index variant over a MULTI-DECADE span by
    calling us_gate1_panel/canada_gate1_panel and build_index() ONE
    CALENDAR YEAR AT A TIME, concatenating only the resulting small
    per-DATE index rows -- never materializing the full per-name panel
    across the whole span.

    Added 2026-09-13 for Gate 4 (docs/02_validation_gates.md): Gate 4's
    realized-market-loading diagnostic needs a 56-year (1970-2025)
    market index, but gate_adapters.us_gate1_panel(start, end) called
    directly over that whole span returns ~71M per-name rows as ONE
    DataFrame before this module ever collapses it to ~14,000 per-date
    index rows -- confirmed by isolating the exact call, this crashes
    with a Rust memory allocation failure (a single year's chunk alone
    processes fine; the crash is the final concat of ~56 yearly chunks
    into one big per-name frame, not the per-year computation itself).
    No caller before Gate 4 ever requested a multi-decade span from
    build_index(), so this gap was invisible until now.

    Correctness: us_gate1_panel(chunk_start, chunk_end) is called with
    each FULL CALENDAR YEAR as its own [start, end] range -- identical to
    how a standalone single-year call already behaves (that function
    already reads the prior year's partition file to resolve a lagged
    mkt_cap across the year boundary, per its own docstring and
    tests/unit/test_gate_adapters.py's boundary-carry tests), so no
    additional carry logic is needed here. Each YEAR's cap/coverage-
    cutoff weighting (_capped_weights/_msci_like_weights) is computed
    independently per date regardless of chunking (CLAUDE.md: never
    pooled across dates), so chunking by year cannot change any single
    date's weights -- only how many dates are computed per call.

    DELIBERATE, TEMPORARY DUPLICATION: this reimplements a yearly-
    chunking loop that us_gate1_panel already has internally (its own
    chunking exists for the SAME reason -- a 56-year single call
    crashing on the final per-name concat). Two chunking implementations
    now exist and must agree; the more correct long-term fix is a
    LazyFrame-based us_gate1_panel that lets a caller aggregate before
    ever materializing the full panel (considered and deferred during
    Gate 4 -- too large a signature change to shared, gate-verified code
    to make mid-gate). Note this if a future session is confused by two
    chunkers: this one is superseded once that refactor lands.

    Bit-identical to calling build_index(panel, method) on the UNCHUNKED
    panel over the same span -- see
    tests/market_index/test_build.py::test_build_index_chunked_matches_unchunked_reference.
    """
    from src.data import gate_adapters

    panel_builders = {
        "us": gate_adapters.us_gate1_panel,
        "canada": gate_adapters.canada_gate1_panel,
    }
    if leg not in panel_builders:
        raise KeyError(f"leg must be one of {sorted(panel_builders)}, got {leg!r}")
    build_panel = panel_builders[leg]

    index_chunks = []
    for year in range(start_date.year, end_date.year + 1):
        chunk_start = max(start_date, datetime.date(year, 1, 1))
        chunk_end = min(end_date, datetime.date(year, 12, 31))
        if chunk_start > chunk_end:
            continue
        panel = build_panel(chunk_start, chunk_end)
        if panel.height == 0:
            continue
        index_chunks.append(build_index(panel, method, overrides))

    if not index_chunks:
        return pl.DataFrame(schema={"date": pl.Date, "index_ret": pl.Float64})

    return pl.concat(index_chunks, how="vertical").sort("date")


def _raw_weights(panel: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional raw weight (mkt_cap / sum(mkt_cap)) computed
    independently WITHIN each date -- never pooled across dates. Only rows
    with BOTH a non-null mkt_cap AND a non-null ret contribute -- matches
    gate1_index.build_vw_index()'s own null-guard convention exactly (see
    that function's docstring: a null ret with a non-null mkt_cap is
    "particularly dangerous", since excluding on mkt_cap alone would still
    hand that name a full weight while build_index()'s later
    weight*ret.sum() silently treats its missing return as an effective 0%
    instead of excluding it and renormalizing the rest to sum to 1)."""
    return (
        panel.filter(pl.col("mkt_cap").is_not_null() & pl.col("ret").is_not_null())
        .with_columns(
            (pl.col("mkt_cap") / pl.col("mkt_cap").sum().over("date")).alias("weight")
        )
        .select(["id", "date", "weight"])
    )


def _single_pass_cap_and_redistribute(panel: pl.DataFrame, cap: float) -> pl.DataFrame:
    """ONE cap-and-redistribute pass, no iteration. Not used by
    build_index() -- exists only so a test can prove the iterative design
    in _capped_weights() is actually necessary (i.e. that a single pass can
    leave a name above `cap` when redistribution itself pushes it over)."""
    weights = _raw_weights(panel)
    over = weights.filter(pl.col("weight") > cap)
    under = weights.filter(pl.col("weight") <= cap)
    excess = over.group_by("date").agg((pl.col("weight") - cap).sum().alias("_excess"))
    under_total = under.group_by("date").agg(pl.col("weight").sum().alias("_under_total"))

    under = under.join(excess, on="date", how="left").join(under_total, on="date", how="left")
    under = under.with_columns(
        (
            pl.col("weight")
            + pl.col("_excess").fill_null(0.0) * pl.col("weight") / pl.col("_under_total")
        ).alias("weight")
    ).select(["id", "date", "weight"])

    capped = over.with_columns(pl.lit(cap).alias("weight")).select(["id", "date", "weight"])
    return pl.concat([capped, under], how="vertical").sort(["date", "id"])


def _capped_weights(panel: pl.DataFrame, cap: float, max_iterations: int) -> pl.DataFrame:
    """Water-filling cap: any name's weight above `cap` is clipped to
    exactly `cap`; the excess is redistributed pro-rata (proportional to
    current weight) across all currently-uncapped names; repeated per date
    until no name exceeds `cap`. Multiple names can breach in the same
    pass, and redistribution can push an already-under-cap name over it --
    both are exercised in tests/market_index/test_build.py. Raises
    RuntimeError if `max_iterations` is exhausted without convergence, OR
    if a date has too few uncapped names left to absorb excess -- sum-to-1
    and max-weight<=cap are jointly infeasible once every name is already
    at the cap and residual excess remains (real dates in this project
    always have far more than ceil(1/cap) names, so this guards a
    synthetic/degenerate panel, not an expected real-data path). Either
    way this raises rather than silently returning an out-of-spec result
    (CLAUDE.md: a wrong number that looks right is the worst possible
    outcome)."""
    weights = _raw_weights(panel)
    frozen = weights.clear()  # empty frame, same schema -- accumulates permanently-capped rows
    remaining = weights
    for _ in range(max_iterations):
        over = remaining.filter(pl.col("weight") > cap)
        if over.height == 0:
            break

        under = remaining.filter(pl.col("weight") <= cap)
        # Per-DATE infeasibility check: a date can have plenty of uncapped
        # names on one day and none on another within the same panel, so
        # this must never be a height check over the whole (multi-date)
        # frame -- that would miss a single bad date hiding among many
        # healthy ones.
        over_dates = set(over["date"].unique().to_list())
        under_dates = set(under["date"].unique().to_list())
        starved_dates = over_dates - under_dates
        if starved_dates:
            raise RuntimeError(
                f"cap={cap} is infeasible on {len(starved_dates)} date(s) "
                f"(e.g. {min(starved_dates)}) -- name(s) still exceed "
                f"the cap but no uncapped name remains on that date to "
                f"absorb the excess (need more names for this cap to be "
                f"feasible)"
            )

        newly_capped = over.with_columns(pl.lit(cap).alias("weight"))
        frozen = pl.concat([frozen, newly_capped], how="vertical")

        excess = over.group_by("date").agg(
            (pl.col("weight") - cap).sum().alias("_excess")
        )
        under_total = under.group_by("date").agg(
            pl.col("weight").sum().alias("_under_total")
        )
        under = under.join(excess, on="date", how="left").join(
            under_total, on="date", how="left"
        )
        remaining = under.with_columns(
            (
                pl.col("weight")
                + pl.col("_excess").fill_null(0.0)
                * pl.col("weight")
                / pl.col("_under_total")
            ).alias("weight")
        ).select(["id", "date", "weight"])
    else:
        still_over = remaining.filter(pl.col("weight") > cap)
        if still_over.height > 0:
            raise RuntimeError(
                f"cap did not converge within {max_iterations} iterations -- "
                f"{still_over.height} name(s) still exceed cap={cap}"
            )

    return pl.concat([frozen, remaining], how="vertical").sort(["date", "id"])


def _msci_like_weights(panel: pl.DataFrame, target_cumulative_weight: float) -> pl.DataFrame:
    """Coverage-cutoff large-cap proxy: within each date independently,
    rank names by raw weight descending, keep names until cumulative raw
    weight reaches `target_cumulative_weight`, then renormalize the KEPT
    subset's weights to sum to 1. Recomputed per date -- the kept-name set
    is not fixed over time (CLAUDE.md: no filtering on full-sample
    properties)."""
    weights = _raw_weights(panel)
    ranked = weights.sort(["date", "weight"], descending=[False, True]).with_columns(
        pl.col("weight").cum_sum().over("date").alias("_cum_weight"),
        pl.col("weight").cum_sum().over("date").shift(1).over("date").fill_null(0.0).alias(
            "_cum_weight_before"
        ),
    )
    # Keep a name if it was needed to REACH the target -- i.e. cumulative
    # weight *before* this name was still short of the target. This keeps
    # exactly the names up through (and including) the one that crosses
    # the threshold, never one extra.
    kept = ranked.filter(pl.col("_cum_weight_before") < target_cumulative_weight)
    kept = kept.with_columns(
        (pl.col("weight") / pl.col("weight").sum().over("date")).alias("weight")
    )
    return kept.select(["id", "date", "weight"])
