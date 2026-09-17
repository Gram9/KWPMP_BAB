"""Parquet caching layer over src/portfolio/longonly_data.py's
build_us_inputs/build_canada_inputs, so repeated backtest/bucket-analysis
runs (parameter sweeps over n_holdings, turnover_k, fringe floors,
beta window, etc.) don't repeatedly pay the expensive panel-build and
multi-variant OLS cost (measured this session: full US history is
~15-20 minutes; longonly.run_longonly_backtest itself, given already-
built inputs, is seconds -- a PURE frame-in/frame-out loop, per that
module's own docstring).

DELIBERATELY separate from longonly_data.py, not merged into it: those
two functions stay PURE (no I/O beyond the WRDS/CHASS reads they already
need, no caching side effects) so they remain directly unit-testable
against synthetic inputs -- the same reasoning
src.data.aqr_bab_loader/src.portfolio.rebalance already follow elsewhere
in this project (cache the expensive orchestration layer, never the
estimator itself). This mirrors CLAUDE.md's own stated reason for NOT
caching inside build_bab_series: a cached result could serve a stale
value during failure-injection testing and make a broken assertion look
like it fired. That concern is about the ESTIMATION function; this
module only wraps the OUTER data-assembly call, and every cache read is
guarded by an explicit manifest match (leg/date-range/variants), so a
different request never silently reads a mismatched cache.

Cache layout, one directory per (leg, start, end):
    data/processed/longonly_cache/<leg>_<start>_<end>/
        manifest.json          -- leg, start, end, variants, generated_at
        betas.parquet           -- variant, formation_date, id, beta
        fringe.parquet          -- formation_date, id, mkt_cap, price
        quarterly_rets.parquet  -- id, month, ret
"""

import datetime
import json
import shutil
import uuid
from pathlib import Path

import polars as pl

from src.portfolio import longonly_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "longonly_cache"


def _cache_dir(leg: str, start: datetime.date, end: datetime.date) -> Path:
    return CACHE_ROOT / f"{leg}_{start.isoformat()}_{end.isoformat()}"


def _manifest_matches(manifest_path: Path, leg: str, start: datetime.date, end: datetime.date, variants: tuple[str, ...]) -> bool:
    if not manifest_path.exists():
        return False
    with open(manifest_path) as f:
        manifest = json.load(f)
    return (
        manifest.get("leg") == leg
        and manifest.get("start") == start.isoformat()
        and manifest.get("end") == end.isoformat()
        and set(manifest.get("variants", [])) == set(variants)
    )


def _write_cache(
    cache_dir: Path,
    leg: str,
    start: datetime.date,
    end: datetime.date,
    variants: tuple[str, ...],
    betas_by_date_per_variant: dict[str, dict[datetime.date, pl.DataFrame]],
    fringe_by_date: dict[datetime.date, pl.DataFrame],
    quarterly_monthly_rets: pl.DataFrame,
) -> None:
    """Writes all 4 cache files into a fresh TEMP directory, then swaps
    it into place as the LAST step -- never writes into cache_dir
    directly. leakage-auditor finding, this session: sequential in-place
    writes (betas.parquet, fringe.parquet, quarterly_rets.parquet,
    manifest.json last) leave a real window where an interrupted write
    (a killed background process -- this session's own repeated pattern
    of stopping long-running driver scripts) can leave a NEW betas.parquet
    beside an OLD fringe.parquet/quarterly_rets.parquet, with the OLD
    manifest.json still validating on the next read (it only records
    leg/start/end/variants, unchanged across a same-parameters rebuild
    triggered by a data refresh) -- reproduced live by the auditor:
    beta from build 2, fringe from build 1, no error, no warning. Under
    CLAUDE.md's "a wrong number that looks right is the worst possible
    outcome," a silently mixed-vintage cache is exactly that.

    rmtree+rename is not fully atomic on Windows, but it narrows the
    interruption window from the full multi-file parquet-write duration
    down to a single directory-swap -- and any interruption INSIDE that
    narrow window leaves either the OLD cache_dir intact (rename never
    happened) or the NEW one fully in place (rename completed), never a
    mix of old and new files."""
    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    beta_rows = []
    for variant, by_date in betas_by_date_per_variant.items():
        for formation_date, df in by_date.items():
            beta_rows.append(
                df.with_columns(
                    pl.lit(variant).alias("variant"),
                    pl.lit(formation_date).alias("formation_date"),
                )
            )
    betas_flat = (
        pl.concat(beta_rows, how="vertical")
        if beta_rows
        else pl.DataFrame(
            schema={"id": pl.Utf8, "beta": pl.Float64, "variant": pl.Utf8, "formation_date": pl.Date}
        )
    )
    betas_flat.write_parquet(tmp_dir / "betas.parquet")

    fringe_rows = [
        df.with_columns(pl.lit(formation_date).alias("formation_date"))
        for formation_date, df in fringe_by_date.items()
    ]
    fringe_flat = (
        pl.concat(fringe_rows, how="vertical")
        if fringe_rows
        else pl.DataFrame(
            schema={"id": pl.Utf8, "mkt_cap": pl.Float64, "price": pl.Float64, "formation_date": pl.Date}
        )
    )
    fringe_flat.write_parquet(tmp_dir / "fringe.parquet")

    quarterly_monthly_rets.write_parquet(tmp_dir / "quarterly_rets.parquet")

    manifest = {
        "leg": leg,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "variants": list(variants),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(tmp_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Swap: remove any existing (old) cache_dir, then rename tmp into
    # place -- the only two operations that touch cache_dir's own path,
    # both fast directory-level ops, not per-file writes.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)


def _read_cache(
    cache_dir: Path, variants: tuple[str, ...]
) -> tuple[dict[str, dict[datetime.date, pl.DataFrame]], dict[datetime.date, pl.DataFrame], pl.DataFrame]:
    betas_flat = pl.read_parquet(cache_dir / "betas.parquet")
    fringe_flat = pl.read_parquet(cache_dir / "fringe.parquet")
    quarterly_monthly_rets = pl.read_parquet(cache_dir / "quarterly_rets.parquet")

    betas_by_date_per_variant: dict[str, dict[datetime.date, pl.DataFrame]] = {
        variant: {} for variant in variants
    }
    for (variant, formation_date), group in betas_flat.group_by(["variant", "formation_date"]):
        if variant in betas_by_date_per_variant:
            betas_by_date_per_variant[variant][formation_date] = group.select(["id", "beta"])

    fringe_by_date: dict[datetime.date, pl.DataFrame] = {}
    for (formation_date,), group in fringe_flat.group_by(["formation_date"]):
        fringe_by_date[formation_date] = group.select(["id", "mkt_cap", "price"])

    return betas_by_date_per_variant, fringe_by_date, quarterly_monthly_rets


def get_us_inputs(
    start: datetime.date,
    end: datetime.date,
    *,
    variants: tuple[str, ...] = longonly_data.DEFAULT_VARIANTS,
    force_rebuild: bool = False,
) -> tuple[dict[str, dict[datetime.date, pl.DataFrame]], dict[datetime.date, pl.DataFrame], pl.DataFrame]:
    """Cache-checked wrapper over longonly_data.build_us_inputs. Reads
    from data/processed/longonly_cache/ if a manifest matches
    (leg="us", start, end, variants exactly); otherwise builds fresh via
    build_us_inputs and writes the cache before returning.

    force_rebuild=True skips the cache read (always rebuilds) but still
    writes the fresh result -- useful when the underlying data changed
    (e.g. data/raw/ was re-pulled) and a stale cache would otherwise be
    silently trusted.
    """
    cache_dir = _cache_dir("us", start, end)
    manifest_path = cache_dir / "manifest.json"

    if not force_rebuild and _manifest_matches(manifest_path, "us", start, end, variants):
        return _read_cache(cache_dir, variants)

    betas_by_date_per_variant, fringe_by_date, quarterly_monthly_rets = longonly_data.build_us_inputs(
        start, end, variants=variants
    )
    _write_cache(
        cache_dir, "us", start, end, variants, betas_by_date_per_variant, fringe_by_date, quarterly_monthly_rets
    )
    return betas_by_date_per_variant, fringe_by_date, quarterly_monthly_rets


def get_canada_inputs(
    start: datetime.date,
    end: datetime.date,
    *,
    variants: tuple[str, ...] = longonly_data.DEFAULT_VARIANTS,
    force_rebuild: bool = False,
) -> tuple[dict[str, dict[datetime.date, pl.DataFrame]], dict[datetime.date, pl.DataFrame], pl.DataFrame]:
    """Cache-checked wrapper over longonly_data.build_canada_inputs. Same
    contract as get_us_inputs, leg="ca"."""
    cache_dir = _cache_dir("ca", start, end)
    manifest_path = cache_dir / "manifest.json"

    if not force_rebuild and _manifest_matches(manifest_path, "ca", start, end, variants):
        return _read_cache(cache_dir, variants)

    betas_by_date_per_variant, fringe_by_date, quarterly_monthly_rets = longonly_data.build_canada_inputs(
        start, end, variants=variants
    )
    _write_cache(
        cache_dir, "ca", start, end, variants, betas_by_date_per_variant, fringe_by_date, quarterly_monthly_rets
    )
    return betas_by_date_per_variant, fringe_by_date, quarterly_monthly_rets
