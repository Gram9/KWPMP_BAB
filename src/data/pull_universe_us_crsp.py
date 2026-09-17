"""
Chunked, one-shot WRDS pull for the full-history US universe, sourced
directly from CRSP's wrds_dsfv2_query view (not Compustat). Not run
inside any loop or on any schedule in the ongoing-operations sense -- run
once, output lands in data/raw/ (gitignored, regenerable, never treated
as source). See CLAUDE.md: data/raw/ is immutable once pulled.

Supersedes the earlier Compustat-based full-history design
(docs/superpowers/specs/2026-09-03-us-full-history-panel-design.md,
marked superseded). This session's investigation found comp.company.sic/
gind/gsubind are not point-in-time (docs/01_data_notes.md Sec14), and
that crsp.wrds_dsfv2_query supplies point-in-time membership,
classification (a materially better REIT signal -- issuertype, not
icbindustry/sic), price, shares, and vwretd back to 1925 -- see the
CRSP-primary design spec's "Findings" section for the full evidence.

Chunked by calendar year (not one unbounded query): the full-window pull
is 74,549,744 rows, live-measured, ~6-11GB estimated in pandas -- too
large/risky for one raw_sql() call (long-running, no partial-progress
resumability). This loop is a memory/reliability chunking strategy for a
SINGLE bulk pull, run once -- not the "one query per point-in-time date"
pattern CLAUDE.md's loop rule warns against (that rule's own evidence,
docs/01_data_notes.md Sec12, is a different failure mode: non-trading-
date silent-empty-result risk from a per-date query pattern).

Distribution-detail columns (dis*-prefixed, ~36 of them) are explicitly
excluded from the SELECT list and DISTINCT is applied on the selected
columns: confirmed this session that a (permno, dlycaldt) pair can
otherwise appear more than once when multiple corporate distribution
events land on the same date (e.g. permno 25099, Integrys Energy, two
same-day liquidating distributions) -- excluding dis* columns + DISTINCT
reduced duplicate (permno, dlycaldt) pairs from 1 to 0 on the full
2015-06-30 test date.

Idempotent: skips any *closed* (strictly-past) year whose output file
already exists, so a crashed or interrupted run can be re-run to fill
in only the missing years, never re-pulling or overwriting a completed
year. The current calendar year is never skip-eligible -- WRDS may
still be backfilling it (e.g. this session's 2026 pull returned 0 rows,
simple data lag, not a bug), so it is always re-queried and its file
always overwritten, on every run, until it becomes a closed year.

Writes are atomic: each year's chunk is written to a `.tmp` sibling of
the final path and moved into place with `os.replace()` (atomic on both
POSIX and Windows) only after the write completes. This guarantees the
skip-check (`output_path.exists()`) never observes a truncated/corrupt
file left behind by a kill mid-write -- it only ever sees a complete
file or no file. A stray `.tmp` file from a prior crash is harmless:
nothing reads it, and skip logic only inspects `output_path`; it is
simply overwritten (its own crash-safety) the next time that year is
(re-)pulled.

Streamed, not buffered in memory as one per-year DataFrame: `raw_sql`'s
own default (return_iter=False) already fetches in chunksize=500_000-row
server-side chunks, but then internally re-`pd.concat`s every chunk into
one full per-year DataFrame before returning it -- for the wider,
31-column schema this project pulls, that reassembly step is exactly
where two live pull attempts crashed with a real
`numpy._core._exceptions._ArrayMemoryError` (once at 1993, once at 1994
after resuming -- a genuine, reproducible memory ceiling as row density
grows through the 1990s, not a fluke). Passing `return_iter=True`
returns the underlying chunk iterator directly, with `date_cols`
(passed through to `pandas.read_sql_query`'s own `parse_dates`) already
applied per-chunk before we ever see it -- each chunk is written to the
year's `.tmp` file via a `pyarrow.parquet.ParquetWriter` as it arrives,
so at most one chunk is ever held in memory, not up to ~1.6M rows
accumulated across a whole dense year.

`chunksize` is explicitly set to 100_000 (down from raw_sql's own
500_000 default). A second round of live pulls hit
`numpy._core._exceptions._ArrayMemoryError` twice more, both INSIDE
pandas' own raw fetch/type-inference step (before this module's chunk
loop ever runs) -- once trying to allocate 122MiB for a 500K x 32
object array, once trying to allocate 7.63MiB, while ~10.5GB was
confirmed genuinely free system-wide (`Get-CimInstance
Win32_OperatingSystem`) at the time of the second crash. Failing a
7.63MiB allocation with 10GB+ free is not a real memory shortage --
it's consistent with heap fragmentation (no single contiguous block
that size available, even though plenty is free in aggregate). Smaller
per-chunk allocation requests are less likely to fail against a
fragmented heap; 100_000 was chosen as a conservative reduction, not
tuned to a specific measured threshold.

Run manually:

    WRDS_USERNAME=<your username> python -m src.data.pull_universe_us_crsp
"""

import os
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import wrds  # type: ignore[import-untyped]
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "us_panel_crsp_full"
CONFIG_PATH = PROJECT_ROOT / "config" / "universe.yaml"

# Distribution-detail columns (dis*-prefixed) are deliberately excluded --
# see module docstring, finding 8 in the design spec.
#
# Beyond the original 18 columns, this list also carries fields anticipated
# for near-term work (Gate 1 index reconstruction, Gate 2/3 liquidity
# filtering, beta estimation) so a second ~40-minute/74.5M-row pull isn't
# needed once that work starts: dlyret/dlyretx (CRSP's own daily returns --
# not computed anywhere else in this pull), dlyvol (trading-volume gate,
# same role cshtrd>0 plays on the Compustat side, per CLAUDE.md/wrds-data),
# cusip/ticker (cross-reference identifiers for joins/debugging),
# dlycumfacpr/dlycumfacshr (cumulative price/share adjustment factors, for
# auditing dlyret or building an independently split-adjusted price series),
# and full daily OHLC (dlyopen/dlyhigh/dlylow/dlyclose) plus dlybid/dlyask,
# not required by BAB's close-to-close construction but cheap to carry
# alongside the rest for future microstructure/liquidity diagnostics.
_COLUMN_NAMES = (
    "permno", "permco", "dlycaldt", "securitynm", "siccd", "icbindustry",
    "issuertype", "securitytype", "securitysubtype", "sharetype", "usincflg",
    "primaryexch", "dlyprc", "shrout", "dlycap", "securitybegdt",
    "securityenddt", "securityactiveflg", "vwretd", "vwretx",
    "dlyret", "dlyretx", "dlyvol", "cusip", "ticker",
    "dlycumfacpr", "dlycumfacshr",
    "dlyopen", "dlyhigh", "dlylow", "dlyclose", "dlybid", "dlyask",
    # Delisting provenance -- carried so delisting handling is VERIFIABLE
    # downstream rather than assumed. dlydelflg='Y' marks the delisting-return
    # row itself; the del*type fields are blanked ('N/A') on that row in the
    # daily view (they live populated in crsp.stkdelists), but are kept here
    # so a reader can confirm that directly instead of rediscovering it.
    "dlydelflg", "delreasontype", "delactiontype", "delstatustype",
)

# Qualified with the `q.` alias: the corrected YEAR_QUERY aliases the outer
# table so the delisting branch's EXISTS subquery can correlate on permno.
SELECT_COLUMNS = ", ".join(f"q.{name}" for name in _COLUMN_NAMES)

# CORRECTED 2026-09-08: the original WHERE clause was the base-descriptor
# filter alone, which silently dropped EVERY delisting-return row in the
# sample -- 31,114 rows, 1965-present, measured live: zero retained in every
# decade. CRSP v2 does not expose a separate `dlret` column the way legacy
# crsp.dsf does; it folds the delisting return into `dlyret` on a row dated
# one trading day AFTER the last trade (stkdelists.deldlydt, vs
# stkdelists.delistingdt for the final trading day), and BLANKS the security
# descriptors on that row: securitytype='N/A', securitysubtype='UNK',
# sharetype='N/A', primaryexch='X'. So every delisting row failed all four
# base predicates. Verified against crsp.stkdelists: on the rows that do
# survive, dlyret == delret exactly.
#
# This is survivorship bias in the direction that flatters BAB: delisting
# returns skew to small, distressed, high-beta names (the short leg). Full
# window: mean -3.22%, 2,164 rows <= -30%, 399 total losses (-100%).
#
# The corrected clause is a two-branch OR:
#   (a) the original base-descriptor filter -- ordinary trading rows;
#   (b) delisting rows (dlydelflg='Y'), admitted ONLY when that permno
#       satisfies (a) somewhere in its own history. Branch (b) cannot be
#       gated on the descriptors (they are blanked) and must not be gated on
#       primaryexch='X' alone -- 'X' also covers 1,370,373 NON-delisting
#       rows. dlydelflg='Y' <=> primaryexch='X' AND a delisting row was
#       verified exactly (31,114 both ways), but the EXISTS subquery is what
#       keeps this from admitting the delisting rows of securities that were
#       never in our universe (ETFs, ADRs, foreign): 31,114 delisting rows
#       total, 24,480 belong to our permnos, 6,634 correctly excluded.
#
# Live-validated on 2015 before this change shipped: 1,009,895 rows old ->
# 1,010,194 new, delta +299 == exactly the delisting rows added (strict
# superset, no ordinary row gained or lost), and all 8 known-missing
# delisting returns recovered at the right date with the right value.
YEAR_QUERY = f"""
    select distinct {SELECT_COLUMNS}
    from crsp.wrds_dsfv2_query q
    where (
            (q.securitytype = %(securitytype)s
             and q.securitysubtype = %(securitysubtype)s
             and q.sharetype = %(sharetype)s
             and q.usincflg = %(usincflg)s)
            or
            (q.dlydelflg = 'Y' and exists (
               select 1 from crsp.wrds_dsfv2_query b
               where b.permno = q.permno
                 and b.securitytype = %(securitytype)s
                 and b.securitysubtype = %(securitysubtype)s
                 and b.sharetype = %(sharetype)s
                 and b.usincflg = %(usincflg)s))
          )
      and q.dlycaldt >= %(year_start)s
      and q.dlycaldt < %(year_end)s
"""

# Explicit target schema every chunk is cast to before writing (see the
# chunk-write loop below) -- confirmed directly against a real populated
# year's on-disk schema (data/raw/us_panel_crsp_full/year=2025/part.parquet).
# raw_sql(return_iter=True) yields chunks as pandas nullable extension
# dtypes (dtype_backend="numpy_nullable"); pa.Table.from_pandas() on a
# single all-empty chunk (a genuinely empty year, e.g. the current,
# still-in-progress calendar year before WRDS has data for it) infers
# pyarrow Null dtype for every non-date column instead of the real
# Int64/Utf8/Float64 types. Casting every chunk to this fixed schema --
# not just special-casing the empty case -- makes every year's file
# correct by construction, with a uniform schema across the whole
# partitioned dataset regardless of which year a given file came from.
_EMPTY_YEAR_SCHEMA = pa.schema([
    ("permno", pa.int64()),
    ("permco", pa.int64()),
    ("dlycaldt", pa.timestamp("ns")),
    ("securitynm", pa.string()),
    ("siccd", pa.int64()),
    ("icbindustry", pa.string()),
    ("issuertype", pa.string()),
    ("securitytype", pa.string()),
    ("securitysubtype", pa.string()),
    ("sharetype", pa.string()),
    ("usincflg", pa.string()),
    ("primaryexch", pa.string()),
    ("dlyprc", pa.float64()),
    ("shrout", pa.int64()),
    ("dlycap", pa.float64()),
    ("securitybegdt", pa.timestamp("ns")),
    ("securityenddt", pa.timestamp("ns")),
    ("securityactiveflg", pa.string()),
    ("vwretd", pa.float64()),
    ("vwretx", pa.float64()),
    ("dlyret", pa.float64()),
    ("dlyretx", pa.float64()),
    ("dlyvol", pa.float64()),
    ("cusip", pa.string()),
    ("ticker", pa.string()),
    ("dlycumfacpr", pa.float64()),
    ("dlycumfacshr", pa.float64()),
    ("dlyopen", pa.float64()),
    ("dlyhigh", pa.float64()),
    ("dlylow", pa.float64()),
    ("dlyclose", pa.float64()),
    ("dlybid", pa.float64()),
    ("dlyask", pa.float64()),
    ("dlydelflg", pa.string()),
    ("delreasontype", pa.string()),
    ("delactiontype", pa.string()),
    ("delstatustype", pa.string()),
])


# Columns added by the 2026-09-08 delisting-return/permco fix. A year file
# lacking any of these predates the fix and must not be silently skipped.
_REQUIRED_NEW_COLUMNS = {"permco", "dlydelflg"}


class StaleSchemaError(RuntimeError):
    """Raised when an existing year-partition file predates the
    delisting-return/permco fix. The idempotent skip is a resumability
    feature for a CRASHED run of the SAME query -- it is not safe across a
    query change, because it would leave pre-fix years permanently stale
    while new years get the corrected filter, producing a panel that is
    silently half-fixed (CLAUDE.md: a wrong number that looks right is the
    worst possible outcome)."""


def _existing_file_has_current_schema(path: Path) -> bool:
    """True if `path`'s parquet schema already carries the post-fix columns.
    Reads only the file's metadata footer, not its row data."""
    return _REQUIRED_NEW_COLUMNS.issubset(set(pq.read_schema(path).names))


def _year_range(window_start: str) -> list[int]:
    start_year = date.fromisoformat(window_start).year
    end_year = date.today().year
    return list(range(start_year, end_year + 1))


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["us_crsp_primary"]

    base_params = {
        "securitytype": cfg["securitytype"],
        "securitysubtype": cfg["securitysubtype"],
        "sharetype": cfg["sharetype"],
        "usincflg": cfg["usincflg"],
    }

    years = _year_range(cfg["window_start"])

    db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    try:
        current_year = date.today().year
        for year in years:
            year_dir = RAW_DATA_DIR / f"year={year}"
            output_path = year_dir / "part.parquet"
            is_closed_year = year < current_year
            if is_closed_year and output_path.exists():
                if not _existing_file_has_current_schema(output_path):
                    raise StaleSchemaError(
                        f"{output_path} exists but predates the "
                        f"delisting-return/permco fix (2026-09-08): its schema "
                        f"is missing one or more of {sorted(_REQUIRED_NEW_COLUMNS)}. "
                        f"The idempotent skip below would otherwise leave this "
                        f"year permanently stale -- it would never be re-pulled, "
                        f"and the panel would silently mix fixed and unfixed "
                        f"years. Move or delete the old "
                        f"data/raw/us_panel_crsp_full/ directory and re-run "
                        f"(see docs/worklog.md, 2026-09-08). Refusing to skip."
                    )
                print(f"{year}: already pulled, skipping")
                continue

            params = {
                **base_params,
                "year_start": f"{year}-01-01",
                "year_end": f"{year + 1}-01-01",
            }
            chunk_iter = db.raw_sql(
                YEAR_QUERY,
                params=params,
                date_cols=["dlycaldt", "securitybegdt", "securityenddt"],
                return_iter=True,
                chunksize=100_000,
            )

            year_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_suffix(".tmp")

            # Every chunk is cast to _EMPTY_YEAR_SCHEMA's explicit dtypes
            # (pa.Table.from_pandas(chunk).cast(...)) rather than letting
            # pyarrow infer each chunk's schema independently -- confirmed
            # live that return_iter=True yields exactly one chunk even for
            # a zero-row year (a genuinely empty pandas chunk, not zero
            # chunks), and pa.Table.from_pandas() on an empty chunk infers
            # Null dtype for every non-date column instead of the real
            # Int64/Utf8/Float64 types. Casting every chunk to the same
            # explicit schema fixes the empty-chunk case AND removes any
            # theoretical risk of two populated chunks inferring different
            # dtypes for the same column (e.g. an all-null vs. populated
            # nullable numeric column in different 500K-row pages), since
            # pq.ParquetWriter requires every write_table() call to match
            # the schema it was opened with.
            total_rows = 0
            distinct_permnos: set[int] = set()
            writer = pq.ParquetWriter(tmp_path, _EMPTY_YEAR_SCHEMA)
            try:
                for chunk in chunk_iter:
                    table = pa.Table.from_pandas(chunk, preserve_index=False).cast(
                        _EMPTY_YEAR_SCHEMA
                    )
                    writer.write_table(table)
                    total_rows += len(chunk)
                    distinct_permnos.update(chunk["permno"].tolist())
            finally:
                writer.close()

            os.replace(tmp_path, output_path)
            print(f"{year}: wrote {total_rows} rows, "
                  f"{len(distinct_permnos)} distinct permnos")
    finally:
        db.close()


if __name__ == "__main__":
    main()
