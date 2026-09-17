"""
Loaders for Ken French's data library CSVs (Gate 2:
docs/superpowers/specs/2026-09-08-gate2-size-deciles-design.md), pulled
manually by the user from mba.tuck.dartmouth.edu's data library (not
WRDS -- WRDS has no current mirror confirmed for this session) into
data/raw/ken_french_csvs/ (immutable per CLAUDE.md's data/raw/ rule).

Each of Ken French's CSV exports uses a genuinely different stacked-
table layout (free-text preamble, one or more labeled blocks each with
their own column-header row, blank-line separators) rather than a
single clean table -- these functions are hand-written per file, not a
generic CSV parser, and each raises loudly (AssertionError) if its
expected header text is missing rather than silently misparsing a
differently-formatted re-download.
"""

import calendar
import datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEN_FRENCH_DIR = PROJECT_ROOT / "data" / "raw" / "ken_french_csvs"
DAILY_RETURNS_PATH = KEN_FRENCH_DIR / "Portfolios_Formed_on_ME_daily.csv"
BREAKPOINTS_PATH = KEN_FRENCH_DIR / "ME_Breakpoints.csv"

_VW_BLOCK_HEADER = "Average Value Weighted Returns -- Daily"
_BREAKPOINTS_PREAMBLE_MARKER = "every 5th NYSE ME percentile"
# Field indices (0-based) into each ME_Breakpoints.csv data row:
# 0=YYYYMM, 1=n_firms, 2=p5, 3=p10, 4=p15, 5=p20, 6=p25, 7=p30, 8=p35,
# 9=p40, 10=p45, 11=p50, 12=p55, 13=p60, 14=p65, 15=p70, 16=p75, 17=p80,
# 18=p85, 19=p90, 20=p95, 21=p100. Deciles are the 10/20/.../90th
# percentile columns specifically -- the 5th/15th/25th/etc. columns
# belong to a finer (20-bucket) sort this gate does not use.
_DECILE_FIELD_INDICES = {
    "p10": 3, "p20": 5, "p30": 7, "p40": 9, "p50": 11,
    "p60": 13, "p70": 15, "p80": 17, "p90": 19,
}
_DECILE_COLUMNS = [
    "Lo 10", "Dec 2", "Dec 3", "Dec 4", "Dec 5",
    "Dec 6", "Dec 7", "Dec 8", "Dec 9", "Hi 10",
]
_MISSING_CODES = {-99.99, -999.0}


def load_size_decile_daily_returns() -> pl.DataFrame:
    """Ken French's daily value-weighted size decile returns, long
    format: date, decile (1-10, Lo 10=1 through Hi 10=10), ret (decimal,
    not percent). Reads only the "Average Value Weighted Returns --
    Daily" block of Portfolios_Formed_on_ME_daily.csv -- the file also
    carries an equal-weighted block afterward, which this function must
    never read past (its own column header row is textually identical,
    so a naive "read to EOF" parse would silently double-count and mix
    EW values into VW output).
    """
    lines = DAILY_RETURNS_PATH.read_text().splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if _VW_BLOCK_HEADER in line:
            header_idx = i
            break
    assert header_idx is not None, (
        f"{DAILY_RETURNS_PATH}: expected a line containing "
        f"'{_VW_BLOCK_HEADER}' -- file layout may have changed since "
        "this loader was written against it; refusing to silently "
        "misparse."
    )

    column_header_line = lines[header_idx + 1]
    columns = [c.strip() for c in column_header_line.split(",")]
    decile_col_indices = {name: columns.index(name) for name in _DECILE_COLUMNS}

    dates: list[datetime.date] = []
    deciles: list[int] = []
    rets: list[float | None] = []

    for line in lines[header_idx + 2 :]:
        if not line.strip():
            break  # blank line ends the VW block, before the EW block's own header
        fields = [f.strip() for f in line.split(",")]
        # Suppression reason: source dates are plain YYYYMMDD calendar
        # dates with no timezone semantics -- a naive datetime is
        # intentional and correct here, not an oversight.
        date_val = datetime.datetime.strptime(fields[0], "%Y%m%d").date()  # noqa: DTZ007
        for decile_number, col_name in enumerate(_DECILE_COLUMNS, start=1):
            raw = float(fields[decile_col_indices[col_name]])
            dates.append(date_val)
            deciles.append(decile_number)
            rets.append(None if raw in _MISSING_CODES else raw / 100.0)

    return pl.DataFrame(
        {"date": dates, "decile": deciles, "ret": rets},
        schema={"date": pl.Date, "decile": pl.Int8, "ret": pl.Float64},
    )


def load_nyse_breakpoints() -> pl.DataFrame:
    """Ken French's NYSE size breakpoints, one row per calendar month:
    month_end, n_firms, p10..p90 (9 decile cutpoints, in raw dollars --
    the source file states values in $millions, converted here so
    callers never have to remember that convention). Every calendar
    month is present in the source file (FF recomputes NYSE breakpoints
    monthly for other series) -- callers needing only the annual
    size-decile formation month must filter to month_end.month == 6
    themselves (this loader does not filter, since other Gate 2
    diagnostics may want the full monthly series later).
    """
    lines = BREAKPOINTS_PATH.read_text().splitlines()
    assert lines and _BREAKPOINTS_PREAMBLE_MARKER in lines[0], (
        f"{BREAKPOINTS_PATH}: expected the first line to mention "
        f"'{_BREAKPOINTS_PREAMBLE_MARKER}' -- file layout may have "
        "changed since this loader was written against it; refusing "
        "to silently misparse."
    )

    month_ends: list[datetime.date] = []
    n_firms: list[int] = []
    percentile_cols: dict[str, list[float]] = {name: [] for name in _DECILE_FIELD_INDICES}

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue  # skips the blank separator line and the trailing copyright line
        fields = [f.strip() for f in line.split(",")]
        yyyymm = fields[0]
        year, month = int(yyyymm[:4]), int(yyyymm[4:6])
        last_day = calendar.monthrange(year, month)[1]
        month_ends.append(datetime.date(year, month, last_day))
        n_firms.append(int(fields[1]))
        for name, idx in _DECILE_FIELD_INDICES.items():
            percentile_cols[name].append(float(fields[idx]) * 1_000_000.0)

    data = {"month_end": month_ends, "n_firms": n_firms, **percentile_cols}
    return pl.DataFrame(
        data,
        schema={
            "month_end": pl.Date,
            "n_firms": pl.Int64,
            **{name: pl.Float64 for name in _DECILE_FIELD_INDICES},
        },
    )
