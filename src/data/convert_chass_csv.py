"""
One-shot format conversion of the CHASS (CFMRC) Canadian securities CSVs
into parquet. Pure format transform -- no filtering, no dedup, no
temporal-ordering decisions made here. See CLAUDE.md: data/raw/ is
immutable; this reads the CSVs and writes parquet as siblings, never
touches the source CSVs.

Source files (both have a junk title row before the real header, hence
skiprows=1):
    data/raw/CHASS_Data/csvs/CHASS_Monthly.csv  (~158MB, ~807K rows)
    data/raw/CHASS_Data/csvs/CHASS_Daily.csv    (~3.76GB)

Monthly is small enough to convert in one shot. Daily is chunked and
written year-partitioned (data/raw/us_panel_crsp_full's convention) so
no more than one chunk is ever held in memory -- this project has twice
hit numpy._core._exceptions._ArrayMemoryError converting/pulling
multi-GB frames in one shot (see pull_universe_us_crsp.py docstring),
so the same chunked-write pattern is used defensively here even though
this is a local CSV read, not a WRDS query.

Known data quirks, NOT resolved here (left for the consuming code /
a design decision, since resolving them would be a filtering/dedup
choice, not a format conversion):
    - No PERMNO; only CUSIP. A (cusip, trdate) pair is not always
      unique -- confirmed duplicates exist in even a 200K-row sample
      of the daily file. Do not join on cusip alone without also
      resolving these.
    - The two dividend-related date/flag columns and the trade-date
      column share the raw CHASS prefix "trdate-" twice in the daily
      file's header; pandas auto-suffixes the duplicate on read
      (trdate-Trade Date, trdate-Trade Date.1) -- both are kept as-is.

Run manually:

    python -m src.data.convert_chass_csv
"""

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = PROJECT_ROOT / "data" / "raw" / "CHASS_Data" / "csvs"
PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "CHASS_Data" / "parquet"

MONTHLY_CSV = CSV_DIR / "CHASS_Monthly.csv"
DAILY_CSV = CSV_DIR / "CHASS_Daily.csv"

DAILY_CHUNKSIZE = 500_000


def _atomic_write_table(table: pa.Table, output_path: Path) -> None:
    tmp_path = output_path.with_suffix(".tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, output_path)


def convert_monthly() -> None:
    output_path = PARQUET_DIR / "monthly.parquet"
    if output_path.exists():
        print(f"monthly: already converted, skipping ({output_path})")
        return

    df = pd.read_csv(MONTHLY_CSV, skiprows=1, low_memory=False)
    df["trdate-Trade Date"] = pd.to_datetime(df["trdate-Trade Date"])

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    _atomic_write_table(table, output_path)
    print(f"monthly: wrote {len(df)} rows to {output_path}")


def convert_daily() -> None:
    daily_dir = PARQUET_DIR / "daily"

    reader = pd.read_csv(
        DAILY_CSV,
        skiprows=1,
        low_memory=False,
        parse_dates=["trdate-Trade Date"],
        chunksize=DAILY_CHUNKSIZE,
    )

    year_writers: dict[int, pq.ParquetWriter] = {}
    year_tmp_paths: dict[int, Path] = {}
    year_row_counts: dict[int, int] = {}
    schema: pa.Schema | None = None

    try:
        for chunk in reader:
            chunk["_year"] = chunk["trdate-Trade Date"].dt.year
            for year, year_chunk in chunk.groupby("_year"):
                year_chunk = year_chunk.drop(columns="_year")
                table = pa.Table.from_pandas(year_chunk, preserve_index=False)

                if schema is None:
                    schema = table.schema
                table = table.cast(schema)

                if year not in year_writers:
                    year_dir = daily_dir / f"year={year}"
                    year_dir.mkdir(parents=True, exist_ok=True)
                    output_path = year_dir / "part.parquet"
                    if output_path.exists():
                        raise FileExistsError(
                            f"{output_path} already exists -- delete the "
                            "parquet output dir to reconvert from scratch"
                        )
                    tmp_path = output_path.with_suffix(".tmp")
                    year_tmp_paths[year] = tmp_path
                    year_writers[year] = pq.ParquetWriter(tmp_path, schema)
                    year_row_counts[year] = 0

                year_writers[year].write_table(table)
                year_row_counts[year] += len(year_chunk)
    finally:
        for writer in year_writers.values():
            writer.close()

    for year, tmp_path in year_tmp_paths.items():
        output_path = tmp_path.with_suffix(".parquet")
        os.replace(tmp_path, output_path)
        print(f"daily {year}: wrote {year_row_counts[year]} rows to {output_path}")


def main() -> None:
    convert_monthly()
    convert_daily()


if __name__ == "__main__":
    main()
