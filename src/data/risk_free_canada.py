"""
Canadian risk-free rate loader, per config/risk_free.yaml's `canada`
block (full reasoning lives in that file -- this module honours those
decisions rather than re-deriving them).

CHASS ind2-30 day Return on T-Bills, NOT ind1 (the 91-day annualized
rate in percent -- a different instrument, corr(ind2, ind1/1200) =
0.944, not ~1.0). The monthly parquet is a per-security panel, so ind2
repeats identically across every security row for a given month; this
loader dedupes to one row per month before returning.

ind2 is null for exactly two months (2023-10-31, 2023-11-30). Config's
derived_overrides fills both from a documented, auditable source ("prefer
the file's own ind1 where derivable, else a user-supplied external rate")
and this loader emits an rf_source column (chass /
derived_from_ind1 / derived_from_external_rate) so a reader can always
tell a real ind2 observation from an interpolation on a neighbouring
convention -- these two months must never pass as observed data.
"""

from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "risk_free.yaml"

with CONFIG_PATH.open() as f:
    _CONFIG = yaml.safe_load(f)

_CANADA_CONFIG = _CONFIG["canada"]
CHASS_MONTHLY_PATH = PROJECT_ROOT / _CANADA_CONFIG["path"]
_CHASS_COLUMN = _CANADA_CONFIG["column"]
_DATE_COLUMN = "trdate-Trade Date"
_DERIVED_OVERRIDES = _CANADA_CONFIG["derived_overrides"]
_BASIS_TO_SOURCE = {
    "external_rate": "derived_from_external_rate",
    "ind1": "derived_from_ind1",
}
assert _CANADA_CONFIG["units"] == "decimal", (
    f"{CONFIG_PATH}: canada.units expected 'decimal' -- loader does not "
    "implement any other conversion."
)


def load_canada_rf_chass() -> pl.DataFrame:
    """Canadian monthly RF: date (CHASS's own real trading month-end),
    rf (decimal), rf_source (chass / derived_from_ind1 /
    derived_from_external_rate)."""
    panel = pl.read_parquet(CHASS_MONTHLY_PATH, columns=[_DATE_COLUMN, _CHASS_COLUMN]).with_columns(
        pl.col(_DATE_COLUMN).dt.date().alias("date")
    )

    # ind2 is a macro series repeated identically across every security
    # row within a month -- but nothing upstream guarantees that. Assert
    # it explicitly rather than trusting plain unique() to pick a
    # consistent row (polars gives no ordering guarantee for which
    # duplicate survives unless `keep=` is specified -- see
    # chass_loader.py's own note on this). A month with more than one
    # distinct ind2 value (including a null/non-null split) means the
    # macro-repeat assumption has broken and this must fail loudly
    # rather than silently pick an arbitrary value.
    per_month_distinct_counts = panel.group_by("date").agg(
        pl.col(_CHASS_COLUMN).n_unique().alias("n_distinct")
    )
    bad_months = per_month_distinct_counts.filter(pl.col("n_distinct") > 1)
    assert bad_months.height == 0, (
        f"{CHASS_MONTHLY_PATH}: {bad_months.height} month(s) have more "
        f"than one distinct {_CHASS_COLUMN!r} value across security "
        f"rows (first: {bad_months['date'][0] if bad_months.height else None}) "
        "-- the assumption that this column is a macro series repeated "
        "identically per month no longer holds; refusing to silently "
        "pick an arbitrary row."
    )

    monthly = (
        panel.unique(subset=["date"], keep="first", maintain_order=False)
        .select(
            "date",
            pl.col(_CHASS_COLUMN).alias("rf"),
        )
        .with_columns(pl.lit("chass").alias("rf_source"))
        .sort("date")
    )

    for override in _DERIVED_OVERRIDES:
        month_end = pl.Series([override["month_end"]]).str.to_date()[0]
        source = _BASIS_TO_SOURCE[override["basis"]]
        # Check ALL pre-dedup rows for this month, not the single
        # deduped survivor -- a mixed null/non-null month would dedup to
        # an arbitrary row, and checking only that row could pass this
        # guard even when a real observation exists elsewhere in the
        # month (this is exactly what the per-month distinct-value
        # assert above also catches, but check again here directly
        # against the override so this function doesn't depend on
        # assertion ordering to stay safe).
        month_rows = panel.filter(pl.col("date") == month_end)
        assert month_rows.height > 0, (
            f"{CONFIG_PATH}: derived_overrides month_end "
            f"{override['month_end']!r} not found in "
            f"{CHASS_MONTHLY_PATH} -- expected at least one matching "
            "month-end row to override."
        )
        assert month_rows[_CHASS_COLUMN].null_count() == month_rows.height, (
            f"{CONFIG_PATH}: derived_overrides month_end "
            f"{override['month_end']!r} has a non-null ind2 value in "
            f"{CHASS_MONTHLY_PATH} -- this override was written when "
            "that month was null; the source data has since changed "
            "and this override may no longer be needed or correct."
        )
        monthly = monthly.with_columns(
            pl.when(pl.col("date") == month_end)
            .then(pl.lit(override["value"]))
            .otherwise(pl.col("rf"))
            .alias("rf"),
            pl.when(pl.col("date") == month_end)
            .then(pl.lit(source))
            .otherwise(pl.col("rf_source"))
            .alias("rf_source"),
        )

    return monthly
