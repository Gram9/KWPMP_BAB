"""Monthly rolling OLS beta, docs/04_handoff_lowbeta_longonly.md Sec 4.2.

NOT the Frazzini-Pedersen estimator (src/estimation/beta_fp.py) and not
built on it. FP's beta is assembled from split-window, split-frequency
components (sigma from 1y daily, rho from 5y overlapping-3-day, then
Vasicek-shrunk) -- a deliberate, non-OLS construction, see that module's
own docstring. This module is a plain single-window OLS slope of a
name's trailing MONTHLY returns on the market's trailing MONTHLY returns,
UNSHRUNK (handoff Sec 3.2, user-confirmed: the long-only sort uses raw
beta -- shrinkage is rank-preserving and therefore immaterial to a top-N
selection, it only matters for BAB's leg-scaling, which has no analogue
here). Window lengths and minimum-observation counts come from
config/beta_monthly.yaml (CLAUDE.md: no magic numbers in src/).

Formation timing (CLAUDE.md's core rule): a beta for formation_date uses
ONLY monthly returns with month <= formation_date. This module accepts
whatever `stock_monthly`/`market_monthly` frames a caller passes and
trusts the caller to have already excluded anything after formation_date
-- exactly like beta_fp's own _estimate_beta_fp_core, which does not
re-derive point-in-time-ness itself, it operates on an already-bounded
input. The point-in-time truncation test in this module's own test suite
verifies this by construction (appending future rows to the input and
confirming the returned beta is unchanged), the same style as beta_fp's
own leakage-adapter tests.
"""

import datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "beta_monthly.yaml"


def _load_config(variant: str) -> dict:
    """The named window/min_obs block from config/beta_monthly.yaml.
    Raises KeyError on an unknown variant -- a typo must fail loudly,
    matching beta_fp._load_beta_config()'s convention exactly."""
    with open(CONFIG_PATH) as f:
        all_variants = yaml.safe_load(f)
    return all_variants[variant]


def _ols_beta(y: np.ndarray, x: np.ndarray) -> float | None:
    """Plain OLS slope of y on x (y = alpha + beta*x + e), via
    np.linalg.lstsq -- same numerical approach diagnostics.
    full_sample_market_loading uses, for consistency across this
    project's two OLS call sites. Returns None if x has zero variance
    (a degenerate window where the market itself didn't move -- division
    by a ~0 denominator would silently produce a huge or NaN "beta" that
    looks like real data; refusing beats fabricating)."""
    if np.std(x) == 0.0:
        return None
    design = np.column_stack([np.ones(len(x)), x])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefs[1])


def estimate_beta_monthly(
    stock_monthly: pl.DataFrame,
    market_monthly: pl.DataFrame,
    formation_date: datetime.date,
    *,
    variant: str,
    id_col: str = "id",
) -> pl.DataFrame:
    """Rolling monthly OLS beta for every name in `stock_monthly`, as of
    `formation_date`, using the named config/beta_monthly.yaml variant's
    window_months/min_obs.

    stock_monthly: long format, columns [id_col, "month", "ret"] --
    month is a calendar month-end (pl.Date), ret is that name's monthly
    arithmetic return for that month. Matches the shape
    src.market_index.monthly_panel's builders already produce (id, date,
    ret -- rename date->month before calling, or pass through with month
    already named), and monthly_returns.monthly_arithmetic_returns's
    (permno, month, ret) shape when id_col="permno".

    market_monthly: columns ["month", "ret"] -- the market index's own
    monthly return series (e.g. src.market_index.build.build_index()'s
    output, with its `date`/`index_ret` columns renamed to `month`/`ret`
    by the caller -- this function does not know or care which market
    index variant produced it, matching spec section 7's requirement that
    ANY index variant can be the estimation benchmark, resolved by the
    caller, not hardcoded here).

    formation_date: the window is [formation_date - window_months + 1
    month, formation_date], inclusive on both ends -- ONLY months with
    month <= formation_date are ever read from either input (the
    point-in-time guarantee; see module docstring). Gaps inside the
    window are tolerated (a missing month simply isn't part of that
    name's observation count) -- matches CHASS's ~22%-null monthly
    return column, which this variant must not silently over-exclude.

    Returns id_col, beta (only for names clearing min_obs -- a name
    below min_obs produces NO ROW, never a fabricated beta; matches
    monthly_returns.monthly_arithmetic_returns's "no row rather than a
    fabricated 0.0" convention) and n_obs (the real observation count
    used, for diagnostics).
    """
    cfg = _load_config(variant)
    window_months = cfg["window_months"]
    min_obs = cfg["min_obs"]

    window_start = _months_before(formation_date, window_months - 1)

    market_window = market_monthly.filter(
        (pl.col("month") >= window_start) & (pl.col("month") <= formation_date)
    ).select(["month", pl.col("ret").alias("_mkt_ret")])

    stock_window = stock_monthly.filter(
        (pl.col("month") >= window_start) & (pl.col("month") <= formation_date)
    )

    merged = stock_window.join(market_window, on="month", how="inner").filter(
        pl.col("ret").is_not_null() & pl.col("_mkt_ret").is_not_null()
    )

    rows = []
    for name, group in merged.group_by(id_col):
        name_id = name[0] if isinstance(name, tuple) else name
        n_obs = group.height
        if n_obs < min_obs:
            continue
        y = group["ret"].to_numpy()
        x = group["_mkt_ret"].to_numpy()
        beta = _ols_beta(y, x)
        if beta is None:
            continue
        rows.append({id_col: name_id, "beta": beta, "n_obs": n_obs})

    if not rows:
        return pl.DataFrame(schema={id_col: stock_monthly.schema[id_col], "beta": pl.Float64, "n_obs": pl.Int64})

    return pl.DataFrame(rows)


def _months_before(d: datetime.date, n: int) -> datetime.date:
    """The calendar month-end n calendar months before d's own month
    (d itself assumed a month-end). n=0 returns d's own month-end."""
    total_month_index = (d.year * 12 + (d.month - 1)) - n
    year, month0 = divmod(total_month_index, 12)
    month = month0 + 1
    if month == 12:
        next_first = datetime.date(year + 1, 1, 1)
    else:
        next_first = datetime.date(year, month + 1, 1)
    return next_first - datetime.timedelta(days=1)
