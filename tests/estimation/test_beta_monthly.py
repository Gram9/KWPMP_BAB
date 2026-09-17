"""Tests for src/estimation/beta_monthly.py -- monthly rolling OLS beta,
docs/04_handoff_lowbeta_longonly.md Sec 4.2. NOT the FP estimator
(src/estimation/beta_fp.py) -- see that module's own docstring for why
these are separate, genuinely different calculations.
"""

import datetime

import numpy as np
import polars as pl
import pytest

from src.estimation import beta_monthly


def _month_range(start: datetime.date, n: int) -> list[datetime.date]:
    """n consecutive calendar month-ends starting at start (start itself
    assumed a month-end)."""
    months = [start]
    for _ in range(n - 1):
        months.append(beta_monthly._months_before(months[-1], -1))
    return months


# ---------------------------------------------------------------------------
# Config echo -- catches a typo'd variant name loudly, matching
# beta_fp._load_beta_config's own convention.
# ---------------------------------------------------------------------------


def test_load_config_returns_window_24m_block():
    cfg = beta_monthly._load_config("window_24m")
    assert cfg["window_months"] == 24
    assert cfg["min_obs"] == 16


def test_load_config_returns_window_36m_block():
    cfg = beta_monthly._load_config("window_36m")
    assert cfg["window_months"] == 36
    assert cfg["min_obs"] == 24


def test_load_config_returns_window_60m_block():
    cfg = beta_monthly._load_config("window_60m")
    assert cfg["window_months"] == 60
    assert cfg["min_obs"] == 36


def test_load_config_unknown_variant_raises():
    with pytest.raises(KeyError):
        beta_monthly._load_config("window_12m_typo")


# ---------------------------------------------------------------------------
# _months_before -- the window-boundary arithmetic everything else depends
# on. Tested directly and independently of estimate_beta_monthly.
# ---------------------------------------------------------------------------


def test_months_before_zero_is_identity():
    d = datetime.date(2015, 6, 30)
    assert beta_monthly._months_before(d, 0) == d


def test_months_before_crosses_year_boundary():
    assert beta_monthly._months_before(datetime.date(2015, 1, 31), 1) == datetime.date(
        2014, 12, 31
    )


def test_months_before_36_month_window_start():
    """A 36-month window ending 2015-06-30 starts at 2012-07-31 (35
    months earlier: window_months - 1, per estimate_beta_monthly's own
    docstring)."""
    assert beta_monthly._months_before(datetime.date(2015, 6, 30), 35) == datetime.date(
        2012, 7, 31
    )


# ---------------------------------------------------------------------------
# ABSOLUTE ANCHOR (CLAUDE.md / handoff Sec 7 lesson 2): regressing the
# market's own monthly return on itself must give beta = 1.000000 exactly.
# Correlation/approx checks alone cannot catch a scale or alignment bug --
# this is the anchor external to any estimated quantity.
# ---------------------------------------------------------------------------


def test_market_on_itself_gives_beta_exactly_one():
    months = _month_range(datetime.date(2015, 6, 30), 36)
    rng = np.random.default_rng(7)
    mkt_rets = rng.normal(0, 0.04, 36)

    market_monthly = pl.DataFrame({"month": months, "ret": mkt_rets})
    stock_monthly = pl.DataFrame({"id": ["MKT"] * 36, "month": months, "ret": mkt_rets})

    result = beta_monthly.estimate_beta_monthly(
        stock_monthly, market_monthly, months[-1], variant="window_36m"
    )
    assert result.height == 1
    assert result["beta"][0] == pytest.approx(1.0, abs=1e-9)
    assert result["n_obs"][0] == 36


# ---------------------------------------------------------------------------
# Synthetic recovery: a KNOWN planted beta, zero idiosyncratic noise ->
# exact recovery. Matches test_beta_fp.py's own synthetic-recovery
# convention (fixed seed, zero noise so the check isn't merely "close").
# ---------------------------------------------------------------------------


def test_recovers_planted_beta_with_zero_noise():
    months = _month_range(datetime.date(2015, 6, 30), 36)
    rng = np.random.default_rng(11)
    mkt_rets = rng.normal(0, 0.04, 36)
    true_beta = 0.35
    true_alpha = 0.002
    stock_rets = true_alpha + true_beta * mkt_rets  # zero noise -- exact recovery

    market_monthly = pl.DataFrame({"month": months, "ret": mkt_rets})
    stock_monthly = pl.DataFrame({"id": ["A"] * 36, "month": months, "ret": stock_rets})

    result = beta_monthly.estimate_beta_monthly(
        stock_monthly, market_monthly, months[-1], variant="window_36m"
    )
    assert result.height == 1
    assert result["beta"][0] == pytest.approx(true_beta, rel=1e-9)


def test_recovers_planted_beta_60m():
    months = _month_range(datetime.date(2015, 6, 30), 60)
    rng = np.random.default_rng(13)
    mkt_rets = rng.normal(0, 0.04, 60)
    true_beta = 1.8

    market_monthly = pl.DataFrame({"month": months, "ret": mkt_rets})
    stock_monthly = pl.DataFrame({"id": ["B"] * 60, "month": months, "ret": true_beta * mkt_rets})

    result = beta_monthly.estimate_beta_monthly(
        stock_monthly, market_monthly, months[-1], variant="window_60m"
    )
    assert result.height == 1
    assert result["beta"][0] == pytest.approx(true_beta, rel=1e-9)


# ---------------------------------------------------------------------------
# min_obs boundary: one observation short must NOT emit a beta; exactly
# at min_obs must.
# ---------------------------------------------------------------------------


def test_below_min_obs_emits_no_row():
    """window_36m needs min_obs=24. A name with only 23 real
    observations inside the 36-month window (13 months null/missing --
    CHASS-style gaps) must produce NO ROW, not a fabricated beta from too
    few points."""
    months = _month_range(datetime.date(2015, 6, 30), 36)
    rng = np.random.default_rng(17)
    mkt_rets = rng.normal(0, 0.04, 36)
    stock_rets = list(0.5 * mkt_rets)
    # Null out 13 of the 36 months -- leaves exactly 23 real observations.
    for i in range(13):
        stock_rets[i] = None

    market_monthly = pl.DataFrame({"month": months, "ret": mkt_rets})
    stock_monthly = pl.DataFrame({"id": ["C"] * 36, "month": months, "ret": stock_rets})

    result = beta_monthly.estimate_beta_monthly(
        stock_monthly, market_monthly, months[-1], variant="window_36m"
    )
    assert result.height == 0, (
        f"expected no row for a name with only 23 real observations "
        f"(min_obs=24), got {result.height} row(s): {result}"
    )


def test_at_min_obs_emits_a_row():
    """Same setup as above but only 12 months nulled -- leaves exactly
    24 real observations, which must clear min_obs=24."""
    months = _month_range(datetime.date(2015, 6, 30), 36)
    rng = np.random.default_rng(17)
    mkt_rets = rng.normal(0, 0.04, 36)
    stock_rets = list(0.5 * mkt_rets)
    for i in range(12):
        stock_rets[i] = None

    market_monthly = pl.DataFrame({"month": months, "ret": mkt_rets})
    stock_monthly = pl.DataFrame({"id": ["C"] * 36, "month": months, "ret": stock_rets})

    result = beta_monthly.estimate_beta_monthly(
        stock_monthly, market_monthly, months[-1], variant="window_36m"
    )
    assert result.height == 1, (
        f"expected exactly one row for a name with exactly 24 real "
        f"observations (min_obs=24), got {result.height}"
    )
    assert result["n_obs"][0] == 24


# ---------------------------------------------------------------------------
# Point-in-time truncation: appending FUTURE months to either input must
# NOT change a beta computed as-of an earlier formation_date. Same
# leakage-proof style as beta_fp's own truncation tests.
# ---------------------------------------------------------------------------


def test_future_months_do_not_change_an_earlier_formation_date_beta():
    months_36 = _month_range(datetime.date(2015, 6, 30), 36)
    formation_date = months_36[-1]  # 2015-06-30
    rng = np.random.default_rng(23)
    mkt_rets = rng.normal(0, 0.04, 36)
    true_beta = 0.9
    stock_rets = true_beta * mkt_rets

    market_truncated = pl.DataFrame({"month": months_36, "ret": mkt_rets})
    stock_truncated = pl.DataFrame({"id": ["D"] * 36, "month": months_36, "ret": stock_rets})

    result_truncated = beta_monthly.estimate_beta_monthly(
        stock_truncated, market_truncated, formation_date, variant="window_36m"
    )

    # Extend both inputs 12 months into the FUTURE beyond formation_date,
    # with wildly different (contaminated) values -- if the estimator
    # leaked future data, the beta computed at the SAME formation_date
    # would change.
    future_months = _month_range(beta_monthly._months_before(formation_date, -1), 12)
    contaminated_mkt = rng.normal(0, 5.0, 12)  # huge, unrelated scale
    contaminated_stock = rng.normal(0, 5.0, 12)

    market_full = pl.concat(
        [market_truncated, pl.DataFrame({"month": future_months, "ret": contaminated_mkt})]
    )
    stock_full = pl.concat(
        [
            stock_truncated,
            pl.DataFrame({"id": ["D"] * 12, "month": future_months, "ret": contaminated_stock}),
        ]
    )

    result_full = beta_monthly.estimate_beta_monthly(
        stock_full, market_full, formation_date, variant="window_36m"
    )

    assert result_full["beta"][0] == pytest.approx(result_truncated["beta"][0], rel=1e-12), (
        "beta computed at formation_date changed after appending FUTURE "
        "months to the input -- this is a lookahead leak"
    )
    assert result_full["beta"][0] == pytest.approx(true_beta, rel=1e-9)


def test_future_months_proof_this_test_can_actually_detect_a_leak():
    """Proof the truncation test above isn't vacuous: if formation_date
    itself is moved forward to include the contaminated future months,
    the beta MUST change -- confirming the earlier test's "no change"
    result reflects real point-in-time correctness, not an estimator that
    ignores its inputs entirely."""
    months_36 = _month_range(datetime.date(2015, 6, 30), 36)
    rng = np.random.default_rng(23)
    mkt_rets = rng.normal(0, 0.04, 36)
    true_beta = 0.9
    stock_rets = true_beta * mkt_rets

    future_months = _month_range(datetime.date(2015, 7, 31), 12)
    contaminated_mkt = rng.normal(0, 5.0, 12)
    contaminated_stock = rng.normal(0, 5.0, 12)

    market_full = pl.DataFrame(
        {"month": months_36 + future_months, "ret": list(mkt_rets) + list(contaminated_mkt)}
    )
    stock_full = pl.DataFrame(
        {
            "id": ["D"] * 48,
            "month": months_36 + future_months,
            "ret": list(stock_rets) + list(contaminated_stock),
        }
    )

    result_at_original = beta_monthly.estimate_beta_monthly(
        stock_full, market_full, months_36[-1], variant="window_36m"
    )
    result_at_later = beta_monthly.estimate_beta_monthly(
        stock_full, market_full, future_months[-1], variant="window_36m"
    )

    assert result_at_original["beta"][0] != pytest.approx(
        result_at_later["beta"][0], rel=1e-6
    ), (
        "moving formation_date forward to include the contaminated "
        "months did NOT change beta -- the truncation test above would "
        "be vacuous if this estimator ignores its inputs"
    )
