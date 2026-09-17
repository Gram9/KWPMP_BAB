"""Tests for src/analysis/longonly_vs_market.py -- docs/
06_handoff_section6.md item A: the 20+20 long-only book's excess return
series regressed on the market index its betas were estimated against.

WRITTEN BEFORE THE MODULE (CLAUDE.md workflow).

The handoff is explicit that BOTH halves of the result belong in the
report -- the US BAB long leg showed significant alpha (t=2.99) alongside
INSIGNIFICANT raw outperformance (t=1.17) -- so this module reports a
risk-adjusted result and a raw one side by side, and both are tested.
"""

import datetime

import polars as pl
import pytest

from src.analysis import longonly_vs_market


def _quarters(n: int, start_year: int = 1990) -> list[datetime.date]:
    out = []
    y, m = start_year, 3
    for _ in range(n):
        if m == 12:
            out.append(datetime.date(y, 12, 31))
        elif m in (6, 9):
            out.append(datetime.date(y, m + 1, 1) - datetime.timedelta(days=1))
        else:
            out.append(datetime.date(y, 4, 1) - datetime.timedelta(days=1))
        m += 3
        if m > 12:
            m -= 12
            y += 1
    return out


def _series(quarters, values, col):
    return pl.DataFrame({"quarter": quarters, col: values})


# ---------------------------------------------------------------------------
# ABSOLUTE ANCHORS
# ---------------------------------------------------------------------------


def test_portfolio_identical_to_market_gives_beta_one_alpha_zero_no_outperformance():
    """ABSOLUTE ANCHOR (CLAUDE.md; docs/05_report_spec.md's global rule
    2): regressing the market on itself through the SAME code path the
    report uses must give beta = 1.000000 and alpha = 0.

    This is the anchor that catches a lagged or misaligned join, which
    correlation cannot -- a series regressed against a shifted copy of
    itself can still correlate highly.
    """
    qs = _quarters(24)
    mkt_values = [0.02 + 0.01 * ((i % 7) - 3) for i in range(24)]
    market = _series(qs, mkt_values, "ret_excess")
    port = _series(qs, list(mkt_values), "ret_excess")

    got = longonly_vs_market.longonly_vs_market_stats(port, market, label="anchor")

    assert got["realized_beta"] == pytest.approx(1.0, abs=1e-9), (
        f"market on itself gave beta={got['realized_beta']!r}, not 1.0"
    )
    assert got["alpha"] == pytest.approx(0.0, abs=1e-12), (
        f"market on itself gave alpha={got['alpha']!r}, not 0.0"
    )
    assert got["raw_outperf_mean"] == pytest.approx(0.0, abs=1e-15)


def test_planted_beta_and_alpha_are_recovered():
    """ABSOLUTE ANCHOR #2, the established tests/analysis/ idiom: plant
    port = 0.002 + 0.60*mkt and recover both coefficients.

    Catches a scale error in either series -- a x2 on the market halves
    the recovered beta while leaving correlation at exactly 1.0.
    """
    qs = _quarters(40)
    mkt_values = [0.03 * ((i % 11) - 5) / 5.0 for i in range(40)]
    port_values = [0.002 + 0.60 * m for m in mkt_values]

    market = _series(qs, mkt_values, "ret_excess")
    port = _series(qs, port_values, "ret_excess")

    got = longonly_vs_market.longonly_vs_market_stats(port, market, label="planted")

    assert got["realized_beta"] == pytest.approx(0.60, rel=1e-9), (
        f"planted beta 0.60 recovered as {got['realized_beta']!r}"
    )
    assert got["alpha"] == pytest.approx(0.002, rel=1e-9), (
        f"planted alpha 0.002 recovered as {got['alpha']!r}"
    )


def test_scaling_the_market_halves_the_recovered_beta():
    """The concrete demonstration that this code path is scale-sensitive
    where a correlation is not. Same two series, market doubled: beta
    must halve exactly. CLAUDE.md's stated reason absolute anchors are
    required at all."""
    qs = _quarters(40)
    mkt_values = [0.03 * ((i % 11) - 5) / 5.0 for i in range(40)]
    port_values = [0.002 + 0.60 * m for m in mkt_values]

    base = longonly_vs_market.longonly_vs_market_stats(
        _series(qs, port_values, "ret_excess"),
        _series(qs, mkt_values, "ret_excess"),
        label="base",
    )
    doubled = longonly_vs_market.longonly_vs_market_stats(
        _series(qs, port_values, "ret_excess"),
        _series(qs, [2.0 * m for m in mkt_values], "ret_excess"),
        label="doubled",
    )

    assert doubled["realized_beta"] == pytest.approx(base["realized_beta"] / 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Annualization and Sharpe conventions -- the two silent-misnumber traps
# ---------------------------------------------------------------------------


def test_annualization_is_geometric_not_arithmetic():
    """A mean quarterly excess return of 2.633% annualizes GEOMETRICALLY
    to 10.9553%, not arithmetically to 10.532%.

    Verified against the handoff's own measured headline (10.96% from
    2.633%/qtr), so geometric is the convention this project's existing
    numbers already use. src/analysis/beta_bucket_timeseries.py uses the
    arithmetic mean*4 convention elsewhere; the two differ by ~42bp and
    Sec 6 must not silently disagree with Sec 4.
    """
    qs = _quarters(20)
    # The portfolio mean is pinned at exactly 2.633%/qtr (the handoff's
    # own measured US figure) while the market VARIES -- a constant
    # market makes the OLS design matrix singular and silently yields
    # NaN coefficients, so the regression would not be well-posed.
    # Values VARY but their mean is exactly 0.02633 (the handoff's own
    # measured US figure): a constant series has zero variance, which
    # makes Sharpe divide by zero and the OLS design singular, so the
    # fixture would not exercise a well-posed regression.
    offsets = [0.01 * ((i % 4) - 1.5) for i in range(20)]
    assert abs(sum(offsets)) < 1e-15, "offsets must be mean-zero"
    port = _series(qs, [0.02633 + o for o in offsets], "ret_excess")
    market = _series(qs, [0.02 + 0.01 * ((i % 7) - 3) for i in range(20)], "ret_excess")

    got = longonly_vs_market.longonly_vs_market_stats(port, market, label="ann")

    assert got["port_ann_excess"] == pytest.approx(0.10955310910231564, rel=1e-9), (
        f"annualized to {got['port_ann_excess']!r} -- 0.10532 means the "
        "arithmetic mean*4 convention was used instead of geometric"
    )
    assert got["port_ann_excess"] != pytest.approx(0.10532, rel=1e-6)


def test_sharpe_uses_four_periods_per_year_not_twelve():
    """diagnostics.annualized_sharpe DEFAULTS to periods_per_year=12
    (monthly). Silently reusing that default on a quarterly series
    misannualizes by sqrt(3) -- a real, plausible-looking error CLAUDE.md
    names directly. This pins the quarterly value and asserts it is not
    the monthly one."""
    qs = _quarters(12)
    values = [0.05, -0.02, 0.03, 0.01, -0.04, 0.06, 0.02, 0.00, -0.01, 0.04, 0.03, -0.03]
    port = _series(qs, values, "ret_excess")
    market = _series(qs, [0.02 + 0.01 * ((i % 5) - 2) for i in range(12)], "ret_excess")

    got = longonly_vs_market.longonly_vs_market_stats(port, market, label="sharpe")

    import statistics

    expected_q = statistics.mean(values) / statistics.stdev(values) * (4**0.5)
    expected_m = statistics.mean(values) / statistics.stdev(values) * (12**0.5)

    assert got["port_sharpe"] == pytest.approx(expected_q, rel=1e-9), (
        f"port_sharpe={got['port_sharpe']!r}, expected the quarterly "
        f"annualization {expected_q!r}"
    )
    assert got["port_sharpe"] != pytest.approx(expected_m, rel=1e-6), (
        "Sharpe was annualized with the monthly sqrt(12) default"
    )


# ---------------------------------------------------------------------------
# alpha_t_stat vs beta_t_stat -- a real past mislabelling
# ---------------------------------------------------------------------------


def test_alpha_t_stat_is_the_intercepts_not_the_slopes():
    """docs/04_handoff_lowbeta_longonly.md Sec 10 records a probe that
    reported beta's t-stat as an alpha t-stat. The tell was
    market-on-market showing t=6.8e16 -- what beta's t-stat does when
    beta is exactly 1 and residuals vanish, never what an alpha t-stat
    shows. Both are surfaced here under unambiguous names and this test
    keeps them numerically distinguishable."""
    qs = _quarters(40)
    mkt_values = [0.03 * ((i % 11) - 5) / 5.0 for i in range(40)]
    port_values = [0.002 + 0.60 * m for m in mkt_values]

    got = longonly_vs_market.longonly_vs_market_stats(
        _series(qs, port_values, "ret_excess"),
        _series(qs, mkt_values, "ret_excess"),
        label="tstats",
    )

    assert "alpha_t_stat" in got and "beta_t_stat" in got
    assert got["alpha_t_stat"] != got["beta_t_stat"], (
        "alpha and beta t-statistics are identical, which means one is a "
        "copy of the other"
    )


# ---------------------------------------------------------------------------
# Raw outperformance t-test
# ---------------------------------------------------------------------------


def test_raw_outperformance_t_matches_hand_computed_with_ddof_one():
    """mean(d) / (std(d, ddof=1) / sqrt(n)). A ddof=0 implementation
    inflates the t-statistic, which is the direction that manufactures
    significance."""
    qs = _quarters(5)
    port_values = [0.05, 0.02, 0.04, 0.01, 0.03]
    mkt_values = [0.01, 0.02, 0.005, 0.015, 0.012]

    got = longonly_vs_market.longonly_vs_market_stats(
        _series(qs, port_values, "ret_excess"),
        _series(qs, mkt_values, "ret_excess"),
        label="tt",
    )

    import statistics

    d = [p - m for p, m in zip(port_values, mkt_values)]
    expected = statistics.mean(d) / (statistics.stdev(d) / (5**0.5))

    assert got["raw_outperf_t"] == pytest.approx(expected, rel=1e-9), (
        f"raw_outperf_t={got['raw_outperf_t']!r}, expected {expected!r}"
    )
    assert got["raw_outperf_mean"] == pytest.approx(statistics.mean(d), rel=1e-12)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_n_quarters_is_the_joined_intersection():
    """A market series shorter than the portfolio's must reduce n, not
    silently regress a misaligned overlap."""
    qs = _quarters(20)
    # A VARYING market series: a constant one makes the OLS design matrix
    # singular and the Sharpe denominator zero, which would make this an
    # alignment test that never reaches the alignment logic.
    mkt_values = [0.02 + 0.01 * ((i % 7) - 3) for i in range(20)]
    port = _series(qs, [0.002 + 0.7 * m for m in mkt_values], "ret_excess")
    market = _series(qs[:15], mkt_values[:15], "ret_excess")

    got = longonly_vs_market.longonly_vs_market_stats(port, market, label="join")

    assert got["n_quarters"] == 15, f"n_quarters={got['n_quarters']}, expected 15"
    assert got["first_quarter"] == qs[0]
    assert got["last_quarter"] == qs[14]


def test_sample_range_reports_the_joined_endpoints():
    qs = _quarters(8)
    mkt_values = [0.02 + 0.01 * ((i % 5) - 2) for i in range(8)]
    port = _series(qs[2:], [0.002 + 0.7 * m for m in mkt_values[2:]], "ret_excess")
    market = _series(qs, mkt_values, "ret_excess")

    got = longonly_vs_market.longonly_vs_market_stats(port, market, label="range")

    assert got["first_quarter"] == qs[2]
    assert got["last_quarter"] == qs[7]
    assert got["n_quarters"] == 6


def test_table_has_a_pinned_column_order():
    qs = _quarters(10)
    rows = [
        longonly_vs_market.longonly_vs_market_stats(
            _series(qs, [0.02 + 0.005 * ((i % 4) - 2) for i in range(10)], "ret_excess"),
            _series(qs, [0.01 + 0.004 * ((i % 4) - 2) for i in range(10)], "ret_excess"),
            label=name,
        )
        for name in ("us", "ca")
    ]

    table = longonly_vs_market.longonly_vs_market_table(rows)

    assert table.columns[0] == "label"
    assert table.height == 2
    assert "alpha_t_stat" in table.columns
