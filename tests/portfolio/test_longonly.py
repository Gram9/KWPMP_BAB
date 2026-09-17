"""Tests for src/portfolio/longonly.py -- the quarterly, long-only,
equal-weight low-beta backtester, docs/04_handoff_lowbeta_longonly.md
Sec 4.1.
"""

import datetime

import polars as pl
import pytest

from src.portfolio import longonly


def _betas(ids: list[str], betas: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"id": ids, "beta": betas})


def _fringe(ids: list[str], mkt_caps: list[float], prices: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"id": ids, "mkt_cap": mkt_caps, "price": prices})


def _monthly_rets(rows: list[tuple[str, datetime.date, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [r[0] for r in rows],
            "month": [r[1] for r in rows],
            "ret": [r[2] for r in rows],
        }
    )


_BIG_CAP = 5_000_000_000.0
_PRICE = 50.0

# 6 names, comfortably above min_survivors/n_holdings for small tests.
_IDS = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Formation timing -- the rule that overrides everything (CLAUDE.md). Same
# engineered-magnitude approach as tests/portfolio/test_rebalance.py's own
# formation-timing test: the formation quarter's return is all-zero, the
# HELD quarter's is where all the signal lives, so using the wrong quarter
# is unmistakable in the output.
# ---------------------------------------------------------------------------


def test_holding_quarter_return_is_the_quarter_AFTER_formation():
    formation_date = datetime.date(2020, 3, 31)  # betas formed through Q1
    held_quarter = datetime.date(2020, 6, 30)  # Q2 is what should be held

    betas_by_date = {formation_date: _betas(_IDS[:4], [0.4, 0.6, 0.8, 1.0])}
    fringe_by_date = {
        formation_date: _fringe(_IDS[:4], [_BIG_CAP] * 4, [_PRICE] * 4)
    }

    # Formation quarter (Jan/Feb/Mar 2020) returns are ALL ZERO for every
    # name -- if the loop ever used this quarter's return instead of the
    # held quarter's, the portfolio return would be exactly 0.0.
    formation_months = [
        datetime.date(2020, 1, 31),
        datetime.date(2020, 2, 29),
        datetime.date(2020, 3, 31),
    ]
    # Held quarter (Apr/May/Jun 2020) has real, nonzero signal.
    held_months = [
        datetime.date(2020, 4, 30),
        datetime.date(2020, 5, 31),
        datetime.date(2020, 6, 30),
    ]

    rows = []
    for name in _IDS[:4]:
        for m in formation_months:
            rows.append((name, m, 0.0))
        for m in held_months:
            rows.append((name, m, 0.03))  # nonzero, uniform for simplicity

    quarterly_monthly_rets = _monthly_rets(rows)

    result = longonly.run_longonly_backtest(
        betas_by_date,
        fringe_by_date,
        quarterly_monthly_rets,
        n_holdings=4,
        turnover_k=1,
        min_survivors=4,
        mkt_cap_floor=1e9,
        price_floor=1.0,
    )

    assert held_quarter in result.returns["quarter"].to_list(), (
        "expected a return row keyed on the HELD quarter (2020-06-30), "
        f"got quarters: {result.returns['quarter'].to_list()}"
    )
    ret = result.returns.filter(pl.col("quarter") == held_quarter)["ret"][0]
    assert ret != pytest.approx(0.0, abs=1e-9), (
        "portfolio return is exactly 0.0 -- this is the exact signature "
        "of using the FORMATION quarter's (all-zero) return instead of "
        "the HELD quarter's, the mistake handoff Sec 7 documents"
    )
    # 3 months of 3% compounded (expm1(3*log1p(0.03))) -- exact value
    # pinned independently in test_quarterly_return_compounds_three_
    # months_in_log_space below; here we only need "matches the real
    # compounded value" to confirm this test exercises the real held
    # quarter, not a hand-picked coincidence.
    import math

    assert ret == pytest.approx(math.expm1(3 * math.log1p(0.03)), rel=1e-9)


def test_formation_quarter_never_appears_as_the_held_quarter():
    """Direct structural check: no returns row is EVER keyed on the
    formation_date's own quarter -- the held quarter must always be
    strictly later."""
    formation_date = datetime.date(2020, 3, 31)
    betas_by_date = {formation_date: _betas(_IDS[:4], [0.4, 0.6, 0.8, 1.0])}
    fringe_by_date = {formation_date: _fringe(_IDS[:4], [_BIG_CAP] * 4, [_PRICE] * 4)}

    all_months = [
        datetime.date(2020, 1, 31), datetime.date(2020, 2, 29), datetime.date(2020, 3, 31),
        datetime.date(2020, 4, 30), datetime.date(2020, 5, 31), datetime.date(2020, 6, 30),
    ]
    rows = [(name, m, 0.01) for name in _IDS[:4] for m in all_months]

    result = longonly.run_longonly_backtest(
        betas_by_date, fringe_by_date, _monthly_rets(rows),
        n_holdings=4, turnover_k=1, min_survivors=4,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    quarters = result.returns["quarter"].to_list()
    assert formation_date not in quarters
    assert all(q > formation_date for q in quarters)


def test_every_formation_date_strictly_precedes_its_own_holding_quarter():
    """The general-case check (same shape as tests/leakage/
    test_no_lookahead.py::test_formation_strictly_precedes_holding_period
    for the monthly BAB loop, applied to quarters): for EVERY formation
    date the real loop processes, that date's own held quarter must be
    strictly later -- asserted on actual dates the loop emits (joining
    positions' formation_date against returns' quarter via
    next_quarter_end, exactly how a real caller would reconstruct the
    pairing), not on the intent of the code. With 3 sequential formation
    dates one quarter apart, this exercises the pairing 3 times, not
    once."""
    from src.portfolio.quarterly import next_quarter_end

    dates = [
        datetime.date(2020, 3, 31),
        datetime.date(2020, 6, 30),
        datetime.date(2020, 9, 30),
    ]
    betas_by_date = {d: _betas(_IDS[:4], [0.4, 0.6, 0.8, 1.0]) for d in dates}
    fringe_by_date = {d: _fringe(_IDS[:4], [_BIG_CAP] * 4, [_PRICE] * 4) for d in dates}

    month_list = []
    for year, month in [
        (2020, 1), (2020, 2), (2020, 3), (2020, 4), (2020, 5), (2020, 6),
        (2020, 7), (2020, 8), (2020, 9), (2020, 10), (2020, 11), (2020, 12),
    ]:
        next_month = month + 1
        next_year = year
        if next_month > 12:
            next_month = 1
            next_year += 1
        month_list.append(datetime.date(next_year, next_month, 1) - datetime.timedelta(days=1))

    rows = [(name, m, 0.01) for name in _IDS[:4] for m in month_list]

    result = longonly.run_longonly_backtest(
        betas_by_date, fringe_by_date, _monthly_rets(rows),
        n_holdings=4, turnover_k=1, min_survivors=4,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    assert result.returns.height == 3
    held_quarters = set(result.returns["quarter"].to_list())
    for formation_date in dates:
        expected_quarter = next_quarter_end(formation_date)
        assert expected_quarter in held_quarters
        assert formation_date < expected_quarter, (
            f"formation_date={formation_date} is not strictly before its "
            f"own held quarter={expected_quarter}"
        )


# ---------------------------------------------------------------------------
# Quarterly compounding -- exact value, independently computed.
# ---------------------------------------------------------------------------


def test_quarterly_return_compounds_three_months_in_log_space():
    """3 months of 3% each: expm1(3*log1p(0.03)) = 0.09272700...,
    computed independently in Python, not re-derived from the function
    under test. NOT 0.09 (naive sum) or exactly 0.03 -- proves real
    compounding is happening."""
    import math

    expected = math.expm1(3 * math.log1p(0.03))
    assert expected == pytest.approx(0.092727, rel=1e-6)  # sanity on the hand-computed value

    held_quarter = datetime.date(2020, 6, 30)
    months = [datetime.date(2020, 4, 30), datetime.date(2020, 5, 31), datetime.date(2020, 6, 30)]
    monthly_rets = _monthly_rets([("A", m, 0.03) for m in months])

    result = longonly._quarterly_return(monthly_rets, "id", held_quarter)
    assert result.height == 1
    assert result["ret"][0] == pytest.approx(expected, rel=1e-9)


def test_quarterly_return_compounds_partial_quarter_for_delisted_name():
    """A name with only 2 of 3 months present (e.g. delisted after May,
    with April/May's returns real and June simply absent -- the shape
    monthly_returns.py produces for a real mid-quarter delisting) must
    still produce a row: its 2 real months compounded, with June
    implicitly 0% cash. User-confirmed this session (replacing an
    earlier "exclude entirely" design flagged by leakage-auditor as
    understating downturn losses): expm1(log1p(0.03)+log1p(0.02)) is the
    correct partial-quarter value, independently computed, NOT the same
    as the full 3-month compound A gets."""
    import math

    held_quarter = datetime.date(2020, 6, 30)
    monthly_rets = _monthly_rets(
        [
            ("A", datetime.date(2020, 4, 30), 0.03),
            ("A", datetime.date(2020, 5, 31), 0.02),
            ("A", datetime.date(2020, 6, 30), 0.01),
            ("B", datetime.date(2020, 4, 30), 0.03),
            ("B", datetime.date(2020, 5, 31), 0.02),
            # B is missing June (delisted after May) -- must get a
            # PARTIAL return (Apr+May compounded), not be excluded.
        ]
    )
    result = longonly._quarterly_return(monthly_rets, "id", held_quarter)
    assert set(result["id"].to_list()) == {"A", "B"}

    a_ret = result.filter(pl.col("id") == "A")["ret"][0]
    b_ret = result.filter(pl.col("id") == "B")["ret"][0]
    expected_a = math.expm1(math.log1p(0.03) + math.log1p(0.02) + math.log1p(0.01))
    expected_b = math.expm1(math.log1p(0.03) + math.log1p(0.02))  # June implicitly 0%
    assert a_ret == pytest.approx(expected_a, rel=1e-9)
    assert b_ret == pytest.approx(expected_b, rel=1e-9)
    assert b_ret != pytest.approx(a_ret, rel=1e-6), (
        "B's partial-quarter return must differ from A's full-quarter "
        "return -- if they came out equal, B's June return was somehow "
        "still being read despite being absent from the input"
    )


def test_quarterly_return_produces_no_row_for_name_absent_all_quarter():
    """A name with ZERO real months in the quarter (never listed, or
    fully missing) must produce NO ROW -- distinct from the partial-
    quarter case above. There is nothing to compound, real or partial."""
    held_quarter = datetime.date(2020, 6, 30)
    monthly_rets = _monthly_rets(
        [
            ("A", datetime.date(2020, 4, 30), 0.03),
            ("A", datetime.date(2020, 5, 31), 0.02),
            ("A", datetime.date(2020, 6, 30), 0.01),
            # "C" has no rows at all in this quarter's months.
            ("C", datetime.date(2020, 1, 31), 0.05),  # outside this quarter
        ]
    )
    result = longonly._quarterly_return(monthly_rets, "id", held_quarter)
    assert "C" not in result["id"].to_list()


def test_end_to_end_delisted_holding_contributes_partial_return_not_excluded():
    """Full run_longonly_backtest level (not just _quarterly_return in
    isolation): a HELD name that delists mid-quarter (a large negative
    return in month 2, then no June row at all) must move the
    portfolio's reported quarterly return -- proving the fix reaches the
    actual output, not just the internal helper. Constructed so the
    delisted name's loss is unmistakable if included and invisible if
    excluded: 3 other names return exactly 0% every month, so if the
    delisted name's loss were still being excluded-and-renormalized
    (the old design), the portfolio return would be EXACTLY 0.0."""
    formation_date = datetime.date(2020, 3, 31)
    betas_by_date = {formation_date: _betas(_IDS[:4], [0.1, 0.2, 0.3, 0.4])}
    fringe_by_date = {formation_date: _fringe(_IDS[:4], [_BIG_CAP] * 4, [_PRICE] * 4)}

    rows = [
        # 3 flat names -- 0% every month, all 3 months present.
        ("A", datetime.date(2020, 4, 30), 0.0),
        ("A", datetime.date(2020, 5, 31), 0.0),
        ("A", datetime.date(2020, 6, 30), 0.0),
        ("B", datetime.date(2020, 4, 30), 0.0),
        ("B", datetime.date(2020, 5, 31), 0.0),
        ("B", datetime.date(2020, 6, 30), 0.0),
        ("C", datetime.date(2020, 4, 30), 0.0),
        ("C", datetime.date(2020, 5, 31), 0.0),
        ("C", datetime.date(2020, 6, 30), 0.0),
        # D delists after May: real Apr/May returns (May carries a large
        # negative delisting-style return), no June row at all.
        ("D", datetime.date(2020, 4, 30), 0.0),
        ("D", datetime.date(2020, 5, 31), -0.60),
    ]

    result = longonly.run_longonly_backtest(
        betas_by_date, fringe_by_date, _monthly_rets(rows),
        n_holdings=4, turnover_k=1, min_survivors=4,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    held_quarter = datetime.date(2020, 6, 30)
    ret = result.returns.filter(pl.col("quarter") == held_quarter)["ret"][0]
    assert ret != pytest.approx(0.0, abs=1e-9), (
        "portfolio return is exactly 0.0 -- this is the signature of D's "
        "delisting loss being excluded-and-renormalized-away (the OLD, "
        "now-replaced design) rather than contributing its partial-"
        "quarter return"
    )
    # D's partial return is -0.60 (May only, June implicitly 0% cash);
    # 3 flat names at 0.0 -- equal-weight mean = -0.60 / 4 = -0.15.
    assert ret == pytest.approx(-0.15, rel=1e-9)


# ---------------------------------------------------------------------------
# Turnover mechanic -- the director's "drop worst-ranked K, buy best-ranked
# replacements" spec.
# ---------------------------------------------------------------------------


def test_first_formation_date_selects_n_lowest_beta_names_outright():
    formation_date = datetime.date(2020, 3, 31)
    betas_by_date = {
        formation_date: _betas(_IDS, [0.9, 0.2, 1.5, 0.1, 0.6, 1.2]),
    }
    fringe_by_date = {formation_date: _fringe(_IDS, [_BIG_CAP] * 6, [_PRICE] * 6)}
    held_months = [
        datetime.date(2020, 4, 30), datetime.date(2020, 5, 31), datetime.date(2020, 6, 30),
    ]
    rows = [(name, m, 0.01) for name in _IDS for m in held_months]

    result = longonly.run_longonly_backtest(
        betas_by_date, fringe_by_date, _monthly_rets(rows),
        n_holdings=3, turnover_k=1, min_survivors=3,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    held = set(result.positions.filter(pl.col("formation_date") == formation_date)["id"].to_list())
    # Lowest 3 betas: D(0.1), B(0.2), E(0.6)
    assert held == {"D", "B", "E"}
    turnover_row = result.turnover.filter(pl.col("formation_date") == formation_date)
    assert turnover_row["n_added"][0] == 3
    assert turnover_row["n_dropped"][0] == 0


def test_turnover_k_caps_replacement_to_worst_ranked_held_names():
    """Two formation dates, n_holdings=3, turnover_k=1: at Q2, only the
    SINGLE worst-ranked (highest beta) of the 3 currently-held names may
    be dropped, even though the universe's rank order changed
    substantially -- the other 2 held names must be carried over
    regardless of their new rank, proving turnover_k is actually a cap
    and not a full re-pick."""
    q1 = datetime.date(2020, 3, 31)
    q2 = datetime.date(2020, 6, 30)

    betas_by_date = {
        q1: _betas(_IDS, [0.9, 0.2, 1.5, 0.1, 0.6, 1.2]),
        # At Q2, ranks are completely reshuffled: D and B (held from Q1)
        # are now the TWO WORST betas in the universe -- but turnover_k=1
        # means only ONE of them can be dropped.
        q2: _betas(_IDS, [0.1, 1.8, 0.15, 1.9, 0.2, 0.25]),
    }
    fringe_by_date = {
        q1: _fringe(_IDS, [_BIG_CAP] * 6, [_PRICE] * 6),
        q2: _fringe(_IDS, [_BIG_CAP] * 6, [_PRICE] * 6),
    }
    all_months = [
        datetime.date(2020, 1, 31), datetime.date(2020, 2, 29), datetime.date(2020, 3, 31),
        datetime.date(2020, 4, 30), datetime.date(2020, 5, 31), datetime.date(2020, 6, 30),
        datetime.date(2020, 7, 31), datetime.date(2020, 8, 31), datetime.date(2020, 9, 30),
    ]
    rows = [(name, m, 0.01) for name in _IDS for m in all_months]

    result = longonly.run_longonly_backtest(
        betas_by_date, fringe_by_date, _monthly_rets(rows),
        n_holdings=3, turnover_k=1, min_survivors=3,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    q1_held = set(result.positions.filter(pl.col("formation_date") == q1)["id"].to_list())
    q2_held = set(result.positions.filter(pl.col("formation_date") == q2)["id"].to_list())

    assert q1_held == {"D", "B", "E"}  # lowest 3 at Q1: D(0.1), B(0.2), E(0.6)

    # At Q2, D and B are now ranked worst-of-held (D's new beta=1.9,
    # B's new beta=1.8) -- but turnover_k=1 caps the drop to ONE of
    # them. E's new beta=0.2 keeps it comfortably held regardless.
    dropped_count = len(q1_held - q2_held)
    assert dropped_count == 1, (
        f"turnover_k=1 must drop EXACTLY 1 of the 3 previously-held "
        f"names, got {dropped_count} dropped ({q1_held - q2_held})"
    )
    assert "E" in q2_held  # E's beta improved, should never be dropped

    turnover_row = result.turnover.filter(pl.col("formation_date") == q2)
    assert turnover_row["n_dropped"][0] == 1
    assert turnover_row["n_added"][0] == 1


def test_no_longer_eligible_name_is_always_dropped_regardless_of_turnover_k():
    """A held name that disappears from the fringe/beta cross-section
    entirely (delisted, or fell below the fringe filter) must ALWAYS be
    dropped -- turnover_k caps discretionary replacement, it cannot force
    the portfolio to hold a name with no current beta or price."""
    q1 = datetime.date(2020, 3, 31)
    q2 = datetime.date(2020, 6, 30)

    betas_by_date = {
        q1: _betas(_IDS, [0.9, 0.2, 1.5, 0.1, 0.6, 1.2]),
        # "D" (held from Q1, lowest beta) is ABSENT at Q2 entirely.
        q2: _betas(["A", "B", "C", "E", "F"], [0.3, 0.25, 1.5, 0.4, 1.2]),
    }
    fringe_by_date = {
        q1: _fringe(_IDS, [_BIG_CAP] * 6, [_PRICE] * 6),
        q2: _fringe(["A", "B", "C", "E", "F"], [_BIG_CAP] * 5, [_PRICE] * 5),
    }
    all_months = [
        datetime.date(2020, 1, 31), datetime.date(2020, 2, 29), datetime.date(2020, 3, 31),
        datetime.date(2020, 4, 30), datetime.date(2020, 5, 31), datetime.date(2020, 6, 30),
    ]
    rows = [(name, m, 0.01) for name in _IDS for m in all_months]

    result = longonly.run_longonly_backtest(
        betas_by_date, fringe_by_date, _monthly_rets(rows),
        n_holdings=3, turnover_k=0, min_survivors=3,  # turnover_k=0!
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    q1_held = set(result.positions.filter(pl.col("formation_date") == q1)["id"].to_list())
    q2_held = set(result.positions.filter(pl.col("formation_date") == q2)["id"].to_list())

    assert q1_held == {"D", "B", "E"}
    assert "D" not in q2_held, (
        "D dropped out of the universe entirely at Q2 but turnover_k=0 "
        "should not have been able to prevent its removal -- a name with "
        "no current beta/price cannot be force-held"
    )


# ---------------------------------------------------------------------------
# Skip handling -- thin cross-sections, missing fringe snapshots.
# ---------------------------------------------------------------------------


def test_skips_formation_date_below_min_survivors():
    formation_date = datetime.date(2020, 3, 31)
    betas_by_date = {formation_date: _betas(["A", "B"], [0.4, 0.6])}
    fringe_by_date = {formation_date: _fringe(["A", "B"], [_BIG_CAP] * 2, [_PRICE] * 2)}

    result = longonly.run_longonly_backtest(
        betas_by_date, fringe_by_date, pl.DataFrame(schema={"id": pl.Utf8, "month": pl.Date, "ret": pl.Float64}),
        n_holdings=5, turnover_k=1, min_survivors=5,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    assert formation_date in result.skips
    assert "min_survivors" in result.skips[formation_date]


def test_fringe_filter_excludes_names_below_floors():
    """A name failing EITHER floor (mkt_cap or price) must not be
    selectable, even with an excellent (low) beta."""
    formation_date = datetime.date(2020, 3, 31)
    betas_by_date = {
        formation_date: _betas(_IDS, [0.05, 0.9, 1.5, 0.1, 0.6, 1.2]),
    }
    fringe_by_date = {
        formation_date: _fringe(
            _IDS,
            # A has the BEST beta (0.05) but fails mkt_cap; C fails price.
            mkt_caps=[500_000_000.0, _BIG_CAP, _BIG_CAP, _BIG_CAP, _BIG_CAP, _BIG_CAP],
            prices=[_PRICE, _PRICE, 0.50, _PRICE, _PRICE, _PRICE],
        )
    }
    held_months = [datetime.date(2020, 4, 30), datetime.date(2020, 5, 31), datetime.date(2020, 6, 30)]
    rows = [(name, m, 0.01) for name in _IDS for m in held_months]

    result = longonly.run_longonly_backtest(
        betas_by_date, fringe_by_date, _monthly_rets(rows),
        n_holdings=3, turnover_k=1, min_survivors=3,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    held = set(result.positions.filter(pl.col("formation_date") == formation_date)["id"].to_list())
    assert "A" not in held, "A has the best beta but fails mkt_cap_floor -- must be excluded"
    assert "C" not in held, "C fails price_floor -- must be excluded"
    # Next-best-eligible 3 of the remaining {B:0.9, D:0.1, E:0.6, F:1.2}: D, E, B
    assert held == {"D", "E", "B"}


def test_circuit_breaker_raises_after_max_consecutive_skips():
    dates = [datetime.date(2020, 3, 31), datetime.date(2020, 6, 30), datetime.date(2020, 9, 30)]
    betas_by_date = {d: _betas(["A"], [0.5]) for d in dates}  # always below min_survivors
    fringe_by_date = {d: _fringe(["A"], [_BIG_CAP], [_PRICE]) for d in dates}

    with pytest.raises(RuntimeError):
        longonly.run_longonly_backtest(
            betas_by_date, fringe_by_date,
            pl.DataFrame(schema={"id": pl.Utf8, "month": pl.Date, "ret": pl.Float64}),
            n_holdings=5, turnover_k=1, min_survivors=5,
            mkt_cap_floor=1e9, price_floor=1.0, max_consecutive_skips=2,
        )
