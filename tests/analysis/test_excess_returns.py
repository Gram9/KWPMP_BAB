"""Tests for src/analysis/excess_returns.py -- calendar re-keying,
quarterly risk-free compounding, and raw -> excess conversion for
docs/06_handoff_section6.md items A/B/C.

WRITTEN BEFORE THE MODULE (CLAUDE.md workflow).

Two defect classes drive this file, both measured on real project data
rather than imagined:

1. RE-KEYING. 158 of 552 Canadian risk-free months and 137 of 480
   Canadian market-index months are keyed on CHASS real trading days,
   not calendar month-ends. Joined against calendar quarter-ends without
   re-keying, 118 of 148 Canadian quarters compound only one or two of
   their three months -- and return a plausible number for all 148
   (_scratch/_probe_ca_align_out.txt). Understating compounded rf
   OVERSTATES excess return, alpha and Sharpe.

2. THE 3-MONTH SUM. Summing three monthly rf values instead of
   compounding them differs by ~1bp at this project's rate levels --
   small enough to look right, which is exactly CLAUDE.md's stated worst
   outcome. The anchor below pins the compounded value exactly.
"""

import datetime

import polars as pl
import pytest

from src.analysis import excess_returns


def _monthly(rows: list[tuple[datetime.date, float]], value_col: str = "rf") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [r[0] for r in rows],
            value_col: [r[1] for r in rows],
        }
    )


def _quarterly(rows: list[tuple[datetime.date, float]], value_col: str = "ret") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "quarter": [r[0] for r in rows],
            value_col: [r[1] for r in rows],
        }
    )


_Q1_1990 = datetime.date(1990, 3, 31)
_Q1_1990_CALENDAR_MONTHS = [
    datetime.date(1990, 1, 31),
    datetime.date(1990, 2, 28),
    datetime.date(1990, 3, 31),
]
# The same quarter's months as CHASS would key them -- real last trading
# days, two of the three NOT the calendar month-end. Modelled on real
# observed values (e.g. 2025-11-28 for November 2025).
_Q1_1990_TRADING_MONTHS = [
    datetime.date(1990, 1, 31),
    datetime.date(1990, 2, 27),
    datetime.date(1990, 3, 30),
]


# ---------------------------------------------------------------------------
# rekey_to_calendar_month_end
# ---------------------------------------------------------------------------


def test_rekey_maps_trading_day_to_calendar_month_end():
    """The real observed case: CHASS keys November 2025 on 2025-11-28.
    A no-op re-key leaves it there and the month vanishes from any
    calendar-keyed quarter lookup."""
    frame = _monthly([(datetime.date(2025, 11, 28), 0.00209)])

    got = excess_returns.rekey_to_calendar_month_end(frame, date_col="date")

    assert got["date"].to_list() == [datetime.date(2025, 11, 30)], (
        f"2025-11-28 re-keyed to {got['date'].to_list()}, not 2025-11-30"
    )
    assert got["rf"][0] == 0.00209, "re-keying must not alter the value"


def test_rekey_is_idempotent_on_calendar_month_ends():
    """The US series is already calendar-keyed. Re-keying it must be a
    no-op -- otherwise passing rekey=True uniformly (which is what gives
    the US path the collision guard for free) would corrupt it."""
    frame = _monthly([(m, 0.01) for m in _Q1_1990_CALENDAR_MONTHS])

    got = excess_returns.rekey_to_calendar_month_end(frame, date_col="date")

    assert got["date"].to_list() == _Q1_1990_CALENDAR_MONTHS, (
        f"re-keying an already-calendar series changed it to {got['date'].to_list()}"
    )


def test_rekey_raises_when_two_rows_collapse_onto_one_month():
    """dt.month_end() is a LOSSY map. Two rows in the same calendar month
    would silently become two rows on the same key, and the log-space
    compounding downstream would then compound that month TWICE --
    inflating rf and understating excess return.

    Measured today: the Canadian rf series is 552 rows -> 552 distinct
    month-ends, zero collisions. This guard is for the next data
    refresh. It must RAISE, never dedupe: picking an arbitrary surviving
    row is precisely what src/data/risk_free_canada.py already refuses
    to do for its own ind2 column.
    """
    frame = _monthly(
        [(datetime.date(2026, 1, 2), 0.001), (datetime.date(2026, 1, 30), 0.002)]
    )

    with pytest.raises(ValueError, match="2026-01-31"):
        excess_returns.rekey_to_calendar_month_end(frame, date_col="date")


def test_rekey_preserves_row_count_and_values_for_a_clean_series():
    frame = _monthly([(m, 0.01 * (i + 1)) for i, m in enumerate(_Q1_1990_TRADING_MONTHS)])

    got = excess_returns.rekey_to_calendar_month_end(frame, date_col="date")

    assert got.height == 3
    assert got["rf"].to_list() == [0.01, 0.02, 0.03]
    assert got["date"].to_list() == _Q1_1990_CALENDAR_MONTHS


# ---------------------------------------------------------------------------
# quarterly_risk_free -- ABSOLUTE ANCHOR
# ---------------------------------------------------------------------------


def test_three_months_at_one_percent_compound_to_exactly_0_030301():
    """ABSOLUTE ANCHOR (CLAUDE.md): 1.01**3 - 1 = 0.030301 EXACTLY.

    Discriminates the wrong implementations that matter here:
      - 3-month arithmetic SUM  -> 0.030000 (fails; only 1bp away, the
        'looks right' magnitude this project exists to guard against)
      - percent/decimal units bug -> ~3.03 or ~3.03e-4 (fails)
      - annualized instead of quarterly -> ~0.1268 (fails)
      - correct log-space compounding -> 0.030301 (passes)
    """
    rf = _monthly([(m, 0.01) for m in _Q1_1990_CALENDAR_MONTHS])

    got = excess_returns.quarterly_risk_free(rf, quarters=[_Q1_1990], rekey=True)

    assert got["rf_q"][0] == pytest.approx(0.030301, rel=1e-12), (
        f"three months at rf=1% compounded to {got['rf_q'][0]!r}, not 0.030301 "
        "-- 0.030000 means an arithmetic 3-month sum, the exact method that "
        "was explicitly ruled out"
    )


def test_quarterly_risk_free_rekeys_trading_day_months():
    """Without re-keying, two of these three months miss the calendar
    lookup entirely and the quarter compounds only 1 month -- the
    measured Canadian defect, reproduced in miniature."""
    rf = _monthly([(m, 0.01) for m in _Q1_1990_TRADING_MONTHS])

    got = excess_returns.quarterly_risk_free(rf, quarters=[_Q1_1990], rekey=True)

    assert got["n_months"][0] == 3, (
        f"re-keyed trading-day months gave n_months={got['n_months'][0]}, not 3"
    )
    assert got["rf_q"][0] == pytest.approx(0.030301, rel=1e-12)


def test_quarterly_risk_free_without_rekey_leaves_quarter_partial():
    """The defect this module exists to prevent, asserted directly: with
    rekey=False the same input silently compounds ONE month and returns
    a plausible 0.01, with nothing but n_months to reveal it."""
    rf = _monthly([(m, 0.01) for m in _Q1_1990_TRADING_MONTHS])

    got = excess_returns.quarterly_risk_free(rf, quarters=[_Q1_1990], rekey=False)

    assert got["n_months"][0] == 1, (
        "expected the un-re-keyed series to match only the one genuinely-"
        f"calendar month, got n_months={got['n_months'][0]}"
    )
    assert got["rf_q"][0] == pytest.approx(0.01, rel=1e-12), (
        "and the value it returns is entirely plausible -- which is why "
        "assert_full_quarters, not eyeballing, is the guard"
    )


def test_quarterly_risk_free_drops_rf_source_column():
    """load_canada_rf_chass returns a third rf_source column marking
    derived months. It must not reach the compounder (which expects a
    single value column); the driver records it separately for the
    report footnote."""
    rf = pl.DataFrame(
        {
            "date": _Q1_1990_CALENDAR_MONTHS,
            "rf": [0.01, 0.01, 0.01],
            "rf_source": ["chass", "chass", "derived_from_ind1"],
        }
    )

    got = excess_returns.quarterly_risk_free(rf, quarters=[_Q1_1990], rekey=True)

    assert "rf_source" not in got.columns
    assert got["rf_q"][0] == pytest.approx(0.030301, rel=1e-12)


# ---------------------------------------------------------------------------
# to_excess -- ABSOLUTE ANCHORS
# ---------------------------------------------------------------------------


def test_to_excess_subtracts_exactly():
    """ABSOLUTE ANCHOR: 0.05 - 0.01 = 0.04 exactly. Catches a sign flip
    (0.06) and a geometric de-rating port/(1+rf) (0.039604), which is a
    different and defensible convention but NOT the one this project
    uses -- src/portfolio/legs.py subtracts arithmetically."""
    raw = _quarterly([(_Q1_1990, 0.05)])
    rf = pl.DataFrame({"quarter": [_Q1_1990], "rf_q": [0.01]})

    got = excess_returns.to_excess(raw, rf, value_col="ret")

    assert got["ret_excess"][0] == pytest.approx(0.04, rel=1e-12), (
        f"0.05 - 0.01 gave {got['ret_excess'][0]!r}, not 0.04 -- 0.06 is a "
        "sign flip, 0.039604 a geometric de-rating"
    )


def test_excess_of_the_risk_free_series_against_itself_is_exactly_zero():
    """ABSOLUTE ANCHOR #2, the rf analogue of market-on-market beta = 1:
    feed the compounded rf series AS the portfolio and every excess
    quarter must be 0.0. Catches misalignment (a shifted join leaves
    non-zero residuals) and double-subtraction."""
    quarters = [
        datetime.date(1990, 3, 31),
        datetime.date(1990, 6, 30),
        datetime.date(1990, 9, 30),
    ]
    rows = []
    v = 0.003
    for q in quarters:
        for m in _months_of(q):
            rows.append((m, v))
            v += 0.0004
    rf_monthly = _monthly(rows)

    rf_q = excess_returns.quarterly_risk_free(rf_monthly, quarters=quarters, rekey=True)
    as_portfolio = rf_q.select(pl.col("quarter"), pl.col("rf_q").alias("ret"))

    got = excess_returns.to_excess(as_portfolio, rf_q, value_col="ret")

    for i, v in enumerate(got["ret_excess"].to_list()):
        assert v == pytest.approx(0.0, abs=1e-15), (
            f"quarter {got['quarter'][i]} gave excess {v!r}, not 0.0 -- the rf "
            "series minus itself is non-zero, so the join is misaligned"
        )


def _months_of(quarter_end: datetime.date) -> list[datetime.date]:
    """Independent reconstruction of a quarter's three calendar month-ends
    -- not imported from the module under test."""
    year, end_month = quarter_end.year, quarter_end.month
    out = []
    for offset in (2, 1, 0):
        m, y = end_month - offset, year
        if m <= 0:
            m += 12
            y -= 1
        nxt = datetime.date(y + 1, 1, 1) if m == 12 else datetime.date(y, m + 1, 1)
        out.append(nxt - datetime.timedelta(days=1))
    return out


def test_to_excess_inner_joins_so_an_unmatched_quarter_drops():
    """A quarter present in the portfolio but absent from rf must not
    silently become excess == raw (which is what a left join with a
    null-fill would produce, understating rf to zero)."""
    raw = _quarterly([(_Q1_1990, 0.05), (datetime.date(1990, 6, 30), 0.03)])
    rf = pl.DataFrame({"quarter": [_Q1_1990], "rf_q": [0.01]})

    got = excess_returns.to_excess(raw, rf, value_col="ret")

    assert got.height == 1, (
        f"expected the unmatched quarter to drop, got {got.height} rows -- a "
        "left join here would report a raw return as an excess one"
    )
    assert got["quarter"][0] == _Q1_1990
