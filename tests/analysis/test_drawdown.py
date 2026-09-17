"""Tests for src/analysis/drawdown.py -- docs/06_handoff_section6.md
item B (max drawdown, worst quarter, recovery) for report Sec 6.3.

WRITTEN BEFORE THE MODULE (CLAUDE.md workflow).

Every anchor here is an exact hand-derived number, not a ratio or a
correlation. The two most likely wrong implementations of a max
drawdown both produce plausible negative numbers:

  - summing returns instead of compounding wealth
  - taking min(per-period return) instead of the peak-to-trough decline

so the tests pin values that separate all three.
"""

import datetime

import polars as pl
import pytest

from src.analysis import drawdown


def _returns(rows: list[tuple[datetime.date, float]], value_col: str = "ret") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "quarter": [r[0] for r in rows],
            value_col: [r[1] for r in rows],
        }
    )


def _quarters(n: int, start_year: int = 1990) -> list[datetime.date]:
    """n consecutive calendar quarter-ends from start_year Q1."""
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


# ---------------------------------------------------------------------------
# ABSOLUTE ANCHORS
# ---------------------------------------------------------------------------


def test_minus_fifty_percent_then_flat_gives_exactly_minus_zero_point_five():
    """ABSOLUTE ANCHOR (CLAUDE.md): a single -50% quarter followed by
    flat quarters is a max drawdown of exactly -0.50.

    Catches: a sign-convention flip (+0.50), a percentage/decimal error
    (-50.0), and a max-where-min-is-meant (0.0).
    """
    qs = _quarters(3)
    frame = _returns([(qs[0], -0.50), (qs[1], 0.0), (qs[2], 0.0)])

    got = drawdown.drawdown_stats(frame, value_col="ret", label="synthetic")

    assert got["max_drawdown"] == pytest.approx(-0.50, rel=1e-12), (
        f"max_drawdown={got['max_drawdown']!r}, not -0.50"
    )


def test_two_consecutive_minus_fifty_percents_give_minus_zero_point_seven_five():
    """ABSOLUTE ANCHOR #2, and the test that separates the three
    candidate implementations outright:

      compounded wealth (correct) -> -0.75
      sum of returns              -> -1.00
      min single-period return    -> -0.50

    All three are plausible-looking negative numbers; only the exact
    value distinguishes them.
    """
    qs = _quarters(2)
    frame = _returns([(qs[0], -0.50), (qs[1], -0.50)])

    got = drawdown.drawdown_stats(frame, value_col="ret", label="synthetic")

    assert got["max_drawdown"] == pytest.approx(-0.75, rel=1e-12), (
        f"max_drawdown={got['max_drawdown']!r} -- -1.00 means returns were "
        "summed, -0.50 means the per-period minimum was taken instead of the "
        "peak-to-trough wealth decline"
    )


def test_wealth_path_starts_at_seed_times_first_return():
    """ABSOLUTE ANCHOR #3: the wealth index is seeded at WEALTH_SEED =
    1.0, anchoring the level outside the dataset entirely. A scale bug
    shows up here immediately; a ratio-only check could not see it."""
    qs = _quarters(2)
    frame = _returns([(qs[0], 0.10), (qs[1], 0.10)])

    # Pin the constant itself against a LITERAL first. Comparing the
    # module's output against the module's own constant is
    # self-referential: changing WEALTH_SEED to 100.0 would leave the
    # comparison below passing while every wealth level was 100x wrong
    # (gate-verifier finding 2026-09-15, same idiom as
    # test_downturn_percentile_is_the_pre_registered_ten_percent).
    assert drawdown.WEALTH_SEED == 1.0, (
        f"WEALTH_SEED is {drawdown.WEALTH_SEED!r}, not 1.0 -- the wealth index "
        "is no longer 'value of $1 invested at the start'"
    )

    path = drawdown.drawdown_path(frame, value_col="ret")

    assert path["wealth"][0] == pytest.approx(1.10, rel=1e-12)
    assert path["wealth"][1] == pytest.approx(drawdown.WEALTH_SEED * 1.21, rel=1e-12)
    assert path["running_peak"][0] == pytest.approx(1.10, rel=1e-12)


def test_monotonically_rising_series_has_zero_drawdown():
    """An off-by-one in the running peak (comparing against the NEXT
    period's wealth) would manufacture a small negative drawdown on a
    series that never falls."""
    qs = _quarters(4)
    frame = _returns([(q, 0.05) for q in qs])

    got = drawdown.drawdown_stats(frame, value_col="ret", label="synthetic")

    assert got["max_drawdown"] == pytest.approx(0.0, abs=1e-15), (
        f"a strictly rising series reported max_drawdown={got['max_drawdown']!r}"
    )
    assert got["recovered"] is True, "a series that never fell is trivially recovered"


# ---------------------------------------------------------------------------
# Recovery -- counted from the TROUGH, and the never-recovers convention
# ---------------------------------------------------------------------------


def test_recovery_is_counted_from_trough_not_from_peak():
    """wealth: 1.0 -> 0.5 -> 0.75 -> 1.125. The pre-drawdown peak (1.0)
    is regained at index 2, which is TWO quarters after the trough at
    index 0. Counting from the peak instead would give a different
    number on any series where the decline spans more than one quarter,
    so this pins the convention."""
    qs = _quarters(3)
    frame = _returns([(qs[0], -0.50), (qs[1], 0.50), (qs[2], 0.50)])

    got = drawdown.drawdown_stats(frame, value_col="ret", label="synthetic")

    assert got["recovered"] is True
    assert got["quarters_to_recover"] == 2, (
        f"quarters_to_recover={got['quarters_to_recover']!r}, not 2"
    )
    assert got["trough_quarter"] == qs[0]
    assert got["recovery_quarter"] == qs[2]


def test_never_recovered_reports_none_and_false_not_a_sentinel():
    """The censoring convention. A sentinel integer (-1, 0, 999) in a
    'quarters to recover' column is precisely a wrong number that looks
    right; src/portfolio/diagnostics.py sets the precedent of returning
    None rather than 0 so 'not computed' cannot be misread as
    'computed as zero'.

    quarters_since_trough is populated instead, so the exhibit can say
    'not recovered (N quarters and counting)' rather than a number that
    reads as a completed recovery.
    """
    qs = _quarters(3)
    frame = _returns([(qs[0], -0.50), (qs[1], 0.01), (qs[2], 0.01)])

    got = drawdown.drawdown_stats(frame, value_col="ret", label="synthetic")

    assert got["recovered"] is False
    assert got["quarters_to_recover"] is drawdown.RECOVERY_NOT_ACHIEVED, (
        f"quarters_to_recover={got['quarters_to_recover']!r} -- must be None, "
        "never a sentinel integer"
    )
    assert got["quarters_since_trough"] == 2, (
        "the censored count must still be reported, under its own name"
    )
    assert got["recovery_quarter"] is None


def test_recovery_requires_regaining_the_peak_not_merely_rising():
    """A series that rises after the trough but stops short of the old
    peak has NOT recovered. An implementation testing 'wealth rose' or
    'drawdown shrank' rather than 'wealth >= running peak' would report
    a false recovery."""
    qs = _quarters(3)
    frame = _returns([(qs[0], -0.50), (qs[1], 0.20), (qs[2], 0.20)])

    got = drawdown.drawdown_stats(frame, value_col="ret", label="synthetic")

    assert got["recovered"] is False, (
        "wealth reached 0.72, short of the 1.0 peak -- not a recovery"
    )


# ---------------------------------------------------------------------------
# Worst single period, and path/stats consistency
# ---------------------------------------------------------------------------


def test_worst_period_is_the_single_minimum_return_not_the_drawdown():
    """The two are different questions and the report states both.
    Here max drawdown is -0.75 (compounded) while the worst single
    quarter is -0.50."""
    qs = _quarters(2)
    frame = _returns([(qs[0], -0.50), (qs[1], -0.50)])

    got = drawdown.drawdown_stats(frame, value_col="ret", label="synthetic")

    assert got["worst_period_ret"] == pytest.approx(-0.50, rel=1e-12)
    assert got["max_drawdown"] == pytest.approx(-0.75, rel=1e-12)
    assert got["worst_period_ret"] != got["max_drawdown"], (
        "worst single period and max drawdown must not be conflated"
    )


def test_stats_are_a_reduction_over_the_path():
    """The exhibit chart is drawn from drawdown_path and the headline
    number comes from drawdown_stats. If the two were computed
    separately they could disagree; this asserts they cannot."""
    qs = _quarters(6)
    frame = _returns(
        [
            (qs[0], 0.10),
            (qs[1], -0.30),
            (qs[2], -0.15),
            (qs[3], 0.05),
            (qs[4], 0.25),
            (qs[5], -0.08),
        ]
    )

    path = drawdown.drawdown_path(frame, value_col="ret")
    stats = drawdown.drawdown_stats(frame, value_col="ret", label="synthetic")

    assert path["drawdown"].min() == stats["max_drawdown"], (
        f"path minimum {path['drawdown'].min()!r} != stats max_drawdown "
        f"{stats['max_drawdown']!r} -- chart and headline can disagree"
    )
    assert stats["n_periods"] == path.height == 6


def test_peak_before_first_observation_is_flagged_not_hidden():
    """A series that falls from its very first quarter has its
    pre-drawdown peak at the seed, before any observation. Reporting a
    peak_quarter of None with peak_is_seed=True is honest; silently
    reporting the first quarter as the peak would misdate the
    drawdown."""
    qs = _quarters(2)
    frame = _returns([(qs[0], -0.20), (qs[1], -0.10)])

    got = drawdown.drawdown_stats(frame, value_col="ret", label="synthetic")

    assert got["peak_is_seed"] is True
    assert got["peak_quarter"] is None
    assert got["max_drawdown"] == pytest.approx(0.8 * 0.9 - 1.0, rel=1e-12)


def test_label_is_carried_through():
    qs = _quarters(2)
    frame = _returns([(qs[0], 0.01), (qs[1], 0.01)])

    got = drawdown.drawdown_stats(frame, value_col="ret", label="us_longonly_total")

    assert got["label"] == "us_longonly_total"


def test_empty_series_raises_rather_than_reporting_zero():
    """A zero max drawdown on an empty series would read as 'never fell'
    rather than 'no data'."""
    frame = _returns([])

    with pytest.raises(ValueError):
        drawdown.drawdown_stats(frame, value_col="ret", label="empty")
