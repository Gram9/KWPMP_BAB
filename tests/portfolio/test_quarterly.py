"""Tests for src/portfolio/quarterly.py -- see that module's docstring
for why this exists (handoff Sec 7's _next_month_end_of vs
_next_month_end confusion, applied to quarters)."""

import datetime

import pytest

from src.portfolio.quarterly import next_quarter_end

# ---------------------------------------------------------------------------
# Round-trip over all 4 quarter boundaries, including the Jun/Sep 30-day
# cases and the Dec->Mar year rollover.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start,expected",
    [
        (datetime.date(2015, 3, 31), datetime.date(2015, 6, 30)),
        (datetime.date(2015, 6, 30), datetime.date(2015, 9, 30)),
        (datetime.date(2015, 9, 30), datetime.date(2015, 12, 31)),
        (datetime.date(2015, 12, 31), datetime.date(2016, 3, 31)),
    ],
)
def test_next_quarter_end_advances_correctly(start, expected):
    assert next_quarter_end(start) == expected


def test_next_quarter_end_full_year_round_trip():
    """Chain all four calls starting from 2015-03-31 and land back on
    2016-03-31 -- one calendar year later, exactly 4 quarters advanced."""
    d = datetime.date(2015, 3, 31)
    for _ in range(4):
        d = next_quarter_end(d)
    assert d == datetime.date(2016, 3, 31)


# ---------------------------------------------------------------------------
# THE mistake-shaped test: next_quarter_end must NEVER be an identity.
# This is the direct analogue of the check that would have caught the
# gate4_bab._next_month_end_of / rebalance._next_month_end confusion
# (handoff Sec 7) before it produced a spurious finding.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d",
    [
        datetime.date(2000, 3, 31),
        datetime.date(2000, 6, 30),
        datetime.date(2000, 9, 30),
        datetime.date(2000, 12, 31),
        datetime.date(1986, 12, 31),
        datetime.date(2025, 12, 31),
    ],
)
def test_next_quarter_end_is_never_an_identity(d):
    result = next_quarter_end(d)
    assert result != d, (
        f"next_quarter_end({d}) returned {result} == input -- this "
        "function must ALWAYS advance one quarter. An identity result "
        "here is exactly the class of bug handoff Sec 7 documents: a "
        "formation-date-to-held-quarter join using an identity function "
        "silently joins weights to the FORMATION quarter's own return "
        "instead of the HELD quarter's."
    )
    assert result > d, f"next_quarter_end({d}) returned {result}, which is not later than input"


def test_next_quarter_end_rejects_non_quarter_end_month():
    """d must itself be a quarter-end month (3/6/9/12) -- a caller
    passing an arbitrary month-end (e.g. formation logic bug upstream)
    should fail loudly, not silently round to the nearest quarter."""
    with pytest.raises(ValueError):
        next_quarter_end(datetime.date(2015, 1, 31))
    with pytest.raises(ValueError):
        next_quarter_end(datetime.date(2015, 4, 30))


def test_next_quarter_end_handles_30_vs_31_day_months():
    """Jun/Sep have 30 days, Mar/Dec have 31 -- confirm the day count is
    correct in both directions (30->31 and 31->30), not just that SOME
    date in the next quarter is returned."""
    assert next_quarter_end(datetime.date(2015, 3, 31)).day == 30  # -> Jun
    assert next_quarter_end(datetime.date(2015, 6, 30)).day == 30  # -> Sep
    assert next_quarter_end(datetime.date(2015, 9, 30)).day == 31  # -> Dec
    assert next_quarter_end(datetime.date(2015, 12, 31)).day == 31  # -> Mar
