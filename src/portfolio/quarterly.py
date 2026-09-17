"""Quarter-end date arithmetic for the long-only backtester
(docs/04_handoff_lowbeta_longonly.md Sec 4.1).

THIS FILE EXISTS BECAUSE OF A REAL PAST MISTAKE (handoff Sec 7): a
throwaway diagnostic script used gate4_bab._next_month_end_of (an
IDENTITY function on a month-end) where rebalance.run_backtest uses
rebalance._next_month_end (which ADVANCES one calendar month) -- joining
each formation date's weights to the formation month's OWN returns
instead of the held month's, producing a completely spurious finding
that survived into a committed document before being caught.

next_quarter_end() below is the quarterly analogue of
rebalance._next_month_end -- it ADVANCES, it is never an identity. It
gets its own module (not bolted onto rebalance.py, which is monthly-only
and BAB-shaped) specifically so a reader cannot mistake a quarterly
formation/holding join for a monthly one, or vice versa, the way the two
near-identically-named month-end helpers were mistaken for each other.
"""

import datetime


def next_quarter_end(d: datetime.date) -> datetime.date:
    """The calendar quarter-end immediately following d (d is itself
    assumed to be a quarter-end -- Mar/Jun/Sep/Dec 31, or 30 for Jun/Sep).
    A held-quarter return lookup must use this, never d itself -- exactly
    the same discipline rebalance._next_month_end enforces for the
    monthly BAB loop, extended to quarters.

    Always advances: next_quarter_end(d) != d for every d, by
    construction (see this module's own test suite, which asserts this
    directly rather than only checking specific transitions -- the shape
    of assertion that would have caught the Sec 7 mistake before it
    happened, not after)."""
    if d.month not in (3, 6, 9, 12):
        raise ValueError(
            f"next_quarter_end expects a calendar quarter-end month "
            f"(3, 6, 9, or 12), got {d!r} (month={d.month})"
        )

    next_month = d.month + 3
    next_year = d.year
    if next_month > 12:
        next_month -= 12
        next_year += 1

    if next_month == 12:
        return datetime.date(next_year, 12, 31)
    if next_month in (6, 9):
        # Jun/Sep have 30 days; day+1's month-first minus one day handles
        # this without a hardcoded day-count table.
        following_month_first = datetime.date(next_year, next_month + 1, 1)
        return following_month_first - datetime.timedelta(days=1)
    # next_month == 3
    following_month_first = datetime.date(next_year, 4, 1)
    return following_month_first - datetime.timedelta(days=1)
