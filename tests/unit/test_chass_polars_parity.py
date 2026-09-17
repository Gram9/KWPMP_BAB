"""
Parity tests for the pandas -> polars port of chass_loader.py /
chass_universe.py (Phase A of the Gate 1 plan). Each test feeds the SAME
real input through the old pandas reference path
(src/data/chass_loader_pandas_reference.py,
src/data/chass_universe_pandas_reference.py -- untouched copies of the
pre-port code) and the new polars path (src/data/chass_loader.py,
src/data/chass_universe.py, post-port), and asserts identical output.

Passing the pre-existing pandas-flavored test suite is necessary but not
sufficient for this port: those tests encode expected VALUES (row counts,
named ground-truth cases), not "does polars agree with pandas on every
row." A subtly wrong port (a groupby tie-break, a null-handling difference,
a sort-stability difference) can still produce a plausible-looking number
that happens to match a pinned count by coincidence, or on the specific
cases the existing tests sample, while diverging elsewhere. These tests
close that gap directly, per CLAUDE.md: "a wrong number that looks right
is the worst possible outcome."

Comparison strategy: convert the polars result to pandas
(`pl.DataFrame.to_pandas()`), sort both frames by their identity key plus
date (row order is not semantically meaningful for these outputs except
where explicitly noted), reset index, and compare with
`pandas.testing.assert_frame_equal` (dtype-loose where the two libraries'
native dtypes differ trivially, e.g. polars Int64 vs pandas int64/Int64).
"""

import datetime
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from src.data.chass_loader_pandas_reference import (
    SECURITY_ID_COLUMNS,
)
from src.data.chass_loader_pandas_reference import (
    apply_chass_fund_mlp_reit_exclusion as pd_apply_exclusion,
)
from src.data.chass_loader_pandas_reference import (
    chass_daily_trade_status as pd_trade_status,
)
from src.data.chass_loader_pandas_reference import (
    chass_fund_mlp_reit_classification as pd_classification,
)
from src.data.chass_loader_pandas_reference import (
    filter_chass_daily_traded as pd_filter_traded,
)
from src.data.chass_loader_pandas_reference import (
    load_daily as pd_load_daily,
)
from src.data.chass_loader_pandas_reference import (
    load_monthly as pd_load_monthly,
)
from src.data.chass_universe_pandas_reference import (
    _terminal_row_flags as pd_terminal_row_flags,
)
from src.data.chass_universe_pandas_reference import (
    market_cap_at as pd_market_cap_at,
)
from src.data.chass_universe_pandas_reference import (
    universe_at as pd_universe_at,
)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "CHASS_Data" / "parquet"


def _parquet_available() -> bool:
    return (RAW_DATA_DIR / "monthly.parquet").exists() and any(
        RAW_DATA_DIR.glob("daily/year=*/part.parquet")
    )


pytestmark = pytest.mark.skipif(
    not _parquet_available(),
    reason="data/raw/CHASS_Data/parquet/ not present -- run "
    "src/data/convert_chass_csv.py first",
)

_KEY = SECURITY_ID_COLUMNS + ["trdate-Trade Date"]


def _sorted_reset(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return df.sort_values(by).reset_index(drop=True)


def _assert_frames_match(pandas_result: pd.DataFrame, polars_result, by: list[str]) -> None:
    """polars_result: a pl.DataFrame. Converts to pandas, aligns column
    order/dtypes loosely, sorts both by `by`, and compares."""
    from_polars = polars_result.to_pandas()
    common_cols = list(pandas_result.columns)
    assert set(from_polars.columns) == set(common_cols), (
        f"column mismatch: pandas-only={set(pandas_result.columns) - set(from_polars.columns)}, "
        f"polars-only={set(from_polars.columns) - set(pandas_result.columns)}"
    )
    from_polars = from_polars[common_cols]

    left = _sorted_reset(pandas_result, by)
    right = _sorted_reset(from_polars, by)
    pdt.assert_frame_equal(left, right, check_dtype=False, check_like=False)


# ---------------------------------------------------------------------------
# chass_loader.py parity
# ---------------------------------------------------------------------------


def test_load_monthly_parity():
    from src.data.chass_loader import load_monthly as pl_load_monthly

    pandas_result = pd_load_monthly()
    polars_result = pl_load_monthly()
    assert len(pandas_result) == len(polars_result) == 807_453
    _assert_frames_match(pandas_result, polars_result, by=_KEY)


def test_load_daily_full_parity():
    from src.data.chass_loader import load_daily as pl_load_daily

    pandas_result = pd_load_daily(years=[1980])
    polars_result = pl_load_daily(years=[1980])
    _assert_frames_match(pandas_result, polars_result, by=_KEY)


def test_load_daily_dividend_fanout_collapse_parity():
    """BCC dividend-fanout collapse (module docstring's named case) must
    produce the exact same single surviving row under both paths."""
    from src.data.chass_loader import load_daily as pl_load_daily

    pandas_result = pd_load_daily(years=[1980])
    polars_result = pl_load_daily(years=[1980]).to_pandas()

    pd_row = pandas_result[
        (pandas_result["cusip-CUSIP"] == "087239109")
        & (pandas_result["trdate-Trade Date"] == pd.Timestamp("1980-02-25"))
    ]
    pl_row = polars_result[
        (polars_result["cusip-CUSIP"] == "087239109")
        & (polars_result["trdate-Trade Date"] == pd.Timestamp("1980-02-25"))
    ]
    assert len(pd_row) == len(pl_row) == 1


def test_load_daily_twe_cusip_collision_parity():
    """TWE usage 0/1 (module docstring's named CUSIP-collision case) must
    both survive under both paths, as two distinct rows."""
    from src.data.chass_loader import load_daily as pl_load_daily

    pandas_result = pd_load_daily(years=[1983])
    polars_result = pl_load_daily(years=[1983]).to_pandas()

    for df, label in [(pandas_result, "pandas"), (polars_result, "polars")]:
        rows = df[
            (df["symbol-Ticker"] == "TWE") & (df["trdate-Trade Date"] == "1983-05-26")
        ]
        assert len(rows) == 2, f"{label} lost the TWE CUSIP-collision rows"
        assert set(rows["usage-Usage Number"]) == {0, 1}


def test_exclusion_classification_parity():
    from src.data.chass_loader import (
        chass_fund_mlp_reit_classification as pl_classification,
    )

    monthly = pd_load_monthly()
    pandas_result = pd_classification(monthly)

    from src.data.chass_loader import load_monthly as pl_load_monthly

    polars_monthly = pl_load_monthly()
    polars_result = pl_classification(polars_monthly)

    _assert_frames_match(
        pandas_result, polars_result, by=SECURITY_ID_COLUMNS
    )


def test_exclusion_applied_row_count_parity():
    from src.data.chass_loader import (
        apply_chass_fund_mlp_reit_exclusion as pl_apply_exclusion,
    )
    from src.data.chass_loader import (
        load_monthly as pl_load_monthly,
    )

    pandas_result = pd_apply_exclusion(pd_load_monthly())
    polars_result = pl_apply_exclusion(pl_load_monthly())
    assert len(pandas_result) == len(polars_result) == 598_251


def test_daily_trade_status_parity():
    from src.data.chass_loader import (
        chass_daily_trade_status as pl_trade_status,
    )
    from src.data.chass_loader import (
        load_daily as pl_load_daily,
    )

    pandas_daily = pd_load_daily(years=[1980])
    polars_daily = pl_load_daily(years=[1980])

    pandas_status = pd_trade_status(pandas_daily)
    polars_status = pl_trade_status(polars_daily)

    pandas_counts = pandas_status.value_counts().to_dict()
    polars_counts = (
        polars_status.to_pandas().value_counts().to_dict()
        if not isinstance(polars_status, pd.Series)
        else polars_status.value_counts().to_dict()
    )
    assert pandas_counts == polars_counts


def test_filter_chass_daily_traded_parity():
    from src.data.chass_loader import (
        filter_chass_daily_traded as pl_filter_traded,
    )
    from src.data.chass_loader import (
        load_daily as pl_load_daily,
    )

    pandas_daily = pd_load_daily(years=[1980])
    polars_daily = pl_load_daily(years=[1980])

    pandas_result = pd_filter_traded(pandas_daily)
    polars_result = pl_filter_traded(polars_daily)
    _assert_frames_match(pandas_result, polars_result, by=_KEY)


# ---------------------------------------------------------------------------
# chass_universe.py parity
# ---------------------------------------------------------------------------


def test_universe_at_parity_real_month():
    from src.data.chass_loader import load_monthly as pl_load_monthly
    from src.data.chass_universe import universe_at as pl_universe_at

    month_end = datetime.date(2015, 6, 30)

    pandas_result = pd_universe_at(pd_load_monthly(), month_end=month_end)
    polars_result = pl_universe_at(pl_load_monthly(), month_end=month_end)
    _assert_frames_match(pandas_result, polars_result, by=SECURITY_ID_COLUMNS)


def test_universe_at_parity_morgan_hydrocarbons_delisting_month():
    """Named ground-truth case: MHI present at 1996-10-31, absent at
    1996-11-30, under both paths."""
    from src.data.chass_loader import load_monthly as pl_load_monthly
    from src.data.chass_universe import universe_at as pl_universe_at

    pandas_monthly = pd_load_monthly()
    polars_monthly = pl_load_monthly()

    for month_end, expect_present in [
        (datetime.date(1996, 10, 31), True),
        (datetime.date(1996, 11, 30), False),
    ]:
        pandas_result = pd_universe_at(pandas_monthly, month_end=month_end)
        polars_result = pl_universe_at(polars_monthly, month_end=month_end).to_pandas()

        pandas_present = (
            (pandas_result["symbol-Ticker"] == "MHI")
            & (pandas_result["usage-Usage Number"] == 0)
        ).any()
        polars_present = (
            (polars_result["symbol-Ticker"] == "MHI")
            & (polars_result["usage-Usage Number"] == 0)
        ).any()
        assert pandas_present == polars_present == expect_present


def test_terminal_row_flags_parity_excluding_known_null_symbol_divergence():
    """NOT a bit-for-bit parity test -- this is the one function where
    pandas and polars are EXPECTED to disagree, and that disagreement is
    a real bug fix, not a porting error.

    National Bank of Canada (cusip 633067103, symbol-Ticker is null in
    the source parquet, 552 monthly rows 1980-2025, a genuine currently-
    listed major bank -- confirmed present and correctly handled by both
    universe_at() and market_cap_at() under both pandas and polars) was
    silently dropped by the pandas reference's _terminal_row_flags():
    pandas's groupby() defaults to dropna=True, which drops the entire
    null-symbol group rather than treating it as its own group. Polars's
    group_by() has no such default -- it keeps the null-key group. The
    polars port is deliberately NOT made to match this pandas behavior;
    the pandas behavior is the bug (see docs/01_data_notes.md for the
    full writeup). Decided with the user 2026-09-05: fix in the port,
    don't preserve the gap.

    Pinned counts: pandas 7,536 groups (drops the null-symbol group),
    polars 7,537 groups (includes it). If this ever changes, it means
    either the source data changed or one of the two group_by
    implementations' null-handling defaults changed -- investigate,
    don't just update the pinned numbers.
    """
    from src.data.chass_loader import load_monthly as pl_load_monthly
    from src.data.chass_universe import _terminal_row_flags as pl_terminal_row_flags

    pandas_monthly = pd_load_monthly()
    polars_monthly = pl_load_monthly()

    pandas_result = pd_terminal_row_flags(pandas_monthly)
    polars_result = pl_terminal_row_flags(polars_monthly).to_pandas()

    assert len(pandas_result) == 7_536
    assert len(polars_result) == 7_537

    nbc_in_pandas = pandas_result["symbol-Ticker"].isna().any()
    nbc_in_polars = polars_result["symbol-Ticker"].isna().any()
    assert not nbc_in_pandas, (
        "pandas reference unexpectedly includes the null-symbol group -- "
        "the known bug this test documents may have been fixed upstream; "
        "re-investigate this test's premise."
    )
    assert nbc_in_polars, (
        "polars port unexpectedly dropped the null-symbol (National Bank "
        "of Canada) group -- this is the specific regression this test "
        "exists to catch."
    )

    # Every OTHER group (i.e. every non-null symbol) must still match
    # row-for-row between the two paths -- the divergence is isolated to
    # exactly the one null-symbol group, not a wider correctness gap.
    # Both sides are already plain pandas frames here (polars_result was
    # converted via .to_pandas() above), so this compares directly rather
    # than going through _assert_frames_match (which expects its second
    # argument to still be a pl.DataFrame it converts itself).
    pandas_non_null = _sorted_reset(
        pandas_result.dropna(subset=["symbol-Ticker"]), SECURITY_ID_COLUMNS
    )
    polars_non_null = _sorted_reset(
        polars_result.dropna(subset=["symbol-Ticker"]), SECURITY_ID_COLUMNS
    )
    pdt.assert_frame_equal(pandas_non_null, polars_non_null, check_dtype=False)


def test_market_cap_at_parity_real_month():
    from src.data.chass_loader import load_daily as pl_load_daily
    from src.data.chass_loader import load_monthly as pl_load_monthly
    from src.data.chass_universe import market_cap_at as pl_market_cap_at

    month_end = datetime.date(2015, 6, 30)

    pandas_monthly = pd_load_monthly()
    pandas_daily = pd_load_daily(years=[2015])
    pandas_mkt_cap, pandas_coverage = pd_market_cap_at(
        pandas_monthly, pandas_daily, month_end=month_end
    )

    polars_monthly = pl_load_monthly()
    polars_daily = pl_load_daily(years=[2015])
    polars_mkt_cap, polars_coverage = pl_market_cap_at(
        polars_monthly, polars_daily, month_end=month_end
    )

    _assert_frames_match(pandas_mkt_cap, polars_mkt_cap, by=SECURITY_ID_COLUMNS)
    _assert_frames_match(pandas_coverage, polars_coverage, by=SECURITY_ID_COLUMNS)


def test_market_cap_at_parity_pembina_hand_computed_value():
    """Named ground-truth case: PPL usage 2 mkt_cap must match the same
    hand-computed value (40.58 * 3,404,490.0 * 100) under both paths."""
    from src.data.chass_loader import load_daily as pl_load_daily
    from src.data.chass_loader import load_monthly as pl_load_monthly
    from src.data.chass_universe import market_cap_at as pl_market_cap_at

    month_end = datetime.date(2015, 6, 30)
    expected = 40.58 * 3_404_490.0 * 100

    pandas_mkt_cap, _ = pd_market_cap_at(
        pd_load_monthly(), pd_load_daily(years=[2015]), month_end=month_end
    )
    polars_mkt_cap, _ = pl_market_cap_at(
        pl_load_monthly(), pl_load_daily(years=[2015]), month_end=month_end
    )
    polars_mkt_cap = polars_mkt_cap.to_pandas()

    pandas_row = pandas_mkt_cap[
        (pandas_mkt_cap["symbol-Ticker"] == "PPL") & (pandas_mkt_cap["usage-Usage Number"] == 2)
    ]
    polars_row = polars_mkt_cap[
        (polars_mkt_cap["symbol-Ticker"] == "PPL") & (polars_mkt_cap["usage-Usage Number"] == 2)
    ]
    assert pandas_row["mkt_cap"].iloc[0] == pytest.approx(expected, rel=1e-6)
    assert polars_row["mkt_cap"].iloc[0] == pytest.approx(expected, rel=1e-6)
