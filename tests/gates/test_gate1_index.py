"""
Tests for src/gates/gate1_index.py -- Gate 1 (docs/02_validation_gates.md):
rebuild the CRSP value-weighted market return from our own universe and
returns, compare to vwretd. Should match to rounding.

Per the Gate 1 implementation plan, Phase C: this is the credibility gate
for the whole US leg. src/data/gate_adapters.py's us_gate1_panel() already
gives every column this needs (lagged mkt_cap for weights, ret for that
day) -- Gate 1 itself is index_ret[date] = sum(mkt_cap_i * ret_i) /
sum(mkt_cap_i) across all names present that date, compared against
vwretd from the same underlying panel.

First real run targets 2015 only (confirmed with the user 2026-09-07) --
a bounded window to get a fast pass/fail signal before committing to the
full 1965-present history.
"""

import datetime

import polars as pl
import pytest

from src.data.chass_loader import MONTHLY_PATH
from src.data.universe_panel import RAW_DATA_DIR
from src.gates import gate1_index

pytestmark = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)


def test_build_vw_index_returns_one_row_per_date():
    panel = pl.DataFrame(
        {
            "id": ["A", "B", "A", "B"],
            "date": [
                datetime.date(2015, 1, 2),
                datetime.date(2015, 1, 2),
                datetime.date(2015, 1, 5),
                datetime.date(2015, 1, 5),
            ],
            "price": [10.0, 20.0, 11.0, 19.0],
            "shares": [100.0, 50.0, 100.0, 50.0],
            "mkt_cap": [1000.0, 1000.0, 1000.0, 1000.0],
            "ret": [0.05, -0.02, 0.1, -0.05],
        }
    )
    index = gate1_index.build_vw_index(panel)
    assert set(index.columns) == {"date", "index_ret"}
    assert index.height == 2


def test_build_vw_index_is_weight_weighted_average_of_returns():
    """Two names, equal mkt_cap (equal weight 0.5 each) on 2015-01-02 --
    index return must be the simple average of the two returns."""
    panel = pl.DataFrame(
        {
            "id": ["A", "B"],
            "date": [datetime.date(2015, 1, 2), datetime.date(2015, 1, 2)],
            "price": [10.0, 20.0],
            "shares": [100.0, 50.0],
            "mkt_cap": [1000.0, 1000.0],
            "ret": [0.10, -0.02],
        }
    )
    index = gate1_index.build_vw_index(panel)
    row = index.filter(pl.col("date") == datetime.date(2015, 1, 2))
    assert row.height == 1
    expected = 0.5 * 0.10 + 0.5 * -0.02
    assert row["index_ret"][0] == pytest.approx(expected, rel=1e-9)


def test_build_vw_index_weights_by_mkt_cap_not_equally():
    """Unequal mkt_cap: name A is 3x name B's weight (750/250 split).
    Index return must reflect that skew, not a simple average."""
    panel = pl.DataFrame(
        {
            "id": ["A", "B"],
            "date": [datetime.date(2015, 1, 2), datetime.date(2015, 1, 2)],
            "price": [10.0, 20.0],
            "shares": [100.0, 50.0],
            "mkt_cap": [750.0, 250.0],
            "ret": [0.10, -0.02],
        }
    )
    index = gate1_index.build_vw_index(panel)
    row = index.filter(pl.col("date") == datetime.date(2015, 1, 2))
    expected = 0.75 * 0.10 + 0.25 * -0.02
    assert row["index_ret"][0] == pytest.approx(expected, rel=1e-9)
    # Sanity: must NOT equal the simple (equal-weight) average.
    simple_average = 0.5 * 0.10 + 0.5 * -0.02
    assert row["index_ret"][0] != pytest.approx(simple_average, rel=1e-9)


def test_build_vw_index_ignores_null_mkt_cap_rows():
    """A row with null mkt_cap (no resolvable lagged price) must not
    contribute to the weight denominator or the weighted sum."""
    panel = pl.DataFrame(
        {
            "id": ["A", "B"],
            "date": [datetime.date(2015, 1, 2), datetime.date(2015, 1, 2)],
            "price": [10.0, 20.0],
            "shares": [100.0, 50.0],
            "mkt_cap": [1000.0, None],
            "ret": [0.10, -0.99],
        }
    )
    index = gate1_index.build_vw_index(panel)
    row = index.filter(pl.col("date") == datetime.date(2015, 1, 2))
    assert row["index_ret"][0] == pytest.approx(0.10, rel=1e-9)


def test_build_vw_index_ignores_null_ret_rows():
    """A row with a null ret (but a valid mkt_cap) must not contribute to
    the index at all -- neither the weighted sum nor the weight
    denominator -- not be silently treated as a 0% return while its full
    market-cap weight stays in the denominator. Constructed so that the
    two failure modes disagree: if name B's null ret were wrongly folded
    in as 0%, the index would be 0.5 * 0.10 + 0.5 * 0.0 = 0.05; the
    correct answer, with B excluded entirely, is name A's own return
    alone (as if B were absent from the panel)."""
    panel = pl.DataFrame(
        {
            "id": ["A", "B"],
            "date": [datetime.date(2015, 1, 2), datetime.date(2015, 1, 2)],
            "price": [10.0, 20.0],
            "shares": [100.0, 50.0],
            "mkt_cap": [1000.0, 1000.0],
            "ret": [0.10, None],
        }
    )
    index = gate1_index.build_vw_index(panel)
    row = index.filter(pl.col("date") == datetime.date(2015, 1, 2))
    assert row["index_ret"][0] == pytest.approx(0.10, rel=1e-9)


def test_build_vw_index_with_extra_group_col_computes_index_ret_per_group():
    """Generalization for Gate 2: build_vw_index(panel, extra_group_cols=["decile"])
    must compute the SAME weighted-average formula independently within
    each decile group, not mix mkt_cap across deciles into one pooled
    weight."""
    panel = pl.DataFrame(
        {
            "id": ["A", "B", "C", "D"],
            "date": [datetime.date(2015, 1, 2)] * 4,
            "price": [10.0, 20.0, 30.0, 40.0],
            "shares": [100.0, 50.0, 10.0, 10.0],
            "mkt_cap": [750.0, 250.0, 100.0, 100.0],
            "ret": [0.10, -0.02, 0.5, -0.5],
            "decile": [1, 1, 2, 2],
        }
    )
    result = gate1_index.build_vw_index(panel, extra_group_cols=["decile"])
    assert set(result.columns) == {"date", "decile", "index_ret"}
    assert result.height == 2

    decile_1 = result.filter(pl.col("decile") == 1)
    expected_1 = 0.75 * 0.10 + 0.25 * -0.02
    assert decile_1["index_ret"][0] == pytest.approx(expected_1, rel=1e-9)

    decile_2 = result.filter(pl.col("decile") == 2)
    expected_2 = 0.5 * 0.5 + 0.5 * -0.5
    assert decile_2["index_ret"][0] == pytest.approx(expected_2, rel=1e-9)


def test_build_vw_index_default_behavior_unchanged():
    """Existing Gate 1 callers (run_gate1_us, run_gate1_canada) call
    build_vw_index(panel) with no second argument -- must still return
    exactly {date, index_ret}, unchanged from before this generalization."""
    panel = pl.DataFrame(
        {
            "id": ["A", "B"],
            "date": [datetime.date(2015, 1, 2), datetime.date(2015, 1, 2)],
            "price": [10.0, 20.0],
            "shares": [100.0, 50.0],
            "mkt_cap": [1000.0, 1000.0],
            "ret": [0.10, -0.02],
        }
    )
    result = gate1_index.build_vw_index(panel)
    assert set(result.columns) == {"date", "index_ret"}


def test_run_gate1_us_2015_reports_high_correlation():
    """The actual Gate 1 run: build the index from real 2015 US data,
    compare against vwretd from the same panel. Per
    docs/02_validation_gates.md, expect a match to rounding -- this test
    asserts a high correlation threshold rather than exact equality,
    since the self-built universe (REIT/MLP-excluded, major-exchange-
    only) is not byte-identical to vwretd's own universe construction.

    Threshold raised 2026-09-09 (Phase B) from >0.9 to >0.995: the
    achieved value on the corrected (re-pulled, delisting-rows-present)
    panel is 0.99756 (docs/02_validation_gates.md), so 0.995 leaves
    ~0.0003 of re-run-noise margin while still catching a regression that
    knocks off the third decimal. Correlation alone is NOT sufficient --
    see the absolute-diagnostic tests below; it is location- and
    scale-invariant and cannot see a level or scale defect
    (docs/02_validation_gates.md's cross-cutting finding)."""
    result = gate1_index.run_gate1_us(
        start_date=datetime.date(2015, 1, 1), end_date=datetime.date(2015, 12, 31)
    )
    assert "correlation" in result
    assert result["correlation"] > 0.995, (
        f"Gate 1 correlation {result['correlation']} is below the 0.995 "
        "credibility threshold -- see docs/02_validation_gates.md's "
        "debug order: (1) universe definition, (2) delisting returns, "
        "(3) estimator windows, (4) weighting, (5) leg scaling. Check "
        "realized market loading first."
    )


def test_run_gate1_us_2015_comparison_frame_has_no_gaps():
    """Every trading day the self-built index has a value for must also
    have a vwretd value to compare against, and vice versa -- an
    unexplained gap on either side would be a date-alignment bug
    (docs/02_validation_gates.md's Gate 1 gotcha list).

    **Correction (2026-09-09, gate-verifier review):** the null-count
    assertions below were structurally unable to fail -- build_vw_index()
    already excludes null mkt_cap/ret before aggregating, and the inner
    join in run_gate1_us() drops any unmatched date from `comparison`
    entirely, so `comparison["index_ret"]`/`["vwretd"]` can never contain
    a null regardless of whether a real date-alignment gap exists. The
    two diagnostics that actually detect a gap -- n_index_only,
    n_vwretd_only -- were computed and returned in the result dict but
    never asserted on. Added those, plus n_dates == 252 (the full 2015
    trading-day count), so a silent window truncation or a one-sided
    date drop now has an assertion capable of catching it."""
    result = gate1_index.run_gate1_us(
        start_date=datetime.date(2015, 1, 1), end_date=datetime.date(2015, 12, 31)
    )
    comparison = result["comparison"]
    assert comparison["index_ret"].null_count() == 0
    assert comparison["vwretd"].null_count() == 0
    assert result["n_index_only"] == 0, (
        f"{result['n_index_only']} dates present in our own index but "
        "missing from vwretd -- date-alignment bug."
    )
    assert result["n_vwretd_only"] == 0, (
        f"{result['n_vwretd_only']} dates present in vwretd but missing "
        "from our own index -- date-alignment bug."
    )
    assert result["n_dates"] == 252, (
        f"n_dates={result['n_dates']}, expected 252 (full 2015 trading "
        "calendar) -- a silent window truncation would show up here "
        "without necessarily moving correlation or the absolute bounds."
    )


def test_run_gate1_us_2015_absolute_diff_bounds():
    """Correlation is location- and scale-invariant (docs/02_validation_gates.md's
    cross-cutting finding: a +50bp/day bias and a 1.50x scale error both
    score corr=1.0000000000 on this project's own 2015 vwretd series).
    These bounds are the non-scale-invariant half of Gate 1 -- added
    Phase B 2026-09-09, corrected same day after gate-verifier review.

    **Correction 1:** the original version asserted only `abs(diff.mean())`
    (the SIGNED mean) against 2e-4. gate-verifier found, and independent
    re-measurement confirmed, that a signed mean is blind to any mean-zero
    or sign-symmetric error -- summation cancels it exactly. Fixed by
    adding a true mean-absolute-deviation bound alongside the signed mean
    (kept, since it still catches a pure constant bias cleanly and
    cheaply).

    **Correction 2:** the first fix's mean-abs-diff bound (1e-3) and the
    original max-abs-diff bound (5e-3) were BOTH re-checked against a
    5%-return-scale injection (k=1.05) and both passed -- neither actually
    caught it, despite the docstring at the time claiming otherwise.
    Measured sweep (k=1.00 baseline through k=1.10):
        k=1.00: mean_abs=5.684e-04  max_abs=3.023e-03  (baseline)
        k=1.01: mean_abs=6.022e-04  max_abs=3.391e-03
        k=1.03: mean_abs=6.868e-04  max_abs=4.127e-03
        k=1.04: mean_abs=7.243e-04  max_abs=4.495e-03
        k=1.05: mean_abs=7.929e-04  max_abs=4.863e-03
        k=1.10: mean_abs=1.099e-03  max_abs=6.704e-03
    Max-abs-diff is the more sensitive of the two to this error class and
    was tightened from 5e-3 to 4e-3 (~1.32x baseline) so it catches k>=1.04
    (~1.3x margin over baseline, well short of what a 5% return-scale
    error produces). Mean-abs-diff is kept at 1e-3 -- it does not reliably
    catch a return-scale error below k~1.08 on this leg, and that is
    stated here explicitly as a known limitation rather than claimed as
    coverage it doesn't have; its job is the sign-symmetric/dispersion
    error class the signed mean can't see, not small scale errors (that's
    max-abs-diff's job).

    Thresholds derived from the measured values on the corrected panel
    (docs/02_validation_gates.md, re-verified 2026-09-09): signed mean
    diff 7.485e-05, mean absolute diff 5.684e-04, max abs diff 3.023e-03.
    Signed-mean bound kept at 2e-4 (~2.7x the achieved value -- a
    +50bp/day constant bias pushes this to ~5e-3, an order of magnitude
    of margin)."""
    result = gate1_index.run_gate1_us(
        start_date=datetime.date(2015, 1, 1), end_date=datetime.date(2015, 12, 31)
    )
    diff = result["comparison"]["diff"]
    signed_mean_diff = diff.mean()
    mean_abs_diff = diff.abs().mean()
    max_abs_diff = diff.abs().max()
    assert abs(signed_mean_diff) < 2e-4, (
        f"Gate 1 US signed mean diff {signed_mean_diff} exceeds 2e-4 -- a "
        "level bias correlation cannot detect (docs/02_validation_gates.md "
        "cross-cutting finding)."
    )
    assert mean_abs_diff < 1e-3, (
        f"Gate 1 US mean absolute diff {mean_abs_diff} exceeds 1e-3 -- a "
        "sign-symmetric error the signed mean alone would miss by "
        "cancellation."
    )
    assert max_abs_diff < 4e-3, (
        f"Gate 1 US max abs diff {max_abs_diff} exceeds 4e-3 on some single "
        "trading day -- check for a scale or unit error correlation would miss."
    )


def test_2015_delisting_rows_present_in_panel():
    """Direct, non-scale-invariant check that the corrected CRSP panel
    (not the pre-fix us_panel_crsp_full.old/) actually carries delisting
    rows for 2015 -- the specific defect that forced Gate 1's US PASS
    retraction (docs/02_validation_gates.md). Reads the raw partition
    file directly rather than through us_gate1_panel(), which does not
    carry dlydelflg/dlyret at this granularity.

    Anchors (verified live against data/raw/us_panel_crsp_full/year=2015,
    Phase B 2026-09-09): 245 rows with dlydelflg='Y' within calendar 2015
    specifically (the year=2015 partition file itself spans 2014-01-02
    through 2015-12-31 due to boundary widening elsewhere in the
    pipeline, and holds 437 dlydelflg='Y' rows unfiltered -- filtering to
    the calendar year is required to reproduce the documented 245).
    Permno 89888 carries dlyret=-0.958333 on 2015-03-19, a real
    distressed-delisting return entirely absent from the old panel."""
    from src.data import universe_panel

    files = universe_panel._year_partition_files(2015)
    panel = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlydelflg", "dlyret"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(datetime.date(2015, 1, 1)).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(datetime.date(2015, 12, 31)).cast(pl.Datetime("ns")))
        )
        .collect(engine="streaming")
    )
    n_delisted = panel.filter(pl.col("dlydelflg") == "Y").height
    assert n_delisted == 245, (
        f"Expected 245 dlydelflg='Y' rows in calendar 2015, got {n_delisted} -- "
        "the delisting-row-recovery fix (dc4e92e) may not be present in "
        "data/raw/us_panel_crsp_full/."
    )

    row = panel.filter(
        (pl.col("permno") == 89888)
        & (pl.col("dlycaldt") == pl.lit(datetime.date(2015, 3, 19)).cast(pl.Datetime("ns")))
    )
    assert row.height == 1, "permno 89888 not found on 2015-03-19 in the panel."
    assert row["dlyret"][0] == pytest.approx(-0.958333, abs=1e-6)


def test_2015_delisting_rows_reach_the_gate1_panel():
    """test_2015_delisting_rows_present_in_panel (above) proves delisting
    rows exist ON DISK in the corrected parquet. It does NOT prove they
    reach build_vw_index() -- it reads the raw partition file directly,
    bypassing us_gate1_panel()'s membership join and mkt_cap-null filter
    entirely. This test closes that gap.

    Why it matters (gate-verifier finding, 2026-09-09): correlation and
    all three absolute-diff bounds in this file were checked by directly
    injecting the historical defect -- stripping delisting rows from the
    INDEX computation itself (not the panel) -- and NONE of the four
    moved outside noise (correlation 0.9975595944981395 unchanged to 6
    decimal places, all three absolute bounds moved by <1e-5). The
    detection floor for this statistic set is between 0.5% and 1% of
    names/day; real delisting rates never exceed 0.055%/year even in the
    worst measured decade. A delisting-handling regression 10-20x worse
    than the one that forced the original retraction would still pass
    every other test in this file. Only a structural count check --
    this one -- actually guards that defect class.

    232 of 245 dlydelflg='Y' rows in calendar 2015 reach the panel, not
    245 -- the 13 gap is fully accounted for, not unexplained: 10 are
    excluded because the name carried issuertype='REIT' at its PRIOR
    month-end snapshot (the point-in-time universe membership date --
    universe_panel.py's reit_issuertype filter working as designed, not
    a leak), and 3 (permnos 29103, 92507, 92150) were not eligible at
    all at their prior month-end (zero rows in that day's snapshot --
    absent from the panel before the delisting question is even
    reached). Traced permno-by-permno this session; see
    docs/worklog.md. Pinned at exactly 232, not `>0` or `>=`: a `>0`
    bound would pass on a single surviving row and reproduces the
    known "our_n_firms > 0" class of vacuous assertion this project has
    already been burned by once."""
    from src.data import gate_adapters, universe_panel

    files = universe_panel._year_partition_files(2015)
    delist = (
        pl.scan_parquet(files)
        .select(["permno", "dlycaldt", "dlydelflg"])
        .filter(
            (pl.col("dlycaldt") >= pl.lit(datetime.date(2015, 1, 1)).cast(pl.Datetime("ns")))
            & (pl.col("dlycaldt") <= pl.lit(datetime.date(2015, 12, 31)).cast(pl.Datetime("ns")))
            & (pl.col("dlydelflg") == "Y")
        )
        .collect(engine="streaming")
        .with_columns(
            pl.col("permno").cast(pl.Utf8).alias("id"),
            pl.col("dlycaldt").cast(pl.Date).alias("date"),
        )
        .select(["id", "date"])
    )

    panel = gate_adapters.us_gate1_panel(
        datetime.date(2015, 1, 1), datetime.date(2015, 12, 31)
    )
    matched = panel.join(delist, on=["id", "date"], how="inner")

    assert matched.height == 232, (
        f"{matched.height} of 245 dlydelflg='Y' rows reached the Gate 1 US "
        "panel (us_gate1_panel() output), expected exactly 232. This gap "
        "is invisible to correlation and to all three absolute-diff bounds "
        "in this file (measured directly via injection, not assumed) -- a "
        "drop here from 232 toward 0 would mean delisting handling "
        "regressed with nothing else in this file able to catch it. If "
        "this count changed for an EXPLAINED reason (e.g. the REIT "
        "exclusion rule changed), update this pin deliberately and note "
        "why; do not loosen it to `> 0`."
    )

    # A second, independent anchor: at least one of the matched rows must
    # be a real distressed-delisting return, not merely present with a
    # placeholder/zero value -- confirms the surviving rows carry real
    # economic content, not just row-count parity. Threshold set at -0.8,
    # not -0.9 (gate-verifier second-pass review, 2026-09-09): of the 232
    # matched rows, only ONE (permno 89888) is below -0.9 -- a single-row
    # dependency that would break this assertion for a non-regression
    # reason if that one delisting were ever reclassified upstream. Four
    # rows clear -0.8, giving the same "real distressed return present"
    # guarantee with less fragility.
    assert matched.filter(pl.col("ret") < -0.8).height > 0, (
        "No matched delisting row has a return below -0.8 -- expected "
        "permno 89888's -0.958333 on 2015-03-19 (or an equivalent "
        "distressed case) to be among the 232 that reach the panel."
    )


# ---------------------------------------------------------------------------
# Canada leg (Gate 1 plan Phase D) -- CHASS's total-return series (ind7)
# only exists at MONTHLY grain, so the self-built daily VW index must be
# chain-linked up to monthly before comparison. See
# docs/superpowers/specs/2026-09-07-gate1-canada-handoff.md for the full
# reasoning on why this can't mirror run_gate1_us's daily-grain approach.
# ---------------------------------------------------------------------------

pytestmark_chass = pytest.mark.skipif(
    not MONTHLY_PATH.exists(),
    reason="data/raw/CHASS_Data/parquet/monthly.parquet not present",
)


def test_compound_daily_index_to_monthly_known_answer():
    """Synthetic daily index returns with a hand-computed compounded
    answer -- not real data, per the handoff doc's TDD instruction for
    this piece specifically. Two months, three trading days each."""
    daily_index = pl.DataFrame(
        {
            "date": [
                datetime.date(2015, 1, 2),
                datetime.date(2015, 1, 5),
                datetime.date(2015, 1, 6),
                datetime.date(2015, 2, 2),
                datetime.date(2015, 2, 3),
            ],
            "index_ret": [0.01, -0.02, 0.03, 0.05, -0.01],
        }
    )
    monthly = gate1_index.compound_daily_index_to_monthly(daily_index)
    assert set(monthly.columns) == {"month_end", "monthly_ret"}
    assert monthly.height == 2

    jan = monthly.filter(pl.col("month_end") == datetime.date(2015, 1, 6))
    expected_jan = (1.01 * 0.98 * 1.03) - 1
    assert jan.height == 1
    assert jan["monthly_ret"][0] == pytest.approx(expected_jan, rel=1e-9)

    feb = monthly.filter(pl.col("month_end") == datetime.date(2015, 2, 3))
    expected_feb = (1.05 * 0.99) - 1
    assert feb.height == 1
    assert feb["monthly_ret"][0] == pytest.approx(expected_feb, rel=1e-9)


def test_compound_daily_index_to_monthly_month_end_is_last_trading_day():
    """month_end must be the last actual date present in each calendar
    month's data (e.g. 2015-01-06 in the synthetic case above), never a
    naive calendar month-end (2015-01-31) -- CHASS's own real trading
    dates gotcha, same one already solved in canada_gate1_panel()."""
    daily_index = pl.DataFrame(
        {
            "date": [datetime.date(2015, 1, 2), datetime.date(2015, 1, 6)],
            "index_ret": [0.01, 0.02],
        }
    )
    monthly = gate1_index.compound_daily_index_to_monthly(daily_index)
    assert monthly["month_end"].to_list() == [datetime.date(2015, 1, 6)]


@pytestmark_chass
def test_run_gate1_canada_2015_reports_correlation():
    """The actual Gate 1 Canada run: build the daily VW index from real
    2015 CHASS data, compound to monthly, compare against ind7 (S&P/TSX
    Composite Monthly Total Return Index) converted to pct-change. Per
    the handoff doc, do not expect the US leg's 0.9976 -- CHASS's own
    universe construction, the accepted no-delisting-return gap
    (docs/01_data_notes.md section 6), and monthly-vs-daily compounding
    noise are real, expected sources of divergence here.

    Threshold raised 2026-09-09 (Phase B) from >0.5 to >0.995: the
    achieved value is 0.99812 (docs/02_validation_gates.md). Note this
    result rests on only 12 monthly observations -- thin evidence,
    flagged as an open item for Phase C (extend the window), not
    something this threshold alone fixes."""
    result = gate1_index.run_gate1_canada(
        start_date=datetime.date(2015, 1, 1), end_date=datetime.date(2015, 12, 31)
    )
    assert "correlation" in result
    assert result["n_dates"] > 0
    assert result["correlation"] > 0.995, (
        f"Gate 1 Canada correlation {result['correlation']} is below the "
        "0.995 credibility threshold -- check universe definition, ind7 "
        "pct-change direction, and month-end date alignment before "
        "trusting this number."
    )


@pytestmark_chass
def test_run_gate1_canada_2015_absolute_diff_bounds():
    """Non-scale-invariant half of the Canada leg, mirroring the US leg's
    equivalent test -- correlation alone cannot detect a level or scale
    defect (docs/02_validation_gates.md cross-cutting finding).

    **Correction 1 (2026-09-09, gate-verifier review):** originally
    asserted only the SIGNED mean, which cancels for any mean-zero or
    return-scale error -- confirmed by independent re-measurement, and
    worse than the US leg's version of this problem: sweeping a positive
    return-scale factor k across this leg does not just shrink the signed
    mean toward zero, it crosses zero and goes NEGATIVE:
        k=1.00: signed_mean= 3.485e-04  mean_abs=1.173e-03  max_abs=3.076e-03
        k=1.01: signed_mean= 2.780e-04  mean_abs=1.210e-03  max_abs=3.483e-03
        k=1.03: signed_mean= 1.369e-04  mean_abs=1.367e-03  max_abs=4.297e-03
        k=1.05: signed_mean=-4.634e-06  mean_abs=1.590e-03  max_abs=5.112e-03
        k=1.10: signed_mean=-3.601e-04  mean_abs=2.408e-03  max_abs=7.153e-03
    A k=1.05 return-scale error would have passed the OLD signed-mean-only
    bound outright (signed mean ~0). Added a true mean-absolute-diff bound.

    **Correction 2:** the first fix's mean-abs-diff bound (2e-3) and the
    original max-abs-diff bound (5e-3) were re-checked against the sweep
    above -- mean-abs-diff at 2e-3 does not catch k=1.05 (1.590e-3 < 2e-3),
    so it is tightened to 1.2e-3 (catches k>=1.03, ~1.02x margin over the
    achieved 1.173e-03 -- deliberately tight, since this leg's only n=12
    monthly grain means there is little re-run noise to guard against on
    a fixed historical panel, and this bound needs real sensitivity to
    compensate for the signed mean's demonstrated blind spot on this leg).
    Max-abs-diff at 5e-3 already catches k=1.05 (5.112e-3) with the
    existing bound, so it is left unchanged; the US leg's asymmetry
    (needing to tighten max-abs-diff instead of mean-abs-diff) does not
    carry over here -- the two legs are not assumed to share a threshold
    derivation. Signed-mean bound kept at 1e-3 (~2.9x the achieved
    3.485e-04 -- still catches a pure constant bias cleanly)."""
    result = gate1_index.run_gate1_canada(
        start_date=datetime.date(2015, 1, 1), end_date=datetime.date(2015, 12, 31)
    )
    diff = result["comparison"]["diff"]
    signed_mean_diff = diff.mean()
    mean_abs_diff = diff.abs().mean()
    max_abs_diff = diff.abs().max()
    assert abs(signed_mean_diff) < 1e-3, (
        f"Gate 1 Canada signed mean diff {signed_mean_diff} exceeds 1e-3 -- "
        "a level bias correlation cannot detect."
    )
    assert mean_abs_diff < 1.2e-3, (
        f"Gate 1 Canada mean absolute diff {mean_abs_diff} exceeds 1.2e-3 -- "
        "a sign-symmetric or return-scale error the signed mean alone would "
        "miss by cancellation (this leg's signed mean can even move TOWARD "
        "zero under a scale error -- see this test's docstring)."
    )
    assert max_abs_diff < 5e-3, (
        f"Gate 1 Canada max abs diff {max_abs_diff} exceeds 5e-3 in some "
        "single month -- check for a scale or unit error correlation would miss."
    )
