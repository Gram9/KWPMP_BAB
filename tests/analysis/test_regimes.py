"""Tests for src/analysis/regimes.py -- docs/06_handoff_section6.md
item C / report Sec 6.4, downturns vs expansions.

WRITTEN BEFORE THE MODULE (CLAUDE.md workflow).

The headline anchor here is unusual and deliberate: it parses
docs/07_regime_preregistration.md and asserts the module's constants
equal what that document says. The definitions were pre-registered
BEFORE any regime statistic was computed, precisely because this project
has retracted two gates whose statistics were chosen after seeing the
data. Transcribing the windows into code makes the doc decorative; this
test makes it load-bearing -- drift becomes a red test rather than a
report error.
"""

import datetime
import re
from pathlib import Path

import polars as pl
import pytest

from src.analysis import regimes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION_DOC = PROJECT_ROOT / "docs" / "07_regime_preregistration.md"


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


def _series(quarters, values, col="ret_excess"):
    return pl.DataFrame({"quarter": quarters, col: values})


# ---------------------------------------------------------------------------
# ABSOLUTE ANCHOR -- anchored OUTSIDE the dataset, in the pre-registration
# document itself.
# ---------------------------------------------------------------------------


def test_crisis_windows_match_the_preregistration_document():
    """ABSOLUTE ANCHOR (CLAUDE.md): CRISIS_WINDOWS must equal, label for
    label and date for date, the table in
    docs/07_regime_preregistration.md.

    That document states it was written before any regime statistic was
    computed and may not be revised afterwards. Without this test the
    constants in regimes.py could quietly drift toward whichever windows
    flatter the strategy, which is exactly the failure mode this project
    has already suffered twice. THE DOC WINS: if this test fails, fix
    the module, not the document.
    """
    text = PREREGISTRATION_DOC.read_text(encoding="utf-8")

    # Rows look like: | `label` | 1973-01-01 .. 1974-12-31 |
    pattern = re.compile(
        r"\|\s*`([a-z0-9_]+)`\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\.\.\s*(\d{4}-\d{2}-\d{2})\s*\|"
    )
    found = pattern.findall(text)

    assert len(found) >= 5, (
        f"parsed only {len(found)} crisis window(s) from {PREREGISTRATION_DOC.name} "
        "-- the document's Definition 1 table is missing or its format changed, "
        "so this anchor is no longer checking anything"
    )

    from_doc = tuple(
        (
            label,
            datetime.date.fromisoformat(lo),
            datetime.date.fromisoformat(hi),
        )
        for label, lo, hi in found
    )

    assert regimes.CRISIS_WINDOWS == from_doc, (
        "regimes.CRISIS_WINDOWS has drifted from the pre-registered "
        f"definitions.\n  module: {regimes.CRISIS_WINDOWS}\n  document: {from_doc}\n"
        "The document is authoritative -- fix the module."
    )


def test_downturn_percentile_is_the_pre_registered_ten_percent():
    """Pinned as a literal, the same idiom target_beta.TARGET_BETAS uses
    for its user-confirmed grid."""
    assert regimes.DOWNTURN_PERCENTILE == 0.10
    assert regimes.DOWNTURN_INTERPOLATION == "linear"


# ---------------------------------------------------------------------------
# Definition 1 -- the quarter-assignment rule
# ---------------------------------------------------------------------------


def test_quarter_ending_inside_a_window_is_labelled():
    """The end-date clause: 1974-06-30 falls inside 1973-01-01..1974-12-31."""
    got = regimes.label_crisis_regimes([datetime.date(1974, 6, 30)])

    labels = got.filter(pl.col("regime").is_not_null())["regime"].to_list()
    assert labels == ["oil_shock_1973_74"], f"got {labels}"


def test_short_window_fully_contained_in_a_quarter_is_labelled():
    """THE LOAD-BEARING CLAUSE. Black Monday (1987-10-01..1987-10-31) and
    COVID (2020-03-01..2020-03-31) are each SHORTER than a quarter and
    end mid-quarter, so an end-date-only rule matches ZERO quarters for
    both and the exhibit is silently empty.

    The pre-registration document states the containment clause
    explicitly for exactly this reason.
    """
    got = regimes.label_crisis_regimes(
        [datetime.date(1987, 12, 31), datetime.date(2020, 3, 31)]
    )

    by_quarter = dict(zip(got["quarter"].to_list(), got["regime"].to_list()))
    assert by_quarter[datetime.date(1987, 12, 31)] == "black_monday_1987", (
        "1987-12-31 did not pick up Black Monday -- the containment clause "
        "is missing, and this crisis will show zero quarters in the report"
    )
    assert by_quarter[datetime.date(2020, 3, 31)] == "covid_2020"


def test_gfc_window_spans_two_quarters():
    """2008-09-01..2009-03-31 covers both 2008-12-31 and 2009-03-31. A
    first-match-only or single-quarter implementation would drop one."""
    got = regimes.label_crisis_regimes(
        [datetime.date(2008, 12, 31), datetime.date(2009, 3, 31)]
    )

    labels = got.filter(pl.col("regime") == "gfc_2008_09")["quarter"].to_list()
    assert sorted(labels) == [datetime.date(2008, 12, 31), datetime.date(2009, 3, 31)], (
        f"GFC matched {labels}, expected both quarters"
    )


def test_quarter_outside_every_window_gets_a_null_regime():
    got = regimes.label_crisis_regimes([datetime.date(1995, 6, 30)])

    assert got.height == 1
    assert got["regime"][0] is None


def test_canada_style_sample_reports_out_of_sample_not_zero():
    """The pre-registration document's Canada caveat, made executable:
    the Canadian leg starts 1989-03-31, so the 1973 and 1987 windows have
    ZERO Canadian quarters. That is a structural absence, not a finding
    of 'no effect', and the exhibit must render it as such rather than as
    a blank or a zero."""
    canadian_quarters = _quarters(80, start_year=1989)

    coverage = regimes.crisis_coverage(canadian_quarters)

    by_label = dict(zip(coverage["regime"].to_list(), coverage["in_sample"].to_list()))
    assert by_label["oil_shock_1973_74"] is False
    assert by_label["black_monday_1987"] is False
    assert by_label["dotcom_2000_02"] is True, (
        "the dot-com window is inside the Canadian sample and must be in_sample"
    )
    # Every window appears, including the ones with no matching quarter.
    assert coverage.height == len(regimes.CRISIS_WINDOWS)


# ---------------------------------------------------------------------------
# Definition 2 -- the objective threshold
# ---------------------------------------------------------------------------


def test_threshold_labels_exactly_ten_percent_of_a_uniform_series():
    """ABSOLUTE ANCHOR #2: on 100 evenly-spaced quarters the 10th
    percentile is an exactly known value, so exactly 10 quarters must be
    labelled downturn, and the quarter sitting ON the breakpoint must be
    included (the pre-registered rule is at-or-below).

    Catches an inclusive/exclusive flip, a wrong interpolation method,
    and a percentile taken over the wrong axis.
    """
    qs = _quarters(100, start_year=1900)
    values = [(-0.50 + 0.01 * i) for i in range(100)]
    market = _series(qs, values)

    got = regimes.label_threshold_regimes(market, value_col="ret_excess")

    n_down = got.filter(pl.col("regime") == regimes.DOWNTURN_LABEL).height
    assert n_down == 10, (
        f"{n_down} quarters labelled downturn on a uniform 100-quarter series, "
        "expected exactly 10 -- an off-by-one here is an inclusive/exclusive "
        "boundary error"
    )


def test_threshold_includes_a_quarter_exactly_on_the_breakpoint():
    """The pre-registered tie rule is '<=', so a quarter landing exactly
    on the 10th percentile is a downturn."""
    # 11 values so the 10th percentile lands EXACTLY on an observation
    # (linear interpolation of 11 points puts p10 on index 1.0), rather
    # than between two -- otherwise no quarter sits on the breakpoint and
    # the tie rule is never exercised.
    qs = _quarters(11, start_year=1900)
    values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    market = _series(qs, values)

    got = regimes.label_threshold_regimes(market, value_col="ret_excess")
    threshold = got["threshold"][0]

    on_breakpoint = got.filter(pl.col("ret_excess") == threshold)
    assert on_breakpoint.height >= 1
    assert set(on_breakpoint["regime"].to_list()) == {regimes.DOWNTURN_LABEL}, (
        "a quarter exactly on the breakpoint was labelled expansion -- the "
        "tie rule must be at-or-below"
    )


def test_threshold_follows_the_market_not_the_portfolio():
    """label_threshold_regimes must split on the MARKET's own
    distribution. An implementation reading the portfolio instead would
    make the strategy look defensive by construction -- the regime split
    would be defined by the very series being evaluated."""
    # A VARYING market with one clear worst quarter. A near-constant
    # series (19 identical values plus one outlier) puts the 10th
    # percentile ON the repeated value, so every quarter is at-or-below
    # it -- correct behaviour, but it would not test what this test
    # claims to test.
    qs = _quarters(20, start_year=1900)
    market_values = [0.01 + 0.002 * i for i in range(19)] + [-0.30]
    market = _series(qs, market_values)

    got = regimes.label_threshold_regimes(market, value_col="ret_excess")

    downturns = got.filter(pl.col("regime") == regimes.DOWNTURN_LABEL)["quarter"].to_list()
    # 10% of 20 quarters = 2: the -0.30 outlier plus the lowest ordinary
    # quarter. Both are the MARKET's own worst, which is the point.
    assert downturns == [qs[0], qs[19]], (
        f"downturn quarters were {downturns}, expected the market's two "
        "weakest quarters"
    )

    # The discriminating half: hand the SAME function a portfolio series
    # whose weak quarters sit elsewhere. The labels must not move, because
    # the split is defined by the market alone.
    portfolio = _series(qs, [-0.40, -0.35] + [0.05] * 18)
    from_portfolio = regimes.label_threshold_regimes(portfolio, value_col="ret_excess")
    assert from_portfolio.filter(pl.col("regime") == regimes.DOWNTURN_LABEL)[
        "quarter"
    ].to_list() == [qs[0], qs[1]], (
        "sanity check on the fixture: the portfolio's own weak quarters are "
        "q0 and q1, which differ from the market's q0 and q19 -- so the two "
        "inputs genuinely discriminate"
    )


def test_every_quarter_gets_exactly_one_threshold_label():
    qs = _quarters(20, start_year=1900)
    market = _series(qs, [0.01 * i for i in range(20)])

    got = regimes.label_threshold_regimes(market, value_col="ret_excess")

    assert got.height == 20
    assert set(got["regime"].to_list()) == {regimes.DOWNTURN_LABEL, regimes.EXPANSION_LABEL}


# ---------------------------------------------------------------------------
# regime_comparison
# ---------------------------------------------------------------------------


def test_regime_comparison_reports_the_pre_registered_statistics():
    qs = _quarters(8, start_year=1900)
    port = _series(qs, [0.05, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.03])
    market = _series(qs, [0.03, 0.01, -0.05, 0.02, 0.03, 0.02, 0.01, 0.02])
    labels = pl.DataFrame(
        {
            "quarter": qs,
            "regime": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )

    got = regimes.regime_comparison(
        port, market, labels, port_col="ret_excess", market_col="ret_excess", label="x"
    )

    for column in (
        "label",
        "regime",
        "n_quarters",
        "port_mean_excess",
        "mkt_mean_excess",
        "diff_mean",
        "diff_t_stat",
        "hit_rate",
    ):
        assert column in got.columns, f"missing pre-registered column {column!r}"
    assert set(got["regime"].to_list()) == {"a", "b"}


def test_regime_comparison_diff_t_stat_matches_hand_computed():
    """mean(d)/(std(d, ddof=1)/sqrt(n)). ddof=0 would inflate it, which
    is the direction that manufactures significance."""
    qs = _quarters(4, start_year=1900)
    port_values = [0.05, 0.02, 0.04, 0.01]
    mkt_values = [0.01, 0.01, 0.02, 0.00]
    labels = pl.DataFrame({"quarter": qs, "regime": ["r"] * 4})

    got = regimes.regime_comparison(
        _series(qs, port_values),
        _series(qs, mkt_values),
        labels,
        port_col="ret_excess",
        market_col="ret_excess",
        label="x",
    )

    import statistics

    d = [p - m for p, m in zip(port_values, mkt_values)]
    expected = statistics.mean(d) / (statistics.stdev(d) / (4**0.5))

    assert got["diff_t_stat"][0] == pytest.approx(expected, rel=1e-9)


def test_hit_rate_counts_strictly_greater_than():
    """The pre-registered wording is 'portfolio > market'. A quarter
    where the two are exactly equal must NOT count as a hit."""
    qs = _quarters(4, start_year=1900)
    port_values = [0.05, 0.02, 0.02, 0.01]
    mkt_values = [0.01, 0.02, 0.02, 0.05]  # equal in quarters 2 and 3
    labels = pl.DataFrame({"quarter": qs, "regime": ["r"] * 4})

    got = regimes.regime_comparison(
        _series(qs, port_values),
        _series(qs, mkt_values),
        labels,
        port_col="ret_excess",
        market_col="ret_excess",
        label="x",
    )

    assert got["hit_rate"][0] == pytest.approx(0.25, rel=1e-12), (
        f"hit_rate={got['hit_rate'][0]!r}; only 1 of 4 quarters is a strict "
        "win, so ties are being counted as hits"
    )


def test_single_quarter_regime_reports_n_one_and_null_t_stat():
    """With 5 crisis episodes this WILL fire on real data. A
    1-observation t-statistic emitted as inf or 0.0 would be read as a
    finding; it must be None."""
    qs = _quarters(3, start_year=1900)
    labels = pl.DataFrame({"quarter": qs, "regime": ["solo", "other", "other"]})

    got = regimes.regime_comparison(
        _series(qs, [0.05, 0.02, 0.01]),
        _series(qs, [0.01, 0.01, 0.01]),
        labels,
        port_col="ret_excess",
        market_col="ret_excess",
        label="x",
    )

    solo = got.filter(pl.col("regime") == "solo")
    assert solo["n_quarters"][0] == 1
    assert solo["diff_t_stat"][0] is None, (
        f"a 1-quarter regime reported diff_t_stat={solo['diff_t_stat'][0]!r}, "
        "which a reader would take as a real statistic"
    )


def test_regime_comparison_ignores_quarters_with_a_null_regime():
    """Crisis labelling leaves non-crisis quarters null; those must not
    become their own pooled 'regime' row."""
    qs = _quarters(4, start_year=1900)
    labels = pl.DataFrame({"quarter": qs, "regime": ["r", None, None, "r"]})

    got = regimes.regime_comparison(
        _series(qs, [0.05, 0.02, 0.04, 0.01]),
        _series(qs, [0.01, 0.01, 0.02, 0.00]),
        labels,
        port_col="ret_excess",
        market_col="ret_excess",
        label="x",
    )

    assert got.height == 1
    assert got["regime"][0] == "r"
    assert got["n_quarters"][0] == 2
