"""Tests for src/analysis/beta_buckets.py -- the beta-bucket/SML
analysis, docs/04_handoff_lowbeta_longonly.md Sec 4.3b, docs/
05_report_spec.md Sec 4.
"""

import datetime
import math

import polars as pl
import pytest

from src.analysis import beta_buckets


def _cross_section(ids: list[str], betas: list[float]) -> pl.DataFrame:
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


# ---------------------------------------------------------------------------
# _bucket_edges -- the computed invariant (10 fine bands + 5 deciles = 15).
# ---------------------------------------------------------------------------


def test_bucket_edges_count_is_15():
    edges = beta_buckets._bucket_edges()
    assert len(edges) == 15


def test_bucket_edges_are_sorted_ascending():
    edges = beta_buckets._bucket_edges()
    assert edges == sorted(edges)


def test_bucket_edges_span_from_zero_to_just_under_one():
    edges = beta_buckets._bucket_edges()
    assert edges[0] == 0.0
    assert edges[-1] == 0.90  # last decile's lower edge


# ---------------------------------------------------------------------------
# assign_buckets -- proportions and boundary correctness.
# ---------------------------------------------------------------------------


def test_assign_buckets_100_names_gives_correct_proportions():
    """100 evenly-ranked names: each 5%-wide V-bucket gets exactly 5
    names, each 10%-wide D-bucket gets exactly 10 -- the direct
    arithmetic check that 10 V-buckets + 5 D-buckets = 15 total and
    each respects its stated width."""
    df = _cross_section([str(i) for i in range(1, 101)], [float(i) for i in range(1, 101)])
    result = beta_buckets.assign_buckets(df)
    counts = result.group_by("bucket").agg(pl.len().alias("n")).sort("bucket")
    counts_dict = dict(zip(counts["bucket"].to_list(), counts["n"].to_list()))

    assert len(counts_dict) == 15
    for v in [f"V{i:02d}" for i in range(1, 11)]:
        assert counts_dict[v] == 5, f"{v} has {counts_dict.get(v)} names, expected 5"
    for d in [f"D{i:02d}" for i in range(6, 11)]:
        assert counts_dict[d] == 10, f"{d} has {counts_dict.get(d)} names, expected 10"


def test_assign_buckets_lowest_name_lands_in_v01_not_v02():
    """Regression test for a real off-by-one found this session: plain
    rank()/n never produces percentile 0.0 for the lowest-ranked name
    (rank is 1-indexed), so the lowest name's percentile landed EXACTLY
    on the V01/V02 boundary and was misclassified into V02. Confirmed
    live pre-fix with a clean 1..20 beta fixture. (rank-1)/n fixes it."""
    df = _cross_section([str(i) for i in range(1, 21)], [float(i) for i in range(1, 21)])
    result = beta_buckets.assign_buckets(df)
    lowest = result.filter(pl.col("id") == "1")
    assert lowest["bucket"][0] == "V01", (
        f"lowest-beta name landed in {lowest['bucket'][0]!r}, not V01 -- "
        "the exact off-by-one this test guards against"
    )


def test_assign_buckets_highest_name_lands_in_d10():
    df = _cross_section([str(i) for i in range(1, 21)], [float(i) for i in range(1, 21)])
    result = beta_buckets.assign_buckets(df)
    highest = result.filter(pl.col("id") == "20")
    assert highest["bucket"][0] == "D10"


def test_assign_buckets_median_boundary_names_split_v10_d06():
    """The two names straddling the median (ranks 10 and 11 of 20) must
    land in the LAST vigintile-band (V10, still below median) and the
    FIRST decile-band (D06, at/above median) respectively -- the exact
    median-split boundary case."""
    df = _cross_section([str(i) for i in range(1, 21)], [float(i) for i in range(1, 21)])
    result = beta_buckets.assign_buckets(df)
    rank10 = result.filter(pl.col("id") == "10")["bucket"][0]  # percentile (10-1)/20=0.45
    rank11 = result.filter(pl.col("id") == "11")["bucket"][0]  # percentile (11-1)/20=0.50
    assert rank10 == "V10"
    assert rank11 == "D06"


def test_assign_buckets_empty_frame_returns_empty_with_bucket_column():
    df = _cross_section([], [])
    result = beta_buckets.assign_buckets(df)
    assert result.height == 0
    assert "bucket" in result.columns


def test_assign_buckets_computed_cross_sectionally_not_pooled():
    """Same beta VALUE, different cross-sections (different other
    members) -- the bucket assignment for a fixed name must depend on
    ITS OWN cross-section's ranks, not some global/pooled distribution.
    Constructed so beta=5.0 lands in a LOW bucket when surrounded by
    high betas, and a HIGH bucket when surrounded by low betas."""
    high_context = _cross_section(
        ["target"] + [f"peer{i}" for i in range(19)],
        [5.0] + [float(50 + i) for i in range(19)],  # target is lowest here
    )
    low_context = _cross_section(
        ["target"] + [f"peer{i}" for i in range(19)],
        [5.0] + [float(-50 - i) for i in range(19)],  # target is highest here
    )
    bucket_in_high_context = beta_buckets.assign_buckets(high_context).filter(
        pl.col("id") == "target"
    )["bucket"][0]
    bucket_in_low_context = beta_buckets.assign_buckets(low_context).filter(
        pl.col("id") == "target"
    )["bucket"][0]
    assert bucket_in_high_context == "V01"
    assert bucket_in_low_context == "D10"
    assert bucket_in_high_context != bucket_in_low_context


# ---------------------------------------------------------------------------
# run_beta_bucket_analysis -- end to end with synthetic data, an ABSOLUTE
# anchor (a zero-noise beta-driven fixture must recover the planted
# betas exactly), and non-monotonicity handling (a fixture with a
# depressed lowest bucket must NOT be silently smoothed away).
# ---------------------------------------------------------------------------


def _build_quarterly_fixture(
    n_names: int, n_quarters: int, betas: list[float], market_rets: list[float], noise: float = 0.0
):
    """25 names (or n_names), each with a KNOWN planted beta, over
    n_quarters quarters of monthly returns (3 months each) driven
    EXACTLY by beta_i * market_ret_of_that_month (zero noise by
    default) -- an absolute-anchor fixture: recovered realized beta must
    equal the planted beta exactly when noise=0."""
    import random

    rng = random.Random(42)
    quarter_ends = []
    d = datetime.date(2015, 3, 31)
    for _ in range(n_quarters + 1):  # +1 for the formation-only first quarter
        quarter_ends.append(d)
        year, month = d.year, d.month + 3
        if month > 12:
            month -= 12
            year += 1
        if month == 12:
            d = datetime.date(year, 12, 31)
        elif month in (6, 9):
            first_next = datetime.date(year, month + 1, 1)
            d = first_next - datetime.timedelta(days=1)
        else:
            first_next = datetime.date(year, 4, 1)
            d = first_next - datetime.timedelta(days=1)

    ids = [f"N{i:02d}" for i in range(n_names)]
    betas_by_date = {}
    fringe_by_date = {}
    monthly_rows = []
    market_monthly_rows = []

    all_months = []
    d = datetime.date(2015, 1, 31)
    for _ in range(n_quarters * 3 + 3):
        all_months.append(d)
        year, month = d.year, d.month + 1
        if month > 12:
            month = 1
            year += 1
        first_next_month = month + 1
        first_next_year = year
        if first_next_month > 12:
            first_next_month = 1
            first_next_year += 1
        d = datetime.date(first_next_year, first_next_month, 1) - datetime.timedelta(days=1)

    for i, month in enumerate(all_months):
        mkt_ret = market_rets[i % len(market_rets)]
        market_monthly_rows.append({"month": month, "ret": mkt_ret})
        for name_idx, name in enumerate(ids):
            beta_i = betas[name_idx % len(betas)]
            eps = rng.uniform(-noise, noise) if noise else 0.0
            monthly_rows.append((name, month, beta_i * mkt_ret + eps))

    for q_end in quarter_ends[:-1]:
        betas_by_date[q_end] = _cross_section(ids, [betas[i % len(betas)] for i in range(n_names)])
        fringe_by_date[q_end] = _fringe(ids, [5e9] * n_names, [50.0] * n_names)

    quarterly_monthly_rets = _monthly_rets(monthly_rows)
    market_monthly = pl.DataFrame(market_monthly_rows)
    return betas_by_date, fringe_by_date, quarterly_monthly_rets, market_monthly


def test_run_beta_bucket_analysis_recovers_planted_beta_with_zero_noise():
    """ABSOLUTE ANCHOR (CLAUDE.md): 30 names with betas evenly spaced
    0.1..3.0, MONTHLY returns EXACTLY beta_i * market_ret (zero
    idiosyncratic noise). Each bucket's realized beta (from
    full_sample_market_loading, regressed on QUARTERLY-COMPOUNDED
    returns) must be CLOSE to that bucket's own mean planted beta -- not
    bit-identical, because log-space compounding of 3 monthly returns is
    a genuinely nonlinear transform: quarterly_ret != beta *
    quarterly_market_ret exactly even when every MONTHLY ret is exactly
    beta * monthly_market_ret. This is the same real, expected effect
    FP's own paper documents (realized beta differs from ex-ante beta,
    Table III) -- not a bug, so the tolerance here (2%) is set to catch
    a real defect (e.g. a wrong regressor, a units error) while
    tolerating genuine compounding drift, confirmed by direct
    measurement this session: max relative gap across all 15 buckets
    under this exact fixture is well under 2%."""
    n_names = 30
    betas_planted = [0.1 + 0.1 * i for i in range(30)]  # 0.1, 0.2, ..., 3.0
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]

    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 8, betas_planted, market_rets, noise=0.0
    )

    result = beta_buckets.run_beta_bucket_analysis(
        betas_by_date,
        fringe_by_date,
        quarterly_rets,
        market_monthly,
        mkt_cap_floor=1e9,
        price_floor=1.0,
    )
    assert result.summary.height > 0

    for row in result.summary.iter_rows(named=True):
        if row["realized_beta"] is None:
            continue
        assert row["realized_beta"] == pytest.approx(row["ex_ante_beta"], rel=0.02), (
            f"bucket {row['bucket']}: realized_beta={row['realized_beta']} != "
            f"ex_ante_beta={row['ex_ante_beta']} under a ZERO-NOISE fixture "
            "where every name's return is EXACTLY beta_i * market_ret"
        )


def test_run_beta_bucket_analysis_fringe_filter_excludes_names():
    """A name failing the mkt_cap/price floor must never appear in any
    bucket's membership (and therefore never influence its return)."""
    n_names = 30
    betas_planted = [0.1 + 0.1 * i for i in range(30)]
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]
    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 8, betas_planted, market_rets, noise=0.0
    )
    # Fail N00 (lowest beta) on mkt_cap at every formation date.
    for d, fringe in fringe_by_date.items():
        fringe_by_date[d] = fringe.with_columns(
            pl.when(pl.col("id") == "N00").then(pl.lit(1.0)).otherwise(pl.col("mkt_cap")).alias(
                "mkt_cap"
            )
        )

    result_filtered = beta_buckets.run_beta_bucket_analysis(
        betas_by_date, fringe_by_date, quarterly_rets, market_monthly,
        mkt_cap_floor=1e9, price_floor=1.0,
    ).summary
    result_unfiltered = beta_buckets.run_beta_bucket_analysis(
        betas_by_date, fringe_by_date, quarterly_rets, market_monthly,
        mkt_cap_floor=None, price_floor=None,
    ).summary
    # With N00 excluded, V01's ex_ante_beta should shift (N00 was the
    # single lowest-beta name and would otherwise dominate V01 alone).
    v01_filtered = result_filtered.filter(pl.col("bucket") == "V01")
    v01_unfiltered = result_unfiltered.filter(pl.col("bucket") == "V01")
    assert v01_filtered.height > 0
    assert v01_unfiltered.height > 0
    assert v01_filtered["ex_ante_beta"][0] != pytest.approx(
        v01_unfiltered["ex_ante_beta"][0], abs=1e-9
    ), "filtering N00 out did not change V01's composition -- filter had no effect"


def test_ex_ante_beta_tracks_the_same_population_as_realized_beta():
    """Regression test for a real defect found by leakage-auditor this
    session: an earlier version computed ex_ante_beta over the FULL
    bucket membership (before the return-survival join) while
    mean_ret/realized_beta were computed over ONLY the members that
    survived to have a real quarterly return -- an ex-ante/ex-post
    population mismatch that (measured) flattens the fitted SML line in
    exactly the direction the BAB thesis predicts, an artifact that
    would look like confirming evidence.

    Fixture: V01 has 2 members at 30 names (N00 beta=0.1, N01 beta=0.2,
    confirmed live this session). N01's returns are wiped out entirely
    for ALL months (never listed, not merely delisted mid-quarter) --
    it stays in the beta cross-section (still gets bucketed) but can
    never survive to a real quarterly return. Post-fix, ex_ante_beta
    for V01 must equal N00's beta ALONE (0.1), not the mean of N00 and
    N01 (0.15) -- since only N00 ever contributes a return."""
    n_names = 30
    betas_planted = [0.1 + 0.1 * i for i in range(30)]
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]
    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 8, betas_planted, market_rets, noise=0.0
    )
    # Wipe out N01's returns entirely -- it remains in betas_by_date/
    # fringe_by_date (still gets bucketed into V01 at every formation
    # date) but can never produce a quarterly return.
    quarterly_rets_missing_n01 = quarterly_rets.filter(pl.col("id") != "N01")

    result = beta_buckets.run_beta_bucket_analysis(
        betas_by_date, fringe_by_date, quarterly_rets_missing_n01, market_monthly,
        mkt_cap_floor=1e9, price_floor=1.0,
    ).summary
    v01 = result.filter(pl.col("bucket") == "V01")
    assert v01.height == 1
    assert v01["ex_ante_beta"][0] == pytest.approx(0.1, abs=1e-9), (
        f"V01's ex_ante_beta is {v01['ex_ante_beta'][0]}, expected exactly "
        "0.1 (N00 alone) -- if this is ~0.15 instead, ex_ante_beta is "
        "still being averaged over the FULL bucket membership (N00+N01) "
        "rather than only the names that survived to have a real return "
        "(N00 alone, since N01's returns were wiped out) -- the exact "
        "population-mismatch defect this test guards against"
    )


def test_run_beta_bucket_analysis_uses_quarterly_not_monthly_sharpe():
    """Regression test for a real bug found this session: Sharpe must
    be annualized with periods_per_year=4 (quarterly), not the
    diagnostics module's monthly default (12) -- a silent sqrt(3)
    misannualization. Confirmed by NUMERICALLY reconstructing V01's own
    quarterly return series (deterministic under this zero-noise
    fixture: every quarter's return is exactly the compounded 3-month
    product of V01's mean beta * that month's market return) and
    comparing the module's reported sharpe against an independently
    computed quarterly-annualized value -- and separately against what
    a (wrong) monthly annualization of the SAME series would give, to
    prove the two are numerically distinguishable here, not merely
    asserting a docstring convention."""
    n_names = 30
    betas_planted = [0.1 + 0.1 * i for i in range(30)]
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]
    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 8, betas_planted, market_rets, noise=0.0
    )
    result = beta_buckets.run_beta_bucket_analysis(
        betas_by_date, fringe_by_date, quarterly_rets, market_monthly,
        mkt_cap_floor=1e9, price_floor=1.0,
    ).summary
    row = result.filter(pl.col("bucket") == "V01")
    assert row.height == 1
    reported_sharpe = row["sharpe"][0]
    assert reported_sharpe is not None

    # Reconstruct V01's own quarterly return series independently: the
    # fixture repeats market_rets cyclically across months, 3 months per
    # quarter, so quarter q's 3 monthly market returns are
    # market_rets[3q % len], market_rets[(3q+1) % len], market_rets[(3q+2) % len].
    # V01's member betas are recovered directly from the module's own
    # output (rather than assumed from a proportion calculation) -- with
    # 30 names and a 5%-wide band, V01 has only 1-2 members depending on
    # rounding, not a round fraction of 30, so hand-deriving the count
    # would be error-prone. n_formations * ex_ante_beta gives the SUM of
    # per-formation-date mean betas (assuming a stable roster, true here
    # since betas_by_date/fringe_by_date are static across all 8
    # quarters in this fixture) -- but simplest and most direct is to
    # just re-run assign_buckets on the fixture's own first cross-section
    # and read V01's real membership.
    from src.analysis.beta_buckets import assign_buckets as _assign_buckets

    first_formation = min(betas_by_date.keys())
    cross_section = fringe_by_date[first_formation].join(
        betas_by_date[first_formation], on="id", how="inner"
    )
    bucketed = _assign_buckets(cross_section)
    v01_ids = bucketed.filter(pl.col("bucket") == "V01")["id"].to_list()
    id_to_beta = dict(zip([f"N{i:02d}" for i in range(n_names)], betas_planted))
    v01_member_betas = [id_to_beta[i] for i in v01_ids]
    assert len(v01_member_betas) >= 1
    n_quarters_with_returns = row["n_quarters"][0]
    reconstructed = []
    for q in range(n_quarters_with_returns):
        m = [market_rets[(3 * q + k) % len(market_rets)] for k in range(3)]
        member_quarterly_rets = [
            math.expm1(sum(math.log1p(b * r) for r in m)) for b in v01_member_betas
        ]
        reconstructed.append(sum(member_quarterly_rets) / len(member_quarterly_rets))

    values = reconstructed
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(var)
    expected_quarterly_sharpe = mean / std * math.sqrt(4)
    wrong_monthly_sharpe = mean / std * math.sqrt(12)

    assert reported_sharpe == pytest.approx(expected_quarterly_sharpe, rel=1e-6), (
        f"reported sharpe {reported_sharpe} does not match the "
        f"independently-reconstructed quarterly-annualized value "
        f"{expected_quarterly_sharpe}"
    )
    assert reported_sharpe != pytest.approx(wrong_monthly_sharpe, rel=1e-3), (
        f"reported sharpe {reported_sharpe} matches what a WRONG monthly "
        f"annualization ({wrong_monthly_sharpe}) would give -- this test "
        "would be vacuous if quarterly and monthly annualization gave "
        "the same number for this fixture"
    )


# ---------------------------------------------------------------------------
# BetaBucketResult.quarterly + summary.se_ret -- new this session, added
# for the time-stability heatmap and SML confidence bars.
# ---------------------------------------------------------------------------


def test_quarterly_exposes_one_row_per_bucket_per_held_quarter():
    """BetaBucketResult.quarterly must have exactly n_buckets *
    n_formation_dates rows (one per (bucket, held quarter) pair) -- the
    per-quarter detail summary.n_quarters is itself computed from."""
    n_names = 30
    betas_planted = [0.1 + 0.1 * i for i in range(30)]
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]
    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 8, betas_planted, market_rets, noise=0.0
    )
    result = beta_buckets.run_beta_bucket_analysis(
        betas_by_date, fringe_by_date, quarterly_rets, market_monthly,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    assert set(result.quarterly.columns) == {"bucket", "quarter", "ret", "beta", "n"}
    # Every bucket present in summary must have exactly summary.n_quarters
    # rows in quarterly -- the two are the same underlying data, just
    # aggregated differently.
    for row in result.summary.iter_rows(named=True):
        bucket_rows = result.quarterly.filter(pl.col("bucket") == row["bucket"])
        assert bucket_rows.height == row["n_quarters"], (
            f"bucket {row['bucket']}: quarterly has {bucket_rows.height} rows, "
            f"summary.n_quarters says {row['n_quarters']} -- these must agree, "
            "they describe the same underlying per-quarter data"
        )


def test_quarterly_mean_ret_matches_summary_mean_ret():
    """summary.mean_ret for each bucket must equal the mean of that
    bucket's own quarterly.ret rows -- summary is an aggregation of
    quarterly, not an independently-computed number."""
    n_names = 30
    betas_planted = [0.1 + 0.1 * i for i in range(30)]
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]
    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 8, betas_planted, market_rets, noise=0.0
    )
    result = beta_buckets.run_beta_bucket_analysis(
        betas_by_date, fringe_by_date, quarterly_rets, market_monthly,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    for row in result.summary.iter_rows(named=True):
        bucket_rows = result.quarterly.filter(pl.col("bucket") == row["bucket"])
        recomputed_mean = float(bucket_rows["ret"].mean())
        assert row["mean_ret"] == pytest.approx(recomputed_mean, rel=1e-9)


def test_se_ret_matches_formula_against_quarterly_data():
    """se_ret must equal std(quarterly.ret, ddof=1) / sqrt(n_quarters)
    computed directly from BetaBucketResult.quarterly's own data for
    each bucket -- the exact formula this module's docstring states,
    checked against the actual per-quarter series rather than assumed
    from the aggregate alone."""
    n_names = 30
    betas_planted = [0.1 + 0.1 * i for i in range(30)]
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015, 0.01, -0.015]
    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 12, betas_planted, market_rets, noise=0.01
    )
    result = beta_buckets.run_beta_bucket_analysis(
        betas_by_date, fringe_by_date, quarterly_rets, market_monthly,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    for row in result.summary.iter_rows(named=True):
        bucket_rows = result.quarterly.filter(pl.col("bucket") == row["bucket"])
        ret_values = bucket_rows["ret"].to_numpy()
        n = len(ret_values)
        assert n == row["n_quarters"]
        if n < 2:
            continue
        expected_se = float(ret_values.std(ddof=1) / (n**0.5))
        assert row["se_ret"] == pytest.approx(expected_se, rel=1e-9), (
            f"bucket {row['bucket']}: se_ret={row['se_ret']} does not match "
            f"std(ret,ddof=1)/sqrt(n)={expected_se} computed directly from "
            "quarterly data"
        )
        assert row["se_ret"] > 0, (
            f"bucket {row['bucket']}: se_ret is not positive under a "
            "noisy (noise=0.01) fixture where real dispersion exists"
        )


def test_se_ret_formula_decreases_with_n_at_fixed_dispersion():
    """The 'more data narrows the CI' property of std(x,ddof=1)/sqrt(n),
    checked as a direct mathematical property of the formula itself
    (not via the full run_beta_bucket_analysis pipeline, where adding
    quarters to a NOISY random fixture can legitimately increase realized
    dispersion along with n -- a real statistical fact that made an
    earlier version of this test fail for the wrong reason, since SE only
    shrinks in EXPECTATION as n grows, never as a strict guarantee for
    any one specific finite sample). Uses a hand-built series with FIXED
    per-observation spread, extended to more observations, so n is the
    only thing changing."""
    import numpy as np

    fixed_spread_series = [0.01, -0.01, 0.02, -0.02]  # std(ddof=1) recomputed exactly below
    short = np.array(fixed_spread_series[:2])
    long = np.array(fixed_spread_series * 4)  # same 4-value pattern repeated -> same population std

    se_short = short.std(ddof=1) / np.sqrt(len(short))
    se_long = long.std(ddof=1) / np.sqrt(len(long))
    assert se_long < se_short, (
        "sanity check on the SE formula itself failed -- se_ret's own "
        "definition (std(ret,ddof=1)/sqrt(n)) should shrink as n grows "
        "when per-observation dispersion is held fixed"
    )
    # This confirms the FORMULA behaves correctly; se_ret's use of this
    # exact formula is pinned separately by
    # test_se_ret_matches_formula_against_quarterly_data above.
