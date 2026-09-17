"""Tests for src/analysis/target_beta.py -- target-beta portfolios,
user-confirmed this session as a distinct view from the percentile-rank
SML buckets (src/analysis/beta_buckets.py).
"""

import datetime

import polars as pl
import pytest

from src.analysis import target_beta


def _cross_section(ids: list[str], betas: list[float], mkt_caps: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"id": ids, "beta": betas, "mkt_cap": mkt_caps})


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
# TARGET_BETAS -- the user-confirmed grid, pinned as a literal so any
# future edit to the generating expression is caught immediately.
# ---------------------------------------------------------------------------


def test_target_betas_matches_user_confirmed_grid():
    assert target_beta.TARGET_BETAS == [
        0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
        1.05, 1.15, 1.25, 1.50, 1.75, 2.00,
    ]


def test_target_betas_has_14_entries():
    assert len(target_beta.TARGET_BETAS) == 14


# ---------------------------------------------------------------------------
# assign_target_beta_groups -- membership size, closest-20 correctness,
# multi-target membership, empty-frame handling.
# ---------------------------------------------------------------------------


def test_assign_target_beta_groups_gives_20_members_per_target():
    """50 names, betas spread 0.1..2.5 -- large enough that every one of
    the 14 targets has at least 20 candidates within range. Each target's
    group must have exactly N_HOLDINGS_PER_TARGET (20) members."""
    n = 50
    ids = [f"N{i:02d}" for i in range(n)]
    betas = [0.1 + (2.5 - 0.1) * i / (n - 1) for i in range(n)]
    mkt_caps = [1e9] * n
    df = _cross_section(ids, betas, mkt_caps)

    result = target_beta.assign_target_beta_groups(df, "beta")
    counts = result.group_by("bucket").agg(pl.len().alias("n"))
    counts_dict = dict(zip(counts["bucket"].to_list(), counts["n"].to_list()))

    assert len(counts_dict) == 14
    for t in target_beta.TARGET_BETAS:
        label = target_beta._target_label(t)
        assert counts_dict[label] == 20, f"{label} has {counts_dict.get(label)} members, expected 20"


def test_assign_target_beta_groups_selects_the_mathematically_closest_20():
    """For target=0.75, the group's members must be EXACTLY the 20 names
    with the smallest |beta - 0.75| -- verified by independently computing
    the true closest-20 set via a plain Python sort and comparing."""
    n = 50
    ids = [f"N{i:02d}" for i in range(n)]
    betas = [0.1 + (2.5 - 0.1) * i / (n - 1) for i in range(n)]
    mkt_caps = [1e9] * n
    df = _cross_section(ids, betas, mkt_caps)

    result = target_beta.assign_target_beta_groups(df, "beta")
    got = set(result.filter(pl.col("bucket") == "T0.75")["id"].to_list())

    true_closest = sorted(zip(ids, betas), key=lambda p: abs(p[1] - 0.75))[:20]
    expected = {p[0] for p in true_closest}

    assert got == expected


def test_assign_target_beta_groups_allows_multi_target_membership():
    """A name sitting between two adjacent targets can be among the
    closest-20 for BOTH -- the shared aggregation core relies on this
    (a name is not exclusively owned by one group, unlike assign_buckets).
    """
    n = 50
    ids = [f"N{i:02d}" for i in range(n)]
    betas = [0.1 + (2.5 - 0.1) * i / (n - 1) for i in range(n)]
    mkt_caps = [1e9] * n
    df = _cross_section(ids, betas, mkt_caps)

    result = target_beta.assign_target_beta_groups(df, "beta")
    membership_counts = result.group_by("id").agg(pl.len().alias("n_targets"))
    max_targets = membership_counts["n_targets"].max()

    assert max_targets is not None and max_targets > 1, (
        "expected at least one name to qualify for more than one target's "
        "closest-20 given a dense, evenly-spaced beta grid -- if this is "
        "1, the function may be accidentally partitioning the universe "
        "exclusively like assign_buckets does, rather than allowing "
        "overlapping membership"
    )


def test_assign_target_beta_groups_output_row_count_exceeds_input_when_overlap_exists():
    """Unlike assign_buckets (exactly one row per input row), this
    function's output can have MORE rows than its input -- 14 targets *
    20 members each, with overlap, on a 50-name universe."""
    n = 50
    ids = [f"N{i:02d}" for i in range(n)]
    betas = [0.1 + (2.5 - 0.1) * i / (n - 1) for i in range(n)]
    mkt_caps = [1e9] * n
    df = _cross_section(ids, betas, mkt_caps)

    result = target_beta.assign_target_beta_groups(df, "beta")
    assert result.height == 14 * 20
    assert result.height > df.height


def test_assign_target_beta_groups_empty_frame_returns_empty_with_bucket_column():
    df = _cross_section([], [], [])
    result = target_beta.assign_target_beta_groups(df, "beta")
    assert result.height == 0
    assert "bucket" in result.columns


def test_assign_target_beta_groups_fewer_than_20_eligible_names_returns_all_of_them():
    """If the cross-section has fewer than 20 names total, every target's
    group must contain all of them (never padded with fabricated rows)."""
    n = 5
    ids = [f"N{i:02d}" for i in range(n)]
    betas = [0.3, 0.5, 0.7, 0.9, 1.1]
    mkt_caps = [1e9] * n
    df = _cross_section(ids, betas, mkt_caps)

    result = target_beta.assign_target_beta_groups(df, "beta")
    for t in target_beta.TARGET_BETAS:
        label = target_beta._target_label(t)
        group = result.filter(pl.col("bucket") == label)
        assert group.height == n, f"{label} has {group.height} members, expected all {n}"


# ---------------------------------------------------------------------------
# Tie-breaking: user-confirmed rule is LARGER mkt_cap wins a tie on
# |beta - target|. Constructed with a genuinely exact float tie (two
# names sharing the identical beta value, so |beta - target| is computed
# identically bit-for-bit for both -- NOT derived from target +/- delta
# arithmetic, which this session confirmed does NOT reliably produce
# exact ties due to IEEE 754 addition/subtraction asymmetry: e.g.
# abs(0.85 - 0.55) == 0.29999999999999993 while
# abs(0.25 - 0.55) == 0.30000000000000004 -- NOT equal despite both
# being "target +/- 0.30").
# ---------------------------------------------------------------------------


def test_assign_target_beta_groups_tie_break_prefers_larger_mkt_cap():
    """19 names clearly closer to target=0.55 than the remaining two
    (N19, N20), which share the IDENTICAL beta value (0.90) -- an exact
    tie on |beta - target| by construction, not by arithmetic coincidence.
    N19 has the larger mkt_cap (5e9 vs 2e9) and must win the 20th spot;
    N20 must be excluded."""
    n_clear = 19
    ids = [f"N{i:02d}" for i in range(n_clear)] + ["N19", "N20"]
    betas = [0.55 + 0.01 * i for i in range(n_clear)] + [0.90, 0.90]
    mkt_caps = [1e9] * n_clear + [5e9, 2e9]
    df = _cross_section(ids, betas, mkt_caps)

    result = target_beta.assign_target_beta_groups(df, "beta")
    t055 = set(result.filter(pl.col("bucket") == "T0.55")["id"].to_list())

    assert "N19" in t055, "larger-mkt_cap name (N19) should win the tie and be included"
    assert "N20" not in t055, "smaller-mkt_cap name (N20) should lose the tie and be excluded"


def test_assign_target_beta_groups_tie_break_is_deterministic_across_calls():
    """The same exact-tie fixture, called repeatedly, must always resolve
    the tie the same way -- not an artifact of polars' internal sort
    stability varying by chance."""
    n_clear = 19
    ids = [f"N{i:02d}" for i in range(n_clear)] + ["N19", "N20"]
    betas = [0.55 + 0.01 * i for i in range(n_clear)] + [0.90, 0.90]
    mkt_caps = [1e9] * n_clear + [5e9, 2e9]
    df = _cross_section(ids, betas, mkt_caps)

    results = []
    for _ in range(5):
        result = target_beta.assign_target_beta_groups(df, "beta")
        t055 = set(result.filter(pl.col("bucket") == "T0.55")["id"].to_list())
        results.append("N19" in t055 and "N20" not in t055)

    assert all(results), "tie-break outcome was not stable across repeated calls"


# ---------------------------------------------------------------------------
# run_target_beta_analysis -- end to end, absolute-anchor zero-noise
# recovery test mirroring test_beta_buckets.py's own pattern.
# ---------------------------------------------------------------------------


def _build_quarterly_fixture(
    n_names: int, n_quarters: int, betas: list[float], market_rets: list[float], noise: float = 0.0
):
    """Same construction as test_beta_buckets.py's own fixture builder --
    n_names each with a KNOWN planted beta, monthly returns EXACTLY
    beta_i * market_ret_of_that_month (zero noise by default)."""
    import random

    rng = random.Random(42)
    quarter_ends = []
    d = datetime.date(2015, 3, 31)
    for _ in range(n_quarters + 1):
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
        betas_by_date[q_end] = pl.DataFrame(
            {"id": ids, "beta": [betas[i % len(betas)] for i in range(n_names)]}
        )
        fringe_by_date[q_end] = _fringe(ids, [5e9] * n_names, [50.0] * n_names)

    quarterly_monthly_rets = _monthly_rets(monthly_rows)
    market_monthly = pl.DataFrame(market_monthly_rows)
    return betas_by_date, fringe_by_date, quarterly_monthly_rets, market_monthly


def test_run_target_beta_analysis_recovers_planted_beta_with_zero_noise():
    """ABSOLUTE ANCHOR (CLAUDE.md): 50 names with betas evenly spaced
    0.1..2.5, monthly returns EXACTLY beta_i * market_ret (zero
    idiosyncratic noise). Each target's realized beta (regressed on
    quarterly-compounded returns) must be close to the target itself --
    not exact, for the same log-compounding-nonlinearity reason
    documented in test_beta_buckets.py's own version of this test."""
    n_names = 50
    betas_planted = [0.1 + (2.5 - 0.1) * i / (n_names - 1) for i in range(n_names)]
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]

    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 8, betas_planted, market_rets, noise=0.0
    )

    result = target_beta.run_target_beta_analysis(
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
            f"target {row['bucket']}: realized_beta={row['realized_beta']} != "
            f"ex_ante_beta={row['ex_ante_beta']} under a ZERO-NOISE fixture"
        )


def test_run_target_beta_analysis_ex_ante_beta_tracks_the_requested_target():
    """The whole point of this module: ex_ante_beta for target T should
    be close to T itself, since the 20 closest names by construction
    average out near T -- PROVIDED the universe has at least 20 names
    genuinely local to T on both sides. Uses 200 names spread 0.0..2.5
    (spacing ~0.0126) rather than 50 (spacing ~0.049): confirmed by
    direct calculation that with only 50 names, a 20-name window's
    minimum-beta edge (0.1) sits within 20 ranks of every target from
    0.25 through 0.55, so those targets' closest-20 sets are IDENTICAL
    (clamped against the grid's own low edge) rather than each centering
    locally on its own target -- a fixture-density artifact, not a
    defect in assign_target_beta_groups (whose closest-20 selection is
    separately verified exactly by
    test_assign_target_beta_groups_selects_the_mathematically_closest_20).
    200 names removes the clamping for every target except the true
    extremes (0.25 near the low edge of 0.0, 2.00 near the high edge of
    2.5), which are excluded below for the same reason."""
    n_names = 200
    betas_planted = [0.0 + 2.5 * i / (n_names - 1) for i in range(n_names)]
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]

    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 8, betas_planted, market_rets, noise=0.0
    )

    result = target_beta.run_target_beta_analysis(
        betas_by_date, fringe_by_date, quarterly_rets, market_monthly,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    label_to_target = {target_beta._target_label(t): t for t in target_beta.TARGET_BETAS}
    edge_targets = {0.25, 2.00}
    checked_any = False
    for row in result.summary.iter_rows(named=True):
        t = label_to_target[row["bucket"]]
        if t in edge_targets:
            continue
        checked_any = True
        assert row["ex_ante_beta"] == pytest.approx(t, abs=0.05), (
            f"{row['bucket']}: ex_ante_beta={row['ex_ante_beta']} not close to "
            f"requested target {t}"
        )
    assert checked_any


def test_run_target_beta_analysis_produces_14_targets():
    n_names = 50
    betas_planted = [0.1 + (2.5 - 0.1) * i / (n_names - 1) for i in range(n_names)]
    market_rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]
    betas_by_date, fringe_by_date, quarterly_rets, market_monthly = _build_quarterly_fixture(
        n_names, 8, betas_planted, market_rets, noise=0.0
    )
    result = target_beta.run_target_beta_analysis(
        betas_by_date, fringe_by_date, quarterly_rets, market_monthly,
        mkt_cap_floor=1e9, price_floor=1.0,
    )
    assert result.summary.height == 14
    assert set(result.summary["bucket"].to_list()) == {
        target_beta._target_label(t) for t in target_beta.TARGET_BETAS
    }
