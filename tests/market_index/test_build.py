"""
Tests for src/market_index/build.py (Phase F2a, docs/03_roadmap.md).

Three market index variants, spec 00_spec.md section 7, all consuming the
common gate_adapters schema (id, date, mkt_cap, ret -- mkt_cap already
LAGGED prior-day-close x shares):

- vw_uncapped: thin wrapper over gate1_index.build_vw_index() (Gate-1-
  validated, wired not rewritten).
- vw_capped_10pct: water-filling 10% single-name cap with pro-rata
  redistribution, iterated to convergence. Required because Nortel reached
  ~28% of the self-built Canadian index in July 2000 (verified against the
  real panel below).
- vw_msci_like: 85% cumulative-coverage large-cap proxy on total mkt_cap
  (no free-float data exists in this project).

Every assertion here must be capable of failing -- see the injected-error
tests, which mirror the project's gate-verifier standard even though this
module isn't a formal "Gate".
"""

import datetime

import polars as pl
import pytest

from src.data.universe_panel import RAW_DATA_DIR
from src.market_index import build


def _two_name_panel(mkt_caps: list[float], rets: list[float], date=None) -> pl.DataFrame:
    date = date or datetime.date(2015, 1, 2)
    return pl.DataFrame(
        {
            "id": ["A", "B"],
            "date": [date, date],
            "mkt_cap": mkt_caps,
            "ret": rets,
        }
    )


# ---------------------------------------------------------------------------
# vw_uncapped -- must be byte-identical to gate1_index.build_vw_index()
# ---------------------------------------------------------------------------


def test_uncapped_matches_gate1_build_vw_index_exactly():
    from src.gates import gate1_index

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
            "mkt_cap": [1000.0, 2000.0, 1100.0, 1900.0],
            "ret": [0.05, -0.02, 0.1, -0.05],
        }
    )
    expected = gate1_index.build_vw_index(panel)
    actual = build.build_index(panel, method="vw_uncapped")
    assert actual["index_ret"].to_list() == pytest.approx(expected["index_ret"].to_list())


# ---------------------------------------------------------------------------
# vw_capped_10pct -- synthetic panels, unit-level cap logic
# ---------------------------------------------------------------------------


def test_capped_index_below_cap_is_unchanged():
    """11 equal-weighted names (~9.09% each) -- nobody exceeds the 10% cap,
    so capping must be a no-op and the capped index must equal the
    uncapped VW average exactly. (A 2-name panel can't test this: with
    only 2 names, one of them is always >= 50% raw weight, guaranteed to
    exceed any 10% cap.)"""
    n = 11
    panel = pl.DataFrame(
        {
            "id": [f"N{i}" for i in range(n)],
            "date": [datetime.date(2015, 1, 2)] * n,
            "mkt_cap": [100.0] * n,
            "ret": [0.10 if i % 2 == 0 else -0.02 for i in range(n)],
        }
    )
    capped = build.build_index(panel, method="vw_capped_10pct")
    uncapped = build.build_index(panel, method="vw_uncapped")
    assert capped["index_ret"][0] == pytest.approx(uncapped["index_ret"][0], rel=1e-9)


def _eleven_name_panel(dominant_mkt_cap: float, rest_mkt_cap_each: float, rets=None):
    """11 names: one dominant name plus 10 equal-sized others -- enough
    names (>= ceil(1/0.10) = 10) for a 10% cap to be FEASIBLE (sum-to-1
    and max-weight<=cap are jointly infeasible below that count -- see
    test_capped_raises_when_cap_infeasible_for_name_count)."""
    n_rest = 10
    ids = ["DOM"] + [f"R{i}" for i in range(n_rest)]
    mkt_caps = [dominant_mkt_cap] + [rest_mkt_cap_each] * n_rest
    if rets is None:
        rets = [1.0] + [0.0] * n_rest
    return pl.DataFrame(
        {
            "id": ids,
            "date": [datetime.date(2015, 1, 2)] * (n_rest + 1),
            "mkt_cap": mkt_caps,
            "ret": rets,
        }
    )


def test_capped_index_clips_dominant_name_to_exactly_cap():
    """DOM is 90% of mkt_cap pre-cap (10 other equal names split the other
    10%) -- must be clipped to exactly 10%, with the other 10 names
    absorbing the redistributed excess pro-rata."""
    panel = _eleven_name_panel(dominant_mkt_cap=900.0, rest_mkt_cap_each=10.0)
    result = build.build_index(panel, method="vw_capped_10pct")
    # DOM capped at exactly 10% with ret=1.0; the other 90% of weight sits
    # on names with ret=0.0 -- index_ret is the weighted average using
    # POST-CAP weights, not the raw ones.
    expected = 0.10 * 1.0 + 0.90 * 0.0
    assert result["index_ret"][0] == pytest.approx(expected, rel=1e-9)


def test_capped_weights_sum_to_one_after_capping():
    """Direct invariant check via the internal weight function, not just
    the aggregated index return -- confirms capping doesn't silently lose
    or double-count weight mass."""
    panel = _eleven_name_panel(dominant_mkt_cap=900.0, rest_mkt_cap_each=10.0)
    weights = build._capped_weights(panel, cap=0.10, max_iterations=50)
    total = weights.select(pl.col("weight").sum()).item()
    assert total == pytest.approx(1.0, abs=1e-9)


def test_capped_no_name_exceeds_cap_after_capping():
    panel = _eleven_name_panel(dominant_mkt_cap=900.0, rest_mkt_cap_each=10.0)
    weights = build._capped_weights(panel, cap=0.10, max_iterations=50)
    assert weights["weight"].max() <= 0.10 + 1e-9


def test_capped_redistribution_is_pro_rata_not_uniform():
    """DOM at 80% raw weight, capped to cap=0.70 (releasing 10pp of
    excess), plus two UNEQUAL uncapped sink names: BIG at 15% and SMALL at
    5% (3:1 ratio), each comfortably under 0.70 even after absorbing a
    share of the 10pp excess. A large cap (0.70, not the project's 10%) is
    used purely so an uncapped-and-unequal sink pair can exist without
    itself breaching the cap after redistribution -- this test is about
    the redistribution ARITHMETIC in isolation, not about the project's
    real 10% cap value (that's covered by the Nortel anchor and the other
    capped_10pct tests above).

    This is the discriminating case a uniform-redistribution bug would NOT
    be caught by: with only equal-sized uncapped names elsewhere in this
    file, pro-rata and uniform redistribution happen to produce identical
    numbers, so this test exists specifically because those tests would
    pass even if `weight` proportionality were silently dropped from the
    redistribution formula.
    """
    panel = pl.DataFrame(
        {
            "id": ["DOM", "BIG", "SMALL"],
            "date": [datetime.date(2015, 1, 2)] * 3,
            "mkt_cap": [800.0, 150.0, 50.0],  # raw weights: 0.80 / 0.15 / 0.05
            "ret": [1.0, 1.0, 1.0],
        }
    )
    cap = 0.70
    weights = build._capped_weights(panel, cap=cap, max_iterations=50)
    dom_weight = weights.filter(pl.col("id") == "DOM")["weight"][0]
    big_weight = weights.filter(pl.col("id") == "BIG")["weight"][0]
    small_weight = weights.filter(pl.col("id") == "SMALL")["weight"][0]
    assert dom_weight == pytest.approx(cap, abs=1e-9)
    # BIG:SMALL raw ratio is 150:50 = 3:1 -- pro-rata redistribution must
    # preserve that ratio exactly. A uniform (per-head) split would instead
    # move the two towards equal shares (1:1).
    assert big_weight / small_weight == pytest.approx(3.0, rel=1e-6)
    assert weights.select(pl.col("weight").sum()).item() == pytest.approx(1.0, abs=1e-9)


def test_capped_raises_when_cap_infeasible_for_name_count():
    """Only 2 names, 90/10 split: capping the dominant name to 10% would
    require the other name to absorb 90%, itself breaching the cap -- with
    only 2 names there is no third name to redistribute to, so sum-to-1
    and max-weight<=10% are jointly infeasible. Must raise, not silently
    return a result where one name exceeds the cap (CLAUDE.md: a wrong
    number that looks right is the worst possible outcome). Real Canadian
    dates always have far more than 10 names, so this never fires in
    production -- it's a synthetic-edge-case guard."""
    panel = _two_name_panel(mkt_caps=[900.0, 100.0], rets=[1.0, 0.0])
    with pytest.raises(RuntimeError):
        build._capped_weights(panel, cap=0.10, max_iterations=50)


def _twelve_name_panel_two_dominant():
    """12 names: A=40%, B=40%, and 10 equal others splitting the remaining
    20% (2% each). Enough names (12 >= ceil(1/0.10) = 10) for a 10% cap to
    be feasible even with two simultaneous breaches."""
    n_rest = 10
    ids = ["A", "B"] + [f"R{i}" for i in range(n_rest)]
    mkt_caps = [400.0, 400.0] + [20.0] * n_rest
    rets = [1.0, 1.0] + [0.0] * n_rest
    return pl.DataFrame(
        {
            "id": ids,
            "date": [datetime.date(2015, 1, 2)] * (n_rest + 2),
            "mkt_cap": mkt_caps,
            "ret": rets,
        }
    )


def test_capped_iterative_convergence_multiple_names_breach_simultaneously():
    """A and B both at 40% raw weight (12 names total, 10 small others
    splitting the remaining 20%) -- capping A and B to 10% each releases
    60pp total, redistributed pro-rata across the 10 small names. This
    exercises two simultaneous breaches in one pass."""
    panel = _twelve_name_panel_two_dominant()
    weights = build._capped_weights(panel, cap=0.10, max_iterations=50)
    assert weights["weight"].max() <= 0.10 + 1e-9
    assert weights.select(pl.col("weight").sum()).item() == pytest.approx(1.0, abs=1e-9)
    a_weight = weights.filter(pl.col("id") == "A")["weight"][0]
    b_weight = weights.filter(pl.col("id") == "B")["weight"][0]
    assert a_weight == pytest.approx(0.10, abs=1e-9)
    assert b_weight == pytest.approx(0.10, abs=1e-9)
    # The 10 small names absorb the 60pp excess equally: each started at
    # 2%, ends at 2% + 60%/10 = 8% -- still under cap, no second pass
    # needed for THIS panel (the forced-second-pass case is covered
    # separately below).
    r0_weight = weights.filter(pl.col("id") == "R0")["weight"][0]
    assert r0_weight == pytest.approx(0.08, abs=1e-9)


def _fourteen_name_panel_forces_second_pass():
    """14 names: A=70%, B=15%, C=10%, and 11 equal others splitting the
    remaining 5% (~0.4545% each). Capping A to 10% releases 60pp,
    redistributed pro-rata across B/C/the 11 others (weights 15/10/5, i.e.
    50%/33.3%/16.7% of the uncapped 30% pool) -- B's new weight would be
    15% + 0.50*60% = 45% in a SINGLE pass, itself exceeding 10% and
    requiring a second capping pass. Enough total names (14 >= 10) for the
    cap to remain feasible throughout."""
    n_rest = 11
    ids = ["A", "B", "C"] + [f"R{i}" for i in range(n_rest)]
    mkt_caps = [70.0, 15.0, 10.0] + [5.0 / n_rest] * n_rest
    rets = [1.0, 1.0, 1.0] + [0.0] * n_rest
    return pl.DataFrame(
        {
            "id": ids,
            "date": [datetime.date(2015, 1, 2)] * (n_rest + 3),
            "mkt_cap": mkt_caps,
            "ret": rets,
        }
    )


def test_capped_iterative_convergence_forces_second_pass():
    """A single redistribution pass would leave B at 45%, violating the
    cap -- this is the exact failure mode the iterative (not single-pass)
    design exists to catch."""
    panel = _fourteen_name_panel_forces_second_pass()
    weights = build._capped_weights(panel, cap=0.10, max_iterations=50)
    assert weights["weight"].max() <= 0.10 + 1e-9, (
        "a name exceeded the cap after redistribution -- single-pass "
        "redistribution is insufficient here"
    )
    assert weights.select(pl.col("weight").sum()).item() == pytest.approx(1.0, abs=1e-9)
    a_weight = weights.filter(pl.col("id") == "A")["weight"][0]
    b_weight = weights.filter(pl.col("id") == "B")["weight"][0]
    assert a_weight == pytest.approx(0.10, abs=1e-9)
    assert b_weight == pytest.approx(0.10, abs=1e-9), (
        "B should be capped in a SECOND pass after absorbing excess from A "
        "pushed it past 10% in the first pass"
    )


def test_capped_raises_if_iteration_guard_exhausted():
    """max_iterations too low to converge must raise, not silently return
    an out-of-spec result -- CLAUDE.md: a wrong number that looks right is
    the worst possible outcome."""
    panel = _fourteen_name_panel_forces_second_pass()
    with pytest.raises(RuntimeError):
        build._capped_weights(panel, cap=0.10, max_iterations=0)


def test_capped_naive_single_pass_redistribution_would_fail_the_cap_invariant():
    """Proves the max-weight-<=cap assertion is CAPABLE of failing --
    CLAUDE.md's gate-verifier standard applied to this module. Simulates
    the single-pass (non-iterative) redistribution directly and shows it
    produces a name above the cap on the same panel that forces a second
    pass above, i.e. the exact injected defect the iterative design
    guards against."""
    panel = _fourteen_name_panel_forces_second_pass()
    single_pass = build._single_pass_cap_and_redistribute(panel, cap=0.10)
    assert single_pass["weight"].max() > 0.10, (
        "expected the single-pass injection to violate the cap -- if it "
        "doesn't, the iterative test above isn't actually discriminating"
    )


# ---------------------------------------------------------------------------
# vw_msci_like -- coverage cutoff
# ---------------------------------------------------------------------------


def test_msci_like_keeps_largest_names_until_coverage_target():
    """Four names, weights 40/30/20/10. Cumulative from largest: 40, 70,
    90, 100. Target 0.85 is crossed at the third name (90% >= 85%), so
    exactly three names are kept -- the smallest (10%) is dropped."""
    panel = pl.DataFrame(
        {
            "id": ["A", "B", "C", "D"],
            "date": [datetime.date(2015, 1, 2)] * 4,
            "mkt_cap": [400.0, 300.0, 200.0, 100.0],
            "ret": [1.0, 1.0, 1.0, 0.0],
        }
    )
    kept = build._msci_like_weights(panel, target_cumulative_weight=0.85)
    assert set(kept["id"].to_list()) == {"A", "B", "C"}


def test_msci_like_renormalizes_kept_subset_to_sum_one():
    panel = pl.DataFrame(
        {
            "id": ["A", "B", "C", "D"],
            "date": [datetime.date(2015, 1, 2)] * 4,
            "mkt_cap": [400.0, 300.0, 200.0, 100.0],
            "ret": [1.0, 1.0, 1.0, 0.0],
        }
    )
    kept = build._msci_like_weights(panel, target_cumulative_weight=0.85)
    assert kept.select(pl.col("weight").sum()).item() == pytest.approx(1.0, abs=1e-9)


def test_msci_like_excludes_smallest_name_return_from_index():
    """D (weight 10%, ret 0.0) must be EXCLUDED -- if it leaked in, the
    index return would be pulled toward 0 relative to the all-1.0 kept
    subset. This is the assertion's failure-mode check: an index that
    accidentally includes D would read 0.9*1.0 + 0.1*0.0 = 0.9 instead of
    the correct 1.0."""
    panel = pl.DataFrame(
        {
            "id": ["A", "B", "C", "D"],
            "date": [datetime.date(2015, 1, 2)] * 4,
            "mkt_cap": [400.0, 300.0, 200.0, 100.0],
            "ret": [1.0, 1.0, 1.0, 0.0],
        }
    )
    result = build.build_index(panel, method="vw_msci_like")
    assert result["index_ret"][0] == pytest.approx(1.0, abs=1e-9)


def test_msci_like_cross_sectional_not_pooled():
    """The kept-name set must be recomputed PER DATE, not fixed from a
    full-panel ranking (CLAUDE.md: no filtering on full-sample properties).
    Date 1: D is largest and must be kept. Date 2: D is smallest and must
    be dropped, even though it was kept on date 1."""
    panel = pl.DataFrame(
        {
            "id": ["A", "B", "C", "D"] * 2,
            "date": [datetime.date(2015, 1, 2)] * 4 + [datetime.date(2015, 1, 5)] * 4,
            "mkt_cap": [10.0, 10.0, 10.0, 400.0] + [400.0, 300.0, 200.0, 10.0],
            "ret": [0.0, 0.0, 0.0, 1.0] + [1.0, 1.0, 1.0, 0.0],
        }
    )
    result = build.build_index(panel, method="vw_msci_like").sort("date")
    day1 = result.filter(pl.col("date") == datetime.date(2015, 1, 2))["index_ret"][0]
    day2 = result.filter(pl.col("date") == datetime.date(2015, 1, 5))["index_ret"][0]
    assert day1 == pytest.approx(1.0, abs=1e-9), "D should be kept (largest) on day 1"
    assert day2 == pytest.approx(1.0, abs=1e-9), "D should be dropped (smallest) on day 2"


# ---------------------------------------------------------------------------
# Null ret with non-null mkt_cap -- leakage-auditor finding (Phase F2a):
# gate1_index.build_vw_index()'s own docstring calls this "particularly
# dangerous" (a null ret with non-null mkt_cap must be excluded and the
# REST renormalized to sum to 1 -- never treated as a silent 0% return
# while still counting the name's full weight mass). _raw_weights() must
# match that contract for capped/msci_like, not just inherit it via the
# vw_uncapped passthrough.
# ---------------------------------------------------------------------------


def test_capped_excludes_null_ret_name_and_renormalizes_rest():
    """11 names (feasible at cap=0.10): one has a non-null mkt_cap but a
    NULL ret. That name must be excluded entirely -- not counted in the
    weight base at all -- and the other 10 renormalized to sum to 1 among
    themselves. If it leaked in with its mkt_cap weight intact but a
    silently-zeroed return, the index would be pulled toward 0 relative to
    the correct all-nonzero-return result."""
    n_rest = 10
    panel = pl.DataFrame(
        {
            "id": ["NULLRET"] + [f"R{i}" for i in range(n_rest)],
            "date": [datetime.date(2015, 1, 2)] * (n_rest + 1),
            "mkt_cap": [50.0] + [50.0] * n_rest,
            "ret": [None] + [1.0] * n_rest,
        }
    )
    weights = build._capped_weights(panel, cap=0.10, max_iterations=50)
    assert "NULLRET" not in weights["id"].to_list()
    assert weights.select(pl.col("weight").sum()).item() == pytest.approx(1.0, abs=1e-9)

    result = build.build_index(panel, method="vw_capped_10pct")
    # All 10 remaining names have ret=1.0 -- the index must equal 1.0
    # exactly. If NULLRET's weight leaked in with an effective 0% return,
    # this would read ~0.909 (10/11) instead.
    assert result["index_ret"][0] == pytest.approx(1.0, abs=1e-9)


def test_msci_like_excludes_null_ret_name_and_renormalizes_rest():
    panel = pl.DataFrame(
        {
            "id": ["NULLRET", "A", "B", "C"],
            "date": [datetime.date(2015, 1, 2)] * 4,
            "mkt_cap": [400.0, 300.0, 200.0, 100.0],
            "ret": [None, 1.0, 1.0, 1.0],
        }
    )
    kept = build._msci_like_weights(panel, target_cumulative_weight=0.85)
    assert "NULLRET" not in kept["id"].to_list()

    result = build.build_index(panel, method="vw_msci_like")
    assert result["index_ret"][0] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Config-driven, not hardcoded (CLAUDE.md: no magic numbers in src/)
# ---------------------------------------------------------------------------


def test_build_index_reads_cap_from_config_not_hardcoded():
    """A custom, non-default cap value must actually change the result --
    proves cap is read from the passed config rather than a hardcoded 0.10
    inside build.py. Uses the 11-name panel (feasible at the default 10%
    cap); a 2-name panel would be infeasible at 10% (see
    test_capped_raises_when_cap_infeasible_for_name_count) so it can't
    serve as the default-cap side of this comparison."""
    panel = _eleven_name_panel(dominant_mkt_cap=900.0, rest_mkt_cap_each=10.0)
    default_cap = build.build_index(panel, method="vw_capped_10pct")
    custom_cap = build.build_index(
        panel, method="vw_capped_10pct", overrides={"cap": 0.50}
    )
    assert default_cap["index_ret"][0] != pytest.approx(custom_cap["index_ret"][0])
    # At cap=0.50, DOM (90% raw) is clipped to exactly 50%; the other 50%
    # spreads across the 10 equal-weighted names (all ret=0.0), none of
    # which individually exceeds 50%, so no further capping occurs.
    expected_custom = 0.50 * 1.0 + 0.50 * 0.0
    assert custom_cap["index_ret"][0] == pytest.approx(expected_custom, rel=1e-9)


def test_unknown_method_raises():
    panel = _two_name_panel(mkt_caps=[900.0, 100.0], rets=[1.0, 0.0])
    with pytest.raises(KeyError):
        build.build_index(panel, method="not_a_real_index")


# ---------------------------------------------------------------------------
# Real-data anchor -- Nortel, 2000-07-27 (verified directly against the
# Canadian panel: pre-cap weight ~0.2794, matching spec's "roughly a third"
# claim). Skipped if raw CHASS/CRSP data isn't present, matching the
# existing tests/gates/ convention.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/ not present",
)


def test_nortel_2000_07_27_is_capped_to_exactly_10pct():
    from src.data import gate_adapters

    panel = gate_adapters.canada_gate1_panel(
        datetime.date(2000, 7, 1), datetime.date(2000, 7, 31)
    )
    target_date = datetime.date(2000, 7, 27)
    day_panel = panel.filter(pl.col("date") == target_date)
    assert day_panel.height > 0, "expected trading data on 2000-07-27"

    total_mkt_cap = day_panel.select(pl.col("mkt_cap").sum()).item()
    nortel_mkt_cap = day_panel.filter(pl.col("id") == "NT_2").select("mkt_cap").item()
    pre_cap_weight = nortel_mkt_cap / total_mkt_cap
    assert pre_cap_weight > 0.10, (
        f"expected Nortel's pre-cap weight to exceed 10% on {target_date}, "
        f"got {pre_cap_weight:.4f} -- anchor date may need updating"
    )
    assert pre_cap_weight == pytest.approx(0.2794, abs=0.01)

    capped_weights = build._capped_weights(day_panel, cap=0.10, max_iterations=50)
    nortel_post_cap = capped_weights.filter(pl.col("id") == "NT_2")["weight"][0]
    assert nortel_post_cap == pytest.approx(0.10, abs=1e-6)
    assert capped_weights.select(pl.col("weight").sum()).item() == pytest.approx(
        1.0, abs=1e-9
    )


# ---------------------------------------------------------------------------
# build_index_chunked -- added 2026-09-13 for Gate 4 (docs/02_validation_
# gates.md): a 56-year US index built via a single us_gate1_panel() call
# crashes on the final per-name concat (see that function's own docstring).
# build_index_chunked processes one calendar year at a time, keeping only
# the small per-date index rows. See build_index_chunked's own docstring
# for why this duplicates (temporarily) us_gate1_panel's own chunking.
# ---------------------------------------------------------------------------


def test_build_index_chunked_matches_unchunked_reference():
    """The ONE thing chunking could get wrong: us_gate1_panel(chunk_start,
    chunk_end) is called with each FULL CALENDAR YEAR as its own range,
    relying on that function's own boundary-carry (already tested in
    tests/unit/test_gate_adapters.py) to resolve a correct lagged mkt_cap
    across each year boundary. A single-year span cannot exercise this
    (there is only one chunk) -- this test spans THREE full calendar
    years (2015-2017) specifically to cross two year boundaries, and
    asserts the chunked and unchunked builds are bit-identical, not
    merely close."""
    from src.data import gate_adapters

    start_date, end_date = datetime.date(2015, 1, 1), datetime.date(2017, 12, 31)

    unchunked_panel = gate_adapters.us_gate1_panel(start_date, end_date)
    unchunked_index = build.build_index(unchunked_panel, "vw_uncapped")

    chunked_index = build.build_index_chunked(
        "us", start_date, end_date, "vw_uncapped"
    )

    assert chunked_index.height > 0, "test setup: expected a non-empty real index"
    assert chunked_index.height == unchunked_index.height, (
        f"chunked index has {chunked_index.height} rows, unchunked has "
        f"{unchunked_index.height} -- a year boundary is dropping or "
        "duplicating dates"
    )

    from polars.testing import assert_frame_equal

    assert_frame_equal(
        chunked_index.sort("date"), unchunked_index.sort("date"), check_row_order=True
    )


def test_build_index_chunked_raises_on_unknown_leg():
    with pytest.raises(KeyError):
        build.build_index_chunked(
            "mars", datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), "vw_uncapped"
        )


def test_build_index_chunked_first_trading_day_of_year_has_no_gap():
    """Direct check on the failure mode a broken year-boundary handoff
    would cause: the first trading day of a chunked-in year must have a
    real index_ret, not be silently absent because that day's names
    couldn't resolve a lagged mkt_cap across the boundary."""
    chunked_index = build.build_index_chunked(
        "us", datetime.date(2015, 1, 1), datetime.date(2016, 12, 31), "vw_uncapped"
    )
    dates_2016 = chunked_index.filter(pl.col("date").dt.year() == 2016)["date"]
    assert dates_2016.len() > 0, "expected real 2016 dates in the chunked index"
    first_day_2016 = dates_2016.min()
    row = chunked_index.filter(pl.col("date") == first_day_2016)
    assert row.height == 1
    assert row["index_ret"][0] is not None, (
        f"index_ret is null on {first_day_2016}, the first trading day of "
        "2016 -- the year-boundary chunk handoff lost this date's coverage"
    )
