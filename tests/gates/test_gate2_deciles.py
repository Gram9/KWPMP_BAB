"""Tests for src/gates/gate2_deciles.py -- Gate 2
(docs/superpowers/specs/2026-09-08-gate2-size-deciles-design.md):
build Fama-French-style value-weighted size decile portfolios from our
own CRSP-primary universe/market-cap data and correlate against Ken
French's own published size-decile series.
"""

import datetime

import polars as pl
import pytest

from src.data.universe_panel import RAW_DATA_DIR
from src.gates import gate2_deciles

pytestmark = pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason="data/raw/us_panel_crsp_full/ not present",
)


def test_assign_deciles_buckets_by_nyse_breakpoints():
    """Fabricated breakpoints (via monkeypatch of nyse_breakpoints_at)
    and fabricated same-day market caps -- assert each permno lands in
    the correct 1-10 bucket, including a value exactly on a breakpoint
    (must fall in the LOWER decile per standard convention: a value
    equal to a cutpoint is <= that cutpoint, not > it)."""

    def fake_breakpoints(month_end):
        # 9 cutpoints -> 10 buckets: (-inf,10], (10,20], (20,30], ...,
        # (80,90], (90, inf).
        return [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    def fake_same_day_mkt_cap(month_end):
        return pl.DataFrame(
            {
                "permno": [1, 2, 3, 4, 5],
                "mkt_cap": [5.0, 10.0, 15.0, 95.0, 100.0],
            }
        )

    def fake_universe_at(month_end):
        # One permco per permno (no share-class collapsing) -- this
        # fixture is about breakpoint bucketing, not permco aggregation,
        # which test_assign_deciles_collapses_share_classes_by_permco
        # covers separately.
        return pl.DataFrame({"permno": [1, 2, 3, 4, 5], "permco": [1, 2, 3, 4, 5]})

    import src.gates.gate2_deciles as module

    orig_bp = module.nyse_breakpoints_at
    orig_cap = module._same_day_mkt_cap_at
    orig_universe = module.universe_panel.universe_at
    try:
        module.nyse_breakpoints_at = fake_breakpoints
        module._same_day_mkt_cap_at = fake_same_day_mkt_cap
        module.universe_panel.universe_at = fake_universe_at

        result = module.assign_deciles(datetime.date(2015, 6, 30))
    finally:
        module.nyse_breakpoints_at = orig_bp
        module._same_day_mkt_cap_at = orig_cap
        module.universe_panel.universe_at = orig_universe

    result = result.sort("permno")
    # mkt_caps [5, 10, 15, 95, 100] against breakpoints
    # [10,20,...,90]: 5<=10 -> decile 1; 10<=10 -> decile 1 (on-breakpoint,
    # lower decile per docstring); 15<=20 -> decile 2; 95>90 -> decile 10
    # (bucket 9 is (80,90], bucket 10 is (90,inf), per this test's own
    # fake_breakpoints comment); 100>90 -> decile 10.
    assert result["decile"].to_list() == [1, 1, 2, 10, 10]


def test_assign_deciles_collapses_share_classes_by_permco():
    """Two permnos sharing one permco (a Berkshire-A/B-style multi-share-
    class company) must be assigned the SAME decile, based on their
    SUMMED market cap -- not each permno's own individual cap. This is
    the exact bug Phase C fixes: aggregating by permno alone would put
    a low-cap share class in a low decile and a high-cap share class in
    a high decile, even though they're one company."""

    def fake_breakpoints(month_end):
        return [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    def fake_same_day_mkt_cap(month_end):
        # permno 1 and 2 share permco 100: individually 6.0 and 7.0 --
        # EACH ALONE falls in decile 1 (<=10.0) -- but SUMMED (13.0) falls
        # in decile 2 ((10,20]). A permno-level (unfixed) implementation
        # would assign both permno 1 and 2 to decile 1; the correct
        # permco-level implementation assigns both to decile 2. This is
        # the discriminating case: it fails under the old per-security
        # aggregation and passes only once shares classes are summed.
        # permno 3/4 (permco 200/300, single-class companies) are
        # unaffected controls, included to confirm the fix doesn't
        # disturb ordinary single-share-class assignment.
        return pl.DataFrame(
            {
                "permno": [1, 2, 3, 4],
                "mkt_cap": [6.0, 7.0, 60.0, 45.0],
            }
        )

    def fake_universe_at(month_end):
        return pl.DataFrame(
            {
                "permno": [1, 2, 3, 4],
                "permco": [100, 100, 200, 300],
            }
        )

    import src.gates.gate2_deciles as module

    orig_bp = module.nyse_breakpoints_at
    orig_cap = module._same_day_mkt_cap_at
    orig_universe = module.universe_panel.universe_at
    try:
        module.nyse_breakpoints_at = fake_breakpoints
        module._same_day_mkt_cap_at = fake_same_day_mkt_cap
        module.universe_panel.universe_at = fake_universe_at

        result = module.assign_deciles(datetime.date(2015, 6, 30))
    finally:
        module.nyse_breakpoints_at = orig_bp
        module._same_day_mkt_cap_at = orig_cap
        module.universe_panel.universe_at = orig_universe

    result = result.sort("permno")
    # permco 100 (permnos 1+2): summed cap 13.0, in (10,20] -> decile 2 for
    # BOTH -- neither permno's OWN cap (6.0, 7.0) crosses the 10.0
    # breakpoint alone, only their sum does. A permno-level (unfixed)
    # implementation would put both in decile 1 instead.
    # permco 200 (permno 3): cap 60.0, in (50,60] -> decile 6.
    # permco 300 (permno 4): cap 45.0, in (40,50] -> decile 5.
    assert result["permno"].to_list() == [1, 2, 3, 4]
    assert result["decile"].to_list() == [2, 2, 6, 5]


def test_nyse_breakpoints_at_2015_06_30_is_ascending_and_real():
    """Real-data smoke test: 9 cutpoints, strictly ascending (a real
    invariant -- decile boundaries must never tie or invert), all
    positive (market caps can't be negative)."""
    breakpoints = gate2_deciles.nyse_breakpoints_at(datetime.date(2015, 6, 30))
    assert len(breakpoints) == 9
    assert all(b > 0 for b in breakpoints)
    assert breakpoints == sorted(breakpoints)
    assert len(set(breakpoints)) == 9  # strictly ascending, no ties


def test_assign_deciles_2015_06_30_covers_full_major_exchange_universe():
    """Every permno eligible per universe_at() gets a decile 1-10 --
    the NYSE-only breakpoint computation must not itself restrict WHO
    gets assigned, only what cutpoints are used."""
    from src.data import universe_panel

    eligible = universe_panel.universe_at(datetime.date(2015, 6, 30))
    assigned = gate2_deciles.assign_deciles(datetime.date(2015, 6, 30))

    assert set(assigned["decile"].unique().to_list()) <= set(range(1, 11))
    # Every assigned permno must be a real eligible permno (no phantom rows).
    assert set(assigned["permno"].to_list()) <= set(eligible["permno"].to_list())


def test_build_decile_membership_holds_assignment_for_full_year():
    """A permno assigned decile 3 at 2015-06-30 must appear tagged for
    EVERY holding month July 2015 through June 2016 (12 months), not
    just July."""
    membership = gate2_deciles.build_decile_membership(start_year=2015, end_year=2015)

    holding_months = membership.filter(pl.col("_holding_year").is_in([2015, 2016])).select(
        ["_holding_year", "_holding_month"]
    ).unique().sort(["_holding_year", "_holding_month"])

    expected = [(2015, m) for m in range(7, 13)] + [(2016, m) for m in range(1, 7)]
    actual = list(
        zip(holding_months["_holding_year"].to_list(), holding_months["_holding_month"].to_list())
    )
    assert actual == expected


def test_build_decile_daily_panel_has_common_schema_plus_decile():
    panel = gate2_deciles.build_decile_daily_panel(
        start_date=datetime.date(2015, 7, 1), end_date=datetime.date(2015, 7, 31)
    )
    assert set(panel.columns) == {"id", "date", "price", "shares", "mkt_cap", "ret", "decile"}
    assert panel.height > 0
    assert panel["decile"].null_count() == 0
    assert panel["mkt_cap"].null_count() == 0  # same null-exclusion discipline as Gate 1


def test_build_decile_daily_panel_mkt_cap_is_lagged_not_same_day():
    """Strong check distinguishing this from Task 4's same-day snapshot:
    the DAILY panel's mkt_cap on 2015-07-01 for permno 10104 must equal
    the PRIOR trading day's (2015-06-30) own price x shares x
    SHROUT_UNITS_MULTIPLIER, independently recomputed here by querying
    the raw parquet directly (same technique as
    test_us_panel_mkt_cap_uses_immediately_prior_trading_day in
    tests/unit/test_gate_adapters.py) -- not just "not equal to a naive
    same-day computation" (that older, weaker assertion would pass even
    under a regressed shift(0) if the spot-checked row's price happened
    to be flat day-over-day; confirmed here that 10104's price actually
    moved between 2015-06-29 (40.42), 2015-06-30 (40.30), and 2015-07-01
    (40.24), so a same-day-vs-lagged mixup is guaranteed to be caught,
    not just probably caught)."""
    from src.data.universe_panel import RAW_DATA_DIR, SHROUT_UNITS_MULTIPLIER

    prior_day_row = (
        pl.scan_parquet(str(RAW_DATA_DIR / "year=2015" / "part.parquet"))
        .select(["permno", "dlycaldt", "dlyprc", "shrout"])
        .filter(
            (pl.col("permno") == 10104)
            & (pl.col("dlycaldt") == pl.lit(datetime.date(2015, 6, 30)).cast(pl.Datetime("ns")))
        )
        .collect()
    )
    assert prior_day_row.height == 1
    expected_mkt_cap = (
        abs(prior_day_row["dlyprc"][0]) * prior_day_row["shrout"][0] * SHROUT_UNITS_MULTIPLIER
    )

    panel = gate2_deciles.build_decile_daily_panel(
        start_date=datetime.date(2015, 7, 1), end_date=datetime.date(2015, 7, 31)
    )
    sample = panel.filter(
        (pl.col("id") == "10104") & (pl.col("date") == datetime.date(2015, 7, 1))
    )
    assert sample.height == 1
    assert sample["mkt_cap"][0] == pytest.approx(expected_mkt_cap, rel=1e-9)


def test_run_gate2_us_reports_correlation_for_all_ten_deciles():
    """The actual Gate 2 run: build decile VW indices from real
    2015-07 through 2016-06 US data (one full FF formation year),
    compare against Ken French's own published series.

    **Corrected 2026-09-09 (gate-verifier review, Phase C):** raised from
    0.98 to the ACTUAL documented gate criterion in docs/02_validation_gates.md
    ("Expect > 0.99") -- the prior 0.98 threshold was a documented-criterion
    mismatch with no stated justification for the gap, not a deliberate
    margin decision. Live per-decile minimum measured this session: 0.9964
    (decile 2), still 3.6x the distance from 0.99 that 0.98 gave, so this
    costs no real margin.

    Correlation alone does NOT establish this gate -- see the per-decile
    absolute-diff-bound test below, and note (gate-verifier finding,
    2026-09-09) that correlation is proven BLIND to the permno/permco
    defect this gate exists to catch: injecting the historical defect
    (reverting to permno-level aggregation) moves decile-1 correlation by
    only ~5.6e-4 (0.99636 defective vs 0.99692 fixed), a factor of ~30
    below this test's own margin. Correlation here catches gross
    decile-assignment scrambling or date misalignment; it does not catch
    the defect this gate was retracted over. See
    test_run_gate2_us_breakpoint_berkshire_external_anchor below for the
    assertion that actually does."""
    result = gate2_deciles.run_gate2_us(
        start_date=datetime.date(2015, 7, 1), end_date=datetime.date(2016, 6, 30)
    )
    assert set(result["correlation_by_decile"].keys()) == set(range(1, 11))
    for decile, correlation in result["correlation_by_decile"].items():
        assert correlation > 0.99, (
            f"Gate 2 decile {decile} correlation {correlation} is below "
            "the 0.99 credibility threshold documented in "
            "docs/02_validation_gates.md -- see that doc's Gate 1 debug "
            "order for the general failure-mode checklist (unlagged "
            "weights first, then ret vs retx, then delisting handling, "
            "then weighting)."
        )


def test_run_gate2_us_reports_no_date_alignment_gaps():
    result = gate2_deciles.run_gate2_us(
        start_date=datetime.date(2015, 7, 1), end_date=datetime.date(2016, 6, 30)
    )
    for decile in range(1, 11):
        assert result["n_index_only_by_decile"][decile] == 0, (
            f"decile {decile} has dates in our own index missing from "
            "Ken French's series -- date-alignment bug"
        )
        # gate-verifier finding, 2026-09-09: n_dates_by_decile was
        # computed and returned but never asserted on -- a defect that
        # halved a decile's panel would leave n_index_only at 0 (nothing
        # in OUR index is missing from FF's side) and correlation on the
        # remaining half would still likely clear 0.99. This is a free,
        # meaningful row-count anchor closing that gap.
        assert result["n_dates_by_decile"][decile] == 253, (
            f"decile {decile} has {result['n_dates_by_decile'][decile]} "
            "dates in the comparison, expected 253 (full 2015-07 through "
            "2016-06 trading calendar) -- a silent panel truncation would "
            "show up here without necessarily moving correlation."
        )


def test_run_gate2_us_per_decile_absolute_diff_bounds():
    """Correlation is location- and scale-invariant (docs/02_validation_gates.md's
    cross-cutting finding) and gate-verifier confirmed directly (2026-09-09)
    that it is specifically blind to the permno/permco defect this gate
    exists to catch -- injecting that defect moves decile-1 correlation by
    only ~5.6e-4 against a much larger margin. Gate 1 already carries
    signed/mean-abs/max-abs bounds (tests/gates/test_gate1_index.py); Gate 2
    did not, despite Phase B's own summary claiming bounds were added "on
    both Gate 1 legs" -- Gate 2 was never actually included. This closes
    that gap with the same three-statistic pattern, per decile.

    Thresholds derived from the measured values on the corrected,
    permco-aggregated panel (2026-09-09): signed mean ranges
    -5.99e-05..2.17e-05, mean-abs ranges 6.17e-05..4.55e-04, max-abs ranges
    5.62e-04..9.66e-03 (decile 2 is the largest, a single-day event, not a
    systematic issue -- every other decile's max-abs is under 2e-3).
    Bounds set with real margin above the observed range, not fitted to
    it: signed <2e-4 (Gate 1's own bound, ~3.3x the largest observed
    decile value), mean-abs <1e-3 (~2.2x), max-abs <1.2e-2 (~1.24x the
    single decile-2 outlier, ~6x every other decile). A +50bp/day bias
    injection (gate-verifier, 2026-09-09) floors mean-abs at ~5e-3,
    comfortably above the 1e-3 bound -- confirmed to fire."""
    result = gate2_deciles.run_gate2_us(
        start_date=datetime.date(2015, 7, 1), end_date=datetime.date(2016, 6, 30)
    )
    comparison = result["comparison"]
    for decile in range(1, 11):
        diff = comparison.filter(pl.col("decile") == decile)["diff"]
        signed_mean = diff.mean()
        mean_abs = diff.abs().mean()
        max_abs = diff.abs().max()
        assert abs(signed_mean) < 2e-4, (
            f"decile {decile} signed mean diff {signed_mean} exceeds 2e-4 "
            "-- a level bias correlation cannot detect."
        )
        assert mean_abs < 1e-3, (
            f"decile {decile} mean absolute diff {mean_abs} exceeds 1e-3 "
            "-- a sign-symmetric error the signed mean alone would miss."
        )
        assert max_abs < 1.2e-2, (
            f"decile {decile} max abs diff {max_abs} exceeds 1.2e-2 on "
            "some single trading day."
        )


def test_run_gate2_us_breakpoint_comparison_present():
    result = gate2_deciles.run_gate2_us(
        start_date=datetime.date(2015, 7, 1), end_date=datetime.date(2016, 6, 30)
    )
    bp = result["breakpoint_comparison"]
    assert bp.height == 9  # 9 decile cutpoints (10th through 90th percentile)
    assert set(bp.columns) == {"percentile", "our_breakpoint", "ff_breakpoint"}

    # our_n_firms/ff_n_firms are single-scalar firm-count diagnostics
    # (not per-percentile columns on breakpoint_comparison, since a firm
    # count isn't a per-percentile value) -- confirms load_nyse_breakpoints()'s
    # parsed-but-previously-discarded n_firms actually surfaces somewhere.
    assert "our_n_firms" in result
    assert "ff_n_firms" in result
    assert isinstance(result["our_n_firms"], int)
    assert result["our_n_firms"] > 0


def test_nyse_company_caps_berkshire_external_anchor():
    """External, real-world ground-truth anchor (gate-verifier
    recommendation, 2026-09-09) -- the direct analogue of the
    Apple/ExxonMobil anchor that caught the shrout 1000x units bug. Every
    other Gate 2 assertion compares two internally-derived quantities
    (our breakpoints vs FF's, our firm count vs FF's); this is the first
    assertion anchored on a fact external to BOTH datasets.

    Berkshire Hathaway on 2015-06-30 traded as two NYSE share classes
    under one company: permno 17778 (BRK.A, $204,850.00/share x 825
    shares) and permno 83443 (BRK.B, $136.11/share x 1,227,452 shares x
    1000 SHROUT_UNITS_MULTIPLIER), both under permco 540. Combined
    implied market cap ~$336.07B, matching Berkshire's well-documented
    real-world market cap on that date (public record: ~$335-340B range
    depending on intraday timing).

    This is the assertion gate-verifier found actually discriminates the
    permno/permco defect on REAL data (every other real-data assertion in
    this file was shown NOT to): under permno-level aggregation, BRK.A
    alone prices at ~$169M (825 shares -- Berkshire deliberately never
    split A shares, so share count is tiny) and BRK.B alone at ~$167B;
    neither individually matches the combined figure, and BRK.A's
    absurdly small SHARE COUNT (not price) means a permno-level company
    caps table would show two wildly different entries for one company
    instead of one correct combined entry."""
    caps = gate2_deciles._nyse_company_caps_at(datetime.date(2015, 6, 30))
    berkshire = caps.filter(pl.col("permco") == 540)
    assert berkshire.height == 1, (
        "permco 540 (Berkshire Hathaway) not found as a single row in "
        "NYSE company caps -- expected one combined row for both share "
        "classes."
    )
    mkt_cap = berkshire["mkt_cap"][0]
    assert 3.2e11 < mkt_cap < 3.6e11, (
        f"Berkshire Hathaway combined market cap {mkt_cap:.4e} on "
        "2015-06-30 is outside the [$320B, $360B] real-world-anchored "
        "range -- expected ~$336.07B (both share classes summed under "
        "permco 540). A value near $167B (BRK.B alone) or $169M (BRK.A "
        "alone, since BRK.A's share count is tiny) would indicate the "
        "permco aggregation regressed back to per-security caps."
    )


def test_run_gate2_us_firm_count_within_tolerance():
    """our_n_firms vs ff_n_firms was computed, printed, and never
    asserted on before Phase B (docs/02_validation_gates.md's
    cross-cutting finding) -- this is one of the two absolute,
    non-scale-invariant diagnostics that were sitting unused.

    **Correction (2026-09-09, gate-verifier review):** the original
    docstring claimed a permno/permco share-class defect "roughly doubles
    the affected count." Independently measured directly against the
    2015-06-30 NYSE universe: 1358 permnos map to 1337 distinct permcos --
    only 21 names, 1.55% of the count, not a doubling. This tolerance
    (5%) does NOT discriminate the permno/permco defect Gate 2 was
    actually retracted over: our_n_firms would read 1358 with the defect
    present and ~1337 once Phase C aggregates by permco, and BOTH values
    clear a 5% band against ff_n_firms=1319 (2.96% and 1.36% respectively).
    A tolerance on a permno count structurally cannot catch a permno-vs-
    permco bug, because it uses the defective unit as its own yardstick --
    see test_run_gate2_us_firm_count_permno_permco_gap_is_small below for
    a diagnostic that measures the actual defect directly instead.

    This test still has value as a coarse universe-definition check (a
    REIT-exclusion-toggle mistake or a wrong exchange filter would move
    firm count by double digits of percent, well outside 5%) -- kept for
    that purpose, but no longer claimed to cover the permno/permco case."""
    result = gate2_deciles.run_gate2_us(
        start_date=datetime.date(2015, 7, 1), end_date=datetime.date(2016, 6, 30)
    )
    our_n = result["our_n_firms"]
    ff_n = result["ff_n_firms"]
    assert ff_n is not None, "No Ken French breakpoint row matched the formation date."
    relative_gap = abs(our_n - ff_n) / ff_n
    assert relative_gap < 0.05, (
        f"our_n_firms={our_n} vs ff_n_firms={ff_n}, relative gap "
        f"{relative_gap:.4f} exceeds the 5% tolerance -- see "
        "docs/02_validation_gates.md Gate 2 for the universe-definition "
        "debug order."
    )


def test_run_gate2_us_firm_count_permno_permco_gap_is_small():
    """Direct measurement of the permno/permco defect itself, independent
    of gate2_deciles.py's production code (which does not carry permco
    at all yet -- that's Phase C's fix, not Phase B's). Reads permco
    straight from the raw panel for the same NYSE-only, same-day-market-
    cap-eligible permno set nyse_breakpoints_at() uses, so this is a
    faithful measurement of what our_n_firms WOULD become post-Phase-C,
    without requiring the aggregation fix to exist yet.

    This is a recorded-value diagnostic, not a defect gate (a small gap
    is the expected, already-diagnosed state -- see docs/03_roadmap.md
    Phase C) -- the assertion exists so a future change to the loader or
    universe filter that suddenly inflates share-class duplication (e.g.
    a broken CCM link producing many spurious extra permnos per permco)
    is caught, since that would be a NEW defect distinct from the one
    already on record. Bound set at 5% (permno count relative to distinct
    permco count) -- the measured gap is 1.55% (1358 permnos, 1337
    permcos on 2015-06-30); 5% is a >3x margin, wide enough that ordinary
    variation doesn't trip it but a share-class duplication bug an order
    of magnitude worse than today's would."""
    from src.data import universe_panel

    formation_date = datetime.date(2015, 6, 30)
    eligible = universe_panel.universe_at(formation_date)
    nyse_only = eligible.filter(pl.col("primaryexch") == "N")
    same_day_caps = gate2_deciles._same_day_mkt_cap_at(formation_date)
    joined = nyse_only.join(same_day_caps, on="permno", how="inner")
    permnos = joined["permno"].to_list()

    files = universe_panel._year_partition_files(formation_date.year)
    snapshot_day = universe_panel._resolve_last_trading_day(formation_date, files)
    raw = (
        pl.scan_parquet(files)
        .select(["permno", "permco", "dlycaldt"])
        .filter(
            (pl.col("dlycaldt") == pl.lit(snapshot_day).cast(pl.Datetime("ns")))
            & (pl.col("permno").is_in(permnos))
        )
        .unique()
        .collect(engine="streaming")
    )
    n_permno = raw["permno"].n_unique()
    n_permco = raw["permco"].n_unique()
    relative_gap = (n_permno - n_permco) / n_permco
    assert relative_gap < 0.05, (
        f"permno count {n_permno} vs distinct permco count {n_permco}, "
        f"relative gap {relative_gap:.4f} exceeds 5% -- share-class "
        "duplication has grown well beyond the diagnosed 2026-09-09 "
        "baseline (1.55%); re-investigate before assuming this is still "
        "the known Phase C issue."
    )


def test_run_gate2_us_breakpoint_p30_hump_diagnosis_partially_confirmed():
    """**RESULT OF THE PHASE C RE-INVESTIGATION (2026-09-09), replacing
    the prior tripwire (test_run_gate2_us_breakpoint_p30_hump_still_present).**

    The original tripwire's own contract (see its git history) said: if
    permco aggregation lands and the p30 hump does NOT flatten, stop and
    re-investigate rather than proceeding. It landed. The hump did NOT
    fully flatten -- it shrank. Per that contract, this triggered a real
    stop-and-investigate, not a silent pass. Findings from that
    investigation, in order:

    1. **The permco fix itself is independently verified correct**, not
       just inferred from the firm-count/breakpoint movement: the 21
       aggregated permcos are all genuine dual-class companies (Berkshire
       A/B, CBS A/B, Molson Coors A/B, Brown-Forman A/B, Constellation
       A/B, Wiley A/B, Lennar A/B), and
       test_nyse_company_caps_berkshire_external_anchor confirms the
       mechanism against real-world ground truth (Berkshire's actual
       ~$336B market cap), independent of Ken French's numbers entirely.
    2. **The permco fix measurably worked**: every one of the nine
       breakpoint gaps shrank (roughly halved), and our_n_firms moved
       from 1358 (2.96% from FF's 1319) to 1337 (1.36% from FF's 1319) --
       the predicted DIRECTION and rough magnitude of movement, matching
       the diagnosis.
    3. **It did not fully resolve the gap.** Post-fix gaps: p10 +0.88%,
       p20 -2.65%, p30 -2.95% (still the largest), p40 -1.05%, p50
       -1.01%, p60 -0.48%, p70 -0.03%, p80 -0.16%, p90 -1.10%. The
       pre-registered prediction in this test's former pytest.skip
       ("expected to collapse to <=1.1% above p10") did NOT hold --
       actual gaps at p20/p30/p90 remain 1-3%, 1-3x the predicted bound.
    4. **A second, unidentified factor remains.** 18 of our 1337 NYSE
       companies (1.36%) still don't reconcile against FF's 1319, and
       this project's working hypothesis that REIT exclusion is "the
       sole difference, and removing it would widen the gap" does not
       fully hold up under direct measurement: FF's count (1319) is
       BELOW our REIT-EXCLUDED count (1337) -- if REIT treatment were
       the only difference and FF includes REITs while we exclude them,
       FF's count should be well ABOVE ours, not below it. Traced and
       ruled out as candidates this session: quantile-interpolation
       convention (checked all five polars conventions, gap magnitude
       unchanged), snapshot date resolution (resolves to 2015-06-30
       itself, no off-by-one), and a minimum-price/listing-age filter on
       our side (14 sub-$1-or-null-price NYSE CORP names exist and are
       included, matching decile-1's expected composition -- no evidence
       FF excludes low-price names, and this project's local docs/specs
       have no citation of French's own methodology establishing one).
       This project's local repository has no sourced citation of Ken
       French's own universe-construction methodology beyond what's
       inferred from gap-direction tests -- see docs/02_validation_gates.md's
       REIT residual discussion for the full record.

    **Conclusion: the diagnosis was directionally correct and the fix is
    verified sound, but NOT the complete explanation.** A second,
    currently-unidentified ~1.36%-of-firm-count factor also contributes.
    This is recorded as an open item (docs/02_validation_gates.md), not
    silently absorbed into a loosened tolerance. This test replaces the
    prior binary tripwire with a assertion on the ACTUAL magnitude of
    improvement (fix works, materially, but isn't total) rather than a
    fragile "is p30 still the argmax" check -- the former tripwire's
    11.7% lead margin (against its own 10% bar) was already flagged by
    gate-verifier as a near-flip, uninterpretable signal either way."""
    result = gate2_deciles.run_gate2_us(
        start_date=datetime.date(2015, 7, 1), end_date=datetime.date(2016, 6, 30)
    )
    bp = result["breakpoint_comparison"]
    assert bp["ff_breakpoint"].null_count() == 0, (
        "No Ken French breakpoint row matched the formation date -- the "
        "magnitude comparison below would be vacuous without this guard."
    )
    bp = bp.with_columns(
        ((pl.col("our_breakpoint") - pl.col("ff_breakpoint")) / pl.col("ff_breakpoint"))
        .abs()
        .alias("rel_gap")
    )
    max_gap = bp["rel_gap"].max()
    mean_gap = bp["rel_gap"].mean()

    # The fix must show REAL improvement over the pre-Phase-C permno-level
    # state (max gap 6.72%, mean gap 3.77%) -- proven capable of catching
    # a regression back to permno-level via injection this session (see
    # docs/worklog.md): reverting _nyse_company_caps_at to permno-level
    # grouping reproduces max_gap=6.72%, well outside this bound.
    assert max_gap < 0.04, (
        f"Max breakpoint relative gap {max_gap:.4f} exceeds 4% -- expected "
        "well under the pre-fix 6.72% ceiling if permco aggregation is "
        "still working. A regression toward permno-level aggregation "
        "would push this back toward 6-7%."
    )
    assert mean_gap < 0.02, (
        f"Mean breakpoint relative gap {mean_gap:.4f} exceeds 2% -- "
        "expected well under the pre-fix 3.77% average if permco "
        "aggregation is still working."
    )
