# Worklog

Dated entries. Two lines is enough. Purpose: reconstructing *why* a choice was
made when writing the KW report, without digging through chat history.

## 2026-09-12 — F2c: rebalance.py monthly loop + F4 run_pipeline; Phase F and E3 complete

Same day as the F2b entry below, resumed after it. `src/portfolio/
rebalance.py::run_backtest` wires weights → legs → diagnostics across a
date range, advancing one calendar month forward from each formation date
for the held return. Thin universe / thin leg / missing-return-coverage
months are skipped with a recorded reason, never silently zeroed.

**Real bug, caught by the leakage suite on its first real execution, not a
designed-in test.** Before the missing-coverage guard, a held month where
every weighted name lacked return data summed over an empty list and
returned an exact `0.0` — a fabricated observation that would have
silently entered the BAB series. `test_point_in_time_reproducibility`
caught it immediately once `run_pipeline` was wired (E3 had been blocked
on this all along). Fixed by skipping the month.

`run_pipeline` (`tests/leakage/test_no_lookahead.py`) now wires
`rebalance.run_backtest` for real: pandas<->polars adapter, one formation
date per calendar month, `estimate_betas` wraps `_estimate_beta_fp_core`
directly (Gate 3's own math, not reimplemented). Both previously-blocked
E3 tests (`test_point_in_time_reproducibility`,
`test_shuffled_signal_produces_no_alpha`) pass. Stop hook widened back to
all of `tests/leakage/`.

`test_formation_strictly_precedes_holding_period` (open item (d)) is also
implemented this session: asserts `max(estimation_date) <
min(holding_return_date)` against `rebalance.run_backtest`'s own real
per-period output on production's `_resolve_window_starts` convention.

Suite: **293 passed / 0 failed / 2 skipped** (`python run.py test`), up
from 276/2/3. Phase F is complete; Phase G (Gate 4) is next.

## 2026-09-12 — F2b (partial): beta_fp wiring, liquidity filter, weighting/leg-scaling/diagnostics

**Closed a live D10 violation.** `beta_fp._market_log_return_series()` and
`_daily_log_returns()` now take required keyword-only `leg`/`market_index`
args instead of hardcoding `gate1_index.build_vw_index()` — `fp_baseline_ca`
(names `market_index: vw_capped_10pct`) was silently getting UNCAPPED betas
until this landed. No default value on purpose: a `vw_uncapped` default
reproduces the exact bug silently. `leg="canada"` on `_daily_log_returns`
raises `NotImplementedError` — the Canadian stock-side beta path (non-permno
identity through `_estimate_beta_fp_core`'s joins) still doesn't exist;
only the market-index half of the Canada leg is wired.

**Measured, not assumed, before pinning:** capping Nortel moves Canadian
`sigma_m` by 0.00134 absolute (9.73% relative) around its 2000-07-27 peak —
a real, material gap. US capped-vs-uncapped is bit-identical (diff 0.0, no
US name nears 10%) and is kept as an explicit *negative control*, not
evidence; `vw_msci_like` vs `vw_uncapped` (diff 7.7e-5) is the real US-side
proof the parameter is read, since the 85% coverage rule bites on every
panel regardless of concentration.

**Liquidity filter built** (`src/portfolio/liquidity.py`,
`config/liquidity.yaml`): two independent gates — `n_traded_days >= 120`
within the sigma window, and formation-date lagged close `>= $1.00` — wired
into `estimate_beta_fp()` as real-panel-only output filtering (US only,
same as `universe_at()`/`market_cap_at()`), never inside
`_estimate_beta_fp_core` (which is also used on synthetic securities with
no volume/price to check). Excludes 116 real names at 2015-06-30. The
roadmap's cited anchors (Lehman 2014, Circuit City) aren't reachable in
this panel's coverage (Lehman delisted Sept 2008; has zero 2014 rows) —
substituted two real, verified cases found by searching the actual data:
Watsco Inc Class B (permno 46068, 94/252 traded days in 2014) for the
volume gate, Kowabunga Inc (permno 90608, genuinely traded at $0.34-$0.35
in Dec 2009/Jan 2010) for the price floor. Unskipped both
`tests/leakage/test_no_lookahead.py` liquidity tests using these.

**`src/portfolio/` started:** `weights.py` (rank_deviation_from_median only
— value weighting explicitly deferred to F2c), `legs.py`
(asymmetric_inverse_beta + leg_ex_ante_beta), `diagnostics.py` (all 6 spec
§8 diagnostics, no toggle parameter by construction — verified via
signature inspection in its own test). Every new assertion in this session
proven capable of failing via targeted injection, then reverted (`git
status` clean after each) — full list in the F2b handoff note below.

**Stopped at an explicit boundary, per user decision.** `rebalance.py` (the
monthly loop), F4's `run_pipeline` wiring, and
`test_formation_strictly_precedes_holding_period` (open item (d)) are NOT
done. Suite: 276 passed / 2 failed / 3 skipped, up from 234/2/5 at session
start — the 2 failures are the same pre-existing `run_pipeline`-blocked E3
tests (not new). See `docs/03_roadmap.md` F2b section for the exact handoff
state and what the next session should read first.

## 2026-09-12 — F2a: market indices (uncapped / capped 10% / MSCI-like)

`config/market_index.yaml` + `src/market_index/build.py` +
`tests/market_index/test_build.py` (20 tests). All three spec §7 variants
built behind one entry point, `build_index(panel, method)`, consuming the
same common-schema panel (`gate_adapters.py`'s `id, date, mkt_cap, ret` —
`mkt_cap` already lagged) every downstream caller uses:

- `vw_uncapped` — thin pass-through to `gate1_index.build_vw_index()`
  (Gate-1-validated, wired not rewritten).
- `vw_capped_10pct` — new water-filling 10% single-name cap, iterated to
  convergence, pro-rata redistribution. **Verified against the real
  Canadian panel:** Nortel reached **27.94%** of the self-built index on
  **2000-07-27** (measured directly, not assumed — matches spec's "roughly
  a third" claim), capped to exactly 10.000000% with weights still summing
  to 1.0.
- `vw_msci_like` — new 85% cumulative-coverage large-cap proxy on total
  market cap (no free-float data exists anywhere in this project, so this
  targets MSCI's coverage-target *rule* rather than replicating free-float
  itself — recorded as a deliberate proxy, not a real MSCI reproduction).

**One real defect found during testing, not just weak tests.** The
capping/coverage weight functions (`_raw_weights`) filtered only on
non-null `mkt_cap`, not also on non-null `ret` — `gate1_index.build_vw_index()`
already guards this exact case (its own docstring calls a null-ret/non-null-
mkt_cap row "particularly dangerous": a naive filter lets the name's full
weight mass into the denominator while `.sum()` silently drops its
contribution from the numerator, handing it an effective 0% return instead
of excluding it and renormalizing the rest to 1). `vw_uncapped` inherited
the guard via delegation; `vw_capped_10pct`/`vw_msci_like` did not, since
they compute weights independently. Found by `leakage-auditor`, fixed same
session, two new tests added and proven to fail against the reintroduced
defect and pass against the fix.

**One design gap surfaced and resolved with the user, not silently
patched:** a strict per-name cap is mathematically infeasible below
`ceil(1/cap)` names (2 names can never both sit ≤10% and sum to 1). Real
Canadian dates always have far more names than this floor, so it never
fires in production — but the algorithm now raises `RuntimeError` naming
the infeasibility (per-date, not a whole-panel check) rather than silently
returning a result with a name over cap. Confirmed by
`test_capped_raises_when_cap_infeasible_for_name_count`.

**Assertions proven capable of failing** (three injections, each reverted
after confirming): (1) pro-rata redistribution replaced with uniform
per-head split — caught by
`test_capped_redistribution_is_pro_rata_not_uniform`, a test added
specifically because the file's other capping tests all happened to use
equal-sized uncapped names and would NOT have caught this; (2) the
per-date infeasibility check removed — caught by
`test_capped_raises_when_cap_infeasible_for_name_count`; (3) the MSCI
coverage boundary changed from "cumulative weight *before* this name"
to "cumulative weight *including* this name" (an off-by-one on which name
crosses the threshold) — caught by two of the four MSCI tests. `git
status` confirmed clean after each revert.

Suite: 234 passed / 2 failed / 5 skipped (`python run.py test`) — up from
216 passed at the Phase D baseline; the 2 failures are the same
pre-existing `run_pipeline`-blocked E3 tests, not new. `lint`/`typecheck`
clean (one pre-existing `yaml` stub warning shared with `beta_fp.py`, not
new).

**Out of scope for F2a, by design:** wiring `market_index.yaml` names into
beta estimation (`beta_fp.py` still hardcodes `gate1_index.build_vw_index()`
directly) or into leg-scaling hedges — both are F2b, once `market_index`
is a resolvable name a run can request through the actual pipeline rather
than called directly as a library function.

## 2026-09-10 — Phase D close-out: Gate 3 PASS after five blockers and two review rounds

**What this entry is really about.** Gate 3 was "built and committed" at
`d8aacfe` and looked finished. It was not. Two independent review rounds found
five blockers, and underneath two of them sat **real estimator defects**, not
merely weak tests. The gate now passes, but the interesting content here is
how much of what was already committed and documented turned out to be wrong.

**Defect 1: window boundaries were calendar days, not trading days.**
`sigma_window_days=252` / `rho_window_days=1260` are trading-day counts —
bab-methodology states them as "1 year" / "5 years", and 252 trading days/year
is the standard convention. The implementation subtracted them as calendar
days, so the sigma window spanned **~173** actual trading days and the rho
window **~866** instead of 252 and 1260. The minimum-observation gates were
therefore far more binding than the spec intended: **1528 of 3772 names were
wrongly excluded on rho** at 2015-06-30. Fixed by resolving boundaries from
the market's own trading-day calendar. This is the defect that mattered most
and it had been sitting in a committed, documented, green-tested module.

**Defect 2: `rolling_sum` bridged trading gaps.** Found by `leakage-auditor`
in the final round. `rolling_sum` counts *rows*, not trading days, so for a
name with a hole in its series (halt, suspension, relisting) the window at the
resume date summed the resume-day return together with pre-gap returns —
fabricating an "overlapping 3-day return" that actually spanned the whole
absence, then correlating it against a genuine 3-day market return. Not
lookahead (every summand is in the past) but a real measurement error, and
concentrated in exactly the halted/illiquid names that populate BAB's low-beta
long leg. Measured: 88 of 5279 names affected, 363 bridged observations, max
|rho| error 0.0421 (permno 88421: 0.1376 → 0.1797), mean 0.0001. **The initial
severity estimate was too low and was revised upward on re-measurement:** the
worst name runs `sigma_i/sigma_m = 6.04`, so that rho error implies a
`beta_shrunk` error of **0.1525 — 42% of Test C's entire cross-sectional SD
(0.3617)** on one name, not the ~0.02–0.03 first estimated. Fixed with a
calendar-span guard; membership and all exclusion counts unchanged.

**The false claim in `d8aacfe`.** That commit's message stated
`estimate_beta_fp_diagnostics` "shares the same trading-day boundary
resolution logic" as `_estimate_beta_fp_core`. **That was false.** They were
two separately-written copies, and they had *already diverged* twice: the
short-panel fallback branch differed, and core clipped stock data at
`<= lookback_end` while the diagnostic clipped at `< month_end` with no
`lookback_end` concept at all. Demonstrated: with the market series ending 3
days before the stock series, the diagnostic admitted 255 sigma-window
observations where core admitted 252, reporting `n_excluded_sigma: 0` for a
name the estimator was in fact excluding. A third copy of related logic (the
grouped `rolling_sum`) had been flagged by an earlier review round and never
consolidated. Extracted `_resolve_window_starts` as the single definition;
proven by perturbing it **once** and confirming the core estimator (3309 → 0
rows) *and* both diagnostic counts (1328 → 1438, 1706 → 5294) move together.

**The unfalsifiable assertion.** `test_estimate_beta_fp_diagnostics_reports_exclusions`
asserted only `n_excluded_sigma >= 0` and `n_excluded_rho >= 0` — both
non-negative by construction (`len()` of a set difference), so the test could
not fail under any defect. Proven rather than argued: a halved-window
injection produced `n_excluded_rho = 5294` against a 3071-name cross-section
— an exclusion count **larger than the entire candidate universe**, i.e.
physically impossible — and the test stayed green. Same failure mode as the
retracted `our_n_firms`/`ff_n_firms` gate.

The first fix was a two-sided bracket, and **`gate-verifier` then showed the
bracket was itself insufficient**: `max(a,b)` and `a+b` are both *symmetric*,
so transposing the sigma and rho labels satisfied it exactly as well as the
correct order — an algebraic invariance no threshold repairs. Five further
defects also passed it (halved `rho_min_obs`, halved `sigma_min_obs`, doubled
windows, two compensating mis-set minimums). Closed properly by returning
`n_excluded_either` (the union) and asserting **exact** equality against what
the estimator drops (1985 == 1985, set symmetric difference 0), plus pinning
both counts at `(1328, 1706)` to break the symmetry.

**Test C was far weaker than its own documentation claimed.** Three
assertions: a VW-mean band, an SD band, and a named-anchor check on 3 permnos.
All of the following passed all three: scrambling all 3068 non-anchor names;
a **full rank reversal** with just the 3 anchors restored; scrambling the
below-median-cap half; reverse-ranking the bottom cap quartile. A complete
inversion of BAB's long/short legs across 3071 names passed as long as 3
values were put back. SD cannot move at all — permuting a multiset cannot
change its standard deviation, so it measured **exactly 0.3617** under every
permutation.

Closed with a full-cross-section rank correlation against an independently
computed OLS beta. **Two things about it are worth recording honestly.**
First, `gate-verifier` showed the comparator is **not** independent: it shares
the estimator's ρ factor *bit-for-bit* (`max|rho_est − rho_comp| = 0.000e+00`),
so ranks reduce to `rank(ρ·σ^1y)` vs `rank(ρ·σ^5y,3d)` — differing only in
which volatility multiplies a shared correlation. The consequence is
directional: collapsing the windows makes those the same object and drives the
statistic **up**, not down (correct 0.8507; `sigma_window 252→1260` **0.9847**).
The correct configuration scored *lower* than the defect, so a floor alone was
not evidence — an upper bound (`< 0.95`) was required. Second, rank correlation
is blind to **tail-local** permutations, which is precisely where BAB trades:
reversing ranks inside the top and bottom 20% scores 0.8140, above the floor,
while bottom-decile membership collapses from 83.4% to 16.0%. Added a
bottom-decile overlap assertion (`> 0.65`) for that. **Bottom decile only,
and for a measured reason:** legitimate bottom-decile overlap runs 76.6–85.6%
across four formation dates, but legitimate *top*-decile overlap runs only
43.5–56.0% (high-beta names are where the 1y and 5y volatility windows
disagree most), so the ">70% on both deciles" originally suggested would have
failed on correct code.

**The VW-mean episode, including the part that does not flatter the work.**
The band is `0.7 < x < 1.3`, and the test's comment implied that had been
derived. It had not. The actual finding is that the legitimate baseline
(1.0536) sits *further* from 1.0 than an inverted-weighting defect does
(1.0471) — which is a statement that VW-mean **cannot discriminate that defect
class at all**, not a derivation of any width. Worse: on encountering that,
**the initial instinct was to widen the tolerance to force a pass**, and that
was caught only on review before being acted on. Widening a threshold to
accommodate a baseline that sits further from the theoretical value than the
defect does would have been exactly the move that produced this project's two
earlier retractions. The band is now recorded as **a judgement call with round
numbers**, with its measured catches (wrong shrink target 0.6536, dropped
`sigma_m` 0.4050, 1.25x scale 1.3170), its misses (inverted weighting 1.0471,
equal weighting 1.1216, double shrinkage 1.0321, 0.80x scale 0.8429), and the
fact that it tolerates a **+23% / −34%** uniform scale error — so it must not
be counted as scale defense.

**Stale numbers, and the rule now applied.** Test C's comments cited
pre-window-fix values as current: Apple 0.8497 (actual **0.9364**), Southern Co
0.5938 (**0.6711**), ConEd 0.6344 (**0.7122**), "baseline 0.9869, sd 0.3191"
(**1.0536 / 0.3617**). The rank-reversal evidence had been measured before the
window fix and never re-derived — the conclusion survived re-derivation, the
cited evidence did not. Every figure was re-measured. One number supplied in
the task brief also failed to reproduce: dropped `sigma_m` was given as 0.0084
but measures **0.4050** through `beta_shrunk` (the 0.4 shrinkage floor
dominates once `beta_raw` collapses to ~0.01); 0.0084 is the *un-shrunk*
`beta_raw` figure. Same verdict either way, but the measured value is what
shipped. Rule now enforced: **no number in a comment unless measured against
the current implementation.**

**NaN vs null.** `is_not_null()` does *not* filter NaN in polars, and
`.count()` counts NaN as a valid observation — so a name with 50 real
observations and 202 NaNs would clear `sigma_min_obs=120` on a count of 252.
Production was already safe, but **by ordering rather than by design**
(`_daily_log_returns` filters nulls before the log transform; the real
2015-06-30 panel measures 0 NaN / 0 inf). Hardened anyway via a `_real_obs()`
predicate on every gate and every std/corr input. `gate-verifier` then found
the *same confusion one layer down on the output side*: a zero-variance
(halted/flat) name clears both min-obs gates on count, then yields `rho = NaN`
from `pl.corr` and hence `beta_shrunk = NaN`, and the smoke test's
`null_count() == 0` could not see it. Such names are now excluded outright.
Again safe by data, not design — the real panel has no such name, so only a
synthetic fixture catches it.

**A test that passed for the wrong reason.** Worth recording as a process
note: the first draft of the market-ends-early regression test **passed against
the known-divergent implementation**. The divergence was genuinely present
(255 vs 252 observations) but a name at that count clears `sigma_min_obs=120`
either way, so no membership decision flipped and the reconciliation bracket
was satisfied. Rewritten to sit the observation count exactly *on* the
threshold, where the 3 disputed days are decisive. A test that passes for a
reason unrelated to the property it claims to check is the same class of
problem as the gates this project has retracted.

**`leakage-auditor`: no leak, no survivorship bias.** Contaminating every row
at/after `month_end` (511,214 stock rows, 129 market rows) and separately
appending 399 fabricated future days left the output **bit-identical**. Output
filtering via `universe_at`/`market_cap_at` changes membership only — **0 beta
values altered** — with 238 names getting betas computed while ineligible at
`month_end`, confirming history is genuinely unfiltered (spec decision 1
upheld). 1,926 names whose series end before 2014 contribute estimation
history and correctly appear nowhere in the output. E2's truncation property
verified directly (max abs diff 0.0).

**Open items recorded, not absorbed.** (a) Reversing ranks inside the **10%**
tails scores 0.8452 *and* 83.4% bottom-decile overlap — indistinguishable from
correct code by every statistic in Test C. (b) **Gate 3 has no absolute anchor
external to the dataset**: every check compares numbers from this panel against
other numbers from the same panel, so a +50bp/day common-mode bias leaves the
rank statistic bit-identical (0.8507), a 1.5x market scale error scores 0.8507,
and a 1-day market misalignment scores 0.8755 — *above* baseline — and all
three pass every assertion. Gate 4's AQR series is the first real external
anchor. (c) The diagnostic models neither `universe_at()` nor
`market_cap_at()` (core 3309 rows vs public 3071), and counts rho observations
pre-join where core counts post-join. (d) E2's adapter labels beta at `t` using
`t`'s own return, the opposite convention from production — not reachable until
Phase F, settle it by implementing the skipped
`test_formation_strictly_precedes_holding_period`.

**Verification.** Every new or changed assertion demonstrated to FAIL under an
injected defect, via temporary uncommitted `tests/estimation/conftest.py`
monkeypatches or in-place `src/` reverts, deleted after each with `git status`
confirmed clean. Suite: **216 passed / 2 failed / 5 skipped**
(`python run.py test`), up from 211 passed at session start. The 2 failures are
E3's `run_pipeline` stubs, unchanged. `ruff` clean on both changed files (the
4 remaining repo-wide lint errors and 31 mypy errors are pre-existing, in files
not touched this session — missing `yaml`/`openpyxl` stubs and two
`date.today()` calls in the pull script).

**Not done this session:** C3 (Gate 0c re-run against CHASS) still outstanding.
The three Gate 3 open items above are deliberately left open rather than
papered over. `data/raw/` was not verified as re-pulled after `dc4e92e` —
outside this diff, settle by counting rows where `dlyret` is non-null while
the security descriptors are blank (~31,114 expected).

## 2026-09-09 — Gate 2 US +18-firm residual: time-boxed investigation, NARROWED

**Question:** is the +1.36%/18-firm gap vs Ken French clustered (a rule
difference) or scattered (CCM-link noise)? Read-only, `data/raw/` untouched,
no `src/` changes.

**Descriptor fields (all uniform, ruled out):** `issuertype`, `securitytype`,
`securitysubtype`, `sharetype` (already known going in) plus two more checked
this session, `usincflg` and `securityactiveflg` — both 1358/1358 uniform at
2015-06-30 NYSE/REIT-excluded. CUSIP-first-character scan for CINS/foreign-
issuer coding: 0 of 1358 alphabetic-first CUSIPs. Foreign incorporation
(spec §11 item 3's old candidate) is not applicable here — that note is about
Compustat `priusa` vs CRSP `shrcd` on the now-superseded Compustat pipeline,
not this CRSP-primary panel, and `usincflg='Y'` is already enforced at pull
time.

**Listing-age hypothesis, tested directly, not confirmed at single-cutoff
resolution:** 66 of 1358 NYSE names were listed within the trailing 12 months
of 2015-06-30 (close to the 64 already measured). Scanned every cutoff from
30 to 365 days in 10-day steps, alone and combined with below-p10-breakpoint
membership — **no cutoff isolates exactly 18 names**; the all-names count
jumps 14→19 crossing the 90-120 day mark with nothing landing on 18, and the
below-p10-decile subset sits at 1 or 4 for a wide band, never 18. The
smallest 40 names by market cap are dominated by OLD distressed 2015
energy/coal names (Walter Energy list. 1995, Alpha Natural Resources list.
2005, Arch Coal list. 1988), not recent IPOs — only 2 of the smallest 40 are
2014-era listings, none are 2015 IPOs. Recently-listed names are spread
across deciles 1-10 (13/8/8/17/6/5/5/0/1/1), not concentrated at the bottom.
**So a single hard listing-age cutoff does not by itself explain the 18.**

**Decisive new finding: the gap is stable, not noisy, across 36 consecutive
months (2014-01 through 2016-12).** `our_n_firms - ff_n_firms` ranges 13-24
(mean 17.9), moves by at most ~3-4 firms month to month, and traces a smooth
secular trend (13 in Jan 2014 -> ~15-20 through 2015 -> 20-24 by end 2016) —
not a noisy, independently-redrawn-each-month series, which is what CCM
link-history gaps would look like. This is the strongest evidence gathered
this session and argues against "scattered / CCM noise" as the explanation.
It does NOT, by itself, identify the rule — French publishes only aggregate
`n_firms` and breakpoints per month, never per-firm identity, so there is no
way to diff his actual firm list against ours from data already in this
repo.

**Conclusion: NARROWED, not identified.** Eliminated: every panel descriptor
field (6 of 6 checked), CINS/foreign-issuer CUSIP coding, and "listing-age
under any single fixed cutoff" as a complete explanation. Still standing,
not confirmed: some structural, slowly-trending exclusion mechanism whose
size (13-24 firms) and secular drift are consistent with new-issuance flow
in direction and rough magnitude, even though no single age cutoff reproduces
it exactly — possibly a softer or differently-shaped eligibility rule (e.g.
requiring a minimum trading-history length measured in returns/trading days
rather than calendar days, which this session did not test) rather than a
sharp calendar cutoff. **Single next check worth running:** locate Ken
French's own sourced data-library methodology documentation (still not
available in this repo or located this session) rather than continuing to
infer the rule shape from count-diffing alone — count-diffing has now been
pushed about as far as it can go without that source.

**Relevance to the BAB research universe, the part that matters:** if the
mechanism is listing-age-shaped (even if not the exact cutoff tested here),
the same ~1-2% of the US universe reappears every formation month at Gate 4,
not just at this one 2015-06-30 snapshot — the 36-month series above shows
it is a persistent, not one-off, feature of the panel. It does not block
Gate 4 and does not change any research-pipeline code (spec-driven deliberate
choices, e.g. REIT exclusion, are not in question here). Diagnosis, not
action: no code or config changed this session.

**Not investigated (out of scope, time-boxed):** dividend/split-adjustment
edge cases, and a full per-firm identity reconciliation (blocked on the
missing French methodology source above).

## 2026-09-09 — Phase C: Gates 1 & 2 US re-derived and PASS

**Gate 1 US: PASS.** Task 1 investigation: correlation on the corrected panel
(0.9975595945) barely moved from the pre-fix value (0.9975581018) despite
22,801 recovered delisting rows. Traced permno 89888's 2015-03-19 row
end-to-end — it DOES reach `build_vw_index()` (membership resolves at the
PRIOR month-end, so the delisting day's own `primaryexch='X'` never hits the
exchange filter). Confirmed at scale: 232/245 `dlydelflg='Y'` rows reach the
index, cumulative |contribution to index_ret| over 2015 is 0.000224 (2.2bp) —
value-weighting legitimately mutes tiny distressed names, not survivorship
bias surviving the fix. `gate-verifier` then found the sharper problem:
because the effect is this small, correlation and Phase B's absolute-diff
bounds are ALL structurally blind to the entire delisting-handling defect
class (injecting "delisting rows stripped from the index" moved none of the
four statistics outside noise; a 10-20x worse bug would still pass). Closed
with `test_2015_delisting_rows_reach_the_gate1_panel`, pinned at exactly 232
(not `>0`), verified to fire on both full and partial (207/232) regressions
via two independent injection rounds (self-check + gate-verifier second pass
with a graded sensitivity sweep). Also loosened the companion distressed-return
anchor from `<-0.9` (a single-row dependency) to `<-0.8` (4 rows), and added
previously-unasserted `n_index_only`/`n_vwretd_only`/`n_dates` checks.

**Gate 2 US: PASS, one open item.** Task 2: added `permco` to
`universe_panel.py`'s `_PANEL_COLUMNS`; `gate2_deciles.py`'s breakpoint and
decile-assignment logic now aggregates by company (permco) not security
(permno), matching Ken French's convention. Firm count 1358→1337 (2.96%→1.36%
vs FF's 1319); breakpoint max gap 6.72%→2.95%. `leakage-auditor` found the
diff clean (point-in-time snapshot preserved, no double-counting, no forward
reach). `gate-verifier` first pass then found the same class of problem as
Gate 1: 6 of 6 real-data assertions passed GREEN when the exact retracted
permno-level defect was reinjected — only a fabricated-data unit test caught
it, and a +50bp/day bias injection passed the whole suite. Closed with
`test_nyse_company_caps_berkshire_external_anchor` (Berkshire Hathaway's real
~$336B market cap, permco 540 — the direct analogue of the Apple/ExxonMobil
anchor that caught the `shrout` units bug; also the ONLY assertion in the
file that catches a uniform units error, since every returns-based statistic
here is mathematically invariant to one), `test_run_gate2_us_per_decile_absolute_diff_bounds`
(mirroring Gate 1's pattern, which Gate 2 never actually had despite Phase
B's summary claiming otherwise), and raised correlation 0.98→0.99 (the
actually-documented bar; cost no real margin). Task 3: the p30-hump tripwire's
own written contract said "if the hump doesn't flatten, stop and
re-investigate" — it shrank (6.72%→2.95%) but did not flatten, which is a
real stop condition, not something to wave through. Re-investigation found
the permco fix independently correct (Berkshire anchor, real dual-class
companies) but incomplete: 18 of 1337 firms (1.36%) still don't reconcile
against FF, and — contrary to this project's prior working assumption — FF's
count (1319) sits BELOW our REIT-excluded count (1337), which is inconsistent
with REIT treatment being the sole remaining difference. Ruled out this
session: quantile-interpolation convention, snapshot-date off-by-one, a
minimum-price filter (14 real sub-$1 2015 energy/coal names, correctly
included). No sourced citation of Ken French's own methodology exists in this
repo or was locatable this session — the REIT-sole-difference claim was
always an inference from gap direction, now shown incomplete. Retired the
binary argmax tripwire and its companion skip, replaced with
`test_run_gate2_us_breakpoint_p30_hump_diagnosis_partially_confirmed`
(max gap <4%, mean <2% — between the fixed and defective states, still
discriminating). Task 4: REIT exclusion kept as-is; measured all four
permno/permco × REIT-excluded/included combinations — including REITs
widens the gap 10x (15.01%/13.34%), confirming exclusion is correct and the
residual +1.36% is a partial, not-fully-understood offset, not clean
agreement.

Both gates independently re-verified by a second `gate-verifier` pass with
its own re-injections (confirmed every number above, including a graded
partial-regression sweep for Gate 1 and a `shrout`-units-bug reinjection for
Gate 2's Berkshire anchor) before being recorded as PASS. Full suite:
`python run.py unit` / equivalent 188 passed, 4 failed (pre-existing E2/E3
leakage stubs, out of scope), 5 skipped. `git status` clean throughout — all
injections were monkeypatches or temporary edits, applied, verified, then
reverted, confirmed byte-identical to the intended diff each time.

C3 (Gate 0c re-run against CHASS) not done this session. Canada leg of Gate 1
not extended beyond n=12 months. Both remain outstanding.

## 2026-09-09 — Phase B: gate statistics made able to fail

Added absolute, non-scale-invariant assertions to Gate 1 (both legs) and
Gate 2, per `docs/03_roadmap.md` Phase B. No gate marked PASS this session —
that's Phase C. First-draft thresholds were wrong in two ways the
`gate-verifier` subagent caught and independent re-measurement confirmed;
final state below is post-correction. Recording the mistake, not just the
fix, since the wrong version would have looked identically green.

**Gate 1.** US correlation `>0.9`→`>0.995` (achieved 0.99756), Canada
`>0.5`→`>0.995` (achieved 0.99812). New direct-panel check: calendar-2015
`dlydelflg='Y'` count == 245 (the `year=2015` partition file itself has 437
unfiltered — spans into 2014 via boundary widening — filtering to calendar
year reproduces the documented 245), permno 89888 `dlyret≈-0.958333` on
2015-03-19. `gate-verifier` verdict: **sound**, no changes.

**Gate 1 absolute-diff bounds — first draft was wrong, both legs.** The
first version asserted only `abs(diff.mean())` (the SIGNED mean). Both
`gate-verifier` and independent re-measurement confirmed a signed mean
cancels under any mean-zero or sign-symmetric error, and a `ret x1.05`
return-scale injection exposed it concretely: US signed mean stayed at
7.75e-05 (bound 2e-4, passed easily); Canada's signed mean didn't just
shrink toward zero, it crossed and went negative across the sweep
(k=1.00 → +3.49e-4, k=1.05 → -4.6e-6, k=1.10 → -3.60e-4) — a k=1.05 error
on that leg would have passed the original bound outright. Fixed by adding
a true mean-absolute-diff bound to both legs, kept alongside (not
replacing) the signed mean, which still catches pure constant bias
cheaply. **Second-pass mistake, caught before committing:** the first fix's
mean-abs-diff bound and the original max-abs-diff bound were both
re-checked against the same `ret x1.05` injection and BOTH still passed —
the "this is now caught" claim in the first draft's docstring was false.
Swept k=1.00 through 1.10 on both legs to find where each statistic
actually moves and picked thresholds from that sweep, not round numbers:
US tightened `max_abs_diff` 5e-3→4e-3 (catches k≥1.04); Canada tightened
`mean_abs_diff` 2e-3→1.2e-3 (catches k≥1.03) since max-abs-diff already
caught k=1.05 there. The two legs needed different assertions tightened —
not assumed to share a derivation. Final bounds: US signed-mean `<2e-4`,
mean-abs `<1e-3`, max-abs `<4e-3`; Canada signed-mean `<1e-3`, mean-abs
`<1.2e-3`, max-abs `<5e-3`. Re-verified: `ret x1.05` now fails both legs'
absolute-diff tests while leaving both legs' correlation tests green, as
intended (correlation is deliberately still the blind statistic; the
absolute bounds are what carries the burden).

**Gate 2.** New live tripwire: p30 breakpoint gap is the largest of the
nine (today it is), with a required >10% lead over the runner-up (measured
lead is 21.3% — p30 6.72% vs p40 runner-up 5.54%) — the bare-argmax version
of this check would flip on ordinary noise given how close some
percentiles run. Also added `bp["ff_breakpoint"].null_count() == 0` before
computing gaps: `gate-verifier` found the original version compared
`None == pytest.approx(None)`, which is `True` — an all-null benchmark
(unreachable today; FF coverage spans 1925–2026, but not structurally
impossible) would have passed this test with zero real data behind it.
Breakpoint dollar tolerance is `pytest.skip`, not `xfail` — gaps are -1.6%
to -6.7%, humped at p30, a diagnosed unresolved `permno`/`permco` defect
(Phase C); any bound wide enough to pass today (>7%) is too wide to catch
a real regression. `xfail` was rejected: nothing about the assertion fails
today, it's simply not assertable yet, and an XPASS is exactly the
ambiguous signal this project keeps getting burned by.

**Gate 2 firm-count tolerance — the docstring's premise was wrong.** The
first draft justified a 5% band by claiming a permno/permco share-class
bug "roughly doubles" the affected count. Measured directly against the
2015-06-30 NYSE universe: 1358 permnos map to 1337 distinct permcos — 21
names, 1.55%, not a doubling. **This tolerance does not discriminate the
defect it was written to catch**: both the pre-fix value (1358, gap 2.96%)
and the Phase-C-fixed value (~1337, gap 1.36%) clear a 5% band against
`ff_n_firms`=1319. Kept the test for its real remaining value (a
REIT-exclusion or exchange-filter mistake moves firm count by double
digits of percent, well outside 5%), corrected the docstring to say so
honestly, and added `test_run_gate2_us_firm_count_permno_permco_gap_is_small`
— a direct measurement of the permno-vs-distinct-permco gap itself (reads
`permco` straight from the raw panel, since `gate2_deciles.py`'s production
code doesn't carry it yet — that's Phase C's fix, not Phase B's), asserted
under 5% as a diagnostic tripwire against the defect getting worse, not a
correctness bar.

**Injected-error demonstrations** (temporary `tests/gates/conftest.py`
monkeypatches, deleted after each; `git status` clean after each):

- **+50bp/day bias on every `ret`:** US correlation unaffected to 10
  decimals (0.99756 — confirms it's still blind to level bias); US
  absolute-diff bounds failed (mean diff → 0.00507). Canada correlation
  **did** drop, to 0.9839 — below the new 0.995 bar — because monthly
  compounding is nonlinear, so Canada's correlation is less bias-blind than
  US's daily-grain one. Not anticipated going in; worth remembering before
  leaning on Canada's correlation alone. Canada absolute-diff bounds also
  failed hard (mean diff 0.1096).
- **1.10x scale on `mkt_cap`** (both Gate 1's panels and Gate 2's
  `_same_day_mkt_cap_at`): all Gate 1 assertions unaffected (scale cancels
  out of VW weighting, both legs) and Gate 2 firm-count unaffected (scale
  doesn't change which permnos are counted) — both as predicted. Gate 2's
  p30-hump test **did** fail: the scale moved the largest relative gap to a
  different percentile (0.082 vs p30's 0.026) rather than leaving the
  profile flat. This was an open question going in, not a pre-committed
  prediction, and is a genuine finding: a uniform scale on top of an
  already-nonuniform gap does not reproduce a flat-gap signature.
- **Delisting rows stripped** (`dlydelflg=='Y'` filtered out of every
  `scan_parquet` call): only the delisting-count test fired (245→0);
  nothing else moved, since no other assertion reads `dlydelflg`.
- **5% return-scale on `ret` (`k=1.05`), added after gate-verifier review:**
  both legs' absolute-diff-bound tests fail (US: mean-abs 7.93e-4 clears the
  old max-abs bound but the tightened `max_abs<4e-3` catches it at 4.86e-3;
  Canada: `mean_abs<1.2e-3` catches it at 1.59e-3), both legs' correlation
  tests stay green, as intended.

`gate-verifier` subagent run against both gate modules and their tests
before closing this out. Its report named four real issues (the two
signed-mean-cancellation gaps, the firm-count tolerance premise, the
hump-test null-vacuity hole) and one thin-margin concern (the hump test's
1.21x argmax lead) — all fixed above, all independently re-verified against
live data rather than taken on the subagent's word alone (per CLAUDE.md:
treat subagent findings as input, not gospel).

Full suite: 184 passed, 4 failed (same E2/E3 stubs as the Phase A baseline —
`estimate_betas`, `run_pipeline`, correctly out of scope), 6 skipped (up from
5 — the new Gate 2 breakpoint-tolerance skip). `python run.py lint` clean.

## 2026-09-09 — A1 verified PASS; E1 wired (build_universe)

**A1.** Re-pull complete: 62 partitions, 74,559,940 rows, 22,801 delisting
rows. Reconciled the 22,801 vs the pre-shipping 24,480 estimate live against
WRDS: 24,480 was raw `COUNT(*)` windowed to `dlycaldt >= 1965-01-01`, no
`DISTINCT`; adding `DISTINCT (permno, dlycaldt)` collapses it to exactly
22,801 (same same-day-multiple-distribution-event mechanism the pull script's
docstring documents for 2015's 299→245). Not a partial pull or lost data —
decade sum matches the total exactly. `year=2026` partition is 0 rows by
design (current year, WRDS data lag, never skip-eligible) — confirmed
`StaleTradingDayError` actually fires against the real empty partition
(month_end=2026-08-31) and that a near-boundary date (2026-01-02) still
resolves correctly to 2025's last trading day. Recommend keeping
`us_panel_crsp_full.old/` until Gates 1/2 are re-derived (Phase C), per the
roadmap's existing dependency — nothing in this session's findings changes
that. Full suite: 6 failures → 4 (both survivorship-adjacent ones now pass),
174 → 178 passed.

**E1.** Implemented `build_universe` as a thin shim over `universe_at()`.
Found the delisting-return fix is invisible to universe *membership* —
verified all 22,801 recovered delisting rows carry `primaryexch='X'`, so
`universe_at()` (which reads only listing-spell columns, never
`dlyret`/`dlydelflg`) produces identical output pre- and post-fix. Kept
`test_universe_contains_eventually_delisted_names` (Lehman) as a legitimate
but non-discriminating survivorship check, documented why, and added
`test_delisting_return_row_recovered` to check the actual recovered row
directly — proven to fail against `us_panel_crsp_full.old/` and pass against
the fixed panel.

Swapped the mirror test's fixture from Shopify (gvkey "023650") to Fitbit
(permno 15390): Shopify is Canadian-incorporated and was never pulled at all
(`usincflg='N'`, zero rows in the panel), so it would have passed for the
wrong reason regardless of date logic. leakage-auditor then found the Fitbit
swap alone still didn't fix the underlying problem: `universe_at()` snapshots
a single trading day before `_apply_pit_bounds` ever runs, and the real panel
has zero rows with `dlycaldt < securitybegdt` (verified), so a not-yet-listed
permno is excluded by snapshot absence, not by the `securitybegdt` bound —
mutation-tested (deleting/inverting `_apply_pit_bounds`) confirms the test's
outcome doesn't change either way. Added
`test_pit_bounds_exclude_not_yet_listed`, a direct unit test against
`_apply_pit_bounds` with a synthetic row, proven to fail against an injected
inverted-comparison bug.

`python run.py test`: 178 passed, 4 failed (all `estimate_betas`/
`run_pipeline` stubs — E2/E3, correctly out of scope), 5 skipped.
`run.py lint`: clean on the touched file (4 pre-existing errors elsewhere,
unrelated). Stop hook in `.claude/settings.json` untouched (points at
different, not-yet-existing files; not widened to `test_no_lookahead.py`).

## 2026-09-02 — Data source discovery (Gates 0, 0b, 0c PASS)
Switched from planned CRSP+CHASS hybrid to Compustat NA for both countries.
Canadian universe rule settled: `fic='CAN' AND tpci='0' AND iid = prican` —
2,614 names in test month (924 TSX / 1,689 TSXV). `exchg` 7=TSX, 9=TSXV
confirmed via blue-chip cross-ref and market-cap distribution. CRSP `dlret`
verified sound across distress and merger cases. Repo docs + Claude Code
config created.

## 2026-09-02 — First src/ code: US universe construction wired
Built `src/data/universe.py` + `pull_universe.py`, wiring the Open Item 3
REIT/MLP exclusion (previously notebook-only) into real code. Investigation
along the way changed the design from what was originally planned:

- **Single-day `cshtrd > 0` gate rejected.** Tested against a 5-day window:
  43% of names zero-volume on one specific day actually traded elsewhere
  that week. Confirms spec §3's existing design — liquidity belongs in the
  estimation window as a multi-day threshold, not a universe-membership
  screen. Not built yet.
- **New: major-exchange filter added (`exchg` in 11/12/14 — NYSE/NYSE
  MKT/NASDAQ).** OTC (`exchg=19`) is 51% of the raw US priusa universe test
  month, larger than NASDAQ+NYSE combined — cannot be traded under the fund's
  (endowment) mandate. New open item, not previously in spec §3.
- **Two new data-integrity traps found and logged (spec §11 items 7, 8):**
  `comp.company.dldte`/`costat` are current-database-state, not
  point-in-time — lookahead risk if used for historical universe filtering.
  `comp.secd.cshoc` can be wrong by orders of magnitude (UNB Corp case:
  implied $3.46B market cap vs. real ~$4-5M, confirmed via FDIC call report
  equity capital ~$10M) — needs a CRSP `shrout` cross-check before `cshoc` is
  trusted for VW weighting or size deciles. See `01_data_notes.md` §9.
- Final US universe (2015-06, major exchanges, REIT/MLP excluded): 3,867 →
  3,534 names. Cached to `data/raw/us_{universe,reference}_2015_06.parquet`.
  7 new unit tests pass; pre-existing leakage-suite stub failures (6,
  `NotImplementedError` — adapters never wired to a real pipeline) are
  unchanged from before this session, not a regression.

## 2026-09-02 — Canadian universe construction (spec, plan, subagent-driven build)
Built via brainstorming → spec → writing-plans → subagent-driven-development:
5 tasks (pull script, exchange-distribution confirmation, `exchanges` param
on `build_universe`, REIT/MLP/trust diagnosis, docs update), each
independently implemented and reviewed with real-data re-verification, plus
a final whole-branch review (caught and fixed one real bug: bad `exchanges`
input silently returned an empty universe instead of raising). Diagnosed
Canadian REIT/MLP rule: `sic=='6798'` union anchored LP-name regex minus
fund-SIC exclusions — 44 of 2,518 excluded, 2,474 remaining. Two US-side
signals (MLP GICS code, bare TRUST/FUND regex) tested against real Canadian
data and explicitly rejected for false positives (pipeline megacaps,
non-trust companies). Branch `canada-universe-construction`, PR opened.

## 2026-09-02 — Out-of-sample diagnostic, both countries, 4 new months
Ran the 2015-06 diagnostic pattern against 1985/1995/2005/2025 (US and
Canada), no code changes. REIT/MLP rule held up structurally across all
months in both countries, zero new false positives. **Confirmed a real
lookahead trap**: `comp.company.gind`/`gsubind` (GICS) are current-database-
state, not point-in-time — same class as the already-known `dldte`/`costat`
issue, but quieter (plausible-looking code, not an obvious null/future
date). Not a bug in today's single-cached-month pipeline; will matter the
moment a point-in-time multi-month panel gets built. Also found: exact-
`datadate` queries have no trading-calendar fallback (1985-06-30 is a
Sunday, returned 0 rows silently); `cshoc` has large null stretches pre-
1997 (US) and in the 1995 Canadian pull. Three new spec §11 open items (9,
10, 11). No change to committed 2015-06-30 numbers/config.

## 2026-09-02 — US market cap via CRSP, replacing cshoc (resolves items #7/#11, US only)
Built `src/data/market_cap.py` (`build_us_market_cap`): CRSP `dsf`
`abs(prc) * shrout * 1000`, joined `gvkey -> permno` via clean CCM link
(date-bounded, prior-trading-day price), plus `pull_market_cap_us.py` and a
leakage test on the link date-bound logic. Live diagnosis on the 2015-06-30
cached US universe: 3,630/3,867 gvkeys (93.87%) matched; 70/7,263 raw rows
carry the expected negative-`prc` no-trade flag, handled via `abs()`.
UNB Corp (gvkey 062212) could **not** provide the planned before/after
number: it is OTC and absent from the exchange-filtered universe cache, and
its only CCM link row (`linktype='NU'`, no PERMNO) fails the clean-link
filter entirely — genuinely no CRSP coverage, confirmed by direct query,
not a fix defect. Ten spot-checked unmatched gvkeys all show the same real
CCM gap: a clean `LC` link exists before and after 2015-06-30, but an `NR`
("not researched") row covers that specific date. Spec items #7/#11 marked
resolved for the US leg only; Canada (`comp.funda` fallback) is separate
follow-up work. See `01_data_notes.md` §13 for full detail.

## 2026-09-03 — US full-history panel, CRSP-primary (supersedes Compustat-based full-history plan)

Built the full US backtesting panel directly from CRSP (`crsp.wrds_dsfv2_query`),
replacing Compustat as the US-leg membership/classification authority — a
same-day pivot after finding `comp.company.sic`/`gind`/`gsubind` aren't
point-in-time (no usable history pre-1999, `01_data_notes.md` §14). This
plan (`docs/superpowers/plans/2026-09-03-us-full-history-panel-crsp-
primary.md`, spec `docs/superpowers/specs/2026-09-03-us-full-history-panel-
crsp-primary-design.md`) **supersedes** the original Compustat-based
full-history plan/spec (`docs/superpowers/plans/2026-09-03-us-full-history-
panel.md`, `docs/superpowers/specs/2026-09-03-us-full-history-panel-
design.md`) — both kept in git history, not deleted; their status headers
point here.

`src/data/pull_universe_us_crsp.py`: chunked year-by-year pull, 1965-2026,
**~74.5M rows, 1.5GB on disk**. Two real bugs found and fixed during
review: date columns (`dlycaldt`/`securitybegdt`/`securityenddt`) weren't
parsed as dates before parquet write (fixed via `date_cols=` on the WRDS
query); a non-atomic write could leave a corrupt file that the idempotent
skip-check would then treat as permanently complete (fixed with atomic
temp-file + `os.replace()`); the current calendar year was being
skip-cached even though WRDS's current-year data is still incomplete
(fixed by never skip-caching the current year).

`src/data/universe_panel.py`: `universe_at()`/`market_cap_at()`, point-in-time
over the CRSP panel. The brief's own illustrative reference code had two
real bugs, found while implementing against the real pulled panel (not
hypothetical): an eager full-panel read OOMs (fixed with `scan_parquet` +
column projection + year-partition pruning), and a raw spell-bounds filter
(or a naive as-of/backward-fill "fix") produces wrong results — either
duplicate rows per permno, or resurrecting a permno whose listing spell is
nominally still open but hasn't traded in months (confirmed concretely:
permno 69550, Mylan Inc). Fixed by resolving `month_end` to the panel's own
actual last trading day, then taking an exact snapshot on that date.
`dlycap` needs **no** shares-outstanding multiplier — confirmed directly
(ratio ≈ 1.0 against `abs(dlyprc) * shrout`), unlike legacy
`crsp.dsf.shrout`'s thousands convention (`01_data_notes.md` §15). Review
also found a real Critical defect: a permno with a real prior-trading-day
row but a NULL price was passing through with a null `mkt_cap` instead of
being excluded — fixed to match `market_cap.py`'s existing `dropna`
pattern (confirmed concretely: permno 14093, 2015-01-02).

`scripts/item3_crosscheck.py` resolved spec open item #3 with real numbers:
Compustat `priusa` (3,632 permnos) vs. CRSP `usincflg`-based `universe_at()`
(3,772 permnos), overlap 3,225, Compustat-only 407, CRSP-only 547 — all 954
non-overlapping names traced to named mechanisms (REIT/MLP/CEF/foreign-
incorporation exclusions on the Compustat-only side; CCM link gaps,
`fic != 'USA'` foreign-incorporation disagreement — a genuinely new
finding — and `exchg`/multi-share-class-pointer disagreement on the
CRSP-only side), zero unexplained residual. `01_data_notes.md` §16 states
all three legitimate overlap ratios (88.8%/85.5%/77.2% Jaccard) after a
review found the initial framing led with only the most favorable one.

Added `tests/leakage/test_universe_panel_crsp_pit_bounds.py` (4 tests,
`_apply_pit_bounds`/`_mkt_cap_from_panel` boundary conditions) — passed
against the real implementation unmodified, confirming Task 2/3's
`>=`/`<` boundary logic has no leakage gap.

**Canada's full-history pull remains separate, deferred work** (spec's own
Gate 4 sequencing rule) — CRSP has no Canadian coverage, so Canada stays
Compustat-based, unaffected by this session's US-leg CRSP-primary finding.

## 2026-09-04 — Final review fix, post-merge pull crashes, CHASS access noted

Final whole-branch review (Opus) found one Critical the five task-level
reviews missed: `market_cap_at()` filtered on the raw `month_end` while
`universe_at()` resolved to the actual trading day, so on non-trading
`month_end` inputs (~29% of calendar month-ends) both landed on the same
day and `market_cap_at()` silently returned a same-day, unlagged price —
a spec-rule violation that lived entirely in the seam between two
individually-correct functions. Fixed by resolving the trading day once
and sharing it; also fixed two Important findings from the same review
(silent unmatched-name dropping with no coverage log, unbounded stale-
data resolution past real panel coverage) and two Minor findings. All
independently re-verified live before merge.

Post-merge, extending the pull's column set (18→32, per user request —
`dlyret`/`dlyretx`, `dlyvol`, `cusip`/`ticker`, adjustment factors, full
OHLC/bid-ask, anticipating Gate 1-3 needs) triggered a real
`numpy._core._exceptions._ArrayMemoryError` inside pandas' own chunk-
reassembly — fixed via `return_iter=True` + streaming
`pyarrow.parquet.ParquetWriter`. That fix had its own gap (a zero-row
year yields one empty chunk, not zero — the empty-schema fallback never
fired), fixed by casting every chunk to an explicit target schema. Then,
separately, a worktree cleanup deleted the only copy of the freshly-
pulled 3.5GB panel (`data/raw/` is gitignored, only the pull script
merges to git) — re-pulled directly in the main checkout, which then hit
two more small-allocation crashes despite 10.5GB+ confirmed free system-
wide (allocator fragmentation, not real shortage); fixed by lowering
`chunksize` 500k→100k. Full 62-year panel (3.5GB) now lives correctly in
main's `data/raw/us_panel_crsp_full/`, verified end to end, all 62 years
schema-uniform, full test suite clean (55 passed, 0 skipped, same 6
pre-existing unrelated failures).

**User has UofT CHASS access** with high-quality TSX data (OHLC, bid/ask,
S&P/TSX sector + TSXV indices, CFMRC indices) — offered mid-session as a
possible complement to Compustat for the Canada leg, given the same class
of data-quality issues (`cshoc` nulls) that drove the US pivot. Assessed
at a glance: no visible shares-outstanding or point-in-time-classification
field in the pasted schema, but the TSXV/sector index data is genuinely
useful for index construction. Queued as a future evidence-first spike,
not started — user already has ad hoc exploration notebooks
(`research_notebooks/universe_tests.ipynb`, `test.ipynb`) committed.

## 2026-09-04/05 — CHASS spike runs the table: full Canadian-leg pivot from Compustat

The queued CHASS spike (above) ran and went well past a spike — it became
the actual Canadian-leg data source, superseding Compustat's `prican` pull
the same way CRSP-primary superseded Compustat on the US leg. `00_spec.md`
§2 rewritten accordingly (v4); `01_data_notes.md` §17 has the full
investigation; `02_validation_gates.md` Gates 0c/1/2 annotated with status
notes since their Compustat-based results are now superseded, not current.

**CSV→parquet conversion, identity key, dedup** (`src/data/
convert_chass_csv.py`, `src/data/chass_loader.py`). CUSIP turned out not to
be a usable identity key at all in this dataset — different securities can
share a real CUSIP (ticker `TWE` usage 0 and usage 1, two distinct
companies, traded live simultaneously under the same CUSIP for weeks in
1983). `(symbol-Ticker, usage-Usage Number)` — CHASS's own disambiguation
mechanism for ticker reuse — maps 1:1 to CUSIP with zero exceptions and is
used as the key instead. Also resolved a daily dividend-event fan-out
(two rows per trading day collapse to one) and added the `volume>0`
liquidity gate (CHASS's `cshtrd>0` analog; confirmed the "true missing"
sentinel arrives as a literal blank CSV field, not `-9` as documented).

**Fund/REIT/MLP exclusion required full hand classification, not a
formula.** `business-Business` cleanly identifies three exclude-outright
categories, but three others mix real operating companies (Pembina
Pipeline, Franco-Nevada, CI Financial, Algonquin Power) with fund products
under the same TSX label — no field or name regex separates them reliably.
All 239 names hand-classified (`config/chass_fund_classification.csv`),
37 left `UNKNOWN` and deferred to a future market-cap filter rather than
guessed.

**Point-in-time membership without the Ticker History table**
(`src/data/chass_universe.py`). `DICTION.DAT` would give an authoritative
delisting date but isn't accessible via this project's CHASS browser
install — chased partway (checked whether "price hits zero" could
substitute; it can't, false-positives on foreign cross-listed names like
American Express and false-negatives on ordinary illiquidity) then
abandoned in favor of a validated interim design instead: row presence at
monthly grain as the primary signal (CHASS stops emitting rows entirely
once delisted, rather than continuing with placeholder rows), a
terminal-row NaN pattern as corroboration (71%/93.4% true-positive rate,
0/30 false positives), and a CUSIP-continuity check to catch ticker
renames hiding as delistings (tested against the most likely rename
candidates — all turned out to be genuine M&A delistings instead, not
renames). A real implementation bug (domestic filtering applied before
fund exclusion, breaking a stale-CSV safety check) was caught by the
test suite, not assumed away, before merge.

**Market cap** (`chass_universe.py::market_cap_at`) needed a genuine
daily-to-monthly join, not an in-place append — CHASS's monthly file has
no closing-price column at all despite its own documentation describing
one, confirmed against the raw CSV header. Shares-outstanding multiplier
is ×100, not ×1000 as first guessed (caught against Pembina Pipeline's
real ~581M share count before it became a silent 10x error). Verified
against Pembina at 2015-06-30: prior-day close × current-month shares =
$13.82B, matching its known real market cap.

**Accepted, documented, unfixable gap: CHASS has no delisting-return
field.** Unlike CRSP's `dlret`, a security's return series just stops at
its last real trade — confirmed via Bema Gold (fine, stock-for-stock
merger, price converged naturally) and Progressive Waste Solutions (not
fine, last trade $41.40, true buyout price unknown). Creates a real,
permanent US-vs-Canada asymmetry (`01_data_notes.md` §6) — flagged, not
patched.

Two commits, both merged to `main`: loader+exclusion+liquidity-gate, then
universe+market-cap. Full CHASS test suite (36 tests across
`test_chass_loader.py`/`test_chass_universe.py`) passing; leakage suite
and ruff/mypy clean on all new files, no regressions against the
pre-existing baseline.

## 2026-09-05 — CHASS loader/universe ported pandas → polars (Gate 1 plan Phase A)
Brainstormed Gate 1 (VW index reconstruction) sequencing; decided CHASS
needed a polars port first so both country legs share one dataframe
library before Gate 1 code sits on top. Pandas originals preserved
unmodified at `chass_loader_pandas_reference.py`/
`chass_universe_pandas_reference.py` for parity testing
(`test_chass_polars_parity.py`, 13 tests against real full-history data,
not just re-run pinned counts).

**Real bug found via parity, not a porting error:** National Bank of
Canada has `symbol-Ticker=null` in the source parquet; pandas'
`groupby()` defaults to `dropna=True` and silently dropped its group from
`_terminal_row_flags()` — confirmed this predates the port (merged
2026-09-05, before this session). `universe_at()`/`market_cap_at()`
unaffected (boolean filters, not groupby) — confirmed NBC correctly
priced and in-universe under both paths. Decided with user: fix in the
port (polars keeps the null-key group by default), don't preserve the
gap. Full derivation: `01_data_notes.md` §18.

Both test files (`test_chass_loader.py`/`test_chass_universe.py`) rewritten
to polars API, same pinned assertions. 53/53 passing (parity + both
regression suites) against real data; full project suite shows only the
6 pre-existing leakage-stub `NotImplementedError` failures (unchanged, not
a regression). Next: output-normalizing adapter layer (Phase B), then
Gate 1 itself.

## 2026-09-07 — Output-normalizing adapters for both country legs (Gate 1 plan Phase B)
Built `src/data/gate_adapters.py`: `us_gate1_panel()` and
`canada_gate1_panel()`, each wrapping its country's existing `universe_at`/
loader functions and normalizing OUTPUT ONLY to one common schema (`id,
date, price, shares, mkt_cap, ret`) — no changes to either country's
underlying loader/universe module, confirmed as the right call over full
unification (see `01_data_notes.md` §18's sibling discussion).

Design: date-range panel builder, not single-date snapshot — monthly
membership via `universe_at()` (held through the following month, formation
timing per spec §10), daily price/return/mkt_cap read directly from each
country's own daily panel. US return field confirmed to be `dlyret` (total
return), verified to genuinely diverge from `dlyretx` on real AAPL
ex-dividend dates — matches `vwretd`'s convention, the Gate 1 comparison
target.

**Two real bugs found and fixed while building the US adapter, both via
direct verification against real data, not assumption:**
1. Int8/Int32 dtype mismatch on the membership join key (`.dt.month()`
   returns Int8, a bare `pl.lit(int)` defaults to Int32) — polars silently
   returned zero join matches rather than erroring. Fixed with explicit
   dtypes on both sides.
2. Lag computed *after* filtering to membership dropped the one row a
   lag actually needs (a holding-month boundary's last day, e.g. June 30
   feeding July 1's shift, was filtered out before the shift saw it).
   Fixed by computing the lag over the full unfiltered per-name history
   first, filtering to membership only afterward — matches
   `universe_panel.py`'s own existing convention, which the adapter
   should have followed from the start.

Canada adapter built clean on the first real test run (11/11 total tests
passing) — CHASS's own real trading month-end dates (e.g. 2015-05-29, not
calendar 2015-05-31) had to be read from the monthly file directly, since
`chass_universe.universe_at()` does no month-end resolution of its own
(unlike the US side). 119 passed / 5 skipped in the full suite, same 6
pre-existing unrelated leakage-stub failures. Next: Gate 1 itself (US leg
first, vs. `vwretd`).

## 2026-09-07 — Gate 1 US leg: PASS (Gate 1 plan Phase C)

> **⚠️ THIS PASS WAS RETRACTED 2026-09-08 — see the status-audit entry at the
> end of this file.** The run below used a cached panel containing zero
> delisting returns, and its correlation statistic cannot detect that. The
> numbers are real; the conclusion drawn from them is not. Entry retained
> unaltered as the record of what was believed at the time.
Built `src/gates/gate1_index.py`: `build_vw_index()` (sum(mkt_cap*ret)/
sum(mkt_cap) per date from `gate_adapters.us_gate1_panel()`'s output) and
`run_gate1_us()` (compares against CRSP's own `vwretd` from the same
underlying panel). First real run targeted 2015 only (bounded window for
a fast signal before committing to full 1965-present history) — built
directly on the already-verified Phase B adapter, came out clean on the
first real test run with no new bugs found.

**Real result: correlation 0.9976 against `vwretd`, 252 trading days,
zero date-alignment gaps** (`n_index_only=0`, `n_vwretd_only=0`). Mean
diff ~7.4e-5 (no systematic bias), max single-day diff ~0.30 percentage
points landing on 2015-08-26 (the August 2015 selloff, the year's highest-
volatility day) — consistent with the self-built universe's deliberate
REIT/MLP/OTC exclusion mattering most on big moves, not a bug. Clears the
gate's correlation threshold with real margin; not exact-to-rounding
(expected — our universe is a deliberate subset of `vwretd`'s own
CRSP-wide universe), but strongly validates return merge, market-cap
calc, and date alignment together, per the gate's own stated purpose.

Full suite: 125 passed, 5 skipped, same 6 pre-existing unrelated
leakage-stub failures. `ruff`/`mypy` clean. Next: Gate 1 Canada leg
(Phase D, vs. CHASS's own TSX Composite TR series), then Gate 2
(universe vs. Ken French size deciles).

## 2026-09-07 — Gate 1 Canada leg: PASS (Gate 1 plan Phase D)

> **⚠️ QUALIFIED 2026-09-08 (status audit).** This PASS stands — the Canadian
> leg does not share the US delisting defect — but the evidence is thin:
> n=12 monthly observations, on a source with no delisting-return field at
> all. Extend the window before relying on it. The closing sentence "Gate 1 is
> now PASS on both legs" below is **no longer true** as of 2026-09-08.
Structural mismatch investigated before writing code (per
`docs/superpowers/specs/2026-09-07-gate1-canada-handoff.md`): CHASS's
S&P/TSX Composite Total Return series (`ind7`) exists ONLY at monthly
grain (552 rows, 1980-01 through 2025-12, zero nulls, a LEVEL series).
The daily file's `ind1` is a PRICE index (excludes dividends), not a
total-return equivalent -- confirmed this is the same `ret`/`retx` trap
already caught on the US side, and using it as a daily benchmark would
fail for the wrong reason (systematic downward bias from missing
dividends). Conclusion: the comparison must happen at monthly grain,
built by chain-linking the self-built DAILY VW index up to monthly --
not by compounding per-name returns first and then weighting, which is
a different, wrong calculation.

Built `compound_daily_index_to_monthly()` in `src/gates/gate1_index.py`
(TDD: synthetic-data tests with hand-computed compounded answers written
first, confirmed failing before implementation, then passing after).
month_end is the last real trading date present in each calendar month's
data (e.g. 2015-01-30), not a naive calendar month-end -- consistent
with `canada_gate1_panel()`'s own real-date convention. Built
`run_gate1_canada()` mirroring `run_gate1_us()`'s return contract
(`n_index_only`/`n_benchmark_only` in place of the vwretd-specific
names), pulling `ind7` from `chass_loader.load_monthly()` and converting
it from a level series to month-over-month pct change.

**Real result: correlation 0.9982 against the TSX Composite TR series,
12 months (2015), zero date-alignment gaps** (`n_index_only=0`,
`n_benchmark_only=0`). Mean diff ~3.6e-4, max single-month diff ~0.30pp
on 2015-11-30. Tighter than the US leg's 0.9976 -- plausible, since
monthly compounding averages out day-level universe-composition noise.
Built clean on the first real test run against real CHASS data, no new
bugs found.

Full suite: 128 passed, 5 skipped, same 6 pre-existing unrelated
leakage-stub `NotImplementedError` failures (unchanged, not a
regression). `ruff` clean on both changed files; `mypy` clean on
`gate1_index.py` itself (the 4 errors surfaced by `mypy src/gates/
gate1_index.py` are pre-existing in `universe_panel.py`'s schema dict
literals, confirmed present on `main` before this session's changes via
`git stash`). Gate 1 is now PASS on both legs. Next: Gate 2 (universe
vs. Ken French size deciles).

## 2026-09-08 — Real bug found and fixed: US market cap was 1000x too small (Gate 2 Task 6 investigation)

While building Gate 2's breakpoint-comparison diagnostic (compare our own
computed NYSE size cutpoints against Ken French's published ones), every
one of our 9 computed breakpoints came back ~1000x smaller than Ken
French's corresponding value. Traced this to real root cause, not
assumed: `docs/01_data_notes.md` section 15 (2026-09-03) had concluded
`shrout` in the US CRSP-primary panel (`crsp.wrds_dsfv2_query`) needed no
units multiplier, based on `dlycap / (abs(dlyprc) * shrout) ~= 1.0` for
sampled rows. **That check only proves `dlycap` and `dlyprc*shrout` are
internally self-consistent with each other — both are CRSP-computed
quantities, so it cannot catch both being wrong by the same factor
together.** It never anchored against a real-world market cap the way
section 13's original `market_cap.py` verification did (ExxonMobil,
Wells Fargo, live-checked against known public market caps).

Ran that same real-world anchor check against this panel: permno 14593
(Apple), 2015-06-30, `dlyprc=125.425`, `shrout=5,705,400` implies a
$715.6 million market cap under "no multiplier" — but Apple's real
market cap that day was ~$715 **billion**, a factor of exactly 1000.
Cross-checked against permno 11850 (ExxonMobil), same date: implied
~$347.9 million vs. real ~$348 billion — same 1000x gap, ruling out an
Apple-specific stock-split artifact.

**Real result: `shrout` in this CRSP-primary panel IS in thousands of
shares, matching legacy `crsp.dsf.shrout`'s convention — the original
"no multiplier" conclusion was a false negative from checking only
internal ratio consistency, never an external anchor.** Fixed by adding
`SHROUT_UNITS_MULTIPLIER = 1000` to `src/data/universe_panel.py`
(matching `market_cap.py`'s existing constant of the same name/value)
and applying it in `_mkt_cap_from_panel()`; the same fix was applied to
`src/data/gate_adapters.py::us_gate1_panel()` and
`src/gates/gate2_deciles.py`'s two market-cap functions
(`_same_day_mkt_cap_at()`, `build_decile_daily_panel()`). Updated
`docs/01_data_notes.md` section 15 with the correction and updated two
stale hardcoded test expectations in `tests/unit/test_universe_panel_us.py`
and one in `tests/unit/test_gate_adapters.py` that had encoded the
pre-fix (1000x-too-small) values.

**Why Gate 1 US's correlation test (0.9976 vs. `vwretd`) never caught
this:** a uniform 1000x scalar error cancels out of any value-weighted
RATIO (weight = mkt_cap_i / sum(mkt_cap_i)) — correlation and
value-weighting are both scale-invariant to a uniform multiplier on one
side. This is a live instance of CLAUDE.md's own named worst-case: "a
wrong number that looks right is the worst possible outcome." Re-ran
Gate 1 US after the fix to confirm: **correlation unchanged, 0.9975581017753824
to full precision** (same value before and after, as expected for a
ratio-based test) — Gate 1 US's PASS status is unaffected and remains
valid. **[2026-09-08 status audit: the reasoning in this sentence is
correct — the units fix genuinely does not disturb a ratio-based test —
but the conclusion no longer holds. Gate 1 US was retracted the next day
for an unrelated defect (absent delisting returns) that this same
scale-invariance also conceals. "Unaffected by the units fix" was true;
"remains valid" was not.]** Gate 2's breakpoint diagnostic, which is NOT ratio-based (it
compares absolute dollar cutpoints), is what actually surfaced the bug —
after the fix, our computed breakpoints and Ken French's published ones
are in the same order of magnitude (both ~$300M-$25B for deciles 1-9 at
2015-06-30) instead of differing by 1000x.

Full suite re-run after the fix: 146 passed, 5 skipped, same 6
pre-existing unrelated leakage-stub failures (unchanged). `ruff`/`mypy`
clean on all touched files (one pre-existing unrelated `RUF059` finding
and the same 4 pre-existing `universe_panel.py` mypy errors, both
confirmed to predate this session's changes). Gate 2's own decile
correlations (0.9961-1.0000) are unaffected by the fix, as expected —
they were already ratio-based and therefore already correct despite the
underlying units bug. Resuming Gate 2 Task 7 (docs update) next.

## 2026-09-08 — Gate 2 US leg: PASS
Recorded Gate 2's real US-leg result in `docs/02_validation_gates.md`
(docs-only task, no code changes). Methodology: NYSE-only breakpoints
computed at each June 30 formation date, same-day market cap for the
June-30 decile assignment, value-weighted daily index per decile built
from lagged daily weights, compared against Ken French's own published
daily size-decile series over July 2015–June 2016
(`src/gates/gate2_deciles.py::run_gate2_us`). Numbers below are from a
fresh re-run of the verification script against the current code —
i.e. after the `d3b1e7f` market-cap-units fix recorded in the entry
immediately above — not from any earlier pre-fix run.

**Real result:** per-decile correlation against Ken French's published
series, 253 trading days per decile, zero within-window date-alignment
gaps (`n_index_only=0` for every decile):

```
decile 1: corr=0.9963  n_dates=253  n_index_only=0  n_benchmark_only=26043
decile 2: corr=0.9961  n_dates=253  n_index_only=0  n_benchmark_only=26043
decile 3: corr=0.9987  n_dates=253  n_index_only=0  n_benchmark_only=26043
decile 4: corr=0.9984  n_dates=253  n_index_only=0  n_benchmark_only=26043
decile 5: corr=0.9983  n_dates=253  n_index_only=0  n_benchmark_only=26043
decile 6: corr=0.9986  n_dates=253  n_index_only=0  n_benchmark_only=26043
decile 7: corr=0.9991  n_dates=253  n_index_only=0  n_benchmark_only=26043
decile 8: corr=0.9993  n_dates=253  n_index_only=0  n_benchmark_only=26043
decile 9: corr=0.9996  n_dates=253  n_index_only=0  n_benchmark_only=26043
decile 10: corr=1.0000  n_dates=253  n_index_only=0  n_benchmark_only=26043
```

(`n_benchmark_only=26043` is identical across all deciles because it
counts dates in Ken French's full published history outside this
one-year test window, not a gap within it — the comparison inner-joins
on date, so within-window alignment is confirmed by `n_index_only=0`
and `n_dates=253` matching the full trading year.)

All ten deciles clear the gate's > 0.99 threshold. Mean/max-abs diff per
decile (index return minus FF return, decimal units):

```
decile 1: mean=-0.000026  max_abs=0.005369
decile 2: mean=-0.000103  max_abs=0.009633
decile 3: mean=0.000009  max_abs=0.002304
decile 4: mean=0.000028  max_abs=0.002528
decile 5: mean=0.000014  max_abs=0.002661
decile 6: mean=-0.000020  max_abs=0.003041
decile 7: mean=-0.000045  max_abs=0.001883
decile 8: mean=-0.000063  max_abs=0.001892
decile 9: mean=0.000022  max_abs=0.001306
decile 10: mean=0.000005  max_abs=0.000551
```

Breakpoint comparison (formation date 2015-06-30, our computed NYSE
cutpoints vs. Ken French's published ones) — same order of magnitude
post-fix, our values running a few percent below FF's at every
percentile:

```
percentile  our_breakpoint  ff_breakpoint
10          3.08514325e8    3.1353e8
20          6.23606436e8    6.5462e8
30          1.0918e9        1.1705e9
40          1.7581e9        1.8611e9
50          2.6326e9        2.7369e9
60          3.8373e9        3.9258e9
70          6.2985e9        6.5239e9
80          1.0735e10       1.1134e10
90          2.4932e10       2.5497e10
```

Canada's Gate 2 equivalent is deferred — no published FF-style Canadian
size-decile benchmark series exists. Also carried forward Open Item 3's
already-resolved citation (spec §3/§11: `usincflg`-based CRSP-primary
membership vs. Compustat's `priusa`, 77.2% union overlap, every
divergent name traced to a named mechanism) into this gate's doc
section — no new code required.

Full suite: 146 passed, 5 skipped, same 6 pre-existing unrelated
leakage-stub `NotImplementedError` failures (unchanged, not a
regression). `ruff` clean after adding a documented `# noqa: DTZ007` on
`ken_french_loader.py`'s daily-returns date parsing (naive
`datetime.strptime` on a plain YYYYMMDD calendar date with no timezone
semantics -- intentional, not an oversight; found and fixed during the
final whole-branch review pass, not at the time of this entry's
original run) -- not an unqualified "clean" claim, since a suppressed
finding still needs to be named rather than implied absent. `mypy`
clean on `gate2_deciles.py` and `ken_french_loader.py` themselves (the
same 4 pre-existing `universe_panel.py` schema-dict errors surfaced via
the `gate2_deciles.py` import, confirmed unchanged from `main` via `git
stash`). Gate 2 is now PASS on the US leg; Canada's equivalent is
deferred per above. Next: Gate 3 (beta estimator).

## 2026-09-08 — Gate 2 US "PASS" RETRACTED; two real data defects found

Re-derived Gate 2's US result independently (the entry above is now an
unverified claim, not a result). The >0.99 correlations are real numbers
but do not support the conclusion drawn from them. Two genuine defects,
plus a third finding about the gate's own test statistic.

**Defect 1 — delisting returns absent from the entire cached panel.**
`crsp.wrds_dsfv2_query` does not expose a separate `dlret` column the way
legacy `crsp.dsf` does: it folds the delisting return into `dlyret`, on a
row dated one trading day AFTER the last trade (`stkdelists.deldlydt` vs
`stkdelists.delistingdt`), and BLANKS the security descriptors on that row
(`securitytype='N/A'`, `securitysubtype='UNK'`, `sharetype='N/A'`,
`primaryexch='X'`). The pull's WHERE clause required all four descriptors,
so every delisting row failed it. Measured live: **31,114 delisting rows
1965-present, 0 retained, in every decade**. Confirmed in the cache — 2015
has zero `dlyret == -1.0` rows and min `dlyret` −0.8824, while the source
has 471 delisting rows that year; all 10 sampled known cases (e.g. permno
89888, −0.9583 on 2015-03-19) absent, each name's history simply stopping
on its last trading day. Verified against `crsp.stkdelists`: where rows do
survive, `dlyret == delret` exactly. Bias direction flatters BAB —
delisting returns concentrate in small, distressed, high-beta names, i.e.
the short leg. Full window: mean −3.22%, 2,164 rows ≤ −30%, 399 total
losses. Worst decade is the 2020s (−5.55%), overlapping the out-of-sample
period. Note `primaryexch='X'` was previously documented as "confirmed
dead/inactive" — correct, but it is the delisting row, and it carries the
final return.

**Defect 2 — market cap aggregated per share class, not per company.**
Ken French sums share classes to company level before taking NYSE
percentiles; we quantile per-`permno`. Berkshire enters 2015-06-30 as two
names ($169.0B + $167.1B) instead of one $336.1B company. This is the
mechanism behind the breakpoint gap, and it is NOT monotone as the entry
above states — gaps run −1.60/−4.74/−6.72/−5.54/−3.81/−2.26/−3.46/−3.58/
−2.22%, humped at p30, which is a composition signature, not a scale
factor. `permco` was never pulled; `dlycap` is no substitute (also
per-security). Using cusip6 as a permco proxy: gaps collapse to ≤1.1% at
every percentile above p10, and firm count moves 1358 → 1337 against Ken
French's own published 1319. 180 of 3772 names (4.8%) change decile.

**The REIT exclusion is a benchmark-comparability mismatch, not a bug.**
We exclude REITs (159 of 1517 NYSE names) per spec; Ken French excludes
none. Removing our exclusion *widens* the gap (−0.37 to −11.34%), so the
observed −2 to −7% is two errors partially cancelling. Gate 2 as designed
cannot pass cleanly even with perfect data — fixing the pipeline to match
Ken French would make it wrong for the actual research.

**Gate 2's test statistic cannot detect either defect.** Correlation is
location- and scale-invariant: a constant +50bp/day error scores
1.000000, as does a 1.10x scale error. Decile 10 hits 1.0000 because
megacaps are unambiguous, not because the code is right. The two
diagnostics that WOULD have caught defect 2 (`our_n_firms` 1358 vs
`ff_n_firms` 1319) were computed, printed in the entry above, and never
asserted on — `test_gate2_deciles.py` checks only `our_n_firms > 0` and
that breakpoints are ascending. Gate 2 needs bounded relative-error
assertions on breakpoints and firm count, not correlation alone.

**Verified correct, for the record** (each with the check that establishes
it): units (`mc/dlycap` = 1000.0 ± 2e-4 across all NYSE names); price sign
(`.abs()` applied, and moot — zero negative/null/zero prices in all
1,009,828 rows of 2015; CRSP v2 drops the legacy negative-price
convention); exchange/PIT filters (raw NYSE non-REIT 1358 == `universe_at`
NYSE 1358, zero inactive); share-code filter; same-day formation snapshot
(June 30 close for a June 30 decision — not a lookahead violation);
VW weighting and lag mechanism (decile 10 at 1.0000 isolates it as sound).

## 2026-09-08 — Fix CRSP pull: delisting returns + permco

`src/data/pull_universe_us_crsp.py` only; `data/raw/` NOT yet re-pulled.
WHERE clause is now a two-branch OR: the original base-descriptor filter,
plus delisting rows (`dlydelflg='Y'`) admitted only when that permno
qualifies under the base filter somewhere in its own history. The second
branch cannot key on the descriptors (blanked) and must not key on
`primaryexch='X'` alone — 'X' also covers 1,370,373 non-delisting rows.
`dlydelflg='Y'` ⟺ `primaryexch='X'` exactly (31,114 both ways) was
verified; the EXISTS subquery is what excludes delisting rows of
securities never in our universe (31,114 total → 24,480 ours, 6,634
correctly excluded). Added `permco`, plus `dlydelflg`/`delreasontype`/
`delactiontype`/`delstatustype` so delisting handling is verifiable rather
than assumed (the `del*type` fields are blanked on the daily row and live
populated in `crsp.stkdelists` — kept so a reader confirms that directly
instead of rediscovering it).

Also added `StaleSchemaError`: the idempotent skip is resumability for a
crashed run of the SAME query, and is unsafe across a query change — it
would leave pre-fix years permanently stale while new years got the
corrected filter, silently half-fixing the panel. It now refuses to skip a
year file lacking `permco`/`dlydelflg`.

**Validated on 2015 before shipping** (WRDS live, wrote only to
scratchpad): old filter 1,009,895 rows → new 1,010,194, delta **+299 ==
exactly the delisting rows added** (strict superset, no ordinary row
gained or lost); all 8 known-missing delisting returns recovered at the
right date and value. End-to-end through the edited module's own
YEAR_QUERY + schema cast: 1,010,073 rows, 37 columns, `permco` non-null on
every row, 245 delisting rows. The 299 → 245 difference is `SELECT
DISTINCT` collapsing duplicate distribution-detail rows — verified zero
`(permno, dlycaldt)` pairs lost and zero surviving twice, i.e. the
existing dedup working as documented, not data loss.

Suite: 147 passed, 6 failed, 5 skipped — the same 6 `NotImplementedError`
leakage stubs, confirmed identical with the change stashed. `ruff`/`mypy`:
2 + 2 findings, all pre-existing and unchanged from baseline (same stash
check).

**Not yet done — the re-pull itself is the user's call.** Plan: rename the
existing `data/raw/us_panel_crsp_full/` rather than delete it (the pull is
~40 min and has crashed mid-run before, twice, on `_ArrayMemoryError` at
1993/1994 — keep a fallback for the duration of one run), re-pull, verify,
then delete the old directory. Overwrite rather than keep both: the new
pull is a strict superset, and two panels invites reading the stale one.
Gate 2 must be re-run and its PASS re-derived afterward; the size-decile
comparison should switch to `permco` aggregation, and the REIT
comparability question resolved explicitly rather than left implicit.

## 2026-09-08 — Full status audit: Gate 1 US retracted, Gate 0c downgraded, Gate 4 benchmark acquired

Independent audit of every gate marked PASS, treating each worklog claim as
unverified. Docs-only session — **no code written, nothing committed, no
existing worklog entry rewritten.** Prior entries stand as the record of what
was believed at the time; this entry states where that record was wrong.

### Gate 1 US — PASS RETRACTED

The 2026-09-07 Gate 1 US PASS (correlation 0.9976 vs `vwretd`) was computed
against the same cached panel, carrying the same delisting defect, that forced
the Gate 2 retraction earlier today. Re-confirmed independently on the
preserved pre-fix cache:

```
OLD 2015 rows:                    1009828
OLD min dlyret:                  -0.882423
OLD rows with dlyret <= -0.99:           0
```

Zero total losses in a year the source shows 471 delisting rows for.

**The inconsistency this audit corrects:** Gate 2 was retracted on this
evidence while Gate 1 was left green — same cache, same defect, same blind
statistic. That was a gap in the record, not a difference in the evidence.

Quantified the blindness directly rather than restating it. Against the real
2015 `vwretd` series:

```
bias +0.0005/day -> corr 1.0000000000
bias +0.0050/day -> corr 1.0000000000
scale 1.10x      -> corr 1.0000000000
scale 1.50x      -> corr 1.0000000000
```

A +50bp/**day** error (~+12.6%/yr, several times BAB's whole alpha) scores a
perfect 1.0. This is the same invariance that left Gate 1's correlation
bit-identical across the 1000x units fix, and it applies to **both** gates —
the earlier retraction framed it as Gate 2's problem specifically.

Also found: Gate 1's test thresholds are far below its documented bar —
`test_gate1_index.py:196` asserts `> 0.9` (US), `:292` asserts `> 0.5`
(Canada), against a stated criterion of "match to rounding". A Canadian index
correlating 0.55 with the TSX passes the suite today.

Canada leg PASS retained but flagged: n=12 monthly observations is a thin
basis for a Pearson correlation, and CHASS has no delisting-return field at
all (an accepted structural asymmetry, `01_data_notes.md` section 6).

### Gate 0c — downgraded PASS to STALE

The PASS is against the Compustat `prican` rule, replaced by CHASS 2026-09-05.
The gate's own body already said so; the **status-summary table did not** — it
showed a bare check, so a reader scanning the summary got the wrong answer.
Same failure class as the Gate 2 retraction, in miniature. Re-running against
`chass_universe.universe_at()` is outstanding.

### Verified correct, with the check that establishes each

Not everything audited was wrong. Recording what held up, so it is not
re-litigated:

- **Formation timing genuinely lags.** Ran `_month_ends_in_range(2015-01-01,
  2015-12-31)`: returns 13 month-ends starting 2014-12-31, and
  `universe_at(2014-12-31)` maps to holding month 2015-01. No same-month
  membership anywhere in the adapter.
- **Lag-then-filter ordering is correct** in both `us_gate1_panel` and
  `build_decile_daily_panel` — filtering to membership before `shift(1)` would
  break the lag at every holding-month boundary (`gate_adapters.py:173-181`).
- **Null handling in `build_vw_index`** requires BOTH `mkt_cap` and `ret`
  non-null (`gate1_index.py:68`). A non-null-cap/null-ret row would otherwise
  keep full weight in the denominator while contributing nothing to the
  numerator — a silent 0% return.
- **The double-resolution leakage guard is real and genuinely tested.**
  `test_universe_panel_crsp_pit_bounds.py:110-161` asserts the market-cap
  price date is strictly earlier than the resolved snapshot on 2007-06-30 (a
  Saturday). This is a test that can fail.
- **Re-pull is applying the corrected schema.** New partitions carry 37
  columns vs the old 32; added exactly `permco`, `dlydelflg`,
  `delactiontype`, `delreasontype`, `delstatustype`.

### Test-suite state — 23 failures, all attributable, none regressions

`23 failed, 129 passed, 6 skipped`. Confirmed by reading the failure text
rather than assuming: every US-data failure is
`universe_at(2015-06-30) returned 0 rows` — the re-pull has not reached 2015
yet (at `year=1993` when checked). Plus the 6 known leakage stubs.

### Leakage tests enforce nothing today

`test_no_lookahead.py` — four adapters all `raise NotImplementedError`
(lines 19-36), six `pytest.skip`. The truncation test at line 43, described in
its own docstring as "the single most valuable test in this file", has never
executed. CLAUDE.md names `make leakage` a must-pass-before-commit gate; it
cannot currently fail. Prior entries called these "pre-existing unrelated
failures" — accurate per-commit, but the cumulative effect went unstated.

They are not implementable until an estimator and portfolio exist, which makes
Gate 3 the unblocking step. Also: `make test`/`make leakage`/`make gates` do
not exist — there is no Makefile.

### Gate 3 — partially built, on an unmerged branch

Branch `worktree-gate3-beta-estimator`, 3 commits (`beb1a33`, `a0af676`,
`7d8a91c`): `config/beta_estimator.yaml`, `src/estimation/beta_fp.py` (131
lines), `tests/estimation/test_beta_fp.py` (199 lines). Not on `main`, not on
the current branch, no sub-test passing. **The branch code was not reviewed in
this audit** — existence and shape confirmed, correctness not assessed.

Dependency worth flagging: that plan's Architecture section calls
`build_vw_index()` "already-passing", which as of today it is not.

### Gate 4 — benchmark data acquired (user-supplied), largest un-costed risk retired

Until today no AQR data existed in the repo at all; the gate the project is
organised around was defined against a series never fetched. Now present at
`data/raw/AQR_BAB_Historical/Betting Against Beta Equity Factors Monthly.xlsx`
and verified by direct XML parse (neither venv has `openpyxl`):

- Sheet "BAB Factors", header row 19, data from row 20. **USA = col Y,
  CAN = col E.** 13 sheets total.
- USA: **1147 months, zero nulls, 12/31/1930 to 06/30/2026**, decimal returns.
- CAN: **473 non-null, 02/28/1987 to 06/30/2026.**

**AQR publishes a Canadian BAB factor** — not previously known here. This
weakens the "do not run Canadian results until US Gate 4 passes" rule, whose
stated reason was that Canada has no external benchmark. From 1987 it does.

Recorded five loader anchors (first/last value, non-null counts) in
`02_validation_gates.md` — these catch the realistic failure mode, a
column-offset or header-row error.

### Risk-free rate — resolved, config written

Missing on both legs before today; blocks Gate 3's real-data sanity check and
all of Gate 4, since BAB is built on excess returns.

Wrote `config/risk_free.yaml` (config only, no loader). US: Ken French `RF`,
**percent units**, with the two-table parse trap documented (monthly ends line
1205; the annual-factors block begins line 1207 with 4-digit year keys a naive
`read_csv` silently appends). Gate 4 overrides to AQR's own RF sheet, since
AQR built their BAB series with it — close to but not identical with Ken
French's.

Canada: CHASS `ind2-30 day Return on T-Bills`, already a monthly decimal
return, coverage 1980-01-31 to 2025-12-31 matching the equity panel exactly.
Chosen over the BoC Valet API that spec section 4 names — that line predates
the CHASS pivot and is stale rather than a considered rejection. **Not `ind1`**
(91-day annualized rate in percent): measured `corr(ind2, ind1/1200) = 0.944`,
not ~1.0, widening post-2009.

Two null months, both inside the out-of-sample window, filled as flagged
derivations — and they are **not the same case**:

```
2023-10-31: ind2 null AND ind1 null -> external rate 4.93% /1200 = 0.00410833
2023-11-30: ind2 null, ind1 = 5.042 -> file's own ind1  /1200 = 0.00420167
```

The initial instinct was to use `ind1` for both; October has no `ind1` either,
so it keeps an external figure. Both remain a proxy on a neighbouring
convention, never observed `ind2`. Sanity-checked: both land inside the
surrounding real-data envelope (2023-07 to 2024-02 spans 0.003851-0.004222),
so neither introduces a visible discontinuity — a plausibility check, not
evidence of correctness. Loader must emit an `rf_source` column
(`chass`/`derived_from_ind1`/`derived_from_external_rate`).

### US sector data — spec Open Item 9 partially resolved as a side effect

Item 9 records that Compustat's `sic`/`gind`/`gsubind` are current-state, not
point-in-time. True, and still true — but the US leg no longer sources
classification from Compustat. Tested whether CRSP's fields are PIT:

```
permnos (2000/2005/2010/2015):            10425
permnos with >1 distinct siccd:            2526
permnos with >1 distinct icbindustry:      3972
siccd null rate: 0.0    icbindustry null rate: 0.0
```

Genuine per-permno time-variation with zero nulls — the signature of a
point-in-time field. **The CRSP pivot appears to have resolved this item for
the US leg and nobody recorded it.** Two caveats carried into the spec: not
yet confirmed against `crsp.stknames` history (strong evidence, not proof),
and `icbindustry` is a sector/exposure classification rather than a
legal-structure signal — right for sector work, wrong for entity exclusions.

### Canadian sector data — escalated, no source exists

CHASS's only classification field is `business-Business`: **844 distinct
free-text values**, already contaminated enough to need a 239-name
hand-curated CSV just to separate funds from operating companies. Spec section
11 Item 4 calls the Canadian sector-neutral variant the one that matters most
(without it, Canadian BAB is a defensives-vs-resources trade). It currently
cannot be built. Sourcing problem, not a refactor; may not be solvable within
this project's data access. Escalated in the spec rather than left to surface
at Phase 6.

### Config grid — structural gap named before it gets expensive

`config/universe.yaml` expresses ONE universe with scalar values; spec section
8's grid needs named variants. The precedent is already in the same file —
`canada_exchange_sets` (lines 30-32) defines `tsx_only`/`tsx_and_tsxv` as two
named cells for exactly this purpose. Generalise that to the US block and to
size restrictions **before** `src/portfolio/` is written; retrofitting after
is materially more expensive.

### Docs updated this session

`02_validation_gates.md` (status table rewritten with a Why column;
cross-cutting correlation finding; Gate 0c downgraded; Gate 1 US retraction;
Gate 3 real status; Gate 4 benchmark + RF sections; ordered path to Gate 4),
`00_spec.md` (section 4 RF amendment, section 8 config-grid gap, Open Items 4
and 9), `README.md` (stale status and repo layout), plus this entry. Prior
worklog entries left intact.

## 2026-09-08 — A2/A3: AQR BAB loader + both RF loaders (Phase A, parallel to A1)

Built while the CRSP re-pull ran (both are pure file parsing, no dependency
on it). `src/data/aqr_bab_loader.py`, `src/data/risk_free_us.py`,
`src/data/risk_free_canada.py`, new `config/aqr_bab.yaml`, 27 unit tests
(fixture/parsing-logic + real-file anchors matching
`docs/02_validation_gates.md`'s Gate 4 table exactly: USA 1147 non-null
12/31/1930-06/30/2026, CAN 473 non-null from 02/28/1987). AQR loader caches
its parsed output to `data/raw/aqr_bab_factors.parquet` per the roadmap,
mtime-invalidated against the source workbook (user approved writing this
one derived file into `data/raw/`, an exception to its immutability rule
since it's new derived output, not a rewrite of a WRDS pull).

**leakage-auditor found two real defects in `risk_free_canada.py`, both
fixed:**
- `panel.unique(subset=[date])` gave no guarantee which row survives if
  security rows ever disagreed on `ind2` for a month (true today, but
  unenforced) — now asserts `n_unique() <= 1` per month before dedup, and
  a fixture with disagreeing values raises instead of silently picking one.
- The two `derived_overrides` null-check ran against the post-dedup single
  row, so a mixed null/non-null month could dedup to a null row and let the
  hardcoded override value silently overwrite a genuine `ind2` observation.
  Now checks all pre-dedup rows for that month. (In practice the first fix's
  `n_unique` guard already catches this case, since null counts as a
  distinct value — kept both asserts so the second doesn't depend on
  assertion ordering.)

Everything else the auditor checked came back clean: no full-sample
statistics applied retroactively, no survivorship filtering in any of the
three loaders, Ken French's two-table parse trap holds under three
adversarial re-download scenarios (grown/shrunk/coincidentally-realigned),
`rf_source` tagging cannot desync from the value it describes.

`python run.py unit`: 136 passed, 1 skipped, 2 failed — both pre-existing
`test_universe_panel_us.py` staleness checks tied to A1's still-incomplete
CRSP re-pull, not touched by this work. `python run.py lint` / `typecheck`
clean on all new files (remaining errors are pre-existing, in files this
diff didn't touch). Added `types-openpyxl` to `.venv-wrds` for typecheck.

## 2026-09-13 — Gate 4 US: real numbers, characterized not passed

`src/gates/gate4_bab.py` (new) runs `fp_baseline_us` monthly against AQR's
published US factor. Full sample 1970-2025 (599 months): correlation
0.9416, mean 0.641%/mo vs AQR 0.718%, Sharpe 0.705 vs 0.735, market loading
−0.077 (t=−2.79) vs FP's own published −0.06. Full numbers, subperiod
breakdown, and open items in `docs/02_validation_gates.md`'s new Gate 4
result section — not duplicated here.

**Design change mid-implementation:** the plan called for a `fold_back`
delisting-placement mode alongside `as_stamped`. Found wrong, not just
incomplete: two real permnos (22518, 87899) are structurally identical at
the daily-panel level — last trade, then a delisting row one trading day
later crossing a calendar-month boundary — with opposite correct answers,
distinguishable only by a portfolio's OWN formation calendar, which the
data layer cannot see without coupling to portfolio construction. Dropped
`fold_back` entirely; `run_backtest`'s existing missing-coverage skip
(built for the same don't-fabricate-a-return reason) handles both cases
correctly with one mechanism. `characterize_skips()` (73/672 skips, 10.9%,
decade-concentrated in the *earlier* half of the sample) replaces the
originally-planned fold_back-vs-as_stamped delta as the measured finding.

**Four pre-existing bugs surfaced, three the same shape.** Full 56-year run
crashed on a Rust memory allocation failure; root-caused to
`us_gate1_panel` and `_market_log_return_series` each materializing the
FULL multi-decade per-name panel (~71M rows) before ever collapsing to the
~14K-row per-date index actually needed — no prior caller (Gate 1/2) had
ever requested more than a single year, so this was invisible until Gate 4.
Fixed via internal yearly chunking (`us_gate1_panel`) plus a new
`market_index.build.build_index_chunked` (deliberate, documented, temporary
duplication), each re-applying the pre-existing `_LAG_CARRY_DAYS=10`
boundary convention at every chunk edge, each proven bit-identical to an
unchunked reference over a real year-boundary span (`assert_frame_equal` /
exact `max(abs(diff))==0.0`, not approximate — Gate 3's own history shows
pinned-number tests alone are structurally blind to small boundary shifts).
Separately found `canada_gate1_panel`'s `years_needed =
sorted({start.year, end.year})` silently drops every year strictly between
the endpoints — the same bug class already fixed once in
`beta_fp._daily_log_returns` — invisible for the same reason (every
existing caller used single-year spans). One pre-existing test's pinned
values (`test_canada_sigma_m_differs_between_capped_and_uncapped_index`)
had been computed against the buggy, data-starved panel and needed
re-pinning, not just re-running, once the fix supplied real data.
**Pattern worth carrying forward:** any `(start_date, end_date)`-range
function needs a multi-year test by default, not only a single-span one —
three of four bugs this session were exactly this shape.

**Open, not yet done:** injections proving each of the four assertions can
fail; `gate-verifier`; the not-yet-implemented band constants in
`tests/gates/test_gate4_bab.py`'s slow tests; why individual long/short leg
realized returns show near-zero market correlation in isolation despite
healthy ex-ante beta dispersion (logged in `02_validation_gates.md`, not
blocking); whether skips cluster in market-stress months or a particular
leg (required by the original plan, not yet computed). No PASS recorded.

## 2026-09-13 — Gate 4 leg-correlation item: closed, retraction not a bug

Follow-up session investigated the near-zero leg-market-correlation item
above. Root cause: the throwaway diagnostic script
(`_scratch/gate4_leg_beta_drift.py`) used `gate4_bab._next_month_end_of`
(identity on a month-end) where `rebalance.run_backtest` uses
`rebalance._next_month_end` (advances one calendar month) — joined every
formation date's weights to the **formation month's** own return instead of
the held month's. Two near-identically-named helpers, opposite meanings.

Mid-investigation the session briefly concluded this was "a genuine
look-ahead violation" in the pipeline itself, before realizing the cached
parquet it was reading was the prior session's own reconstruction, not
pipeline output. A synthetic isolation of `run_backtest` (Jan +0.10 / Feb
+0.20 / Mar +0.30, formation 2015-01-31 → implied return exactly
`+0.200000`) settled it the other way. Worth naming: the retraction
happened *before* the finding was written up, which is the discipline
CLAUDE.md asks for — catch the misdiagnosis before it becomes a claim, not
after.

Corrected join, full pipeline rebuild (1,917,384 position rows): long leg
realized β 0.737 (ex-ante 0.703, corr 0.824), short leg 1.515 (ex-ante
1.415, corr 0.862) — both FP-shaped (Table III: realized exceeds ex-ante,
gap widens with beta), not the near-zero (0.03-0.05) correlations
originally reported. Bonus: the two beta-scaled legs' own market loadings
(1.0735, 1.1507) difference to exactly −0.0772, mechanically explaining the
gate's headline market-loading number rather than leaving it
"coincidentally-plausible." No `src/` change — docs-only
(`02_validation_gates.md`, `03_roadmap.md`); rebuilt series confirmed
bit-identical to the committed one (max abs diff 0.0). Fast suite: 146
passed, 1 skipped.

Skip-clustering (market-stress / which-leg) remains the one open item from
the original plan. No PASS recorded.

*(Both resolved the following day — skip-clustering computed (neither
effect holds) and Gate 4 US recorded as PASS; see the 2026-09-14 entry
below.)*

## 2026-09-14 — Gate 4 US: bands, injections, gate-verifier/leakage-auditor — PASS

Closed the three gaps left after the 2026-09-13 characterization
(`docs/02_validation_gates.md`'s "From 'characterized' to PASS" section has
the complete account; this entry is the session log).

**`|market_loading_t| < 2.0` could not be applied as planned.** Measured
against the real 599-month comparison frame: AQR's own published US BAB
factor, regressed on the same market series, scores t=−2.165 — outside the
bar. FP's own published −0.06 at this project's own SE(β)=0.0277 gives
t=−2.17 — also outside. The bar rejects the benchmark itself; it measures
sample size (n=599), not defect presence. Replaced with a comparison
against AQR's OWN loading directly: `(ours − aqr)` regressed on the market
gives β=−0.01267, t=−1.258 — indistinguishable, and this is what an
external-anchor gate should actually test.

**Real defect found in the band functions before trusting any injection
evidence.** First implementation recomputed each band LIVE from whatever
`result` dict a caller passed (`max(2*SE, 1.5*measured_gap)` computed from
`result`'s own fields). Tested against a x100 scale injection: the
injection inflates its own gap term by the same x100, widening the band to
match — the assertion never fired. This is the same class of vacuous
assertion CLAUDE.md already warns about (`>0` bounds passing on a single
surviving row), just algebraic rather than structural. Fixed before any
injection was recorded as evidence: all four bands are now frozen
constants in `config/gate4.yaml`, computed once against the real baseline,
never recomputed from the result being judged.

**Injection evidence, 6 total.** Four run against
`_scratch/gate4_legs_fixed.parquet` (independently verified to rebuild the
committed series to 8.3e-17 max abs diff — mathematically identical to a
pipeline re-run for injections downstream of beta estimation): `ret x100`,
`+20bp/month` additive, `beta_long`/`beta_high` swap, `weight_short`
zeroed. Each fired only its predicted assertion(s); baseline passed all
four clean. Two full-pipeline injections (~15 min each, real
`build_bab_series` runs): shuffling all 672 `betas_by_date` keys, and
truncating `monthly_rets` at 2020-12-31 — both tripped
`rebalance.run_backtest`'s own `max_consecutive_skips` circuit breaker
(`RuntimeError`, 7 consecutive skips) before reaching the gate's own
statistics, a stronger catch than planned but one that left the exact
`n_dates`/`n_skips` pins unexercised. Added a milder single-held-month
drop (2025-06-30) as a third full-pipeline run: `n_dates` 599→598,
`n_skips` 73→74 fired correctly while correlation barely moved
(0.94158→0.94150) — the precise scenario the pins exist for.

**`gate-verifier` first pass: INSUFFICIENT, two findings, both fixed.**
(1) Adversarial check (500 randomized `result` dicts, `{}`, `None`)
confirmed the frozen-band fix is real — but the `loading_diff_t_bound`
yaml comment's claim that a t-stat is inherently scale-robust
("self-normalizing... cannot inflate its own denominator") was traced
through the actual OLS math and found FALSE for this bound specifically:
under a scale injection on `ours` alone, `|t_diff|` saturates at ~2.786
rather than diverging; under a uniform scale on both series or an
additive bias, it is exactly invariant. Comment corrected to record this
bound as a redundant secondary check riding on `loading_diff_band`'s
frozen value, not an independently robust statistic — so a future edit
doesn't remove the frozen band on a false premise. (2) `market_loading_aqr`
(the one genuinely external, real-world-checkable number in the function's
output, ≈−0.0645 vs FP's published −0.06) and `n_ours_only` (the
join-asymmetry guard the function's own docstring tells a reader to check)
were both computed and returned but never asserted — added
`-0.15 < market_loading_aqr < 0.0` and `n_ours_only == 0`.

**`leakage-auditor`: no temporal leak, one SUSPECT, fixed.** All three
market-loading regressions (ours/aqr/diff) verified numerically consistent
to ~1e-16 via `β(ours)−β(aqr) == β(ours−aqr)`, which holds only under
identical row ordering — a real falsifiable check. SUSPECT: the pin-tie
test compared two config constants against pure arithmetic on two dates
(`599+73==672`, true forever regardless of pipeline behavior), and
`build_bab_series`'s beta loop silently drops a formation date with an
empty cross-section — reaching neither `returns` nor `skips`. Fixed:
assert `n_dates + n_ours_only + len(skips) == candidates` against the
LIVE run; independently confirmed zero formation dates vanish in the real
run via `_scratch/gate4_positions.parquet`/`gate4_skips.parquet` (599
distinct formation dates + 73 skips = 672 exactly).

**Final state.** 9 fast tests pass (~50s; "12" as originally written here
was a miscount — the file collects 15, 9 fast + 6 slow, re-verified
2026-09-14). 6 real slow tests (no
injection, fully reverted pipeline) pass clean, run twice independently
(before and after the gate-verifier/leakage-auditor fixes, 86 min each) —
identical results both times. `git status` clean after every injection
revert. `lint`/`typecheck` clean on all changed files.

**Status: ✅ PASS.** Note: `docs/02_validation_gates.md`'s Gate 4 section
also carries a skip-clustering follow-up (which leg triggers the
missing-coverage skip, market-stress correlation) added concurrently by a
separate piece of work during this session — not part of this entry, see
that section directly for its own findings (neither effect holds).

**Not done, explicitly out of scope:** Canada leg. Correlation's own
independent evidentiary weight remains unproven (both full-pipeline
injections tripped the circuit breaker before reaching it) — a graded
shuffle under the circuit-breaker threshold would close this; not blocking
since every defect class is covered elsewhere.

## 2026-09-15 — §6 items A/B/C: long-only vs market, drawdown, regimes

Built the three deliverables that make report §6 say anything
(`docs/06_handoff_section6.md` A/B/C). D (benchmark comparison) and E
(sector composition) deferred by user decision; D(c)'s WRDS pull not
attempted.

**Five new modules in `src/analysis/`**, 69 new tests, test-before-code
throughout: `quarterly_compounding` (extracted from
`_grouped_quarterly_analysis`), `excess_returns`, `longonly_vs_market`
(item A), `drawdown` (item B), `regimes` (item C). Driver:
`_scratch/section6_run.py`.

**A real defect found before any code was written.** Canada's market
index and risk-free series are keyed on CHASS real trading days; the
cached quarter labels are calendar quarter-ends. Compounding without
re-keying gives `{2: 110, 3: 30, 1: 8}` months per quarter — 118 of 148
quarters silently partial — while returning a plausible number for all
148 with no error and no missing rows. Measured cost if unguarded: rf
understated 0.25pp/quarter, **Canada Sharpe 0.6205 → 0.7089**. Same trap
`longonly_data.py` already documents for Canadian *stock* returns; the
rf and index series had no such fix. This is why `n_months` is a
first-class output and `assert_full_quarters` exists.

**Extraction acceptance bar met.** Old inline loop vs new helper on the
real US market series: max abs diff **exactly 0.0** across 231 quarters
(`_scratch/_verify_isolate_out.txt`). The committed Sec 4 bucket parquet
differs at ~1e-16, but *unmodified git HEAD reproduces the same
differences* — the non-reproducibility predates this change. Note for
future sessions: that parquet is **not** bit-reproducible run to run, so
any regression test on it needs a tolerance, not equality.

**Four injections, all fired as predicted** (`_scratch/_cmp_*.txt`):
no-rekey-ca (guard raises naming 118 quarters); market×2 (beta ratio
0.4946/0.4917, portfolio stats exactly 1.0000); portfolio+1% (alpha
moves **exactly** +0.010000, beta ratio exactly 1.0000, US alpha_t_stat
1.43→4.25 — the concrete demonstration of correlation-blindness);
rf-as-sum (~26bp annualized, and `raw_outperf` invariant, confirming the
documented claim that rf cancels in the difference).

### gate-verifier: INSUFFICIENT on first pass — four real findings, all fixed

Every one reproduced independently before fixing
(`_scratch/_verify_audit_out.txt`):

1. **`n_months` counted ROWS, not distinct months.** Input (Jan, Jan,
   Feb) with March absent returned `0.3310000000000004` and
   `n_months=3` — byte-identical to a correct quarter — and
   `assert_full_quarters` passed it clean. This defeated the module's
   entire justification. Now a hard error.
2. **The driver's market-on-market anchor could not fail.**
   β = cov(x,x)/var(x) = 1 *identically*; measured, a ×1000 units defect
   scored β=1.0000000000000004. Its docstring claimed it caught "a
   LEVEL/SCALE/ALIGNMENT defect" — that claim was false and is now
   corrected in place. Replaced with external plausibility bounds on
   annualized return/vol, demonstrated to catch ×100, ×1000 and ÷100
   (`_scratch/_verify_new_anchor_out.txt`). **Stated limit: a +50bp/month
   bias still passes** (18.89% is plausible) — these are a units
   tripwire, not a bias detector.
3. **NaN propagated silently.** One null gave `realized_beta=nan`,
   `alpha=nan`, `port_sharpe=nan` with no exception and `n_quarters`
   still reporting the full count. Now `assert_clean_quarterly_series`
   on all three consumers.
4. **Duplicate/gap quarters unguarded.** A duplicated quarter moved
   max drawdown −0.44 → −0.61; a removed quarter moved it −0.44 → −0.30.
   Contiguity is opt-in (`require_contiguous`) since only
   path-dependent statistics need it.

Also fixed the self-referential `WEALTH_SEED` test (compared the module's
output against the module's own constant) and pinned the crisis-quarter
counts, which were printed but never asserted — the `our_n_firms`
pattern. The pin is demonstrated capable of failing: removing the
containment clause drops `black_monday_1987` 1 → 0 and the pin fires
(`_scratch/_verify_crisis_pin_out.txt`).

gate-verifier confirmed sound, and I did not weaken them: the 0.331 /
0.030301 / −0.50 / −0.75 anchors, the doc-parsing anchor (including its
`>=5` guard — verified both drift directions turn it red), the ddof=1
t-stats, the geometric-vs-arithmetic pin, `periods_per_year=4`, and
alpha_t_stat/beta_t_stat separation.

### Measured results (headline 60m window, EXCESS returns)

| | US | Canada |
|---|---|---|
| n quarters | 231 (1968-06-30..2025-12-31) | 148 (1989-03-31..2025-12-31) |
| port ann. excess | 6.26% | 7.49% |
| mkt ann. excess | 7.83% | 7.02% |
| port Sharpe | 0.435 | 0.620 |
| mkt Sharpe | 0.437 | 0.453 |
| realized beta | 0.536 | 0.452 |
| alpha (qtr) | 0.508% | 1.049% |
| **alpha_t_stat** | **1.432** | **2.595** |
| raw outperf. | −0.374%/qtr, t=−0.857 | +0.110%/qtr, t=+0.212 |

Total-return drawdown: US book −39.75% vs market −46.37%; Canada book
−25.80% vs market −37.80%.

**Three findings that contradict prior expectations — reported, not
softened:**

- **US alpha is INSIGNIFICANT** (t=1.43) and US raw outperformance is
  **negative** (t=−0.86). The handoff's expectation of significant alpha
  holds only for Canada (t=2.59), whose raw outperformance is itself
  insignificant (t=0.21).
- **Low beta drew down LESS than the market here** (US −39.7% vs −46.4%),
  the opposite of §5's monthly finding. Different object, different
  frequency. The book's gap vs §5 (−39.75% quarterly vs −52.59% monthly,
  12.8pp) is much larger than the market's (4.0pp) — worth an explicit
  sentence in the report rather than the blanket "different object" note.
- **Downturn outperformance is large** (US +6.60%/qtr, t=+5.05; CA
  +9.02%/qtr, t=+5.45) **but is close to an arithmetic consequence of
  β≈0.45–0.54**, not independent evidence of defensiveness. Expansion
  underperformance is significant for the US (−1.18%/qtr, t=−2.75). The
  spec already forbids the "low beta is defensive" pitch; §6.4 should say
  the downturn result is largely what β<1 implies.

Regime definitions pre-registered in `docs/07_regime_preregistration.md`
BEFORE any regime statistic was computed, and held to that document by a
test that parses it.

Verification: `345 passed, 1 skipped` (analysis + portfolio + unit);
earlier full non-gate sweep `431 passed, 1 skipped`; ruff and mypy clean.
Gates not re-run (slow, untouched). `docs/05_report_spec.md` updated with
both ⚠ corrections and the convention notes.

**Not done:** items D and E; no PASS recorded for any gate.
