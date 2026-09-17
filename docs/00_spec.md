# 00 — Specification & Decision Log (P0 v3)

Every discretionary choice, its justification, and how it gets validated.
Items marked **OPEN** are unresolved and must be closed before Phase 2 ends.

Revision history: v1 (initial scope) → v2 (post data-source discovery) →
**v3 (current, post WRDS diagnostics)**.

---

## 1. Research question

Primary: does the BAB effect hold in Canadian equities out-of-sample relative to
FP's sample end (March 2012), and how much of any measured alpha survives
construction choices rather than depending on them?

Secondary: reproduce FP's US result closely enough to validate the pipeline
against an external benchmark (AQR's published BAB series).

---

## 2. Data sources — RESOLVED (v4: superseded twice since v3)

**v3's Compustat-primary-for-both-countries plan did not survive contact with
real data quality problems on either leg, and has been replaced twice.**
Both replacements are merged and are what the actual codebase runs today;
this section is rewritten (not just annotated) to keep this file an accurate
decision log rather than a record of a plan that was abandoned. Full
derivation for both pivots: `01_data_notes.md`.

**US leg: CRSP-primary** (`crsp.wrds_dsfv2_query`, via
`src/data/pull_universe_us_crsp.py` / `src/data/universe_panel.py`), not
Compustat. Pivoted because `comp.company.sic`/`gind`/`gsubind` turned out not
to be point-in-time (`01_data_notes.md` §14) — a real lookahead risk this
project's core rule (CLAUDE.md) cannot tolerate — while CRSP's
`wrds_dsfv2_query` view supplies point-in-time membership, classification
(`issuertype`, a materially better REIT signal than `icbindustry` — §14),
price, shares, and `vwretd` back to 1925 in one query, with
`usincflg='Y'` reproducing legacy `shrcd IN (10,11)` exactly (§16). Merged
2026-09-04.

**Canadian leg: CHASS/CFMRC** (TSX/CFMRC Summary Information database, via
`src/data/convert_chass_csv.py` / `src/data/chass_loader.py` /
`src/data/chass_universe.py`), not Compustat. v3 explicitly dropped CHASS in
favor of Compustat's Canadian coverage; that coverage turned out to have
real problems this project's own rigor caught before they became silent
bugs — most notably `business-Business`-category fund/REIT/MLP
misclassification requiring full hand-curation (`config/chass_fund_
classification.csv`) and the discovery that CUSIP is not a usable identity
key at all in Canadian data (real CUSIP collisions between unrelated
securities). CHASS also required building its own point-in-time membership
proxy (row-presence at monthly grain, since its authoritative Ticker
History table is not accessible via this project's CHASS browser install)
and surfaced a structural, unfixable gap CRSP does not have: **no
delisting-return field at all** — a security's return series simply stops
at its last real trade with no reconciling value for a merger/bankruptcy
outcome (`01_data_notes.md` §6). This is an accepted, documented asymmetry
between the two legs, not something patched over. Merged 2026-09-05.

**TSXV inclusion is now doubly relevant and still open.** CHASS's Summary
Information database is TSX-only, by construction — TSXV data lives only in
a completely separate intra-day TAQ database starting 2018-01-02, a
different file format not yet investigated. This means the TSXV-inclusion
open item below cannot be resolved from CHASS Summary Information at all
(nor could it from Compustat's `exchg` field, per §3's existing caution
against using `exchg` as an inclusion gate). Any TSXV decision will need
either that separate TAQ database or an accepted scope limitation to
TSX-only.

**CRSP retained for the US leg's validation role** as originally planned:
`crsp.dsf` / `crsp.dsedelist` for delisting-return treatment matching FP's
own construction, and `crsp.ccmxpf_lnkhist` for the return-formula
cross-check — both independent of the CRSP-primary pivot above, which
replaces the *membership/classification* source, not this validation use.

---

## 3. Universe — RESOLVED

**Rule:**

```sql
-- Canada
fic = 'CAN' AND tpci = '0' AND s.iid = c.prican
-- US
fic = 'USA' AND tpci = '0' AND s.iid = c.priusa
```

Exact equality. No wildcard or prefix matching (see `01_data_notes.md` §4 for
why an earlier prefix approach was wrong).

**Do NOT filter on `exchg`.** It is descriptive (which venue), not an inclusion
criterion. Verified: a naive `exchg IN (7,9)` Canadian filter would still work
under the `prican` rule, but `exchg` should not be the gate — it varies by which
security row is selected and is not a reliable membership test.

**No size or price screen in the base universe.** Size restrictions are grid
cells (§8), not screens. Pre-filtering to "top 50% by market cap" — considered in
v1 — was rejected because it converts the central Novy-Marx/Velikov question into
an assumption and breaks comparability with AQR's published series.

**Liquidity handling** is a filter on estimation-window quality, not universe
membership: minimum count of non-zero-volume days within the vol window.
Motivated by two real observed failure modes (`01_data_notes.md` §6).

**OPEN — TSXV inclusion.** ~1,689 of ~2,614 Canadian names in a test month are
TSXV (`exchg = 9`), median market cap ~$3M vs. ~$192M on TSX. Under rank
weighting these would dominate the portfolio. Decide: include (matching FP's
unfiltered approach), exclude, or — preferred — run both as grid cells and
report the delta.

**DIAGNOSED — `priusa` vs. CRSP `shrcd IN (10,11)` (Open Item 3, §11).** Not
equivalent. Single test month (2015-06), Compustat-priusa linked via clean CCM
to 4,174 names vs. CRSP shrcd(10,11) 3,788 names; intersection 3,652 (~88% of
the smaller set). Two systematic divergence mechanisms, confirmed against raw
CRSP `shrcd`/`comp.security` records, not sampling noise:

1. **REITs and MLPs (one-directional, dominant).** 522 names only in
   `priusa`; of the 445 with a live CRSP name record, 184 are CRSP `shrcd=18`
   (REIT) and 145 are `shrcd=71`, plus smaller buckets. Top-30 by size is the
   REIT/MLP universe outright — Simon Property, Prologis, Public Storage,
   AvalonBay, Enterprise Products Partners, Energy Transfer, Magellan
   Midstream. `priusa`'s `tpci='0'` treats these as ordinary common equity;
   CRSP's common-stock definition excludes them by construction.
2. **Multi-share-class primary-flag disagreement (bidirectional, smaller).**
   136 names only in CRSP shrcd(10,11), 100% `shrcd=11`. Confirmed directly on
   Berkshire Hathaway (`priusa` points to `iid='02'`/BRK.B; the CRSP-side name
   in the diff is the other class), Alphabet, and the Liberty Media family
   (LMCA/LMCB/LMCK, LINT) — Compustat's single `priusa` pointer and CRSP's
   per-permno `shrcd=11` flag pick different classes of the same multi-class
   firm. Concentrated in mega-caps; ~3.6% of the CRSP-side universe by count.
3. 3,563 `priusa` gvkeys have no CCM link at all (shells, dormant, subsidiary
   entities) — not comparable to CRSP by construction, excluded from the
   above counts.

**RESOLVED — exclude REITs and MLPs** from the `priusa` universe for the
FP/AQR-comparability run. This is a share-type/entity-structure screen, not
the size/price screen ruled out above — a different question, not settled by
that decision.

**Exclusion mechanics, verified against `comp.company` reference tables
(`comp.r_giccd`, `comp.r_issuetyp`):**

- **REITs:** `sic = '6798'`. Clean, exact — confirmed against all 10 known
  REITs in the item-3 diff (Simon Property, Prologis, AvalonBay, Equity
  Residential, Ventas, Welltower, Macerich, Essex Property, American Tower,
  Crown Castle) with zero misses.
- **MLPs:** no single clean field exists. `tpci` only distinguishes
  common/preferred/etc. (per `comp.r_issuetyp`), not legal entity structure —
  confirmed by checking known MLPs (Enterprise Products, Plains All American,
  Magellan Midstream, Williams Partners, Energy Transfer) all sit at
  `tpci='0'` same as ordinary common stock. SIC alone is unusable: these five
  MLPs scatter across SIC 1311, 4610, 4922, 5171 — codes shared with ordinary
  C-corp oil/gas and pipeline companies, so an SIC-only exclusion would drop
  legitimate common stock, not just fix the definitional gap.
  **Tested rule: union of (a)** `comp.company.gsubind = '10102040'` ("Oil &
  Gas Storage & Transportation", confirmed against `comp.r_giccd` — note this
  is GSUBIND, not GIND: `gind='101020'` ("Oil, Gas & Consumable Fuels") is one
  level up and too broad, catching ordinary E&P/refining C-corps) **and (b)**
  a name-suffix regex on `comp.company.conm`, `[\s\-,]L\.?P\.?$` (name ends in
  an LP designation, e.g. "SUNOCO LP", "AMERIGAS PARTNERS  -LP").

  **Tested against the full priusa universe, 2015-06 (n=7,738):** REIT
  `sic='6798'` → 248 names. MLP GICS rule → 85. MLP name rule → 110. Union →
  133 (62 caught by both, 23 GICS-only, 47 name-only). Total excluded 380;
  remaining universe 7,358 (95.1% of priusa).

  **Two bugs found and fixed during testing, not present in the rule above:**
  an earlier keyword search over GICS descriptions matched `452020` ("Tech
  Hardware, Storage & Peripherals") on the word "Storage" — replaced with the
  hardcoded, verified GSUBIND code. A bare `PARTNERS` alternative in the name
  regex caught ~11 ordinary C-corps with "Partners" in their brand name
  (Artisan Partners Asset Mgmt, Partners Bancorp, Green Brick Partners Inc,
  two bank holding companies) — fixed by anchoring the regex to an actual LP
  suffix at the end of the name. **Spot-checked the resulting 47 name-only
  matches individually post-fix: zero false positives**, every match ends in
  a genuine LP designation (Icahn Enterprises LP, AllianceBernstein Holding
  LP, United Dev Funding III LP, etc). REIT side had zero false positives
  from the start, verified against all 10 known REITs in the item-3 diff.

Diagnosed and tested on one month; not yet wired into pipeline code (`src/`
does not exist yet). See §11 item 3.

---

## 4. Currency — RESOLVED, with one caveat

USD for US-listed, CAD for Canadian-listed. Local currency is cleaner than FP's
USD-unhedged international treatment; because betas are estimated against a
local index and BAB is self-financing within each country, FX largely drops out.

**Dependency:** risk-free rate must match currency. Ken French one-month T-bill
(US); Bank of Canada 3-month T-bill via Valet API (Canada). Mismatching these
produces a small drift easily misattributed to the strategy.

> **AMENDED 2026-09-08 — Canada no longer uses BoC.** The Valet-API line above
> was written when the Canadian leg was Compustat-based; it predates the CHASS
> pivot (§2, merged 2026-09-05) and is stale rather than a considered
> rejection of CHASS. The Canadian risk-free rate is now CHASS's own
> `ind2-30 day Return on T-Bills`, for three verified reasons: it is already a
> monthly decimal return (no annualization step, no day-count convention to
> get wrong); its coverage (1980-01-31 → 2025-12-31, 552 months) matches the
> CHASS equity panel's range exactly, so BoC's deeper history is unusable
> anyway; and it is keyed on CHASS's own **real trading month-ends**
> (2015-01-30, not calendar 2015-01-31) — the same convention
> `gate_adapters.canada_gate1_panel()` and
> `gate1_index.compound_daily_index_to_monthly()` already depend on. A second
> vendor would reintroduce the month-end alignment trap this codebase has
> already hit once.
>
> **Use `ind2`, never `ind1`.** `ind1-91 Day T-Bill Rate` is an annualized
> rate in percent (mean 4.99, max 20.82 in the 1981 spike) — a different
> instrument on a different convention. Measured `corr(ind2, ind1/1200) =
> 0.944`, not ~1.0, and the gap widens post-2009. Do not substitute one for
> the other, and do not use one to validate the other.
>
> Two months (2023-10-31, 2023-11-30) are null in `ind2` and are filled by a
> flagged `annualized_rate / 1200` derivation — a proxy on a neighbouring
> convention, not observed data. Full detail and the loader's flagging
> requirement: `config/risk_free.yaml`.
>
> **US side unchanged** (Ken French), with one addition: Gate 4 specifically
> uses AQR's own RF series instead, since AQR built the BAB factor we
> correlate against using it. See `02_validation_gates.md`, Gate 4.

**Composite construction:** if reporting a North America composite, rescale each
country series to 10% ex-ante volatility on a rolling 3-year estimate and
equal-weight (FP's own multi-asset aggregation method). Do not convert returns.

**Caveat resolved:** interlisted megacaps (RY, TD, SU, ENB, CNQ, BMO, CM, BNS)
were initially suspected to lack native CAD pricing. Verified false — e.g. RY
`iid='01C'` has 10,686 CAD-priced rows from 1983-12-30 to present. No exception
needed.

---

## 5. Cross-listing — RESOLVED

Assign country at issuer level, **fixed for the security's life**, never by
exchange at time *t*. A name that changes primary listing mid-sample must not
jump universes — that creates a phantom entry/exit unrelated to beta.

The `prican`/`priusa` rule in §3 handles this. A dual-listed company has both
flags populated and appears once in each country's universe, on the
correspondingly-priced security.

---

## 6. Minimum history — RESOLVED

**No separate filter.** The estimator's own requirement binds: FP's spec needs
750 non-missing days (3 years) for the correlation window, which dominates any
1-year listing requirement. A 12-month daily OLS variant drops this to 120 days.

**Required diagnostic:** log the count and characteristics of names excluded by
the 750-day minimum each month. Recent IPOs are systematically high-beta and
high-vol, so this mechanically thins the short leg. Part of the estimator
comparison is this selection difference, not just the estimator itself.

---

## 7. Market index — RESOLVED

**Build it; do not use SPX/TSX Composite as the estimation benchmark.**

Rationale: beta is only meaningful against a proxy spanning the universe being
sorted. Estimating against a large-cap index while sorting the full tail gives
small stocks understated betas (low correlation with a large-cap index reads as
low beta), which pushes them into the long leg and turns BAB into a size trade.
Also, BAB is market-neutral *relative to the index used to estimate betas* — so
estimation and evaluation benchmarks must be the same object, or a non-zero
realized loading can't be distinguished from a bug.

**Construction:** value-weighted, total return, **lagged weights** (prior day's
close × shares outstanding), delisting returns included, total shares not free
float.

**Canadian variants required:**
- Uncapped VW (methodologically standard)
- **Capped VW (10% single-name limit)** — Nortel was roughly a third of the TSE
  300 at its 2000 peak, contaminating every Canadian beta estimated on a 5-year
  correlation window from ~1997 to ~2006. Treat capped-vs-uncapped as a result,
  not a footnote.
- MSCI-Canada-like large-cap proxy — reproduces what FP actually did
  internationally, which differs from what they did for the US.

**Note the inconsistency in FP worth exploiting:** for the US they used the CRSP
value-weighted index (spans the universe). For the 19 international markets they
used MSCI local indices (large/mid-cap only) while sorting all available common
stocks. If Canada's result differs materially between these two benchmarks, that
partly explains why FP's Canada number was among their strongest.

---

## 8. Estimators & construction grid

**Estimators** (common interface, config-swappable):
- FP spec: ρ from overlapping 3-day log returns over 5y; σ from 1-day log
  returns over 1y; β̂ = ρ·σᵢ/σₘ; shrink β = 0.6·β̂ + 0.4·1
- 12-month daily OLS, same 0.6 shrinkage

Shrinkage is retained in both. It does not affect ranking (monotone), only leg
scaling — its function is keeping implied leverage honest.

**Required estimator diagnostic:** monthly cross-sectional regression of β̂ on
σᵢ and on ρ separately. Because ρ is estimated over 5y and only σ over 1y,
cross-sectional beta dispersion is expected to be dominated by recent
volatility. Quantify this before interpreting any result — it is the bridge to
the Asness-Frazzini-Gormsen-Pedersen betting-against-correlation decomposition.

**Grid:**

| Axis | Cells |
|---|---|
| Beta estimator | FP spec / 12m daily OLS |
| Weighting | rank-weight (FP) / value-weight |
| Leg scaling | asymmetric 1/β_L vs 1/β_H (FP) / dollar-neutral + explicit market hedge |
| Universe | full / ex-microcap / large-only |
| Canada only | TSXV in / TSXV out; sector-neutral / not |

FP's headline occupies one cell. Where alpha survives across cells and where it
collapses is the finding.

> **STRUCTURAL GAP — assessed 2026-09-08 (status audit).** The current config
> layout cannot express this grid, and the cheapest moment to fix that is
> **before** `src/portfolio/` is written.
>
> - **Beta estimator axis — expressible today.** `config/beta_estimator.yaml`
>   (on the unmerged Gate 3 branch) already uses an `fp_spec:` top-level key,
>   so a sibling `ols_12m:` block is a natural addition. No refactor needed.
> - **Weighting / leg scaling / universe axes — no surface exists.**
>   `src/portfolio/` does not exist yet, so there is nothing to refactor —
>   but `config/universe.yaml` currently expresses exactly ONE universe, with
>   single scalar values for `us_major_exchanges_crsp`, `reit_issuertype`,
>   etc. The ex-microcap and large-only cells, plus any minimum-market-cap
>   knob, need either a named-variant layer
>   (`variants: {full: {...}, ex_microcap: {min_mkt_cap: ...}}`) or a
>   per-run override mechanism.
> - **The precedent already exists in this file's own config.**
>   `canada_exchange_sets` (`config/universe.yaml:30-32`) defines `tsx_only`
>   and `tsx_and_tsxv` as two named cells specifically so the TSXV grid
>   comparison (Open Item 1) can be run. That pattern generalises directly;
>   it simply has not been applied to the US block or to size restrictions.
>
> Retrofitting named variants after the portfolio module is written is
> materially more expensive than choosing the shape up front.

**Monthly logging, every run:** realized market loading, dollars long, dollars
short, ex-ante beta spread, name count, weight fraction in smallest size decile.

---

## 9. Sample periods — RESOLVED

| Run | Window | Purpose |
|---|---|---|
| Common | 1977–present | Headline US-vs-Canada comparison, same period both countries |
| US max | ~1965–present | Subperiod stability; deepest AQR validation overlap |
| Post-publication | 2013–present | The out-of-sample test. FP's Canada sample ends Mar 2012. |

Daily data throughout. **Do not splice monthly-estimated betas for the
pre-1962 era** — a monthly-estimated and daily-estimated beta are different
signals and splicing creates a structural break mid-sample.

---

## 10. Formation timing — RESOLVED

Betas from data through the last trading day of month *t−1*; positions held
through month *t*. Run a skip-one-day variant as robustness, since the same
closing prices otherwise feed both the beta estimate and the first day's return.

---

## 11. OPEN ITEMS

| # | Item | Why it matters | Blocking? |
|---|---|---|---|
| 1 | TSXV inclusion (§3) | Dominates rank-weighted Canadian portfolio | Phase 4 |
| 2 | Income trusts, 2001–2011 | High-payout, low-vol, trust-structured; sit squarely in the low-beta leg. The 2006 tax announcement and 2011 conversion deadline restructured the whole cohort. Include/exclude deliberately and measure their share of the Canadian long leg. | Phase 6 |
| 3 | US universe: does the `priusa` rule reproduce CRSP `shrcd 10/11`? **Resolved (§3): no — exclude REITs (`sic='6798'`) and MLPs (`gsubind='10102040'` ∪ LP-suffix name regex) from the `priusa` universe for the FP/AQR-comparability run. Tested on 2015-06: 380 of 7,738 names excluded, spot-checked with zero false positives.** Multi-share-class primary-flag disagreement (Berkshire, Alphabet, Liberty Media) remains a smaller, undecided bidirectional gap — not addressed by this filter. Rule tested standalone; not yet wired into pipeline code. **Re-resolved 2026-09-03 (Task 4) against the CRSP-primary full-history panel design's actual pipeline output (`universe_at()`, `usincflg='Y'`-based) rather than raw `shrcd`, on 2015-06-30, CCM clean-linked: 3,632 `priusa` permnos, 3,772 `usincflg`-based permnos, overlap 3,225. Stated from all sides, not just the most favorable: Compustat-only 407/3,632 = 11.2% of `priusa`'s own universe, CRSP-only 547/3,772 = 14.5% of `usincflg`'s own universe, Jaccard/union overlap 3,225/4,179 = 77.2% — the often-quoted 3,225/3,632 = 88.8%-of-the-smaller-set figure is the most favorable of these ratios, not the representative one. What makes this low-risk is not the raw overlap percentage under any denominator, but that both "only" buckets are fully explained, zero unexplained residual — Compustat-only (407) is 100% this design's own REIT/MLP/CEF/foreign-incorporation exclusion rules already working as intended; CRSP-only (547) is a mix of known CCM link-history gaps (48) and two real Compustat-vs-CRSP rule disagreements confirmed directly: `fic != 'USA'` foreign-incorporated/US-listed firms (26, e.g. Cayman/Ireland/UK-domiciled tax inversions — a genuinely new finding, not seen in the original diagnosis) and the `exchg`-descriptive-field/multi-share-class-pointer disagreement already flagged in §3 (429, now confirmed directly rather than inferred, includes the Zillow priusa-points-to-non-trading-class case). `priusa` and `usincflg`-based membership diverge by ~12-15% of either side but with every divergent name traced to a named mechanism; this design's choice of `usincflg` is confirmed low-risk on that fully-explained basis. See `01_data_notes.md` §16 for the full per-bucket derivation.** | Determines whether US results are comparable to FP/AQR | Phase 2 |
| 4 | Sector-neutral variant construction | Canadian low-beta = financials/utilities/telecom/pipelines; high-beta = junior energy/mining. Without neutralization this is a defensives-vs-resources trade, not BAB. **ESCALATED 2026-09-08 (status audit) — the Canadian half of this item has NO DATA SOURCE. CHASS's only classification field is `business-Business`, which holds 844 distinct free-text values. It is not a taxonomy, and it is already known to be contaminated enough to require a 239-name hand-curated CSV (`config/chass_fund_classification.csv`) merely to separate funds from operating companies — a far coarser task than sector assignment. So the variation this spec identifies as the most important one on the Canadian leg currently cannot be built at all. This is a data-sourcing problem (GICS via another vendor, or hand-mapping ~7,500 names), NOT a refactor, and it may not be solvable within this project's current data access. It should be raised now rather than discovered at Phase 6, because the answer may change what the Canadian contribution can claim. The US half of this item is fine — see Item 9's 2026-09-08 addendum.** | Phase 6 — **Canadian half at risk, escalate now** |
| 5 | Borrow feasibility for the Canadian short leg | Small TSX/TSXV borrow is thin, expensive, or nonexistent. A long-only or long-biased variant may be the more decision-relevant test for a real Canadian mandate. | Phase 7 |
| 6 | Funding-liquidity proxy for Canada (Props 3/4) | TED has no direct Canadian analog; candidates are CORRA-OIS or bankers' acceptance spreads | Phase 7 |
| 7 | `comp.secd.cshoc` (shares outstanding) can be wrong by orders of magnitude, corrupting market cap silently. Confirmed case: gvkey 062212 (UNB Corp, OTC community bank), `cshoc=22.17M` implies a $3.46B market cap; real-world cap is ~$4-5M — true shares outstanding is ~1000x smaller. Not a `tpci`/security-type issue (ruled out via web search — genuinely common stock). **Resolved for US leg — switched to CRSP `dsf` via CCM (`src/data/market_cap.py`), sidesteps the field rather than characterizing its error rate. 93.87% CCM match rate (3,630/3,867 gvkeys) on the 2015-06-30 cached US universe. UNB Corp itself is untestable as a before/after pair — it has no CRSP/CCM coverage at all (OTC, excluded from the exchange-filtered universe cache; its only CCM link row is `linktype='NU'` with no PERMNO) — see `01_data_notes.md` §13 for the full honest finding. Canada (`comp.funda` fallback) remains open, separate phase.** See `01_data_notes.md` §9, §13. | US leg: Gate 1/2 can proceed. Canada leg still blocks Gate 1/2 (VW index, size deciles). |
| 8 | `comp.company.dldte`/`costat` are current-database-state fields, not point-in-time. 518 of 1,249 names with zero volume across a full 5-day test window carry a `dldte` *after* the pull date — using these fields to filter a historical universe is lookahead. Universe/liquidity logic must be built from `comp.secd` trading history up to date *t* only. See `01_data_notes.md` §9. | Blocks Gate 2/3 |
| 9 | `comp.company.gind`/`gsubind` (GICS) **and `sic`** are current-database-state, not point-in-time — same failure class as Item 8, confirmed via out-of-sample test: 1985/1995/2005/2025 US pulls all show near-zero GICS null rates despite GICS not existing before 1999. Quieter than Item 8 — returns a plausible-looking code, not an obviously-wrong null or future timestamp. **`sic` audited directly 2026-09-03: `comp.company` is confirmed exactly one row per gvkey (58,378/58,378) with no date-versioning on any field except `ipodate`/`dldte`; a separate GICS-history table (`comp.co_hgic`) exists but is unused by the current pipeline, while no SIC-history table exists anywhere in Compustat — `sic` cannot be sourced point-in-time from this database at all. See `01_data_notes.md` §14.** Not a bug in the single-cached-month pipeline; becomes a real lookahead risk the moment a point-in-time multi-month panel depends on `sic`/`gind`/`gsubind` — `src/data/universe_panel.py`'s REIT/MLP exclusion (2026-09-03 full-history-panel plan) drops the `gsubind` signal entirely and flags `sic` as similarly unsafe, keeping only the LP-suffix name regex unconditionally. See `01_data_notes.md` §12, §14. **PARTIALLY RESOLVED 2026-09-08 (status audit): the finding above is about COMPUSTAT and remains true there — but the US leg no longer sources classification from Compustat. CRSP's `wrds_dsfv2_query` carries `siccd` and `icbindustry` on every daily row, and they genuinely vary through time per permno: over 2000/2005/2010/2015, 2,526 of 10,425 permnos show >1 distinct `siccd` and 3,972 show >1 distinct `icbindustry`, with a 0.0 null rate on both. That time-variation is the signature of a point-in-time field, so the CRSP pivot appears to have resolved this item for the US leg as a side effect, and nobody had recorded it. Two caveats: (a) not yet confirmed against `crsp.stknames` history — the daily-row variation is strong evidence, not proof; (b) `icbindustry` is a sector/exposure classification, not a legal-structure signal (it tags CBRE/Zillow as REIT — see `config/universe.yaml`'s `reit_issuertype` comment), so it is the right field for SECTOR work and the wrong one for entity-structure exclusions. This matters for spec §8's sector-neutral grid cell: the US leg has usable PIT sector data in hand; Canada does not (see Item 4).** | Blocks point-in-time panel (in progress, 2026-09-03 plan) |
| 10 | Exact-`datadate` queries in `pull_universe.py`/`pull_universe_canada.py` have no trading-calendar fallback — a non-trading date (confirmed: 1985-06-30, a Sunday) returns 0 rows silently, indistinguishable from a connection failure. Not yet hit in committed code (`TEST_MONTH_END` is a trading day), but will bite any future arbitrary-date extension. See `01_data_notes.md` §12. | Blocks point-in-time panel (not yet started) |
| 11 | `comp.secd.cshoc` has large null stretches in earlier decades, distinct from Item 7's wrong-value bug — US ~100% null before ~1996-97, Canada 100% null in a spot-checked 1995 pull. Blocks any market-cap-dependent logic (VW weighting, size deciles) for these periods without a fallback source (`comp.funda`, or CRSP `shrout` via CCM). **Resolved for US leg (CRSP has no equivalent null-gap problem for this period — the 2015-06-30 test month join has a 93.87% CCM match rate, remaining unmatched gvkeys are genuine per-name CCM link-history gaps, not a systematic null stretch; see `01_data_notes.md` §13). Canada spot-check still shows nulls — separate phase.** See `01_data_notes.md` §12, §13. | US leg: resolved for the test month; still needs the full time-series pull (post-plan follow-up) before the US-max sample period (spec §9, ~1965-present) is fully unblocked. Canada leg unresolved. |
