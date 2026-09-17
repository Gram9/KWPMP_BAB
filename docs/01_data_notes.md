# 01 — Data Notes

Findings from the WRDS diagnostic session. Everything here was verified
empirically against the live database, not taken from documentation.

Several of these cost significant debugging time. They are recorded so they
cost it once.

---

## 1. Schema locations

| What | Where | Trap |
|---|---|---|
| CCM link history | `crsp.ccmxpf_lnkhist` | Lives in the **`crsp`** schema despite the `ccm` prefix in its name. `ccm.ccmxpf_lnkhist` does not exist. |
| Security-level fields | `comp.security` | Only identifiers: `gvkey`, `iid`, `tic`, `cusip`, `exchg`, `tpci` |
| Company-level fields | `comp.company` | Everything describing the business: `conm`, `gsector`, `ggroup`, `gind`, `gsubind`, `sic`, `fic`, `priusa`, `prican`, `prirow`, `dldte`, `dlrsn`, `costat` |
| Daily security prices | `comp.secd` | `prccd`, `ajexdi`, `trfd`, `cshtrd`, `cshoc`, `curcdd` |

**Rule of thumb:** if the field describes the *business*, it's on `company`.
If it describes the *security*, it's on `security`. Getting this wrong produces
`UndefinedColumn` with a helpful `HINT` naming the correct alias.

---

## 2. Return construction from Compustat

Compustat does not provide a return field. Build it:

```
p_adj = (prccd / ajexdi) * trfd
ret   = p_adj / p_adj.shift(1) - 1
```

- **`trfd` already reinvests distributions.** It is the analog of CRSP `ret`,
  not `retx`. Do not separately add dividends back — that double-counts.
- **`ajexdi` can be near-zero but not literally zero** (values like 3e-8 have
  been reported). These display as 0 at reduced precision and explode the
  division. Guard on magnitude, not `== 0`.
- **Blank `trfd` ≠ `trfd = 1`.** Observed on a security's first row. Treat as a
  missing return, not a zero-adjustment day.
- Boundary rows at merger effective dates can carry a populated `trfd` with null
  price/volume (observed: Compaq, 2002-05-06). Auto-computed returns on such
  rows are undefined, not zero.

---

## 3. Canadian exchange codes — verified empirically

| `exchg` | Exchange | Test-month name count | Median mkt cap |
|---|---|---|---|
| 7 | **TSX** | 924 | ~$192M |
| 9 | **TSX Venture** | 1,689 | ~$3M |
| 1 | minor/legacy | 1 | — |
| 11, 12, 14 | NYSE / AMEX / NASDAQ | — | — |
| 19 | Other–OTC | — | — |

Confirmed two independent ways: known blue chips (RY, TD, SU, ENB, CNQ, BMO,
CM, BNS) all land under `7`; and the market-cap distributions differ by roughly
two orders of magnitude at the median (90th pct: ~$4.07B on `7` vs. ~$28.6M
on `9`).

**Note:** the JKP global factor library treats `exchg` values 1, 13, and 19
(among others) as non-"main exchange" codes, consistent with the tiny counts
observed for 1 and 13 here.

---

## 4. The `iid` / `prican` / `priusa` system — READ THIS BEFORE WRITING ANY
   COMPUSTAT QUERY

A single `gvkey` can carry **multiple simultaneously-primary securities** — one
per region (US, Canada, rest-of-world). For an interlisted Canadian company
these are genuinely **separate rows** in `comp.security` with separate pricing
in `comp.secd`:

Example — Royal Bank of Canada (`gvkey = 015633`):

| `iid` | `exchg` | `curcdd` | rows | span |
|---|---|---|---|---|
| `01` | 11 (NYSE) | USD | 7,769 | 1995-10-16 → present |
| `01C` | 7 (TSX) | **CAD** | 10,686 | 1983-12-30 → present |

`priusa = '01'`, `prican = '01C'`. Both are real securities with real price
histories. The `C` is **not** a suffix tag on a flag field — it is part of an
actual `iid`.

**Correct filter: exact equality.** `s.iid = c.prican`.

**Rejected approaches and why:**
- `LIKE s.iid || '%'` and `strpos(prican, iid) = 1` both **over-match**: `'01'`
  is a literal prefix of `'01C'`, so the US-side row passes a Canadian filter.
  This silently restored the duplication the filter was meant to remove
  (2,753 names vs. the correct 2,614).
- `LEFT(prican, 2) = iid` assumes 2-char `iid`, which fails on 3-char values
  like `01C` — produced 1,770 names, dropping TSX/TSXV entirely.

**Verified correct result** (single test month, `fic='CAN'`, `tpci='0'`,
`iid = prican`): **2,614 distinct names** — 924 TSX, 1,689 TSXV, 1 other.

---

## 5. Ticker convention on interlisted names — a real trap

Compustat appends a **trailing period** to the Canadian-side ticker of an
interlisted security:

| Company | US-side `tic` (`iid='01'`) | Canada-side `tic` (`iid='01C'`) |
|---|---|---|
| Royal Bank | `RY` | `RY.` |
| TD | `TD` | `TD.` |
| Enbridge | `ENB` | `ENB.` |

**Never join or filter Canadian entities on `s.tic` without accounting for
this.** A lookup for `'RY'` against the Canadian universe returns empty, which
looks exactly like the company being absent from the universe. This produced a
false "the megacaps are missing" conclusion during the diagnostic session.

Prefer `gvkey` for joins. Use `conm` for human inspection.

---

## 6. Delisting & survivorship

### CRSP — behaves as documented

| Company | `dlstcd` | `dlret` | Read |
|---|---|---|---|
| Lehman | 574 | −0.60 | Distress (500-range), captured |
| Circuit City | 574 | −0.48 | Distress, captured |
| Compaq | 231 | +0.048 | Merger (200-range), near fair value |
| Countrywide | 231 | +0.021 | Merger |

Conclusion: CRSP's `dlret` is populated and economically sensible for both
distress and merger delistings. Shumway (1997)'s missing-delisting-return
correction should still be implemented for coverage gaps, but the mechanism is
working in this data.

### Compustat `dlrsn` codes

| Code | Meaning |
|---|---|
| 01 | Acquisition or merger |
| 02 | Bankruptcy |
| 03 | Liquidation |
| 04 | Reverse acquisition |
| 05 | No longer fits original format |
| 06 | Leveraged buyout |
| 09 | Now private |
| 10 | Other — no longer files with SEC |

**Trap: neither Lehman nor Circuit City is coded `02`.** Both are `10`. A filter
for `dlrsn IN ('02','03')` would miss two of the most famous bankruptcies in the
sample. `10` is a large mixed bucket and must be cross-checked against CRSP
`dlstcd` 500-range rather than trusted alone.

### Company-record closure lags real delisting by years

| Company | Stock stopped trading | Compustat `dldte` | Lag |
|---|---|---|---|
| Lehman | Sep 2008 | 2012-03-07 | 3.5 yr |
| Circuit City | Nov 2008 | 2011-09-29 | ~3 yr |
| Compaq | May 2002 | 2002-05-06 | days |
| Countrywide | Jun 2008 | 2008-07-01 | days |

Clean mergers close near-simultaneously; **bankruptcy/liquidation lags by
years** while the proceeding winds down.

**Consequence:** `costat = 'I'` + populated `dldte` tells you the company record
eventually closed, **not** when the security stopped being investable. Any
universe logic keyed on Compustat's activity flag will keep economically dead
companies in the pool for years.

### Entity continuity differs between databases

Washington Mutual: CRSP kept **PERMNO 81593 alive continuously** through the
holding company's Chapter 11 → WMI Holdings (2012) → Mr. Cooper Group (COOP,
2018). Its `dlstcd = 100` is the still-active placeholder, not a delisting; the
2008 equity wipeout appears in ordinary `ret`, not in `dsedelist`. Compustat
instead **closed the old gvkey** (`016243`, `dlrsn = 05` "no longer fits
original format", 2012-09-19) and opened a successor.

**Implication:** don't QA delisting coverage by spot-checking famous
bankruptcies — parent-shell survival through reorganization is common enough
that some "known failures" aren't delistings at all. Do it systematically: for
every security whose return series stops before the sample end, assert a
corresponding delisting record exists.

### Two distinct bad-data modes in `comp.secd` — both real, both observed

**Zombie quotes (Lehman, Nov–Dec 2014):** 15 consecutive trading days, price
frozen at $0.55, **`cshtrd = 0.0` every day**, implied return exactly 0.0.
A stale indicative quote carried forward for a non-trading security.
→ Zero realized variance ⇒ understated σᵢ ⇒ β pushed toward zero ⇒ **a defunct
bankrupt shell sorts into the low-beta long leg.**

**Sub-penny noise (Circuit City, Sep 2011):** genuinely trading (nonzero volume
daily) at $0.0021–$0.0035 during Chapter 11 wind-down. A one-tick move is a 40%
"return."
→ Inflated σᵢ ⇒ **pushed into the high-beta short leg on pure noise.**

Both failure directions are represented. A filter must catch both. Use these two
names as concrete unit tests once the liquidity filter is built.

**`comp.secd` having a row for a date is not evidence the security traded that
date.** Gate on `cshtrd > 0`, not row presence.

### CHASS (Canadian leg) — no delisting return at all, confirmed structural gap

Unlike CRSP's `dlret`, **CHASS Summary Information has no field that captures
a delisting/merger consideration price.** Confirmed directly against two real
2016-era cash-and-stock-adjacent mergers:

| Company | Last real trading day | Last price/return captured | What actually happened |
|---|---|---|---|
| Bema Gold (BGO) | 2007-02-28 | $7.30, real volume, real daily return | All-stock merger into Goldcorp — price organically converged to deal terms through ordinary trading, so this case happens to be fine |
| Progressive Waste Solutions (BIN) | 2016-05-31 | $41.40, volume spike, May 2016 monthly return = +2.80% (real, computed) | Acquired by Waste Connections; June 2016 row is fully NaN (close/return/shares_out all missing) — **the actual deal consideration, if it differed from the last open-market trade, is nowhere in the data** |

The daily/monthly return series simply **stops** at the last real trading day.
There is no Price Adjustment record (dividend/recap flags, §14 of the CHASS
user's guide) for delisting — those flags cover ordinary corporate actions
(dividends, splits, stock-in-lieu) during a security's *active* life, not
termination events. A security's final return is whatever its last real trade
implies, with no reconciling entry for the gap between that trade and the true
economic outcome (cash payout, exchange ratio, or total loss).

**This creates the same bias direction CRSP's `dlret`/Shumway (1997) exists
to fix, in both directions:**
- Bankruptcy-style delistings disappear at their last (often already-depressed,
  possibly stale/illiquid) trade rather than their true near-zero value —
  understates realized losses.
- Acquisition-style delistings disappear at their last open-market trade,
  which is frequently below the actual deal price (tender offers/mergers
  typically close at a premium to the pre-announcement or pre-halt trade) —
  understates realized gains.

**No fix is available from CHASS Summary Information.** Checked and ruled
out: Price Adjustment records (wrong purpose, active-life corporate actions
only), the Ticker History table (`DICTION.DAT` — gives delisting *dates*, not
delisting *values*, and is separately inaccessible via the CHASS browser GUI
in this project's install, see the Canadian universe pull notes). The CHASS
daily TAQ database (intra-day trade/quote, 2018-01-02+ only) was not
investigated for this — it might narrow the gap for the most recent portion
of the sample by showing the actual last intra-day print/halt, but would not
help at all for the bulk of the historical window and has not been explored.

**Consequence:** the Canadian leg of this project has a real, currently
unresolved asymmetry against the US leg (CRSP-primary `dlret` is populated
and economically sensible, per the CRSP subsection above). Any comparison of
US vs. Canadian delisting-driven return behavior should account for this —
the Canadian side will structurally understate the return impact of both
bankruptcies and buyouts. Treat as an accepted data limitation, not something
to silently patch with an invented value.

---

## 7. CCM link table

Values observed: `linkprim` ∈ {C: 58201, P: 54596, N: 6695, J: 3896};
`linktype` ∈ {NR, NU, LC, LU, LS, LX, LN, LD, NP}.

Standard clean-link filter: `linktype IN ('LC','LU','LS')`, `linkprim NOT IN
('J','N')`.

**`linkprim = 'N'` + `linktype = 'LX'` is a cross-listing marker.** `LX` denotes
a Compustat security trading on a foreign exchange not itself covered by CRSP,
linked to a related CRSP-covered security; `N` is documented as arising from
duplicated links caused by Canadian securities associated with a US-traded firm.
Nearly every sampled `linkprim='N'` row carried `LX`. Potentially useful for
independently identifying dual-listed names.

**Duplicate rows are normal.** CCM splits one continuous link into consecutive
date-range rows at administrative boundaries. Observed for XOM and JNJ — same
PERMNO on both rows, so benign.

**Do not `drop_duplicates()` in the real pipeline.** A company genuinely can be
reassigned to a different PERMNO over its history; blind dedup would discard a
real segment. Join with an explicit `linkdt <= date <= COALESCE(linkenddt,
CURRENT_DATE)` condition, or `pd.merge_asof` on the link windows.

---

## 8. Environment gotchas

**`%` in raw SQL breaks psycopg2.** Any literal `%` — in `ILIKE '%foo%'` or
`LIKE x || '%'` — collides with the DBAPI placeholder parser, producing the
misleading `TypeError: immutabledict is not a sequence`. Fixes: pass patterns as
bound parameters (`ilike %(pat)s` with `params={'pat': f'%{pat}%'}`), or avoid
`%` entirely via `strpos()` / `left()`.

**Postgres aborts the whole transaction on any error.** The WRDS connection is
one long-lived transaction, so after any failed query every subsequent query
raises `PendingRollbackError` until cleared. Fix: `db.connection.rollback()`,
or `db.close(); db = wrds.Connection()`. Wrap it:

```python
def q(sql, **kwargs):
    try:
        return db.raw_sql(sql, **kwargs)
    except Exception:
        db.connection.rollback()
        raise
```

**pandas copy-on-write warnings.** `df.sort_values(...)` can return a view;
subsequent column assignment warns. Use `.copy()` or `.reset_index()` to get an
unambiguous object before assigning.

---

## 9. Liquidity-gate design work (2026-09-02) — two new traps found

While diagnosing what should replace a naive single-day `cshtrd > 0` gate for
universe/estimation-window construction, US priusa panel, 2015-06-24 to
2015-06-30 (5 trading days):

**A single-day zero-volume gate is too blunt to use as-is.** Of 2,209 names
with `cshtrd = 0` on 2015-06-30 specifically: 956 (43%) traded on at least one
other day that same week — a single-day cut would wrongly kill live names that
just happened to be quiet that day. Only 1,253 (57%) were zero-volume across
the *entire* 5-day window. Confirms spec §3's existing design intent: liquidity
belongs in the estimation window as a **minimum-count-of-non-zero-volume-days**
threshold, never a single-date screen.

**`comp.company.dldte`/`costat` are CURRENT-DATABASE-STATE fields, not
point-in-time.** Of the 1,249 names with zero volume on all 5 days in the
window: 518 carry a `dldte` *after* 2015-06-30 (up to 2023) — meaning, as of
the pull date, they had not yet delisted. `costat`/`dldte` reflect the state as
of whenever the table was last refreshed, not state as of any historical date.
**Using `comp.company.costat`/`dldte` to filter a historical universe is
lookahead** — it lets today's knowledge of "this company eventually died"
leak into a period-t membership decision. Any survivorship/liquidity logic
must be built from `comp.secd` trading history up to date *t*, never from
these company-level flags.

**`cshoc` (shares outstanding) can be wrong by orders of magnitude — not a
`tpci` misclassification.** Case: gvkey `062212` (UNB Corp / United National
Bank & Trust, Mount Carmel PA), `iid='01'`, `priusa='01'` — the flagged
primary US listing, ticker `UNPA`, OTC Pink Sheets since 1995. Price sat
frozen at exactly $156.00 for the entire month with `cshtrd=0` every single
day. Implied "market cap" (`prccd * cshoc` = $156 × 22,173,000 shares =
**$3.46B**) is not believable for a single-branch-county community bank —
confirmed against real-world data (current market cap ~$4-5M, ticker still
UNPA on OTC, checked 2026-09-02 via web search). $156 × real cap of ~$4-5M
implies true shares outstanding near ~30,000, roughly **three orders of
magnitude below** the 22.17M in `comp.secd.cshoc`. Initially misdiagnosed as
a `tpci` preferred-stock classification bug — ruled out: `UNPA`'s "PA" is
this company's actual OTC ticker, not a preferred-share suffix, and the
underlying security is genuinely common stock. The real defect is in
Compustat's `cshoc` field for this name, cause unconfirmed (stale/unadjusted
share count, corporate action not reflected, data entry error).

**Consequence for market-cap-based logic (size deciles, VW weighting,
microcap flags):** a bad `cshoc` corrupts market cap directly, silently,
without tripping any price or volume sanity check. **Needs a cross-check
against an independent shares-outstanding source (e.g. CRSP `shrout`) before
`cshoc` is trusted for weighting or size-decile construction** — this is
larger than the immediate liquidity-filter question. Scope unknown: only one
instance confirmed so far; not yet swept across the full universe.

**Independently confirmed three ways, 2026-09-02:**
1. Real-world current market cap ~$4-5M (Yahoo Finance, stockanalysis.com).
2. FDIC call report (RSSD 249416, cert #7631): total bank equity capital
   ~$10M, total assets $167M as of 2026-Q2 — no plausible price-to-book
   multiple for a bank this size reaches $3.46B at any point in its history.
3. `exchg=19` independently verified against WRDS's own `comp.r_ex_codes`
   reference table = "Other-OTC" (confirms `01_data_notes.md` §3's existing
   table was already correct; ruled out a hypothesis that 19 might mean TSX).

Distinct from, and not to be confused with, gvkey/CIK 746481 ("UNIZAN
FINANCIAL CORP", formerly "UNB CORP/OH", Canton OH, SEC filings through
2005) — a same-named but unrelated company surfaced by naive web search.
Compustat's `incorp='OH'` for gvkey 062212 (Mount Carmel PA business
address, `state='PA'`) is a real but separate fact — likely a holding-company
incorporation-state choice — not evidence of confusion with the Ohio entity.
Bloomberg independently tracks it as "UNB Corp/PA", confirming it is
recognized elsewhere as a distinct name from the Ohio company. No SEC EDGAR
10-K found for this entity — consistent with a small OTC Pink Sheets bank
holding company filing with its primary regulator (FDIC/OCC) instead of the
SEC, common below Exchange Act Section 12(g) registration thresholds.

**No separate "primary exchange" field exists on the Compustat side.**
`crsp.dsenames.primexch` (legacy) / `crsp.stksecurityinfohist.primaryexch`
(CIZ format) are CRSP-only. `comp.secd`/`comp.security` expose only `exchg` —
checked directly against both tables' column lists. This is not a gap: the
`priusa`/`prican`/`prirow` system already resolves to one row per region per
gvkey, so `exchg` on that resolved row already serves the role `primexch`
would. A CRSP cross-check of `exchg` against `primexch` would be a genuine
independent validation (same role CRSP plays for delisting returns per spec
§2) but was not run — deferred, not needed to close today's finding.

---

## 10. Canadian universe pull for build_universe (2026-09-02)

Confirmed the `exchg` distribution of the cached Task 1 pull
(`data/raw/can_universe_2015_06.parquet`, `fic='CAN'`, `tpci='0'`,
`s.iid = c.prican`, single `datadate = 2015-06-30` row match):

```
exchg
1       1
7     904
9    1613
Name: count, dtype: Int64

total names: 2518
```

**Compared against Gate 0c (`docs/02_validation_gates.md`, `docs/01_data_notes.md`
§4): 2,614 distinct names (924 TSX / 1,689 TSXV / 1 other).** The Task 1 pull is
96 names lower (2,518 vs. 2,614, a 3.7% gap):

| `exchg` | Task 1 pull | Gate 0c | diff | diff % |
|---|---|---|---|---|
| 1 (other) | 1 | 1 | 0 | 0.00% |
| 7 (TSX) | 904 | 924 | −20 | −2.16% |
| 9 (TSXV) | 1,613 | 1,689 | −76 | −4.50% |
| **total** | **2,518** | **2,614** | **−96** | **−3.67%** |

**Diagnosis: benign, attributable to query shape, not a structural exchange
problem.** The Task 1 query (`src/data/pull_universe_canada.py`) inner-joins
`comp.secd` on an exact `sd.datadate = '2015-06-30'` match to pull price/share
fields alongside the name — any `gvkey`/`iid` satisfying the `fic`/`tpci`/
`prican` base rule but with **no `comp.secd` row printed on that single exact
calendar date** (e.g., halted, thinly-traded-enough to have a gap, or a
quote/date mismatch) silently drops out of the inner join. Gate 0c's original
diagnostic counted distinct names off the base rule directly, independent of
whether a price row exists on one specific day — a different, wider matching
condition. This is consistent with what the split shows:

- Zero duplication (`u.duplicated(subset=['gvkey','iid']).sum() == 0`) — the
  gap is not a join fan-out artifact.
- The `exchg=1` singleton matches exactly (1 vs. 1) — no whole code group
  disappeared.
- Both real exchanges shrink, and **TSXV shrinks proportionally more than
  TSX** (−4.50% vs. −2.16%) — consistent with TSXV's known lower liquidity
  (median mkt cap ~$3M vs. ~$192M on TSX, §3 above) producing more names
  without a printed quote on any single specific day.

No single exchange is missing and the gap is not concentrated in one place —
this is a query-shape artifact of the single-`datadate` join, not a data
integrity or structural problem. Not resolved further here (peripheral to
this task); if a task downstream needs the full 2,614-name base-rule count
(e.g., an exact reconciliation), the fix is to decouple the name-eligibility
join from the price-pull join rather than requiring both in one query.

**`curcdd` check (carried over from Task 1 Step 3):** all 2,518 rows in the
universe parquet have `curcdd == 'CAD'` — zero non-CAD rows found.

---

## 11. Canadian REIT/MLP/income-trust exclusion — derivation

Diagnosed empirically against `data/raw/can_reference_2015_06.parquet`
(n=2,518) and cross-joined to `data/raw/can_universe_2015_06.parquet` for
market-cap sanity checks, following the same evidence-first sequence as the
US derivation (§9/spec §3). **Two of the three US-side signals do not
transfer to Canada and were rejected after testing against real data — not
assumed or ported unchecked.**

### Step 1 — coverage check

```
total: 2518
sic null: 0
gind null: 53
gsubind null: 53
gsector null: 53
```

`sic` is fully populated. GICS fields are null on 53/2,518 = 2.11% of rows —
close to spec §2's existing ~1.5% Canadian GICS-coverage estimate, not a
material blow-up. Proceeded with an SIC-primary, GICS-secondary approach
rather than stopping to rethink the whole design.

### Step 2 — SIC/GICS cross-tab

```
sic == 6798 (US REIT code) count: 43
gsubind == 10102040 (US MLP code) count: 9
```

`sic='6798'` returns a plausible, REIT-sized bucket (43 of 2,518 — same
order of magnitude as the US's 226 of 7,738). **Not near-zero — Canadian
REITs use the identical SIC code as US REITs.** Inspected all 43 names
directly: every one is a real, recognizably-named REIT/real-estate trust
(RioCan, SmartCentres, Boardwalk, the Dream REIT family, Killam Apartment
REIT, etc.), overwhelmingly `gsector=60` (Real Estate) with a couple of
sensible exceptions (Chartwell Retirement Residences under healthcare
`gsector=35`; RFA Financial under financials `gsector=40` — both real
REIT-structured businesses operating in adjacent sectors). **Zero false
positives, mirrors the US SIC signal exactly.**

`gsubind='10102040'` ("Oil & Gas Storage & Transportation", the exact GICS
code the US MLP rule uses) returns only 9 names on the Canadian panel, and
inspecting them by name immediately disqualifies the signal:

```
ENBRIDGE INC, TC ENERGY CORP, TIDEWATER MIDSTREAM INFRASTR, VERESEN INC,
INTER PIPELINE LTD, PEMBINA PIPELINE CORP, KEYERA CORP,
ENBRIDGE INCOME FUND HLDGS, GIBSON ENERGY INC
```

These are Canada's actual pipeline **corporations** — Enbridge, TC Energy,
Pembina, Keyera, Gibson Energy are large-cap C-corps, not MLPs. GICS
sub-industry classifies them here by business activity (pipeline
transportation), not legal entity structure, because Canada does not use
the US MLP structure the same way. **Rejected: applying `gsubind='10102040'`
to Canada would wrongly exclude Canada's pipeline megacaps, not identify
MLP-equivalent entities.**

Also inspected `sic='6799'` ("Investors, NEC", 38 names) as a candidate —
rejected immediately: every name is an ordinary investment-holding/venture-
capital shell (Onex Corp, Senvest Capital, Clairvest Group), not a
REIT/trust bucket.

### Step 3 — name-regex scan for trust/partnership conventions

```python
candidates = r[r['conm'].str.contains(
    r'TRUST|\bFUND\b|\.UN\$| UN\$|REIT| LP\$|L\.P\.\$',
    regex=True, case=False, na=False)]
```

48 matches. Findings by sub-pattern:

- **`.UN$` / ` UN$`** (the real-world TSX income-trust unit-ticker
  convention, e.g. `REI.UN`): **zero matches.** `comp.company.conm` does not
  carry the ticker suffix in the company-name field. Confirmed by the Step 4
  web search on Restaurant Brands Intl LP: its real TSX ticker is `QSP.UN`,
  but `conm` is plain `"RESTAURANT BRANDS INTL LP"`. **Rejected outright —
  zero real hits in this data, not kept just to "be thorough."**
- **`REIT`**: real hits, but every one is already caught by `sic='6798'`.
  Redundant, not incremental.
- **` LP$` / `L\.P\.$`** (anchored, same regex as the US rule): 4 matches —
  `PURE MULTI-FAMILY REIT LP`, `AMERICAN HOTEL INCM PROP LP` (both already
  `sic=6798`), `RESTAURANT BRANDS INTL LP` (`sic=5812`, not caught by SIC),
  and `MACKENZIE MASTER LP` (`sic=6199`) — see Step 5 false-positive check.
- **`TRUST`**: 13 matches, a genuine mix of real income trusts and false
  positives — see Step 5.
- **`FUND`**: heavy false-positive concentration — of the `FUND`-matching
  names, most sit at `sic` 6199/6726 (closed-end funds, split-share corps,
  investment offices), not operating-company income trusts. See Step 5.

### Step 4 — market-cap spot-check of the largest candidates (web-verified)

Joined Step 3's 48 candidates to `can_universe_2015_06.parquet` on `gvkey`
for `prccd * cshoc`. Top 5 by implied market cap, each independently
web-verified (same standard as the UNB Corp check, §9):

| Name | Implied mkt cap (2015-06) | Web-confirmed entity type | Sanity |
|---|---|---|---|
| Restaurant Brands Intl LP | $12.0B | Genuine Ontario limited partnership, formed 2014, TSX ticker `QSP`/`QSP.UN`, indirect holding vehicle for Tim Hortons + Burger King, RBI Inc. is sole GP. | Plausible for a newly-merged multi-billion-dollar QSR group. |
| RioCan REIT | $8.5B | Canada's largest diversified REIT, ticker `REI.UN`. Current (2026) cap ~$6-6.65B per public sources; higher cap in mid-2015 (pre-2018 US-portfolio exit) is directionally consistent. | Sane. |
| SmartCentres REIT (named "Calloway REIT" as of 2015-06; rebranded 2017) | $3.6B | **Exact cross-check**: Calloway REIT's own Q1-2015 supplemental disclosure reports market cap of **$3.98B as of 2015-03-31** (136.7M units × $29.10). Our $3.6B for 2015-06-30 is directionally consistent with a quarter's unit-price movement. | Sane, independently corroborated to the same order of magnitude. |
| Canadian Apt Ppties REIT | $3.27B | Real, TSX-listed apartment REIT (CAPREIT). | Plausible. |
| Boardwalk Real Estate Trust | $2.69B | Real, TSX-listed multi-family REIT (Boardwalk REIT). | Plausible. |

**No UNB-Corp-style data-quality artifact found** — every large-cap
candidate is a genuine REIT/LP with a web-corroborated, economically
sensible market cap. Unlike the US diagnosis, this spot-check surfaced no
separate `cshoc`-type bug.

### Step 5 — false-positive check on TRUST/FUND/LP patterns

**`FUND` and bare `TRUST`, even after excluding the closed-end-fund SIC
codes (`6199`, `6726`), still produce false positives that cannot be
cleanly separated from genuine hits by any single available field:**

- `WESTERN PACIFIC TRUST CO` (`sic=6200`): web-confirmed a **licensed
  trust-company financial institution** (fiduciary/registrar/custodial
  services business, BC/AB/SK), not a real-estate or income trust. Same
  class of false positive as the US's "bare PARTNERS" trap (Artisan
  Partners Asset Mgmt) — an ordinary operating company whose actual
  corporate name happens to contain the trigger word.
- `GOODMAN GOLD TRUST` (`sic=1040`, gold mining — looks like an operating
  miner by SIC): web-confirmed a **closed-end mutual fund** managed by
  GCIC Ltd / Ned Goodman Investment Counsel, not an operating gold-mining
  company. SIC is actively misleading here, ruling out an "exclude fund
  SICs" patch as sufficient on its own.
- The `FUND`-matching names split similarly: `NORANDA INCOME FUND` and
  `DOMINION CITRUS INCOME FUND` look like genuine operating-business income
  funds, but `COXE COMMODITY STRATEGY FUND`, `PENDER GROWTH FUND INC`,
  `BRAND LEADERS INCOME FUND`, `FLOATING RATE INCOME FUND`,
  `PATHFINDER INCOME FUND`, `STONE AGRIBUSINESS FUND` are all closed-end
  investment funds (`sic` 6199/6726), confirmed by the SIC-6199/6726 bucket
  containing an unambiguous cluster of split-share corps and income funds
  (TD Split Inc, 5Banc Split Corp, Faircourt Gold Income Corp, etc.) with no
  further reliable discriminator against the genuine hits.

**Verdict: bare `TRUST` and `FUND` name regexes are rejected**, same
standard as the US "bare PARTNERS" fix — a pattern that still produces
identifiable false positives after the best available tightening is not
shipped.

`Argent Energy Trust` and `Parallel Energy Trust` (both `sic=1311`,
web-confirmed genuine post-2011 SIFT-loophole energy income trusts holding
US oil & gas assets) are real income trusts missed by this decision — no
clean field distinguishes them from the `TRUST` false positives above. They
are **not excluded** by the rule below; see Open Item #2 note.

**The anchored `[\s\-,]L\.?P\.?$` LP-suffix regex (same pattern as the US
rule) has exactly one residual false positive**: `MACKENZIE MASTER LP`
(`sic=6199`), web-confirmed a captive fee-collection vehicle for legacy
Mackenzie mutual-fund sales commissions — it invests solely in a Mackenzie
money-market fund and issues no new units, not an operating LP. **Fix:**
exclude `sic` 6199/6726 from the LP-name match, same fund-SIC exclusion
identified in the TRUST/FUND analysis. This removes Mackenzie Master LP
while keeping all 3 genuine LP matches intact.

### Step 6 — finalized rule

```
is_reit     = sic == '6798'
is_lp_name  = conm matches '[\s\-,]L\.?P\.?$'  AND  sic NOT IN {'6199','6726'}
excluded    = is_reit OR is_lp_name
```

`gsubind='10102040'` and any `TRUST`/`FUND`/`.UN$` name pattern are
deliberately **not** part of the rule — each was tested against real data
and rejected for the reasons above. See `config/universe.yaml`'s
`can_reit_mlp_trust_exclusion` block for the exact values and full
rejection rationale.

### Final counts, 2015-06 Canadian reference panel (n=2,518)

```
is_reit count:    43
is_lp_name count:  3   (2 overlap with is_reit; 1 incremental)
union (excluded): 44
remaining:      2,474
```

The one incremental LP-only catch is Restaurant Brands International LP
(web-confirmed genuine, Step 4). Applied to the raw universe panel
(`can_universe_2015_06.parquet`, n=2,518, all gvkeys present in both the
reference and universe parquets):

| Filter | n before | n excluded | n after |
|---|---|---|---|
| No exchange filter | 2,518 | 44 | 2,474 |
| `tsx_only` (`exchg=7`) | 904 | 42 | 862 |
| `tsx_and_tsxv` (`exchg∈{7,9}`) | 2,517 | 44 | 2,473 |

**Income-trust names are flagged for identification only.** Argent Energy
Trust and Parallel Energy Trust are real income trusts, confirmed genuine
by web search, but excluded from neither this rule nor a defensible
alternative — whether/when income trusts should be excluded from the
Canadian universe at all is spec §11 Open Item #2 (the 2006
tax-announcement / 2011 conversion-deadline question), and remains
deliberately open. This task's rule only removes REITs and the one
Canadian LP-structured entity found; it does not attempt to resolve the
broader income-trust question.

---

## 12. Out-of-sample diagnostic (2026-09-02) — GICS lookahead confirmed, plus two pipeline gaps

Ran the same diagnostic checks (base-rule count, exchange distribution,
sic/gind/gsubind coverage, REIT/MLP/trust exclusion counts, `curcdd` check,
market-cap spot-checks) against 4 new test months — 1985-06-30, 1995-06-30,
2005-06-30, 2025-06-30 — for both US and Canadian universes, comparing
against the existing 2015-06-30 baseline (§9, §11). Reused the identical
query shape from `pull_universe.py`/`pull_universe_canada.py`; nothing in
`src/` was changed for this exercise.

### Finding 1 — `comp.company.gind`/`gsubind` are NOT point-in-time. Confirmed lookahead.

GICS (Global Industry Classification Standard) was introduced by MSCI/S&P
in 1999. The expectation going in was that pre-1999 pulls would show
`gind`/`gsubind` almost entirely null. **They do not:**

```
US reference null rates by month:
1985-06-28  n=2,933  gind null=0.82%  gsubind null=0.82%
1995-06-30  n=4,784  gind null=0.13%  gsubind null=0.13%
2005-06-30  n=4,195  gind null=0.00%  gsubind null=0.00%
2025-06-30  n=3,945  gind null=0.15%  gsubind null=0.15%
```

Spot check, 1985-06-28 US reference pull — real rows, not an artifact:

```
gvkey   conm                          sic   gind    gsubind
001001  A & M FOOD SERVICES INC       5812  253010  25301040
001004  AAR CORP                      5080  201010  20101010
001013  ADC TELECOMMUNICATIONS INC    3661  452010  45201020
```

**Root cause:** `gind`/`gsubind` live on `comp.company` — one row per
`gvkey`, describing the company's *current* state — not on `comp.secd`,
which is genuinely a daily time series. A query against `comp.company` for
a 1985 `gvkey` returns whatever GICS code that company (or its successor
entity under the same `gvkey`) is classified as **today**, not what it was
classified as — or whether the concept even existed — in 1985.

**This is the same failure class as the already-documented `costat`/`dldte`
trap (§9)**: a `comp.company` field silently reflects current database
state, not history, and reads as perfectly ordinary data (a real 6-digit
GICS code, not a null or an obviously-wrong value) — arguably *more*
dangerous than the `costat`/`dldte` case because there is no null or
future-dated timestamp to notice; the value simply looks correct.

**Consequence:** every place this pipeline uses `gind`/`gsubind` —
currently only the REIT/MLP exclusion rule's `gsubind=='10102040'` term for
Canada testing (rejected, not used in the final rule) and the *originally
tested and rejected* US signal — was validated against the single 2015-06
cached month, where "current classification" and "2015 classification" are
close enough not to matter. **This becomes a real lookahead bug the moment
`build_universe` is extended to a genuine multi-month/multi-year
point-in-time panel** (already flagged as future, not-yet-built work in
`universe.py`'s docstring) — a beta-formation-month universe built with
today's GICS codes would let information from decades in the future
silently influence a decision made at that historical date, which is
exactly the rule CLAUDE.md puts above everything else.

**Not urgent today:** the current pipeline only ever reads the single
cached 2015-06 month, where this doesn't bite. Recorded here so it is not
rediscovered the hard way when the point-in-time panel gets built — at that
point, `gind`/`gsubind` must either be dropped from any exclusion rule, or
sourced from a genuinely dated table if WRDS has one (not yet investigated).

### Finding 2 — exact-`datadate` query has no trading-calendar fallback

1985-06-30 is a **Sunday**. Both `pull_universe.py` and
`pull_universe_canada.py`'s exact `sd.datadate = %(month_end)s` match
returned **zero rows** for that date, on both countries, with no error —
indistinguishable from a broken WRDS connection or a genuine data gap.
Confirmed root cause: `comp.secd` has dense, continuous coverage through
all of 1985 (~250-350 US rows/day around that week); the query simply found
no matching row on a non-trading day, silently.

The existing `pull_universe.py`/`pull_universe_canada.py` scripts hard-code
`TEST_MONTH_END = "2015-06-30"`, itself a trading day (Tuesday), so this
gap has not yet been hit in committed code. It will bite the moment either
script (or the eventual point-in-time panel) is pointed at an arbitrary
date without checking it against a trading calendar first.

### Finding 3 — `comp.secd.cshoc` has large null stretches in earlier decades

Distinct from the already-documented UNB Corp `cshoc`-wrong-by-1000x bug
(§9) — this is `cshoc` being **absent**, not wrong:

- US: ~100% null before ~1996-97 (100.0% in the 1985 pull, 99.96% in 1995,
  dropping to 0.9% by 2005, 0.02% by 2025).
- Canada: 100% null specifically in the 1995-06 pull.

**Consequence:** any market-cap-dependent logic (value-weighting, size
deciles, the microcap-weight-fraction diagnostic in spec §8) is unusable
for these earlier periods from `comp.secd` alone. A fallback source (e.g.
`comp.funda`'s annual shares-outstanding field, or CRSP `shrout` via the
CCM link, already planned for the US validation role per spec §2) will be
needed before extending value-weighted construction into this era. Not
blocking today's single-month pipeline; relevant whenever the sample-period
work in spec §9 (US max ~1965-present) actually runs.

## 13. US market cap: CRSP replaces cshoc (spec items #7, #11)

Resolves spec §11 open items #7 (`cshoc` wrong-by-orders-of-magnitude) and
#11 (`cshoc` null stretches) **for the US leg only**. `src/data/market_cap.py`
computes US market cap as `abs(crsp.dsf.prc) * crsp.dsf.shrout * 1000`
(shrout is in thousands), joined `gvkey -> permno` via the clean CCM link
(`linktype IN ('LC','LU','LS')`, `linkprim NOT IN ('J','N')`, date-bounded at
`as_of_date`), using the prior trading day's price per the spec's lagged-
weight rule. This sidesteps `cshoc` entirely rather than trying to
characterize or patch its error rate — CRSP is FP/AQR's own market-cap
source anyway (spec §2).

**Match rate, live WRDS, 2015-06-30 cached US universe:** 3,630 of 3,867
gvkeys (93.87%) matched a valid CCM-linked CRSP row. 237 unmatched, logged
in `coverage_log_df`, never silently dropped.

**UNB Corp (gvkey 062212) does NOT provide a before/after number — genuine
absence of CRSP/CCM coverage, confirmed, not a gap in this fix.** Two
separate facts compound here, checked independently:

1. UNB Corp is absent from `data/raw/us_universe_2015_06.parquet` entirely.
   It is OTC (`exchg=19`, §9 above), and the US universe cache already
   filters to major exchanges only (`exchg IN (11,12,14)` — NYSE/NYSE
   MKT/NASDAQ, spec item #3) before `build_us_market_cap` ever runs. This is
   the existing exchange filter working as designed, not a defect in this
   plan's code.
2. Even bypassing the universe cache and querying CCM directly for gvkey
   `062212` with no date filter, its **only** `crsp.ccmxpf_lnkhist` row is:
   `linktype='NU'`, `lpermno` is null, `linkdt=1996-02-29`, `linkenddt` null.
   `NU` is not in the clean-link set (`LC`/`LU`/`LS`) and carries no PERMNO
   at all — there is no CRSP security to join to. Applying the standard
   clean-link filter at 2015-06-30 returns zero rows.

**UNB Corp genuinely has no usable CRSP coverage, at any date, under any
filter.** This is expected for a single-branch OTC Pink Sheets community
bank — CRSP's coverage universe is exchange-listed securities. The fix's
correctness does not depend on re-deriving this one name's number: the
join mechanism (CCM clean-link, date-bounded, `abs(prc) * shrout`) is
proven correct by `tests/leakage/test_ccm_link_date_bounds.py` (link
date-bound logic, 5/5 passing) and `tests/unit/test_market_cap_us.py`
(4/5 passing on real assertions; the 5th,
`test_unb_corp_market_cap_is_plausible_not_cshoc_implied`, correctly
`SKIP`s for the same reason documented here — UNB Corp is not in the
cache). Per CLAUDE.md ("a wrong number that looks right is the worst
possible outcome"), this note states the true negative plainly rather than
substituting a different name's number to manufacture a before/after pair.

**Multiplier spot-check (large-cap names, live WRDS + cache cross-check,
2015-06-29):** the design spec required verifying `abs(prc) * shrout * 1000`
against known real-world market caps for a large-cap name; this fell out of
scope during this task's own work and is closed here. Four large-cap names
already present in `data/raw/us_market_cap_2015_06.parquet` were checked —
first read from the cache, then independently reproduced with a fresh live
query against `crsp.dsf` joined through `crsp.ccmxpf_lnkhist` (same
clean-link filter and date bound as `pull_market_cap_us.py`'s
`CCM_LINKED_MARKET_CAP_QUERY`) for 2015-06-29 (the prior trading day before
the 2015-06-30 test month end). Both sources agree exactly:

| permno | gvkey  | name              | prc    | shrout (000s) | mkt_cap = abs(prc) * shrout * 1000 |
|-------:|--------|-------------------|-------:|---------------:|------------------------------------:|
| 14593  | 001690 | Apple (AAPL)      | 124.53 |       5,705,400 | $710.49B |
| 10107  | 012141 | Microsoft (MSFT)  |  44.37 |       8,089,575 | $358.93B |
| 11850  | 004503 | ExxonMobil (XOM)  |  82.82 |       4,181,108 | $346.28B |
| 38703  | 008007 | Wells Fargo (WFC) |  56.06 |       5,149,205 | $288.66B |

All four are in line with real-world market capitalizations for these
companies as of late June 2015 (Apple ~$700-740B, Microsoft ~$355-360B,
ExxonMobil ~$340-350B, Wells Fargo ~$285-290B), confirming
`SHROUT_UNITS_MULTIPLIER = 1000` in `src/data/market_cap.py` is correct —
`crsp.dsf.shrout` is indeed reported in thousands of shares, not raw share
counts. This closes the verification the design spec requested twice
(Architecture and Testing sections) that Task 4 did not carry out.

**Negative `prc` (no-trade flag):** 70 of 7,263 raw `crsp.dsf` rows pulled
have negative `prc`. CRSP's documented convention: a negative price is the
bid/ask midpoint stand-in used on a day with no actual trade, not a real
negative price. `build_us_market_cap` takes `abs(prc)` before multiplying
by `shrout`, per the module's existing design (a real liquidity gate on
trading activity is separate, later work — Gate 2/3, not built here).

**Unmatched-gvkey spot-check (10 of 237, from `coverage_log_df`'s
unmatched sample):** `003691` (GT Biopharma), `005849` (BBX Capital),
`007183` (Soluna Holdings), `007662` (Biomerica), `011703` (Flux Power),
`012717` (Anixa Biosciences), `013239` (Koru Medical), `013362` (Naked
Brand Group), `013484` (Soligenix), `013888` (Mr Cooper Group). All ten are
the **same diagnosis, verified directly against `crsp.ccmxpf_lnkhist`**:
each has a genuine `linktype='LC'` clean link both before and after
2015-06-30, but the link-history row that actually covers 2015-06-30 is
coded `linktype='NR'` ("not researched" — a documented CCM linktype value,
§7 above, distinct from the clean-link set) for that specific date range.
CCM itself does not consider the gvkey<->PERMNO link valid at this exact
date for these names, even though one exists on both sides of it. This is
a genuine, verifiable CCM coverage gap for this date — not a join bug in
`build_us_market_cap` or `pull_market_cap_us.py` (confirmed by checking the
full `crsp.ccmxpf_lnkhist` row history for each gvkey with no date filter,
not just the as-of-date-bounded query). No spot-checked case indicated a
join bug; none required stopping to flag before writing this section.

### What did NOT change from this diagnostic

The REIT/MLP/(trust) exclusion rule itself — `sic=='6798'` (both
countries), the anchored LP-name regex, the Canadian fund-SIC carve-out —
**held up structurally across all 4 new months in both countries**, with
counts moving sensibly with known market history (Canadian REIT count
tracked the real 1995→2005 income-trust boom: 3 → 33 excluded) and zero new
false positives found in spot-checks. **No change to the committed
2015-06-30 config values or counts (§9, §11) is warranted by this
diagnostic.** The three findings above are pipeline-robustness gaps for
future multi-month work, not defects in what is currently built and tested.

## 14. Point-in-time audit: `sic` vs. `gind`/`gsubind` (2026-09-03)

§12 already confirmed `comp.company.gind`/`gsubind` (GICS) are
current-database-state, not point-in-time, and flagged this as a real
lookahead trap once a point-in-time multi-month panel gets built (this is
exactly the `src/data/universe_panel.py` work, see the 2026-09-03
full-history-panel plan). This section checks `sic` for the same risk,
live against WRDS, before porting the REIT/MLP exclusion rule into that
panel's point-in-time function.

**`comp.company` is confirmed one row per gvkey, no date-versioning
whatsoever:** `select count(*), count(distinct gvkey)` returns 58,378 /
58,378 — an exact 1:1. Its column list has exactly two date-bounded fields,
`ipodate` and `dldte`; every other descriptive field, including `sic`,
`gind`, `gsubind`, `conm`, `fic`, is a bare current-state value with no
`effdate`/`thrudate`-style companion.

**GICS has a separate historical table; SIC does not.** `comp.co_hgic`
exists (`gvkey`, `ggroup`, `gind`, `gsector`, `gsubind`, `indfrom`,
`indthru` — ~45,860 rows, genuinely date-bounded) — meaning GICS *could*
have been sourced point-in-time, just isn't in the current pipeline (§12's
finding stands: `comp.company.gind`/`gsubind` is the wrong table for that
purpose). A parallel search across every `comp` table with "hist" or "sic"
in its name (`co_acthist`, `co_hgic`, `g_co_hgic`, `g_sec_history`,
`r_siccd`, `sec_history`, `sec_idhist`) found **no SIC-history table at
all**. `r_siccd` is a flat code→description lookup, not history.
`sec_history`/`sec_idhist` are effective-dated, but their `item` values are
`EXCHG`, `EXCHGTIER`, `MKVALINCL`, `PRIHISTCAN`, `PRIHISTUSA` — exchange and
primary-listing history, not SIC.

**Direct check, known test gvkeys** (`030128` Lehman, `062212` UNB Corp —
both already-established names in this project):

| gvkey | conm | sic | gind | gsubind | ipodate | dldte |
|---|---|---|---|---|---|---|
| 030128 | LEHMAN BROTHERS HOLDINGS INC | 6211 | 402030 | 40203020 | 1994-04-29 | 2012-03-07 |
| 062212 | UNB CORP | 6020 | 401010 | 40101015 | (null) | (null) |

One row each, exactly as the count above predicts — there is no second,
earlier `sic` value queryable for either name.

**Conclusion: `sic` carries the identical point-in-time risk as
`gind`/`gsubind`, and is in fact worse-positioned — GICS at least has an
unused historical table (`co_hgic`) that could someday backfill it; SIC has
no such table anywhere in Compustat.** `src/data/universe_panel.py`'s
`universe_at()` must not treat `sic` as safe to use unconditionally across
a multi-decade panel: a name's `sic` reflects Compustat's *current*
classification of that business, applied retroactively to every historical
month, which is a t+1-influences-t violation under CLAUDE.md's rule the
moment a company's line of business actually changed over its history.

**Consequence for the full-history-panel plan's Task 4:** the REIT/MLP
exclusion ported into `universe_at()` cannot rely on `sic` (`reit_sic`
check) or `gsubind` (`mlp_gsubind` check) as point-in-time-safe signals.
Only the LP-suffix name regex on `conm` survives this audit unconditionally
— `comp.company.conm` is also only a current name (a renamed company would
have the same issue), but for the specific MLP-name-regex use case the
risk is a company's *name* changing entity structure denotation over time,
which is a materially narrower failure mode than an entire industry
classification field being wrong for the whole pre-current-classification
history. This is flagged here as an open item, not resolved by dropping
`sic`/`gsubind` alone — see spec's open-items table.

**No change to the committed 2015-06-30 single-month config values (§9,
§11) or code (`src/data/universe.py`) is warranted by this finding** — the
existing single-month pipeline queries `comp.company` for one date only and
was never claiming point-in-time correctness for `sic`/`gind`/`gsubind` in
the first place (see the universe.py module docstring). This finding is
scoped entirely to the new multi-month `universe_panel.py` work.

## 15. `dlycap`'s unit convention, verified directly (2026-09-03, Task 3) — **CORRECTED 2026-09-08, see end of section**

`comp.security_daily`/`crsp.dsf`-successor CRSP-primary panel
(`us_panel_crsp_full`) carries a `dlycap` column alongside `dlyprc` and
`shrout`. Before using `dlycap` as a shortcut for `market_cap_at()`, or
assuming `shrout` here follows the legacy `crsp.dsf.shrout`
thousands-of-shares convention already documented in §13/`market_cap.py`
(`SHROUT_UNITS_MULTIPLIER = 1000`), this was checked directly against the
real pulled data rather than assumed.

Query: for every row on `dlycaldt == 2015-06-30` in
`data/raw/us_panel_crsp_full/year=2015/part.parquet` with non-null
`dlycap`, `dlyprc`, `shrout` (first 20 matching rows), computed
`dlycap / (abs(dlyprc) * shrout)`.

**Result: ratio ≈ 1.0 for all 20 sampled rows** (range 0.9999999999999998
to 1.00000037807555 — floating-point noise only, no systematic deviation).
This was originally read as: `dlycap` is `abs(dlyprc) * shrout` on the
**same day**, with **no unit multiplier** needed — `shrout` in this
CRSP-primary panel is already in raw shares. **This conclusion was wrong
— see the correction below.** The two consequences originally drawn from
it (`market_cap_at()` needs no multiplier; `dlycap` can't be used directly
as a lagged shortcut) — the first is corrected below, the second remains
true (a same-day figure still can't stand in for a lagged one regardless
of its unit convention).

**CORRECTION (2026-09-08, discovered during Gate 2 Task 6):** the
ratio-≈1.0 check above only proves `dlycap` and `abs(dlyprc) * shrout` are
INTERNALLY self-consistent with each other — both are CRSP-computed
quantities, so the check cannot distinguish "both correctly in raw units"
from "both wrong by the same factor together." It never anchored against
a real-world market cap, which §13's original `market_cap.py` verification
(ExxonMobil, Wells Fargo) did do. Running that same kind of anchor check
against this panel: permno 14593 (Apple), 2015-06-30, `dlyprc=125.425`,
`shrout=5,705,400`, `dlycap=7.15599795e8` (~$715.6 million under the "no
multiplier" reading). Apple's real, publicly known market cap that day was
~$715 **billion** — a factor of exactly 1000 higher. Cross-checked against
permno 11850 (ExxonMobil), same date: `dlyprc=83.2`, `shrout=4,181,108`,
implied cap ~$347.9 million under "no multiplier" vs. Exxon's real
~$348 **billion** cap that day — same 1000x gap, ruling out an
Apple-specific split/adjustment artifact.

**Corrected conclusion: `shrout` in this CRSP-primary panel IS in
THOUSANDS of shares, matching legacy `crsp.dsf.shrout`'s convention after
all — the original "no multiplier" finding was a false negative caused by
checking only internal ratio consistency, never an external anchor.**
`src/data/universe_panel.py` now defines its own
`SHROUT_UNITS_MULTIPLIER = 1000` (matching `market_cap.py`'s existing
constant of the same name/value) and applies it in `_mkt_cap_from_panel()`;
`src/data/gate_adapters.py::us_gate1_panel()` and
`src/gates/gate2_deciles.py`'s two market-cap functions were corrected the
same way. A uniform 1000x scalar error cancels out of any VW-weighted
RATIO (weight = mkt_cap_i / sum(mkt_cap_i)) — this is why Gate 1 US's
correlation test (0.9976 vs. `vwretd`) never caught it: correlation and
value-weighting are scale-invariant to a uniform multiplier on one side.
Re-verified after the fix: Gate 1 US's correlation is unchanged (still
0.9976, as expected for a ratio-based test), and Gate 2's breakpoint
diagnostic now matches Ken French's published dollar figures to within
normal universe-definition noise instead of being off by ~1000x. See
`docs/worklog.md`'s 2026-09-08 entry for the full before/after
verification.

## 16. Spec item #3 cross-check: `priusa` vs. `usincflg`-based CRSP membership (2026-09-03, Task 4)

Spec §3/§11 item #3 was previously diagnosed (§3's "DIAGNOSED" block) against
raw CRSP `shrcd IN (10,11)`, unfiltered — that comparison predates this
CRSP-primary full-history panel design and its `universe_at()` module. This
section re-runs the cross-check against what the pipeline actually produces
today: Compustat's `priusa`-based US universe cache
(`data/raw/us_universe_2015_06.parquet`, `fic='USA' AND tpci='0' AND
s.iid=c.priusa`, exchange-filtered to `exchg IN (11,12,14)` at pull time,
`src/data/pull_universe.py`) versus `src/data/universe_panel.py`'s
`universe_at(2015-06-30)` (CRSP `usincflg='Y'`-based, `securitytype='EQTY'`/
`securitysubtype='COM'`/`sharetype='NS'` base rule, major-exchange filter
`Q/N/A`, REIT exclusion by `issuertype`). Joined via `crsp.ccmxpf_lnkhist`,
clean-link filter (`linktype IN ('LC','LU','LS')`, `linkprim NOT IN
('J','N')`), date-bounded at 2015-06-30 — same rules as every other CCM join
in this project (§7, §13). Script: `scripts/item3_crosscheck.py`
(throwaway diagnostic, not imported by any pipeline code).

**Real counts, live WRDS, 2015-06-30:**

| | count |
|---|---:|
| Compustat `priusa` universe (CCM clean-linked) | 3,632 permnos |
| CRSP `usincflg='Y'`-based universe (`universe_at()`) | 3,772 permnos |
| Overlap | 3,225 |
| Compustat-only (`priusa` includes, CRSP excludes) | 407 |
| CRSP-only (CRSP includes, `priusa` excludes) | 547 |

Stated fully, not just from the most flattering angle: 407 of `priusa`'s own
3,632 names (11.2%) are Compustat-only, and 547 of `usincflg`'s own 3,772
names (14.5%) are CRSP-only — roughly one name in seven or eight falls in a
mismatch bucket depending on which universe you ask. Overlap as a share of
the *smaller* set (3,225/3,632 = 88.8%) is the most favorable of three
legitimate ratios; overlap as a share of the *larger* set is 3,225/3,772 =
85.5%; overlap as a share of the *union* (Jaccard index) is
3,225/(3,632+3,772−3,225) = 3,225/4,179 = 77.2%. None of these ratios alone
is "the" answer — they're the same 954 non-overlapping names viewed from
different denominators.

What actually makes this low-risk is not the raw overlap percentage under
any of those denominators — it's that **all 954 non-overlapping names (100%)
were traced to a specific, named, verified mechanism, with zero unexplained
residual**, checked in full (not sampled) against this design's own actual
filters and Compustat's own actual query, per CLAUDE.md ("a wrong number
that looks right is the worst possible outcome" — the discipline here is the
same one that caught the earlier bare-`PARTNERS` false positive, §3). A
fully-explained ~12–15%-of-either-side divergence with known mechanisms is a
materially different (and stronger) claim than an unexplained divergence of
the same magnitude would be. No join bug found in `universe_at()`,
`pull_universe.py`, or the cross-check script itself.

### Compustat-only (407): fully explained by this design's own exclusion rules

All 407 permnos were pulled from CRSP directly (`crsp.wrds_dsfv2_query`,
`issuertype`/`sharetype`/`securitytype`/`securitysubtype`/`usincflg`) and
checked against every filter `universe_at()` applies:

| reason | count |
|---|---:|
| `issuertype='REIT'` (REIT exclusion) | 218 |
| `sharetype` != `'NS'` (units/MLP-style, never entered the CRSP-primary panel — Task 1's pull-time filter requires `sharetype='NS'`) | 118 (`UG`) + 41 (`SB`) + 1 (`CE`) |
| `securitysubtype` != `'COM'` (closed-end funds, `securitytype='FUND'`/`securitysubtype='CEF'`) | 51 |
| `usincflg='N'` (foreign-incorporated, fails the base rule) | 16 |
| **subtotal, explained by base-rule/REIT filters** | **406** |
| `primaryexch='X'` (confirmed dead/inactive: `securityactiveflg='N'`, null price, e.g. permno 91522 — the documented `'X'` case in `config/universe.yaml`) | 1 |
| **total explained** | **407 / 407** |

(Some permnos hit more than one reason simultaneously, e.g. a REIT with
`sharetype='SB'`; the table's categories are the checked conditions, not a
strict partition — every one of the 407 is covered by at least one.)

No unexplained residual. This bucket is `universe_at()`'s intentional design
working as documented (REIT/MLP/CEF/foreign-incorporation exclusion),
confirmed against real per-permno CRSP attributes rather than assumed from
the earlier raw-`shrcd` diagnosis's REIT/MLP finding.

### CRSP-only (547): CCM-mechanics-clean; two real divergence mechanisms in the underlying rules, plus known link-history gaps

Every one of the 547 CRSP-only permnos has *some* CCM link history to a
gvkey (0 with zero link rows at all). Classifying by link coverage at
2015-06-30:

| link coverage at 2015-06-30 | count |
|---|---:|
| Clean type + clean linkprim, covers the date (mechanically "should join") | 499 |
| Clean type, `linkprim IN ('J','N')` covers the date, no clean-linkprim row covers it | 47 |
| No covering link row of any type (link exists but lapsed across this exact date) | 1 |

The 47 `linkprim='J'`/`'N'` cases and the 1 link-history gap are the same
class of CCM coverage gap already documented in §13 (there, `linktype='NR'`;
here, `linkprim` excluded or a lapsed window) — genuine CCM link-table facts,
not a bug in the cross-check's join logic.

**The 499 "should-join" gvkeys were checked directly against
`pull_universe.py`'s own query and are 0/499 present in
`us_universe_2015_06.parquet`** — confirming the gap originates in what
Compustat's `priusa` cache-building query actually returns, not in the CCM
join. Re-querying `comp.company`/`comp.security` for these 499 gvkeys'
`priusa`-pointed security row splits cleanly into two mechanisms:

1. **`fic != 'USA'` (26 of 499, ~5%) — foreign-incorporated, US-exchange-
   listed companies.** `pull_universe.py`'s `c.fic = 'USA'` filter correctly
   excludes these; CRSP's `usincflg='Y'`-based rule does not require US
   *incorporation* the same way — `usincflg` was verified elsewhere
   (`config/universe.yaml`) to reproduce CRSP `shrcd IN (10,11)` exactly,
   and `shrcd=11` does not encode incorporation domicile. Spot-checked
   `fic` values on these 26: `CYM` (Cayman Islands), `IRL` (Ireland),
   `GBR` (UK), `BMU` (Bermuda), `NLD` (Netherlands), `CAN`, `CHE`
   (Switzerland), `VGB` (British Virgin Islands) — the standard
   tax-inversion/redomiciled-shell jurisdictions for US-exchange-primary
   firms. **This is a genuine, previously-undocumented structural
   disagreement between the two universe definitions** (legal domicile vs.
   CRSP's US-common-stock classification), not a data error on either side.
2. **`exchg` field disagreement (429 of 499, ~86%, dominant) — same
   mechanism already flagged in spec §3** ("Do NOT filter on `exchg`... it
   varies by which security row is selected and is not a reliable
   membership test"). 426 of these 429 have `fic='USA'` and a real
   `comp.secd` trading row on 2015-06-30 (real `cshtrd`/`prccd`), but
   `comp.security.exchg` on the `priusa`-pointed security row is `19`
   (Other-OTC) — while CRSP's `primaryexch` for the same underlying company
   is a major exchange (`Q`/`N`/`A`). `pull_universe.py`'s
   `exchg IN (11,12,14)` filter (applied to the `priusa` security row) and
   this design's CRSP-side `us_major_exchanges_crsp: ["Q","N","A"]` filter
   (applied to CRSP's own primary-listing record) are reading two different
   systems' opinions of which venue is primary for the same company, and
   they disagree for these 429 names. Confirmed on a targeted sample of 22
   of these 429 that additionally had `exchg IN (11,12,14)` yet still didn't
   join: every one of the 22 turned out to have `fic != 'USA'` (mechanism 1
   above) — i.e. mechanisms 1 and 2 are not fully independent; `fic` is
   checked first in `pull_universe.py`'s query, and a foreign-incorporated
   name can carry any `exchg` value at all. The remaining ~44 rows (minor
   `exchg` codes 0/17/1/18/13/21, or no `comp.secd` row for the `priusa` iid
   on 2015-06-30 at all — e.g. Zillow, gvkey 187039, whose `priusa` pointer
   selects `iid='05'` with zero trading rows that day while its actively-
   traded Class A `iid='01'` traded normally on NASDAQ — are smaller
   variants of the same "which security row `priusa` selects" family,
   consistent with the multi-share-class primary-flag disagreement spec §3
   already documented for Berkshire/Alphabet/Liberty Media, now confirmed
   directly for a `priusa`-vs-`usincflg` gvkey/permno pair rather than
   inferred.

### Conclusion

`priusa` and `usincflg`-based membership diverge by roughly one name in
seven to eight on either side (Compustat-only 407/3,632 = 11.2% of
`priusa`'s own universe; CRSP-only 547/3,772 = 14.5% of `usincflg`'s own
universe; Jaccard/union overlap 3,225/4,179 = 77.2%; overlap-of-smaller-set
88.8% is the most favorable of these three ratios, not the representative
one). What makes this **low-risk rather than merely small** is **zero
unexplained residual in either divergence bucket** — every one of the 954
non-overlapping names traces to an identified, verified mechanism: this
design's own REIT/MLP/CEF/foreign-incorporation exclusion rules
(Compustat-only, 407/407), or a real `fic`-domicile or
`exchg`-descriptive-field disagreement between the two source systems plus
known CCM link-history gaps (CRSP-only, 547/547). No join bug found in
`universe_at()`, `pull_universe.py`, or `scripts/item3_crosscheck.py`.
**Both `priusa` and `usincflg` are usable universe-membership rules; this
design's choice of `usincflg` (spec §2, CRSP retained for the US leg) is
confirmed low-risk** — the divergence is structural and fully accounted for,
not a hidden defect, even though it is larger in absolute terms (~12–15% of
either side) than the headline 88.8% figure alone would suggest. The one new
finding this cross-check surfaces beyond the earlier §3 diagnosis is the
`fic` incorporation-domicile mechanism (26 names, foreign-incorporated/US-
listed) — worth keeping in mind if a future US-sample comparison to FP/AQR's
own published series shows small membership deltas at the margin.

---

## 17. CHASS (CFMRC) Canadian data source — identity, exclusion, liquidity,
## and point-in-time membership (2026-09-04/05)

CHASS (Canadian Financial Markets Research Center, TSX/CFMRC Database) was
brought in this session as the Canadian-leg data source, replacing the plan
to rely solely on Compustat's `prican`-based pull for Canada. Raw CSVs
(`data/raw/CHASS_Data/csvs/CHASS_{Daily,Monthly}.csv`) converted to
year-partitioned parquet by `src/data/convert_chass_csv.py`
(`data/raw/CHASS_Data/parquet/`). Loader, exclusion, liquidity-gate, and
point-in-time-membership logic all live in `src/data/chass_loader.py` and
`src/data/chass_universe.py` — this section is a pointer to the reasoning
and live-measured numbers behind that code, not a restatement of it; see
those two modules' docstrings for the full derivations.

### CUSIP is not a usable identity key in this dataset

Confirmed against the full 46-year daily file (16,715,944 rows): `(cusip,
date)` has 106,519 duplicate rows across 9,336 groups. Two CUSIPs are
outright placeholders (`000000000`, `305915xxx`) shared by unrelated
securities; the remainder are cases where two genuinely different
securities share one real CUSIP (e.g. ticker `TWE` usage 0 "TRANS-WESTERN
EXPLORATION INC." and usage 1 "TRANSWEST ENERGY INC." both trade live,
independently, under CUSIP `893921106` for weeks in 1983). CHASS's own
"Usage Number" field exists specifically to disambiguate ticker reuse —
`(symbol-Ticker, usage-Usage Number)` is the correct identity key, verified
to map to exactly one CUSIP in 100% of cases across both daily and monthly
files. CUSIP is kept as a plain reference column but never used as a key.
Full derivation: `src/data/chass_loader.py` module docstring.

### Dividend-event fan-out on the daily file

A single (ticker, usage, date) can carry two dividend-event rows (e.g.
"Cash dividend - Regular" + "Cash dividend - Extra" on the same ex-date),
identical on every trading field, differing only in dividend detail.
Resolved by dropping the dividend-specific columns and `drop_duplicates()`
before use — collapses cleanly with zero information loss for price/
return/volume purposes (16,715,944 → 16,713,209 rows). See
`src/data/chass_loader.py::load_daily`.

### ETF/mutual-fund/REIT/MLP exclusion required hand classification

CHASS's `business-Business` field cleanly identifies three exclude-outright
categories (`INVESTMENT FUND`, `INVESTMENT TRUST`, `LIMITED PARTNERSHIP`)
but three others (`MUTUAL FUNDS`, `INVESTMENT COMPANY`, `TRUST FUND`)
genuinely mix real operating companies with fund products under the same
TSX sector label — no field or name regex reliably separates them (tested
and rejected: "SPLIT" in name, CAPITAL SHARES/UNITS suffix heuristics,
`foreign_flag`). Real, large, currently-listed companies were found
misfiled in these categories: Pembina Pipeline, Algonquin Power &
Utilities, Franco-Nevada, Keyera, CI Financial Corp., AGF Management,
Guardian Capital Group. All 239 distinct names across the three
contaminated categories were hand-classified this session — see
`config/chass_fund_classification.csv` and `config/universe.yaml`'s
`chass_fund_mlp_reit_exclusion` block for the full table and reasoning.
`REAL ESTATE` was tested and rejected as a fourth clean category: it is
operating real-estate companies (Brookfield Office Properties, Trizec),
not REITs — the same "sector label catches non-REIT operating companies"
trap already documented for CRSP's `icbindustry` (§14). 37 names remain
`UNKNOWN` (confidence too low to classify), included in the universe as an
explicit open item, not a resolved decision — deferred to a future
market-cap filter per user direction.

### Liquidity gate: `volume-Daily Volume > 0`

Direct analog of this project's `cshtrd > 0` rule. `volume-Daily Volume`
partitions cleanly into three non-overlapping states, verified against the
full year=1980 partition (140,563 / 68,984 / 7,341 of 216,888 total rows,
summing exactly): `> 0` (real trade), `== 0` (genuinely listed, no trade —
closeprice always populated, return always NaN), and `NaN` (true missing —
confirmed this arrives as a **literal blank CSV field**, not the digit
string `-9`, e.g. a "Recapitalization" event for ticker BHG.B on
1980-12-01 has every trading field blank in the raw CSV, matching CHASS's
documented rule that a reclassification zeroes out the return at both `t`
and `t-1`). See `src/data/chass_loader.py::chass_daily_trade_status`.

### Point-in-time membership without the Ticker History table

CHASS's authoritative delisting-date source (`DICTION.DAT`, the Ticker
History/Dictionary table, documented in `CHASS_Documentation.pdf` §3.7) is
not accessible via this project's CHASS browser install — the GUI does not
expose a Ticker History option. Rather than block on resolving that access
problem, an interim proxy was designed and empirically validated directly
against real data:

1. **Row presence at monthly grain** is the primary membership signal — a
   security is a member in month `t` iff it has a monthly row at `t`.
   Verified: Pembina Pipeline (confirmed real, currently listed) has zero
   gaps across 339 monthly rows, 1997–2025; Morgan Hydrocarbons (acquired
   by Talisman Energy, Oct 1996) has monthly rows through 1996-10 and none
   after, in any later year, with its CUSIP never resurfacing under any
   other ticker.
   **Rejected alternative, tested directly:** "price/return hits zero" as
   a delisting proxy. False-positives on foreign cross-listed names with
   light TSX volume (American Express: 15 consecutive zero-price days on
   TSX with zero relation to any real corporate delisting); false-negatives
   from ordinary illiquidity that resolves with real trading resuming
   later (many domestic microcaps show the identical pattern with no
   delisting at all).
2. **Terminal-row NaN corroboration**: `shares_out`/`return` both NaN on a
   name's last-ever monthly row is true for 71.0%/93.4% of plausibly-
   delisted names (last row before 2015) versus 0/30 false positives on a
   sample of still-listed names. A confidence signal, not the primary
   trigger.
3. **CUSIP-continuity rename check**: before treating a stopped
   `(ticker, usage)` as delisted, check whether its CUSIP reappears under
   a different `(ticker, usage)` within ~60 days — guards against a ticker
   rename (not a real exit) being misread as a delisting. Tested against
   the 179 names most likely to hide a rename (those lacking the NaN
   corroboration) — zero matches; that group turned out to be genuine M&A
   delistings with clean trading to the end (Bell Aliant→BCE 2014, Bema
   Gold→Goldcorp 2007, Burlington Resources→ConocoPhillips 2005), not
   renames.

This is an inferred proxy, not an authoritative field, and cannot perfectly
distinguish "genuinely delisted" from "parquet coverage just ends" at the
dataset's 2025 edge. Full derivation and all three mechanisms:
`src/data/chass_universe.py` module docstring.

**Real implementation bug caught by testing, not assumed away:**
`universe_at()` originally applied domestic-only filtering before fund/
REIT/MLP exclusion. This broke `apply_chass_fund_mlp_reit_exclusion`'s
stale-CSV safety check, which compares the full classification CSV's keys
against the input frame — two CSV-listed names (`CDI.A`/`CDI.B`) are
Foreign Firm rows, so filtering domestic-only first silently dropped them
before the exclusion step ever saw them, tripping a false "stale entry"
alarm on every call even though nothing was actually stale. Fixed by
reordering: fund exclusion runs first on the full monthly frame, domestic-
only filtering after. Order doesn't change the final membership result
(both are independent boolean filters) — only this order satisfies the
downstream function's precondition of seeing close to the full universe.

### CHASS has no delisting-return field at all — see §6

The CHASS-specific addition to the delisting/survivorship discussion lives
in §6 above ("CHASS — no delisting return at all, confirmed structural
gap"), not repeated here. Summary: unlike CRSP's `dlret`, CHASS's return
series simply stops at a security's last real trade, with no reconciling
value for the gap to its true economic outcome (merger consideration or
near-total loss). Confirmed against Bema Gold (all-stock merger, happens
to be fine — price converged through ordinary trading) and Progressive
Waste Solutions (acquired 2016 — series just stops at $41.40, no way to
know if that matched the deal price). No fix available from CHASS Summary
Information; accepted as a documented limitation creating a real US-vs-
Canada asymmetry.

---

## 18. Pandas-to-polars port of the CHASS loader/universe modules
(2026-09-05) — real bug found via parity testing, not a porting artifact

`src/data/chass_loader.py` and `src/data/chass_universe.py` were ported
from pandas to polars (Gate 1 prerequisite — see the Gate 1 implementation
plan) so both country legs share one dataframe library before Gate 1 code
is written on top of them. The original pandas versions are preserved,
unmodified, at `src/data/chass_loader_pandas_reference.py` and
`src/data/chass_universe_pandas_reference.py` for parity testing
(`tests/unit/test_chass_polars_parity.py`) — every ported function's
output is compared row-for-row against the pandas original on the same
real input, not just checked against the pre-existing pinned-count test
suite (which encodes expected values, not "does polars agree with pandas
on every row" — a subtly wrong port can still hit a pinned count by
coincidence while diverging elsewhere).

**Found via this process: `_terminal_row_flags()` silently dropped
National Bank of Canada under the pandas original.** `symbol-Ticker` is
null for this name in the source parquet (cusip `633067103`, 552 monthly
rows spanning 1980-01-31 to 2025-12-31, `business-Business='BANKING'`,
`foreign_flag-Foreign Flag='Domestic Firm'` — a genuine, currently-listed
major bank, not a data artifact). pandas's `groupby()` defaults to
`dropna=True`, which drops the entire null-key group rather than treating
it as its own group — so `_terminal_row_flags()` never emitted a row for
this name at all, silently, under the pre-port code that had been merged
and tested since 2026-09-05 (before this port). polars's `group_by()` has
no such default; it keeps the null-key group, which is why the initial
parity test run (13 tests, real full-history data) surfaced a 7,536 vs.
7,537 group-count mismatch on this one function while all 12 other ported
functions matched exactly.

**Scope confirmed narrow, not a wider correctness gap:** `universe_at()`
and `market_cap_at()` both use boolean `.filter()`/join logic, not
`groupby`/`group_by`, so neither has this hazard — confirmed directly
that National Bank of Canada correctly appears in `universe_at(monthly,
2015-06-30)` and gets a correct market cap from `market_cap_at()` under
both the pandas original and the polars port. The bug was confined to the
`_terminal_row_flags()` delisting-corroboration diagnostic (a confidence
signal, not a membership or market-cap input) — it never affected which
names were actually in the tradable universe or their weights.

**Decision (confirmed with the user 2026-09-05): fix in the port, do not
preserve the gap.** The polars port keeps the null-symbol group rather
than matching pandas's behavior — polars's default here is the more
correct one. `test_terminal_row_flags_parity_excluding_known_null_symbol_
divergence()` documents this explicitly (pinned counts: pandas 7,536,
polars 7,537) rather than silently passing or silently diverging — a
future re-run of this test with different counts means either the source
data changed or one of the two libraries' null-handling defaults changed,
and should be investigated, not just re-pinned.

This is the first real bug this project's own leakage/correctness
discipline caught via a deliberate parity-testing step (not a lookahead
bug — a silent-drop bug of exactly the kind CLAUDE.md's "a wrong number
that looks right is the worst possible outcome" warns about) — logged
here as evidence the parity-testing requirement earns its cost.

---

## 19. CRSP v2 hides the delisting return inside `dlyret` — and blanks the
## security descriptors on that row (2026-09-08)

**The trap.** `crsp.wrds_dsfv2_query` has **no `dlret` column**. Grepping the
schema for `dlret`/`delist` returns nothing return-shaped, which reads as
"this view has no delisting returns." That conclusion is wrong. CRSP v2
*folds the delisting return into `dlyret`* on a dedicated row, and blanks
that row's security descriptors:

```
permno  dlycaldt    dlyret     securitytype  subtype  sharetype  usincflg  primaryexch  dlydelflg
11999   2015-06-25  -0.665459  N/A           UNK      N/A        N         X            Y
89888   2015-03-19  -0.958333  N/A           UNK      N/A        N         X            Y
```

Verified against `crsp.stkdelists`: on these rows `dlyret == delret`
**exactly**, every case sampled.

**Two dates, not one.** `stkdelists.delistingdt` is the last *trading* day;
`stkdelists.deldlydt` is the day the delisting return is *posted* — one
trading day later. A join on `delistingdt = dlycaldt` returns **empty**.
Join on `deldlydt`.

**Why this silently destroys a universe pull.** Any WHERE clause requiring
the normal descriptors (`securitytype='EQTY' AND securitysubtype='COM' AND
sharetype='NS' AND usincflg='Y'`) drops **every delisting row in the
database**. Measured live: 31,114 delisting rows 1965-present, **0 pass**,
in every decade. The failure is invisible — each name's history just stops
on its last trading day, and `dlyret` looks fully populated.

**Do not key the exception on `primaryexch='X'`.** `dlydelflg='Y'` ⟺
`primaryexch='X'` holds exactly (31,114 both directions), but `'X'` also
covers **1,370,373 non-delisting rows**. Gate on `dlydelflg='Y'`, and
restrict to your own universe with an EXISTS subquery on `permno` — the
descriptors are unavailable on the row itself (31,114 delisting rows total
→ 24,480 belong to a base-filtered US common-equity universe, 6,634
correctly excluded as ETF/ADR/foreign).

See `src/data/pull_universe_us_crsp.py::YEAR_QUERY` for the corrected
two-branch clause.

**`delreasontype`/`delactiontype`/`delstatustype` are blanked ('N/A') on the
daily row.** They are populated in `crsp.stkdelists`. Reason-level splits
(bankruptcy vs merger) must come from that table, not the daily view.

**Magnitude, full window (delisting rows with a return, n=27,945):** mean
−3.22%, median +0.06%, 15,083 positive (merger premia partly offset
failures) — but 2,164 rows ≤ −30% and 399 total losses (−100%). The tail
concentrates in small, distressed, high-beta names, so omitting these
biases a BAB short leg's performance **upward**. Worst decade is the 2020s
(mean −5.55%).

## 20. `permco` vs `permno` — CRSP's company vs security key (2026-09-08)

`permno` identifies a **security** (one share class); `permco` identifies
the **company**. Berkshire Hathaway is one `permco` (`84670` region) and two
permnos — 17778 (BRK.A) and 83443 (BRK.B). Both are in
`crsp.wrds_dsfv2_query`; this project's original pull took `permno` only.

**Where it matters.** Ken French (and any FF-style size breakpoint) sums
share classes to the company before taking percentiles. Quantiling
per-`permno` splits one large company into two smaller ones and pushes every
percentile down — measured against Ken French's published 2015-06-30 NYSE
breakpoints, gaps of −1.6% to −6.7%, humped at p30 (a composition
signature, not a scale factor). Aggregating to company level collapses them
to ≤1.1% above p10 and moves the NYSE firm count 1358 → 1337 against Ken
French's published 1319.

**`dlycap` is not a shortcut** — it is also per-security (Berkshire A and B
carry separate `dlycap` values), and it is in **$thousands**, consistent with
`abs(dlyprc) * shrout * 1000` (ratio 1000.0 ± 2e-4 across all NYSE names on
2015-06-30).

**cusip6 is a usable proxy when `permco` is unavailable** (issuer prefix;
zero null CUSIPs in every year sampled 1975-2024, and it caught all 22
multi-class NYSE issuers in 2015) — but it is not guaranteed to equal CRSP's
own permco mapping for issuers that changed CUSIP. Prefer `permco`.

Scale of the effect: ~1.1-1.3% of names are multi-class in any recent year
(0.33% in 1975), but their share of total market cap has grown steadily —
0.3% (1975) → 5.1% (2015) → 6.2% (2024). Under **rank weighting** (spec §8
grid) a two-class company also receives roughly double its intended weight,
and its two near-identical return series distort cross-sectional beta
estimation.
