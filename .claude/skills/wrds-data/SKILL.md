---
name: wrds-data
description: WRDS schema locations, Compustat/CRSP field semantics, and verified data traps for this project. Use when writing any SQL against WRDS, building the data layer, constructing the universe, or debugging a query that returns unexpected counts.
---

# WRDS Data Layer

`docs/01_data_notes.md` is authoritative. Everything below was verified
empirically against the live database, not read from documentation.

## Schema locations

| What | Where |
|---|---|
| CCM link history | `crsp.ccmxpf_lnkhist` — **in the `crsp` schema**, not `ccm` |
| Security identifiers | `comp.security` — `gvkey`, `iid`, `tic`, `cusip`, `exchg`, `tpci` |
| Company descriptors | `comp.company` — `conm`, `gsector`, `gind`, `sic`, `fic`, `priusa`, `prican`, `prirow`, `dldte`, `dlrsn`, `costat` |
| Daily prices | `comp.secd` — `prccd`, `ajexdi`, `trfd`, `cshtrd`, `cshoc`, `curcdd` |
| CRSP daily | `crsp.dsf`, `crsp.dsenames`, `crsp.dsedelist`, `crsp.dsi` |

Rule of thumb: describes the *business* → `company`. Describes the *security* →
`security`. Getting it wrong yields `UndefinedColumn` with a `HINT` naming the
right alias.

## Universe rule — exact equality, never prefix matching

```sql
-- Canada
fic = 'CAN' AND tpci = '0' AND s.iid = c.prican
-- US
fic = 'USA' AND tpci = '0' AND s.iid = c.priusa
```

A `gvkey` can have up to three simultaneously-primary securities (US, Canada,
RoW). For interlisted names these are **genuinely separate rows** with separate
pricing. RBC (`gvkey 015633`): `iid='01'` is NYSE/USD, `iid='01C'` is TSX/CAD
with 10,686 rows back to 1983. The `C` is part of a real `iid`, not a flag
suffix.

**Never use `LIKE`, `strpos`, or `LEFT(...)` here.** `'01'` is a literal prefix
of `'01C'`, so prefix matching lets the US row pass a Canadian filter and
restores the duplication the filter exists to remove. Verified counts, single
test month: exact equality → 2,614 names (correct); `strpos` → 2,753 (over-
matches); `LEFT(...,2)` → 1,770 (drops TSX/TSXV entirely).

## Exchange codes (verified two independent ways)

`7` = TSX (median mkt cap ~$192M) · `9` = TSXV (~$3M) · `11/12/14` = NYSE/AMEX/
NASDAQ · `19` = Other-OTC · `1`, `13` = minor/legacy.

**`exchg` is descriptive, not an inclusion filter.** Use the `prican`/`priusa`
rule for universe membership.

## Traps

- **Never join on `tic`.** Compustat appends a trailing `.` to the Canadian-side
  ticker of interlisted names (`RY.` vs `RY`). A lookup for `'RY'` against the
  Canadian universe returns empty, which looks identical to the company being
  absent. Join on `gvkey`.
- **A row in `comp.secd` does not mean the security traded.** Gate on
  `cshtrd > 0`. Lehman printed a frozen $0.55 with zero volume for 15 straight
  days in 2014.
- **`ajexdi` can be near-zero but not zero** (values like 3e-8). Guard on
  magnitude, not `== 0`.
- **Blank `trfd` ≠ `trfd = 1`.** Treat as a missing return.
- **Literal `%` in raw SQL breaks psycopg2** (`TypeError: immutabledict is not a
  sequence`). Use bound params or `strpos()`/`left()`.
- **A failed query poisons the transaction.** Every subsequent query raises
  `PendingRollbackError` until `db.connection.rollback()`.

## Return construction (Compustat has no return field)

```
p_adj = (prccd / ajexdi) * trfd
ret   = p_adj / p_adj.shift(1) - 1
```

`trfd` already reinvests distributions — it is the analog of CRSP `ret`, not
`retx`. Do not add dividends on top.

## Delisting

CRSP `dlret` verified sound: distress (500-range codes) → large negative
(Lehman −0.60, Circuit City −0.48); merger (200-range) → small positive
(Compaq +0.048, Countrywide +0.021).

Compustat `dlrsn`: `01` merger, `02` bankruptcy, `03` liquidation, `04` reverse
acq, `05` no longer fits format, `06` LBO, `09` now private, `10` other/no
longer files. **Neither Lehman nor Circuit City is coded `02`** — both are `10`.
Never filter distress on `dlrsn` alone; cross-check CRSP `dlstcd` 500-range.

Company-record closure lags real delisting by **years** for bankruptcies
(Lehman traded through Sep 2008, `dldte` 2012-03-07). `costat='I'` tells you the
record closed, not when the security stopped being investable.

Entity continuity differs between databases: CRSP kept WaMu's PERMNO 81593
continuously through Chapter 11 → WMI Holdings → Mr. Cooper Group; Compustat
closed the gvkey and opened a successor. Do not QA delisting coverage by
spot-checking famous bankruptcies.

## CCM links

Clean filter: `linktype IN ('LC','LU','LS')`, `linkprim NOT IN ('J','N')`.
`linkprim='N'` + `linktype='LX'` marks foreign-exchange securities linked to a
US-covered security — a cross-listing indicator.

Duplicate rows are normal (CCM splits one link into consecutive date ranges).
**Do not `drop_duplicates()`** — a security can genuinely be reassigned to a
different PERMNO. Join with `linkdt <= date <= COALESCE(linkenddt, CURRENT_DATE)`.