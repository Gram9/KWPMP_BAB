---
name: bab-spec
description: Settled design decisions and unresolved open items for the BAB backtest. Use when making any choice about universe, currency, sample period, market index, filters, or anything that would change what the backtest measures.
---

# Project Spec — Decisions & Open Items

`docs/00_spec.md` is authoritative. This is the summary plus the guardrail.

## The guardrail

**Six decisions are OPEN. Do not resolve any of them silently in code.**

If a task requires one of these to be settled, say so and ask. Picking a
default and proceeding produces a backtest whose results depend on an
unexamined choice — which is exactly the failure this project is structured to
avoid.

| # | Open item | Blocks |
|---|---|---|
| 1 | **TSXV inclusion.** ~1,689 of 2,614 Canadian names are TSXV (median cap ~$3M vs. ~$192M on TSX). Under rank weighting these dominate. Preferred resolution: run both as grid cells. | Phase 4 |
| 2 | **Income trusts, 2001–2011.** High-payout, low-vol, trust-structured; sit squarely in the low-beta leg. The 2006 tax announcement and 2011 conversion deadline restructured the cohort. | Phase 6 |
| 3 | **Does the `priusa` rule reproduce CRSP `shrcd IN (10,11)`?** Determines whether US results are comparable to FP/AQR. | Phase 2 |
| 4 | **Sector-neutral variant.** Canadian low-beta = financials/utilities/telecom/pipelines; high-beta = junior energy/mining. Without neutralization this is a defensives-vs-resources trade, not BAB. | Phase 6 |
| 5 | **Canadian short-leg borrow feasibility.** Small TSX/TSXV borrow is thin or nonexistent. A long-only variant may be the more decision-relevant test. | Phase 7 |
| 6 | **Canadian funding-liquidity proxy** for Props 3/4. No direct TED analog; candidates are CORRA-OIS or BA spreads. | Phase 7 |

## Settled decisions

**Data source.** Compustat North America for both countries. CRSP retained for
the US leg's delisting treatment and the return-formula cross-check. CHASS/CFMRC
dropped.

**Universe.** `fic` + `tpci='0'` + `iid = prican`/`priusa`. **No size or price
screen in the base universe** — size restrictions are grid cells, not screens.
Pre-filtering would convert the central Novy-Marx/Velikov question into an
assumption and break comparability with AQR's published series.

**Currency.** USD for US-listed, CAD for Canadian-listed. Risk-free must match
currency (Ken French 1-mo T-bill; BoC 3-mo T-bill via Valet).

**Cross-listing.** Country assigned at issuer level, **fixed for the security's
life**, never by exchange at time t.

**Minimum history.** No separate filter — the estimator's 750-day correlation
minimum binds. But log excluded-name counts monthly: recent IPOs are
systematically high-beta, so this mechanically thins the short leg.

**Market index.** Build it; do not use SPX/TSX Composite for estimation. Beta is
only meaningful against a proxy spanning the sorted universe, and BAB is
market-neutral *relative to the index used to estimate betas* — so estimation
and evaluation benchmarks must be the same object. Value-weighted, total return,
**lagged weights**, delisting returns included, total shares not free float.

Canadian variants required: uncapped VW, **capped VW (10% single-name limit —
Nortel was ~⅓ of the TSE 300 at its 2000 peak, contaminating every Canadian beta
estimated on a 5-year window from ~1997–2006)**, and an MSCI-like large-cap
proxy to reproduce what FP actually did internationally.

**Sample periods.** Three runs: common window 1977–present (headline US-vs-CA
comparison); US max ~1965–present (subperiod stability, AQR overlap);
2013–present (the out-of-sample test — FP's Canada sample ends Mar 2012).

Daily data throughout. **Do not splice monthly-estimated betas for the pre-1962
era** — different signal, creates a structural break mid-sample.

## Validation gates

Gates 0/0b/0c pass. Gates 1–4 not started. **Gate 4 — correlating the US BAB
series against AQR's published factor — is the credibility gate. Do not run
Canadian results until it passes**, because if the Canadian numbers are wrong
there is no external benchmark to catch it. See `docs/02_validation_gates.md`.