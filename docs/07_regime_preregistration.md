# §6.4 regime definitions — PRE-REGISTERED

**Written 2026-09-15, BEFORE any regime statistic was computed.**

CLAUDE.md and `docs/06_handoff_section6.md` §C both require regime
definitions be fixed before looking at results. This project has retracted
two gates whose statistics were chosen after seeing the data. This file is
the commitment record: it is written and committed *first*, and the
computation reads its definitions rather than restating them.

Nothing in this file may be revised after the first regime number is
produced. If a definition turns out to be unworkable (e.g. zero quarters
qualify), the finding is recorded as an amendment with its own date and
reason — the original definition stays in place, struck through, not
deleted.

---

## Definition 1 — Named historical crises

Fixed calendar windows, taken verbatim from `docs/05_report_spec.md` §6.4
and `docs/06_handoff_section6.md` §C. Chosen because they are externally
recognisable and were named before this analysis existed, not selected
from our own return series.

| Label | Window (inclusive) |
|---|---|
| `oil_shock_1973_74` | 1973-01-01 .. 1974-12-31 |
| `black_monday_1987` | 1987-10-01 .. 1987-10-31 |
| `dotcom_2000_02` | 2000-03-01 .. 2002-10-31 |
| `gfc_2008_09` | 2008-09-01 .. 2009-03-31 |
| `covid_2020` | 2020-03-01 .. 2020-03-31 |

**Quarter assignment rule, stated explicitly because it is a real choice:**
a held quarter belongs to a crisis regime if the quarter's own END DATE
falls inside the window, OR the window is fully contained within the
quarter. The second clause exists because three of the five windows
(1987-10, 2020-03, and the 2008-09..2009-03 boundary months) are shorter
than one quarter and would otherwise match no quarter at all under an
end-date-only rule.

This is a *labelling* rule applied to already-computed quarterly returns.
It involves no look-ahead: the crisis dates are historical fact known to
the reader, and the returns are not re-estimated.

**Canada caveat, stated in advance:** the Canadian leg starts 1989-03-31,
so `oil_shock_1973_74` and `black_monday_1987` will have ZERO Canadian
quarters. That is an expected structural absence, not a finding, and the
exhibit must show it as "n/a — outside sample", never as a blank or a zero.

## Definition 2 — Objective threshold

A held quarter is a **downturn** if the MARKET's quarterly EXCESS return
for that quarter is at or below the 10th percentile of the market's own
quarterly excess returns over the full sample of held quarters for that
leg. Everything else is an **expansion**.

Committed specifics, so no degrees of freedom remain:

- **Series:** the market index used to estimate that leg's betas
  (US `vw_uncapped`, Canada `vw_capped_10pct`) — spec §7's rule that
  estimation and evaluation benchmark are the same object.
- **Excess, not raw**, quarterly, risk-free compounded to quarterly on
  the same log-space path as the returns themselves.
- **Percentile:** 10th, `interpolation="linear"` (polars default).
- **Computed over:** the full sample of held quarters for that leg, each
  leg separately. This is a full-sample statistic and therefore NOT
  tradeable — it is a descriptive regime split, not a signal. The report
  must say so. It does not contaminate the backtest, because no position
  is formed from it; it only labels quarters after the fact.
- **Ties:** at-or-below (`<=`), so a quarter exactly on the breakpoint is
  a downturn.

## Statistics reported per regime, per leg

Fixed in advance, so the choice is not made after seeing which flatters
the strategy:

1. `n_quarters` in the regime
2. portfolio mean quarterly EXCESS return
3. market mean quarterly EXCESS return
4. difference (portfolio − market), and its t-statistic
   `mean(diff) / (std(diff, ddof=1) / sqrt(n))`
5. hit rate: fraction of regime quarters where portfolio > market

## Pre-registered expectation

Recorded so the result cannot be re-narrated afterwards.
`docs/06_handoff_section6.md` §C notes the US low-beta leg has a WORSE max
drawdown than the market (−54.5% vs −51.5%) despite β≈0.74. If the low-beta
book also underperforms the market in downturns, that CONTRADICTS the
"low beta is defensive" pitch — which `docs/05_report_spec.md` already
forbids the report from making. **That outcome is to be reported plainly,
not softened, not buried, and not reframed as a risk-adjusted win.**

The opposite outcome (outperformance in downturns) is equally reportable,
but note in advance: with ~5 crisis episodes, per-regime t-statistics will
be low-powered. A statistically insignificant difference must be reported
as insignificant, in both directions.
