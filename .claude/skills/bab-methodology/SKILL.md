---
name: bab-methodology
description: Frazzini-Pedersen (2014) beta estimator and BAB portfolio construction — exact formulas, windows, minimums, and the reasoning behind each choice. Use when writing or modifying anything in src/estimation/ or src/portfolio/.
---

# BAB Methodology

Frazzini & Pedersen (2014), *Betting Against Beta*, JFE 111(1), 1–25.

## Beta estimator

FP's beta is **not an OLS regression coefficient.** It is assembled from
components estimated on different frequencies and windows:

$$\hat\beta_i = \hat\rho_{i,m} \cdot \frac{\hat\sigma_i}{\hat\sigma_m}$$

| Component | Frequency | Window | Minimum obs |
|---|---|---|---|
| σ̂ᵢ, σ̂ₘ | 1-day log returns | 1 year (252d) | 6 months / 120 days |
| ρ̂ᵢₘ | overlapping 3-day log returns | 5 years (1260d) | 3 years / 750 days |

3-day log return: `r³ᵈ = Σ²ₖ₌₀ ln(1+rₜ₊ₖ)`. Log returns are additive, so this is
a rolling sum. Overlapping windows are intentional; FP do not correct the
resulting serial correlation.

Then shrink toward the cross-sectional mean (Vasicek 1973, weight fixed rather
than estimated):

$$\beta_i = 0.6\,\hat\beta_i^{TS} + 0.4 \cdot 1$$

### Why each piece

- **Split windows:** correlations move more slowly than volatilities. A single
  OLS regression can't give a responsive vol and a stable correlation.
- **3-day overlap:** non-synchronous trading correction (cruder cousin of
  Dimson 1979 / Scholes-Williams 1977). A microcap last traded at 2pm doesn't
  reflect the 3:30pm market move, biasing its daily correlation down.
- **Shrinkage:** outlier control. It is **rank-preserving**, so it does not
  change which stocks land in which leg. Its only function is setting the leg
  scaling (1/β_L, 1/β_H) and thus the implied leverage. Keep it in any variant.

### Required diagnostic

Because ρ is estimated over 5y and only σ over 1y, cross-sectional beta
dispersion is expected to be **dominated by recent volatility**. Regress β̂ on σᵢ
and on ρ separately each month and record the R². Quantify this before
interpreting any result — it is central to the Novy-Marx & Velikov critique and
the bridge to the betting-against-correlation decomposition.

## Portfolio construction

Rank-weight within each leg:

```
z    = cross-sectional rank of beta
zbar = mean rank
k    = 2 / sum(|z - zbar|)
w_H  = k * (z - zbar)⁺        w_L = k * (zbar - z)⁺
```

Each leg's weights sum to 1 by construction. The median split is **automatic**:
`zbar = (n+1)/2` is the median rank, so positive-part clipping puts every
below-median-beta name in the long leg.

Payoff:

$$r^{BAB} = \frac{1}{\beta_L}(r_L - r_f) - \frac{1}{\beta_H}(r_H - r_f)$$

FP's US sample averages **$1.40 long / $0.70 short**. The legs are scaled by
*different* amounts — this asymmetry is the source of the Novy-Marx & Velikov
objection, because once realized betas drift from ex-ante betas the position is
not market-neutral.

## Invariants to assert in code

- Each leg's weights sum to 1
- Shrinkage preserves rank ordering
- Ex-ante portfolio beta is exactly 0 by construction
- Realized market loading should be near 0 (FP report −0.06 for US)

## Log every month

Realized market loading · dollars long · dollars short · ex-ante beta spread
`(β_H − β_L)/(β_H β_L)` · name count · weight fraction in smallest size decile.

That last one is the microcap early-warning: N-M&V report FP's BAB commits
~$1.05 per $1 invested to stocks in the bottom 1% of market cap.

## Formation timing

Betas from data through the last trading day of month t−1; positions held
through month t. **Never same-month.** A skip-one-day variant is a planned
robustness check, since otherwise the same closing prices feed both the beta
estimate and the first day's return.

## Reference numbers (FP Table III, US 1926–Mar 2012)

BAB excess return 0.70%/mo (t=7.12); CAPM alpha 0.73% (t=7.44); 3-factor 0.73%
(t=7.39); 4-factor 0.55% (t=5.59); Sharpe 0.78. Decile betas: ex-ante 0.64→1.70,
realized 0.67→1.85 (realized exceeds ex-ante, gap widens with beta).

Canada (FP Table V, 1984–2012): 1.23%/mo, 4-factor alpha 0.67% (t=2.71),
Sharpe 1.05.

## Variant grid

Never report a single cell as "the" result:

| Axis | Cells |
|---|---|
| Estimator | FP spec / 12m daily OLS (both shrunk 0.6) |
| Weighting | rank-weight / value-weight |
| Leg scaling | asymmetric (FP) / dollar-neutral + explicit market hedge |
| Universe | full / ex-microcap / large-only |
| Canada | TSXV in / out; sector-neutral / not |