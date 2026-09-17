# Betting Against Beta — US & Canada Replication

Code accompanying the KWPMP strategy report *"Does BAB Actually Work?"*.

Replication and out-of-sample extension of Frazzini & Pedersen (2014),
*Betting Against Beta*, JFE 111(1), 1–25. The US BAB replication is the
validation harness; the long-only US and Canadian books are the
contribution the report acts on.

## Scope of this repository

This is a **submission subset** of a larger research repository. It contains
the code, configuration and validation evidence on the path of the numbers
reported in the deck, and nothing else. Exploratory notebooks, superseded
prototypes, one-shot diagnostics and internal sequencing docs are omitted.

**No data is included.** Inputs are licensed WRDS/CRSP (US) and CHASS/CFMRC
(Canada) data, plus published factor files from AQR and Ken French. The code
reads them from `data/raw/`, which is intentionally absent. See
[Reproducing the numbers](#reproducing-the-numbers).

## Two legs, deliberately not unified

The US and Canadian legs use different sources, different identity keys
(`permno` vs CHASS symbol) and different market indices. They are wired as
parallel siblings behind thin adapters (`data/gate_adapters.py`,
`portfolio/longonly_data.py`) rather than forced through one shared schema.
Where a module has a `_pandas_reference` twin, that twin is a preserved
pre-port implementation kept so `tests/unit/test_chass_polars_parity.py` can
compare old against new on identical input — it is test evidence, not dead
code.

## Two different beta estimators

Both are used, for different report sections, and they are not variants of
each other:

- **`estimation/beta_fp.py`** — the Frazzini–Pedersen estimator. Correlation
  from overlapping 3-day log returns over 1,260 trading days, volatility from
  1-day log returns over one year, then shrunk toward one
  (β̂ = 0.6·β + 0.4). Used for the BAB factor.
- **`estimation/beta_monthly.py`** — plain single-window rolling OLS on
  trailing *monthly* returns, **unshrunk**. Used for the long-only books,
  where selection is a top-N rank and shrinkage is rank-preserving and so
  immaterial.

## Repo layout

```
src/
  data/             CRSP/CHASS loaders, universe construction, returns, rf
  estimation/       beta_fp (FP daily, shrunk); beta_monthly (OLS, unshrunk)
  market_index/     project's own value-weighted US/Canada indices
  portfolio/        BAB legs, weighting, rebalance, long-only backtester
  analysis/         buckets, target-beta, drawdown, regimes, excess returns
  gates/            validation-gate implementations (Gates 1, 2, 4)
tests/              unit tests, validation gates, leakage tests
config/             YAML — every parameter here, no magic numbers
docs/               spec, data notes, validation gates, report spec
```

## Which code produces which part of the report

| Report section | Primary modules |
|---|---|
| US BAB factor (p6–7) | `estimation/beta_fp.py`, `portfolio/{legs,weights,rebalance}.py`, `gates/gate4_bab.py` |
| Market index, and its 0.993 correlation vs CRSP VW (p6) | `market_index/build.py`, `market_index/monthly_panel.py`, `gates/gate1_index.py` |
| Beta cross-section buckets (p8) | `analysis/beta_buckets.py`, `analysis/_grouped_quarterly_analysis.py` |
| Bucket stability over sub-periods (p7 alpha decay) | `analysis/beta_bucket_timeseries.py`, `analysis/regimes.py` |
| Long-only 20+20 ranked book (p9) | `portfolio/longonly.py`, `portfolio/longonly_data.py`, `estimation/beta_monthly.py` |
| Target-beta grid (p10) | `analysis/target_beta.py` |
| Drawdown, excess returns, vs-market comparison | `analysis/{drawdown,excess_returns,longonly_vs_market,quarterly_compounding}.py` |
| Risk-free series (AQR, not Ken French, for BAB) | `data/risk_free_us.py`, `data/risk_free_canada.py` |

`portfolio/longonly_cache.py` is a parquet cache over the expensive
panel-build step only; the estimator itself is never cached, so a failure
injection cannot be served a stale value.

## Working rules

- **No magic numbers.** Every window, threshold and shrinkage weight comes
  from `config/`.
- **No lookahead.** Betas are estimated through the last trading day of month
  t−1 and used for month t. Universe membership at t is computable from data
  available at t. Index weights use the prior day's close. Enforced by
  `tests/leakage/`.
- **Write the validation test before the function it validates.**
- **A gate must be able to fail.** Correlation is location- and
  scale-invariant, so every gate pairs it with an absolute, non-ratio
  assertion anchored outside the dataset being validated. Two gates in this
  project were passed and later retracted for exactly this reason; that
  history is preserved in `docs/02_validation_gates.md` rather than tidied
  away.

## Validation status

Authoritative status lives in `docs/02_validation_gates.md`.

| Gate | Status |
|---|---|
| 0, 0b — schema & delisting mechanics | ✅ PASS |
| 0c — Canadian universe rule | ⚠️ STALE — validates a superseded Compustat rule, not the current CHASS path |
| 1 — market index | ✅ US PASS (re-derived 2026-09-09) / 🟡 Canada PASS, thin evidence (n=12) |
| 2 — size deciles | ✅ US PASS (re-derived 2026-09-09), one open item |
| 3 — beta estimator | ✅ PASS (2026-09-10) |
| 4 — full pipeline vs published BAB | ✅ US PASS (2026-09-14), correlation 0.9416 / Canada not started |

Open items are recorded rather than absorbed, and are disclosed here
deliberately:

- Gate 2 carries an unexplained ~1.36% / 18-firm residual against Ken French.
- Gate 0c has never been re-run against `chass_universe.universe_at()`.
- Gate 3 has no absolute anchor external to the dataset.
- The Canadian BAB factor is not replicated; the report's Canadian results are
  the long-only book only, on a universe that differs from AQR's.

`docs/07_regime_preregistration.md` fixes the §6.4 regime definitions in
writing *before* any regime statistic was computed.

## Reproducing the numbers

Requires WRDS credentials and a CHASS/CFMRC extract; neither can be
redistributed here.

```bash
python -m venv .venv-wrds
.venv-wrds/Scripts/pip install -r requirements.txt
```

```bash
python run.py test        # full suite
python run.py unit        # fast, no WRDS data needed
python run.py leakage     # leakage tests
python run.py gates       # validation gates (slow, needs cached data)
python run.py lint        # ruff
python run.py typecheck   # mypy
```

**Expected result in this repository**, with `data/raw/` absent:

```
253 passed, 212 skipped, 11 failed, 5 errors
```

Every one of those 11 failures and 5 errors is a missing-input
`FileNotFoundError` from `data/raw/`, not a logic failure. The same tests
were confirmed passing (38 passed, 1 skipped) in the research repository
with the data cache present.

`run.py` pins the interpreter to `.venv-wrds` deliberately: the project has
historically had a second virtualenv missing `yaml`/`polars`/`pandas`, and
picking it fails in a confusing way.

## Known rough edges

- `mypy src/` reports errors that are overwhelmingly missing third-party type
  stubs (`yaml`, `openpyxl`), not type defects.
- `ruff check` reports a small number of unused-variable warnings in tests.

Both pre-date this submission subset and are unchanged by it.
