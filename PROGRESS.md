# Progress

Running record of what is built, what is verified, and what is still open.

## State

All four phases are written. The test suite passes. The modelling stack has been exercised
end to end on the generated stand in dataset. **No model has been fitted on the real Kaggle
data yet**, so no results are quoted anywhere in the documentation.

## Phase 1: data, model, backtest

Done.

- Repository scaffold, config driven throughout, nothing hardcoded outside `config/config.yaml`
- `scripts/fetch_data.py` for the Kaggle downloads and the Robyn RData file
- `scripts/make_replica_dataset.py` generates a schema faithful stand in, 3051 rows, 26
  divisions, 113 weeks, 6 January 2018 to 29 February 2020, with the Division Z duplication
  reproduced and ground truth parameters written out for a recovery check
- Data quality suite: primary key uniqueness, balanced panel, exact duplicates, missing
  values, negativity, zero variance channels, weekly date continuity, derived column detection
- Division Z fix, with diagnosis separated from repair, and a refusal to guess when duplicate
  keys carry conflicting values
- `Overall_Views` identified as `Paid_Views + Organic_Views` and dropped, tested on the data
  rather than assumed
- Enrichment by week: holidays, Google Trends, FRED CPI, each cached, each degrading to a
  warning rather than a failure, with a bad control correlation diagnostic
- EDA producing seven tables and three figures
- Hierarchical Bayesian model in PyMC: geometric adstock, Hill saturation, lognormal
  non centred channel effects per division, Fourier seasonality, division baselines
- Linear marketing mix model with a grid search over transform parameters, variance inflation
  factors, and a comparison table quantifying the interval width difference
- Rolling origin backtest with expanding windows, plus a seasonal naive baseline

Verified so far:

- NumPy, vectorised NumPy and PyTensor implementations of adstock and Hill agree to machine
  precision
- The model builds, `logp` and `dlogp` compile and evaluate finite, and NUTS samples
- Full data pipeline runs end to end: the Division Z bug is detected, fixed, and the cleaned
  panel passes every blocking check
- 64 tests pass and ruff reports no issues
- The optimiser comparison was corrected mid build: the naive plan was originally scored
  without constraints, which flattered it. It is now reported unconstrained with a
  feasibility flag, projected onto the feasible set, and against the constrained optimum

## Phase 2: validation layers

Done, code complete.

- Robyn `dt_simulated_weekly` benchmark via `pyreadr`, with explicit pass or fail checks on
  effect signs, holdout error and media share of sales
- Secondary dataset generalisation check, with column resolution at runtime and an explicit
  statement that a 300 row result is a software claim rather than a scientific one
- Geo holdout: fit without divisions selected by variability rank, predict them from pooled
  parameters, compare against a media free baseline
- Difference in differences with division clustered standard errors, a minimum treated unit
  guard, a parallel trends pre test, and a placebo distribution over 50 random assignments

Not yet run against real data, because the Kaggle and GitHub downloads have not been
performed.

## Phase 3: money and optimisation

Done, code complete.

- CPM translation with every rate and its basis recorded in config and documented
- Plausibility check comparing implied media spend against revenue, with a suggested uniform
  rescale when it falls outside the range real advertisers occupy
- Constrained optimiser: SLSQP, budget equality, per channel floors and caps, total
  reallocation cap via the exact linear reformulation of the absolute value, twelve restarts
- Binding constraint reporting, so the cost of executability is visible
- Uncertainty on the recommendation by re-solving under posterior draws
- Naive proportional to average ROI comparator
- CPM sensitivity sweep reporting direction stability under cost shocks
- Marginal ROI alongside average ROI
- Streamlit dashboard with constraint sliders, editable CPM inputs, and a curve showing what
  the execution constraint costs

## Phase 4: engineering layer

Done, code complete.

- dbt project on DuckDB, Bronze, Silver, Gold, with Bronze deliberately retaining the
  duplicates so the fix is auditable
- Four singular dbt tests including the Division Z regression test, a full history check, a
  guard against reintroducing `Overall_Views`, and a check that the SQL CPM values have not
  drifted from the config
- Airflow DAG with `dbt test` as a hard gate, a refit short circuit, and a fit task that
  refuses to publish a non converged model
- Docker Compose: dashboard by default, full Airflow stack behind a profile
- Makefile wrapping every step

Not executed yet: `dbt run`, `dbt test` and the Airflow DAG have not been run end to end.
Neither Airflow nor dbt is installed on the machine this was built on, and the Docker stack
that would run them has not been brought up.

## Verification summary

| Check | Result |
| --- | --- |
| Replica dataset shape | 3051 rows, 26 divisions, Division Z at 226 rows, 113 exact duplicates |
| Data quality catches the bug | Primary key and date continuity checks both fail on raw, pass after cleaning |
| Derived column detection | `Overall_Views` identified and dropped |
| Adstock, NumPy against PyTensor | Agree to 1e-16 |
| Hill, NumPy against PyTensor | Agree exactly |
| Model compiles and samples | Yes, finite logp and gradients, zero divergences on the small smoke fit |
| Optimiser | Budget conserved, bounds respected, movement cap respected, uplift non negative |
| Test suite | 64 passing, ruff clean |
| Full phase 1 data path | Runs end to end in about 5 seconds without the model |
| Optimiser against naive plan | Like for like on the same constraint set, optimiser wins |
| CPM plausibility check | Correctly flags the stand in data at 0.6 percent of revenue |

## Still open

1. **Fetch the real data.** `python scripts/fetch_data.py` needs a Kaggle token at
   `~/.kaggle/kaggle.json`. The stand in files in `data/raw/` have to come out first
   (`sample_media_spend.csv`, `REPLICA_NOTICE.txt`, `replica_ground_truth.json`), or the
   loader picks them up instead of the download.

2. **Confirm the real Division Z row counts.** The 226 against 113 figures come from an
   inspection of the source file, not from the pipeline. Once the real download is in place,
   `reports/tables/duplicate_diagnosis.json` confirms or corrects them, and the figure quoted
   in `README.md` follows whatever that file says.

3. **Run the full pipeline.** Phase 1 at full sampling, then phases 2 and 3, roughly an hour
   or two. `reports/tables/model_diagnostics.csv` carries r hat and divergence counts, and
   nothing downstream is worth reading until those look right.

4. **Write up the results.** `docs/executive_summary.md` has a marked placeholder where the
   recommendation goes. The README carries no results section until there are checked numbers
   behind it.

5. **Calibrate the CPM rates.** Phase 3 writes `reports/tables/cpm_plausibility.json`. If
   implied spend falls outside 2 to 25 percent of revenue, the suggested scaling factor goes
   into the `cpm` block in `config/config.yaml` and the phase reruns, with the change recorded
   in `docs/cpm_assumptions.md`.

6. **Decide what to commit.** `data/` and `models/` are gitignored. The generated stand in
   CSV is about 400 KB and the figures a few hundred KB. Committing `reports/tables/*.csv` is
   worth considering so the repository shows results without a reviewer having to run
   anything, but only after step 3.

7. **Run the engineering layer end to end.** `pip install -r requirements-pipeline.txt`, then
   `make dbt`, then `docker compose --profile pipeline up`. The code is written and unit
   tested but has never been executed against a live warehouse and scheduler.

## Known gaps I chose not to close

- The optimiser holds flighting fixed and reallocates only between channels
- The model is additive in level space rather than log linear, traded for unambiguous
  contribution decomposition
- Google Trends is a bad control and is reported both ways rather than resolved
- No price, promotion or competitor variables exist in the source data
- The parameter recovery check only means something on the simulated stand in

All of these are written up in `docs/limitations.md` with the reasoning and what would fix
them.
