# Running the pipeline

## Local, no Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_data.py            # or make_replica_dataset.py
python scripts/run_phase1.py --smoke    # wiring check
python scripts/run_phase1.py            # real run
python scripts/run_phase2.py
python scripts/run_phase3.py
streamlit run app/streamlit_app.py
```

`make help` lists the same steps as shorthand targets.

## Expected runtimes

Measured on a four core laptop. Sampling dominates everything else.

| Step | Smoke | Full |
| --- | --- | --- |
| Data quality, cleaning, EDA | under 10 seconds | under 10 seconds |
| Bayesian fit, 26 divisions | 1 to 3 minutes | 20 to 60 minutes |
| Rolling backtest, 3 folds | 3 to 9 minutes | 1 to 3 hours |
| Optimiser with uncertainty | under 1 minute | 3 to 10 minutes |
| CPM sensitivity sweep | 1 to 3 minutes | 10 to 30 minutes |

The first PyTensor compilation takes a few minutes on its own and is cached afterwards. If
you see a long pause before sampling starts with no output, that is what is happening.

## dbt

```bash
cd dbt
dbt deps --profiles-dir .
dbt run  --profiles-dir .
dbt test --profiles-dir .
```

Builds Bronze, Silver and Gold into `data/processed/mmm.duckdb`. The Silver layer is where
the Division Z duplicates are removed, and `dbt test` is what proves they stayed removed.

## Airflow

```bash
docker compose --profile pipeline up
# webserver at http://localhost:8080, admin / admin
```

The DAG runs weekly on Monday at 06:00. `dbt test` is a hard gate before any fitting. The
fit task is behind a short circuit operator that skips the refit unless at least four new
weeks of data have arrived, because refitting weekly makes the estimates jitter for sampling
reasons rather than market reasons.

Set `MMM_SMOKE=1` in the scheduler environment to run the DAG with minimal sampling, which
is how the compose file is configured by default so the stack is demonstrable without an
hour of compute.

## Airflow dependency installation

Airflow pins a long list of transitive dependencies. Installing this project's requirements
on top without the constraint file will silently upgrade one of them and break the scheduler
at runtime rather than at install time. The Dockerfile uses:

```
pip install -r requirements.txt --constraint \
  "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"
```

If installing Airflow locally rather than in Docker, use the same constraint URL.

## Configuration

Everything lives in `config/config.yaml`. The values most worth changing:

| Key | What it controls |
| --- | --- |
| `cpm.*.value` | Assumed cost per thousand. Every currency figure depends on these |
| `model.bayesian.draws` and `chains` | Sampling effort against runtime |
| `backtest.holdout_weeks` and `n_folds` | Evaluation protocol |
| `optimizer.min_spend_pct_of_current` | Per channel floor |
| `optimizer.max_total_reallocation_pct` | How much of the budget may move |
| `features.adstock.max_lag` | Carryover window in weeks |
| `enrichment.*.enabled` | Turn individual external sources on or off |

Secrets go in `.env`, copied from `.env.example`. Nothing in the repository reads a
credential from anywhere else.
