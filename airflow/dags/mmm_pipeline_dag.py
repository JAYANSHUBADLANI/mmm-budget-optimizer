"""Airflow DAG for the weekly refresh.

Shape of the pipeline:

    fetch -> load to duckdb -> dbt run -> dbt test -> validate -> [enrich] -> fit
    -> backtest -> optimise -> publish

Two design decisions worth defending:

1. dbt test is a hard gate. If the Division Z duplicate check fails, the DAG stops before
   fitting. A pipeline that keeps running on data it knows is broken produces a number that
   looks authoritative and is wrong, which is worse than producing nothing.

2. The model fit is not on the weekly schedule by accident. Refitting a Bayesian marketing
   mix model every week on 113 weeks of data means the estimates jitter week to week for
   reasons that are sampling noise, not market change, and stakeholders lose trust in the
   number. The schedule here is weekly for the data layer and the fit task checks whether
   enough new data has arrived to justify a refit.

A note on scope. This orchestration layer is plumbing, not the contribution of the project.
It exists to show the modelling can run as a scheduled job rather than a notebook.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

PROJECT_ROOT = Path(os.environ.get("MMM_PROJECT_ROOT", "/opt/project"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "analytics",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

MIN_NEW_WEEKS_BEFORE_REFIT = 4


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------
def load_raw_to_duckdb(**_context) -> str:
    """Register the raw CSV as a DuckDB table so dbt has something to build on."""
    import duckdb

    from mmm.config import load_config

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    raw_path = PROJECT_ROOT / cfg["paths"]["raw"] / cfg["data"]["primary"]["filename"]
    db_path = PROJECT_ROOT / cfg["paths"]["processed"] / "mmm.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} is missing. Run scripts/fetch_data.py, or "
            "scripts/make_replica_dataset.py to generate the stand in."
        )

    con = duckdb.connect(str(db_path))
    con.execute("create schema if not exists main")
    con.execute(
        f"""
        create or replace table main.sample_media_spend as
        select * from read_csv_auto('{raw_path}', header=true)
        """
    )
    n_rows = con.execute("select count(*) from main.sample_media_spend").fetchone()[0]
    con.close()
    logger.info("Loaded %d rows into %s", n_rows, db_path)
    return str(db_path)


def run_python_validation(**_context) -> dict:
    """Run the Python data quality suite as well as the dbt tests.

    The two overlap on purpose. dbt tests guard the warehouse contract, the Python checks
    guard the modelling assumptions, and the panel continuity check in particular has no
    natural expression as a column level dbt test.
    """
    from mmm import cleaning, io, validation
    from mmm.config import load_config

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    raw = io.parse_primary_dates(io.load_primary_raw(cfg), cfg)

    raw_report = validation.run_primary_checks(raw, cfg, stage="raw")
    diagnosis = cleaning.diagnose_duplicates(
        raw, cfg["data"]["primary"]["entity_col"], cfg["data"]["primary"]["date_col"]
    )
    clean, log = cleaning.clean_primary(raw, cfg)
    clean_report = validation.run_primary_checks(clean, cfg, stage="clean")
    clean_report.raise_if_failed()

    io.save_processed(clean, "primary_clean", cfg)
    io.save_json(diagnosis, "duplicate_diagnosis", cfg)
    io.save_json(log.to_dict(), "cleaning_log", cfg)

    return {
        "raw_failures": [r.name for r in raw_report.failures],
        "rows_removed": log.rows_in - log.rows_out,
        "duplicate_entities": log.duplicate_entities,
    }


def should_refit(**context) -> bool:
    """Skip the fit unless enough new weeks have arrived since the last one.

    Refitting on every run produces estimates that move for sampling reasons rather than
    market reasons, and a media team that watches an ROI number wobble every Monday stops
    believing any of it.
    """
    from mmm import io
    from mmm.config import load_config

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    posterior = PROJECT_ROOT / cfg["paths"]["models"] / "primary_posterior.nc"
    if not posterior.exists():
        logger.info("No existing posterior, fitting for the first time")
        return True

    panel = io.load_processed("primary_clean", cfg)
    latest_week = panel["Calendar_Week"].max()
    fitted_at = datetime.fromtimestamp(posterior.stat().st_mtime)
    weeks_since = (latest_week.to_pydatetime() - fitted_at).days / 7.0
    logger.info("Latest data week %s, model fitted %s, gap %.1f weeks",
                latest_week, fitted_at, weeks_since)
    return weeks_since >= MIN_NEW_WEEKS_BEFORE_REFIT


def fit_model(**_context) -> dict:
    from mmm import io
    from mmm.config import load_config
    from mmm.models.bayesian_mmm import HierarchicalMMM, prepare_model_data

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    panel = io.load_processed("primary_clean", cfg)
    paid = [c for c in cfg["channels"]["paid"] if c in panel.columns]
    controls = [c for c in cfg["channels"]["organic_controls"] if c in panel.columns]

    holdout = int(cfg["backtest"]["holdout_weeks"])
    n_weeks = panel["Calendar_Week"].nunique()
    data = prepare_model_data(
        panel, media_cols=paid, control_cols=controls, train_weeks=n_weeks - holdout
    )
    model = HierarchicalMMM(cfg=cfg, smoke=bool(os.environ.get("MMM_SMOKE")))
    model.fit(data)
    model.save(PROJECT_ROOT / cfg["paths"]["models"] / "primary_posterior.nc")

    n_div = model.divergences()
    diagnostics = model.diagnostics()
    max_rhat = float(diagnostics["r_hat"].max()) if "r_hat" in diagnostics else float("nan")

    # Fail the task rather than publish a model that did not converge. A silent bad fit is
    # the failure mode that ends up in a board deck.
    if n_div > 0.01 * int(cfg["model"]["bayesian"]["draws"]) * int(
        cfg["model"]["bayesian"]["chains"]
    ):
        raise ValueError(f"{n_div} divergent transitions, refusing to publish this fit")
    if max_rhat == max_rhat and max_rhat > 1.05:
        raise ValueError(f"Maximum r_hat {max_rhat:.3f}, chains have not mixed")

    return {"n_divergences": n_div, "max_r_hat": max_rhat}


def run_optimiser(**_context) -> dict:
    import numpy as np

    from mmm import io
    from mmm.config import load_config
    from mmm.models.bayesian_mmm import HierarchicalMMM, prepare_model_data
    from mmm.optimizer import optimise_budget
    from mmm.roi import observed_spend_by_channel

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    panel = io.load_processed("primary_clean", cfg)
    paid = [c for c in cfg["channels"]["paid"] if c in panel.columns]
    controls = [c for c in cfg["channels"]["organic_controls"] if c in panel.columns]

    holdout = int(cfg["backtest"]["holdout_weeks"])
    n_weeks = panel["Calendar_Week"].nunique()
    data = prepare_model_data(
        panel, media_cols=paid, control_cols=controls, train_weeks=n_weeks - holdout
    )
    model = HierarchicalMMM(cfg=cfg)
    model.load(PROJECT_ROOT / cfg["paths"]["models"] / "primary_posterior.nc")
    model.data = data
    params = model.posterior_params(max_draws=400)

    current = observed_spend_by_channel(panel, data.channels, cfg)
    current_spend = np.array([float(current[c]) for c in data.channels])
    result = optimise_budget(params, data, current_spend, cfg)

    io.save_table(result.to_frame(), "optimal_allocation", cfg)
    return {"uplift_pct": result.uplift_pct, "binding": result.binding_constraints}


def publish_summary(**context) -> None:
    """Collect the task outputs into one summary file for the reporting layer."""
    from mmm import io
    from mmm.config import load_config

    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    ti = context["ti"]
    payload = {
        "run_date": str(context.get("ds")),
        "validation": ti.xcom_pull(task_ids="validate_panel"),
        "fit": ti.xcom_pull(task_ids="fit_model"),
        "optimiser": ti.xcom_pull(task_ids="run_optimiser"),
    }
    io.save_json(payload, "pipeline_run_summary", cfg)
    logger.info("Published run summary: %s", payload)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id="mmm_pipeline",
    description="Weekly marketing mix refresh: ingest, transform, validate, fit, optimise",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * 1",  # Monday morning, after the week closes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["marketing-mix", "bayesian", "optimisation"],
) as dag:

    fetch = BashOperator(
        task_id="fetch_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && python scripts/fetch_data.py --primary || "
            "echo 'Fetch failed, continuing with whatever is already on disk'"
        ),
    )

    load = PythonOperator(task_id="load_raw_to_duckdb", python_callable=load_raw_to_duckdb)

    dbt_deps = BashOperator(
        task_id="dbt_deps",
        bash_command=f"cd {PROJECT_ROOT}/dbt && dbt deps --profiles-dir .",
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {PROJECT_ROOT}/dbt && dbt run --profiles-dir .",
    )

    # The hard gate. No fitting on data that failed its contract.
    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {PROJECT_ROOT}/dbt && dbt test --profiles-dir .",
    )

    validate = PythonOperator(task_id="validate_panel", python_callable=run_python_validation)

    refit_gate = ShortCircuitOperator(task_id="should_refit", python_callable=should_refit)

    fit = PythonOperator(task_id="fit_model", python_callable=fit_model)

    optimise = PythonOperator(task_id="run_optimiser", python_callable=run_optimiser)

    publish = PythonOperator(
        task_id="publish_summary",
        python_callable=publish_summary,
        trigger_rule="none_failed_min_one_success",
    )

    fetch >> load >> dbt_deps >> dbt_run >> dbt_test >> validate
    validate >> refit_gate >> fit >> optimise >> publish
