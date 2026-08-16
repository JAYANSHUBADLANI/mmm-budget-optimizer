#!/usr/bin/env python3
"""Phase 1: load, validate, clean, explore, fit, compare, backtest.

  python scripts/run_phase1.py --smoke     fast wiring check, roughly a minute
  python scripts/run_phase1.py             full sampling, expect tens of minutes
  python scripts/run_phase1.py --no-backtest   skip the refits, fit once only

Everything this produces lands in reports/tables and reports/figures, and the fitted
posterior is saved to models/ so later phases do not need to refit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmm import cleaning, eda, enrichment, io, plots, validation  # noqa: E402
from mmm.config import load_config, repo_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("phase1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 pipeline")
    parser.add_argument("--smoke", action="store_true", help="few draws, for wiring checks only")
    parser.add_argument("--no-backtest", action="store_true")
    parser.add_argument("--no-enrichment", action="store_true")
    parser.add_argument("--skip-model", action="store_true", help="data and EDA steps only")
    args = parser.parse_args()

    cfg = load_config()
    started = time.time()
    summary: dict[str, object] = {"smoke_mode": args.smoke}

    # -- 1. Load raw and check it before touching anything ------------------
    logger.info("Step 1: loading the raw primary dataset")
    raw = io.load_primary_raw(cfg)
    raw = io.parse_primary_dates(raw, cfg)

    raw_report = validation.run_primary_checks(raw, cfg, stage="raw")
    io.save_table(raw_report.to_frame(), "dq_report_raw", cfg)
    summary["raw_shape"] = list(raw.shape)
    summary["raw_check_failures"] = [r.name for r in raw_report.failures]
    logger.info(
        "Raw data failed %d checks and raised %d warnings, which is the expected state "
        "before cleaning",
        len(raw_report.failures),
        len(raw_report.warnings),
    )

    diagnosis = cleaning.diagnose_duplicates(
        raw, cfg["data"]["primary"]["entity_col"], cfg["data"]["primary"]["date_col"]
    )
    io.save_json(diagnosis, "duplicate_diagnosis", cfg)
    summary["duplicate_diagnosis"] = diagnosis

    # -- 2. Clean ------------------------------------------------------------
    # Division Z's conflicting rows are resolved by averaging rather than summing or
    # keeping one source: the averaged values sit close to a typical division's scale,
    # while summing would make Division Z roughly double every other division in the
    # panel. See duplicate_diagnosis.json and the cleaning log for the numbers behind this.
    logger.info("Step 2: cleaning")
    clean, clean_log = cleaning.clean_primary(raw, cfg, on_conflict="average")
    io.save_json(clean_log.to_dict(), "cleaning_log", cfg)
    summary["cleaning_log"] = clean_log.to_dict()

    clean_report = validation.run_primary_checks(clean, cfg, stage="clean")
    io.save_table(clean_report.to_frame(), "dq_report_clean", cfg)
    clean_report.raise_if_failed()
    logger.info("Cleaned data passes every blocking check")

    # -- 3. Enrichment -------------------------------------------------------
    control_cols: list[str] = list(cfg["channels"]["organic_controls"])
    control_cols = [c for c in control_cols if c in clean.columns]
    enriched = clean
    if not args.no_enrichment:
        logger.info("Step 3: enrichment")
        enriched, added = enrichment.enrich_panel(clean, cfg)
        control_cols += added
        summary["enrichment_columns"] = added
        if added:
            collinearity = enrichment.collinearity_with_media(
                enriched, added, list(cfg["channels"]["paid"])
            )
            io.save_table(collinearity, "enrichment_collinearity", cfg)
            flagged = collinearity[collinearity["flag"] != ""]
            if not flagged.empty:
                logger.warning(
                    "These controls correlate above 0.5 with a media channel and may absorb "
                    "media effect:\n%s", flagged.to_string(index=False)
                )
                summary["bad_control_warnings"] = flagged.to_dict("records")
    else:
        summary["enrichment_columns"] = []

    io.save_processed(enriched, "primary_clean", cfg)

    # -- 4. EDA --------------------------------------------------------------
    logger.info("Step 4: exploratory analysis")
    tables = eda.run_eda(enriched, cfg)
    for name, table in tables.items():
        io.save_table(table, name, cfg)

    paid = [c for c in cfg["channels"]["paid"] if c in enriched.columns]
    plots.plot_sales_over_time(enriched, cfg=cfg)
    plots.plot_channel_volumes(enriched, paid, cfg=cfg)
    plots.plot_division_panel(enriched, cfg=cfg)

    if args.skip_model:
        summary["stopped_after"] = "eda"
        io.save_json(summary, "phase1_summary", cfg)
        logger.info("Stopped after EDA as requested, %.1f seconds", time.time() - started)
        return 0

    # -- 5. Bayesian model ---------------------------------------------------
    from mmm.backtest import backtest_bayesian, backtest_linear, baseline_backtest, evaluate
    from mmm.features import carryover_half_life
    from mmm.models.bayesian_mmm import HierarchicalMMM, prepare_model_data
    from mmm.models.linear_mmm import compare_models, fit_linear_mmm
    from mmm.roi import contribution_table, marginal_roi, roi_table

    logger.info("Step 5: fitting the hierarchical Bayesian model")
    holdout = int(cfg["backtest"]["holdout_weeks"])
    n_weeks = enriched[cfg["data"]["primary"]["date_col"]].nunique()

    data = prepare_model_data(
        enriched,
        media_cols=paid,
        control_cols=control_cols,
        entity_col=cfg["data"]["primary"]["entity_col"],
        date_col=cfg["data"]["primary"]["date_col"],
        target_col=cfg["data"]["primary"]["target_col"],
        fourier_order=int(cfg["features"]["seasonality"]["fourier_order"]),
        period_weeks=float(cfg["features"]["seasonality"]["period_weeks"]),
        train_weeks=n_weeks - holdout,
    )
    model = HierarchicalMMM(cfg=cfg, smoke=args.smoke)
    model.fit(data)

    diagnostics = model.diagnostics()
    io.save_table(diagnostics, "model_diagnostics", cfg)
    n_div = model.divergences()
    summary["n_divergences"] = n_div
    max_rhat = float(diagnostics["r_hat"].max()) if "r_hat" in diagnostics else np.nan
    summary["max_r_hat"] = max_rhat
    if n_div > 0 or (not np.isnan(max_rhat) and max_rhat > 1.01):
        logger.warning(
            "Convergence is not clean: %d divergences, maximum r_hat %.4f. Treat the "
            "estimates as provisional and raise target_accept before quoting them.",
            n_div, max_rhat,
        )

    models_dir = repo_root() / cfg["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "primary_posterior.nc")

    params = model.posterior_params(max_draws=500)
    transforms = pd.DataFrame(
        {
            "channel": data.channels,
            "decay_mean": params.decay.mean(axis=0),
            "decay_sd": params.decay.std(axis=0),
            "half_life_weeks": [carryover_half_life(float(d)) for d in params.decay.mean(axis=0)],
            "half_sat_mean": params.half_sat.mean(axis=0),
            "slope_mean": params.slope.mean(axis=0),
        }
    )
    io.save_table(transforms, "fitted_transforms", cfg)

    contrib = contribution_table(params, data, int(cfg["features"]["adstock"]["max_lag"]))
    io.save_table(contrib, "channel_contributions", cfg)

    roi = roi_table(params, data, enriched, cfg, int(cfg["features"]["adstock"]["max_lag"]))
    io.save_table(roi, "channel_roi", cfg)
    plots.plot_roi_comparison(roi, cfg)

    mroi = marginal_roi(params, data, enriched, cfg, int(cfg["features"]["adstock"]["max_lag"]))
    io.save_table(mroi, "channel_marginal_roi", cfg)

    # Parameter recovery, only meaningful when running on the simulated stand in.
    truth_path = repo_root() / cfg["paths"]["raw"] / "replica_ground_truth.json"
    if truth_path.exists():
        with open(truth_path, encoding="utf-8") as handle:
            truth = json.load(handle)
        recovery = pd.DataFrame(
            {
                "channel": data.channels,
                "true_decay": [truth["true_decay"].get(c, np.nan) for c in data.channels],
                "recovered_decay": params.decay.mean(axis=0),
                "true_half_sat": [truth["true_half_sat"].get(c, np.nan) for c in data.channels],
                "recovered_half_sat": params.half_sat.mean(axis=0),
            }
        )
        recovery["decay_absolute_error"] = (
            recovery["recovered_decay"] - recovery["true_decay"]
        ).abs()
        io.save_table(recovery, "parameter_recovery", cfg)
        summary["parameter_recovery_mean_decay_error"] = float(
            recovery["decay_absolute_error"].mean()
        )
        logger.info(
            "Parameter recovery against the simulated ground truth, mean absolute decay "
            "error %.3f", recovery["decay_absolute_error"].mean()
        )

    # -- 6. Linear comparison ------------------------------------------------
    logger.info("Step 6: fitting the linear comparison model")
    linear = fit_linear_mmm(
        enriched,
        media_cols=paid,
        control_cols=control_cols,
        entity_col=cfg["data"]["primary"]["entity_col"],
        date_col=cfg["data"]["primary"]["date_col"],
        target_col=cfg["data"]["primary"]["target_col"],
        cfg=cfg,
    )
    io.save_table(linear.coefficients, "linear_coefficients", cfg)
    io.save_table(linear.vif, "linear_vif", cfg)
    summary["linear_r2"] = linear.r2
    summary["linear_notes"] = linear.notes
    summary["max_media_vif"] = float(linear.vif["vif"].max())

    comparison = compare_models(linear, roi, paid)
    io.save_table(comparison, "method_comparison", cfg)
    if not comparison.empty:
        plots.plot_method_comparison(comparison, cfg)
        summary["median_interval_width_ratio"] = float(
            comparison["interval_width_ratio"].median()
        )
        summary["n_negative_ols_media_coefficients"] = int(
            comparison["ols_sign_negative"].sum()
        )

    # -- 7. Backtest ---------------------------------------------------------
    preds = model.predict(data).mean(axis=0)
    actual = data.y * data.target_scale
    train_end = n_weeks - holdout
    single_split = {
        "train": evaluate(actual[:train_end], preds[:train_end]),
        "holdout": evaluate(actual[train_end:], preds[train_end:]),
    }
    io.save_table(
        pd.DataFrame([{"split": k, **v} for k, v in single_split.items()]),
        "single_split_metrics",
        cfg,
    )
    summary["single_split"] = single_split
    logger.info("Single split holdout metrics: %s", single_split["holdout"])

    if not args.no_backtest:
        logger.info("Step 7: rolling origin backtest, this refits the model per fold")
        bayes_bt, _ = backtest_bayesian(
            enriched, paid, control_cols, cfg, smoke=args.smoke,
            entity_col=cfg["data"]["primary"]["entity_col"],
            date_col=cfg["data"]["primary"]["date_col"],
            target_col=cfg["data"]["primary"]["target_col"],
        )
        io.save_table(bayes_bt, "backtest_bayesian", cfg)

        linear_bt = backtest_linear(
            enriched, paid, control_cols, cfg,
            entity_col=cfg["data"]["primary"]["entity_col"],
            date_col=cfg["data"]["primary"]["date_col"],
            target_col=cfg["data"]["primary"]["target_col"],
        )
        io.save_table(linear_bt, "backtest_linear", cfg)

        base_bt = baseline_backtest(
            enriched, cfg,
            entity_col=cfg["data"]["primary"]["entity_col"],
            date_col=cfg["data"]["primary"]["date_col"],
            target_col=cfg["data"]["primary"]["target_col"],
        )
        io.save_table(base_bt, "backtest_baseline", cfg)
        plots.plot_backtest(bayes_bt, linear_bt, base_bt, cfg=cfg)

        summary["backtest_mean_holdout_mape"] = {
            "bayesian": float(bayes_bt["holdout_mape"].mean()),
            "linear": float(linear_bt["holdout_mape"].mean()),
            "seasonal_naive": float(base_bt["holdout_mape"].mean()),
        }
        logger.info("Mean holdout MAPE: %s", summary["backtest_mean_holdout_mape"])

    summary["runtime_seconds"] = round(time.time() - started, 1)
    io.save_json(summary, "phase1_summary", cfg)
    logger.info("Phase 1 complete in %.1f seconds", time.time() - started)
    if args.smoke:
        logger.warning(
            "This was a smoke run. The numbers in reports/ came from a handful of draws and "
            "are not fit to quote. Rerun without --smoke for results."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
