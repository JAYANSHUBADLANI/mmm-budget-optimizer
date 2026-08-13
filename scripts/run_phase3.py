#!/usr/bin/env python3
"""Phase 3: translate the model into budget terms and solve the reallocation.

  python scripts/run_phase3.py                 uses the posterior saved by phase 1
  python scripts/run_phase3.py --refit --smoke fits first, for a quick end to end check
  python scripts/run_phase3.py --budget 5000000

Reads models/primary_posterior.nc, so run phase 1 first.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmm import io, plots  # noqa: E402
from mmm.config import cpm_rates, load_config, repo_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("phase3")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 pipeline")
    parser.add_argument("--refit", action="store_true", help="fit rather than load the posterior")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--budget", type=float, default=None, help="override the total budget")
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()

    from mmm.models.bayesian_mmm import HierarchicalMMM, prepare_model_data
    from mmm.optimizer import (
        cpm_sensitivity,
        naive_proportional_reallocation,
        optimise_budget,
        optimise_with_uncertainty,
    )
    from mmm.roi import observed_spend_by_channel, roi_table, saturation_curve

    cfg = load_config()
    started = time.time()
    summary: dict[str, object] = {}

    panel = io.load_processed("primary_clean", cfg)
    paid = [c for c in cfg["channels"]["paid"] if c in panel.columns]
    controls = [c for c in cfg["channels"]["organic_controls"] if c in panel.columns]
    controls += [
        c for c in ["n_holidays", "is_major_holiday_week", "is_black_friday_week",
                    "trends_index", "cpi_index"]
        if c in panel.columns
    ]

    holdout = int(cfg["backtest"]["holdout_weeks"])
    n_weeks = panel["Calendar_Week"].nunique()
    data = prepare_model_data(
        panel,
        media_cols=paid,
        control_cols=controls,
        fourier_order=int(cfg["features"]["seasonality"]["fourier_order"]),
        period_weeks=float(cfg["features"]["seasonality"]["period_weeks"]),
        train_weeks=n_weeks - holdout,
    )

    model = HierarchicalMMM(cfg=cfg, smoke=args.smoke)
    posterior_path = repo_root() / cfg["paths"]["models"] / "primary_posterior.nc"
    if args.refit or not posterior_path.exists():
        logger.info("Fitting the model, no saved posterior at %s", posterior_path)
        model.fit(data)
        model.save(posterior_path)
    else:
        logger.info("Loading the saved posterior from %s", posterior_path)
        model.load(posterior_path)
        model.data = data

    params = model.posterior_params(max_draws=500)
    max_lag = int(cfg["features"]["adstock"]["max_lag"])

    # -- current spend under the assumed CPM rates ---------------------------
    rates = cpm_rates(cfg)
    current = observed_spend_by_channel(panel, paid, cfg)
    current_spend = np.array([float(current[c]) for c in data.channels])
    spend_summary = pd.DataFrame(
        {
            "channel": data.channels,
            "total_impressions": [float(panel[c].sum()) for c in data.channels],
            "assumed_cpm": [rates.get(c, 0.0) for c in data.channels],
            "assumed_spend": current_spend,
            "share_of_budget": current_spend / current_spend.sum(),
        }
    )
    io.save_table(spend_summary, "assumed_spend_by_channel", cfg)
    summary["total_assumed_budget"] = float(current_spend.sum())
    logger.info("Total assumed budget %.0f", current_spend.sum())

    # Sanity check the CPM assumptions before anything downstream depends on them.
    from mmm.roi import check_spend_plausibility, suggested_cpm_scaling

    plausibility = check_spend_plausibility(panel, data.channels, "Sales", cfg)
    plausibility["suggested_uniform_cpm_scaling"] = suggested_cpm_scaling(
        panel, data.channels, "Sales", cfg
    )
    io.save_json(plausibility, "cpm_plausibility", cfg)
    summary["cpm_plausibility"] = plausibility
    if not plausibility["plausible"]:
        logger.warning(
            "Assumed spend is %.3f of revenue. Multiplying every CPM by %.1f would put it "
            "at ten percent, which is a more defensible starting point. Relative costs "
            "across channels are unchanged by a uniform rescale, so the reallocation "
            "ranking does not move.",
            plausibility["spend_share_of_revenue"],
            plausibility["suggested_uniform_cpm_scaling"],
        )

    roi = roi_table(params, data, panel, cfg, max_lag)
    io.save_table(roi, "channel_roi_phase3", cfg)

    # -- saturation curves ---------------------------------------------------
    curves = {ch: saturation_curve(params, data, ch, cfg, max_lag=max_lag) for ch in data.channels}
    io.save_table(pd.concat(curves.values(), ignore_index=True), "saturation_curves", cfg)
    plots.plot_saturation_curves(curves, cfg)

    # -- constrained optimisation -------------------------------------------
    logger.info("Solving the constrained reallocation")
    result = optimise_budget(params, data, current_spend, cfg, total_budget=args.budget,
                             max_lag=max_lag)
    frame = result.to_frame()
    io.save_table(frame, "optimal_allocation", cfg)
    plots.plot_budget_allocation(frame, cfg)

    summary["optimiser"] = {
        "success": result.success,
        "message": result.message,
        "total_budget": result.total_budget,
        "predicted_sales_current": result.predicted_sales_current,
        "predicted_sales_optimal": result.predicted_sales_optimal,
        "uplift_absolute": result.uplift,
        "uplift_pct": result.uplift_pct,
        "binding_constraints": result.binding_constraints,
        "notes": result.notes,
    }
    logger.info("Uplift %.2f percent under the stated constraints", result.uplift_pct)

    # -- the naive alternative, for comparison -------------------------------
    #
    # The comparison has to be like for like. Scoring the naive plan without applying the
    # constraints the optimiser obeys would compare a solution to an easier problem, and
    # the naive plan would often win purely because it is allowed to be infeasible. So I
    # report the raw naive plan, flag whether it is executable, and also score the version
    # projected back into the feasible set, which is the fair comparison.
    from mmm.optimizer import check_feasibility, make_objective, project_to_feasible

    objective = make_objective(params, data, current_spend, max_lag)
    naive_raw = naive_proportional_reallocation(roi, current_spend, data.channels)
    naive_feasible = project_to_feasible(
        naive_raw, current_spend, result.total_budget, cfg
    )
    naive_check = check_feasibility(naive_raw, current_spend, result.total_budget, cfg)

    comparison = pd.DataFrame(
        {
            "approach": [
                "current allocation",
                "proportional to average ROI, unconstrained",
                "proportional to average ROI, made executable",
                "constrained optimiser",
            ],
            "predicted_sales": [
                result.predicted_sales_current,
                float(objective(naive_raw)),
                float(objective(naive_feasible)),
                result.predicted_sales_optimal,
            ],
            "respects_constraints": [
                True,
                naive_check["feasible"],
                True,
                True,
            ],
            "share_of_budget_moved": [
                0.0,
                naive_check["share_of_budget_moved"],
                float(np.abs(naive_feasible - current_spend).sum() / (2 * result.total_budget)),
                float(np.abs(result.optimal_spend - current_spend).sum()
                      / (2 * result.total_budget)),
            ],
        }
    )
    comparison["uplift_vs_current_pct"] = (
        100.0
        * (comparison["predicted_sales"] - result.predicted_sales_current)
        / result.predicted_sales_current
    )
    io.save_table(comparison, "allocation_approach_comparison", cfg)
    summary["approach_comparison"] = comparison.to_dict("records")
    summary["naive_plan_violations"] = naive_check["violations"]
    if not naive_check["feasible"]:
        logger.info(
            "The unconstrained naive plan is not executable: %s. Compare the optimiser "
            "against the 'made executable' row, not against the unconstrained one.",
            naive_check["violations"],
        )

    # -- uncertainty on the recommendation -----------------------------------
    logger.info("Re-solving across posterior draws for an interval on the recommendation")
    alloc_summary, uplift_summary = optimise_with_uncertainty(
        params, data, current_spend, cfg, max_lag=max_lag
    )
    io.save_table(alloc_summary, "optimal_allocation_uncertainty", cfg)
    io.save_table(uplift_summary, "uplift_uncertainty", cfg)
    summary["uplift_uncertainty"] = uplift_summary.to_dict("records")

    # -- CPM sensitivity -----------------------------------------------------
    if not args.no_sensitivity:
        logger.info("CPM sensitivity sweep, this re-solves the problem many times")
        sens = cpm_sensitivity(params, data, panel, cfg, max_lag=max_lag)
        io.save_table(sens, "cpm_sensitivity", cfg)
        ok = sens[sens["status"] == "ok"] if "status" in sens else sens
        if not ok.empty:
            direction_cols = [c for c in ok.columns if c.endswith("_direction")]
            stability = {
                c.replace("_direction", ""): float(
                    ok[c].value_counts(normalize=True).max()
                )
                for c in direction_cols
            }
            summary["direction_stability_under_cpm_shocks"] = stability
            logger.info(
                "Share of CPM scenarios where each channel keeps the same recommended "
                "direction: %s", stability
            )

    summary["runtime_seconds"] = round(time.time() - started, 1)
    io.save_json(summary, "phase3_summary", cfg)
    logger.info("Phase 3 complete in %.1f seconds", time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
