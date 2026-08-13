"""Scenario dashboard.

  streamlit run app/streamlit_app.py

The dashboard exists so a planner can change the constraints and see the recommendation
move, rather than being handed one allocation and asked to trust it. The three sliders are
the three levers a real media team argues about: how much can move in total, how far any one
channel can be cut, and how far it can be raised.

It loads the posterior saved by phase 1. It does not fit anything, so it starts fast and
every scenario re-solves the optimisation against the same fitted model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmm import io  # noqa: E402
from mmm.config import DotDict, cpm_rates, load_config, repo_root  # noqa: E402

st.set_page_config(page_title="Media budget scenarios", layout="wide")


@st.cache_resource(show_spinner="Loading the fitted model")
def load_everything():
    from mmm.models.bayesian_mmm import HierarchicalMMM, prepare_model_data

    cfg = load_config()
    panel = io.load_processed("primary_clean", cfg)
    paid = [c for c in cfg["channels"]["paid"] if c in panel.columns]
    controls = [c for c in cfg["channels"]["organic_controls"] if c in panel.columns]
    controls += [
        c
        for c in ["n_holidays", "is_major_holiday_week", "is_black_friday_week",
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
    posterior_path = repo_root() / cfg["paths"]["models"] / "primary_posterior.nc"
    if not posterior_path.exists():
        return cfg, panel, data, None
    model = HierarchicalMMM(cfg=cfg)
    model.load(posterior_path)
    model.data = data
    # 200 draws keeps each scenario responsive. The full posterior is used in the reports.
    return cfg, panel, data, model.posterior_params(max_draws=200)


def main() -> None:
    st.title("Media budget scenarios")
    st.caption(
        "Constrained reallocation against a hierarchical Bayesian marketing mix model. "
        "All currency figures rest on assumed cost per thousand impressions, documented in "
        "docs/cpm_assumptions.md."
    )

    cfg, panel, data, params = load_everything()
    if params is None:
        st.error(
            "No fitted posterior found at models/primary_posterior.nc. "
            "Run 'python scripts/run_phase1.py' first, or 'python scripts/run_phase1.py "
            "--smoke' for a fast wiring check."
        )
        return

    from mmm.optimizer import optimise_budget
    from mmm.roi import check_spend_plausibility, marginal_roi, roi_table

    max_lag = int(cfg["features"]["adstock"]["max_lag"])
    rates = cpm_rates(cfg)
    # -- sidebar controls ----------------------------------------------------
    st.sidebar.header("Constraints")
    max_realloc = st.sidebar.slider(
        "Share of total budget allowed to move", 0.0, 0.60,
        float(cfg["optimizer"]["max_total_reallocation_pct"]), 0.05,
        help="The execution constraint. Zero means no change is permitted.",
    )
    min_pct = st.sidebar.slider(
        "Floor per channel, as a share of current spend", 0.0, 1.0,
        float(cfg["optimizer"]["min_spend_pct_of_current"]), 0.05,
        help="Stops the optimiser from proposing that a channel be switched off.",
    )
    max_pct = st.sidebar.slider(
        "Cap per channel, as a multiple of current spend", 1.0, 3.0,
        float(cfg["optimizer"]["max_spend_pct_of_current"]), 0.1,
        help="Reflects real inventory and team capacity limits.",
    )
    budget_multiplier = st.sidebar.slider(
        "Total budget, relative to current", 0.5, 1.5, 1.0, 0.05
    )

    st.sidebar.header("Assumed CPM")
    st.sidebar.caption(
        "These are assumptions, not measured costs. Change one and the recommendation "
        "changes, which is the honest way to see how much the answer depends on them."
    )
    adjusted_rates = {}
    for channel in data.channels:
        adjusted_rates[channel] = st.sidebar.number_input(
            channel.replace("_", " "), min_value=0.01, value=float(rates.get(channel, 1.0)),
            step=0.5, format="%.2f",
        )

    scaled_spend = np.array(
        [float(panel[c].sum()) / 1000.0 * adjusted_rates[c] for c in data.channels]
    )
    total_budget = float(scaled_spend.sum() * budget_multiplier)

    scenario_cfg = DotDict(
        {
            **cfg,
            "optimizer": {
                **cfg["optimizer"],
                "min_spend_pct_of_current": min_pct,
                "max_spend_pct_of_current": max_pct,
                "max_total_reallocation_pct": max_realloc,
                "n_restarts": 4,
            },
        }
    )

    # -- plausibility banner -------------------------------------------------
    plaus = check_spend_plausibility(panel, data.channels, "Sales", cfg)
    if not plaus["plausible"]:
        st.warning(
            f"Assumed media spend is {plaus['spend_share_of_revenue']:.2%} of revenue. "
            "Real advertisers typically sit between 2 and 25 percent. The channel ranking is "
            "unaffected by a uniform rescale of every CPM, but the absolute ROI levels below "
            "should be read as relative, not literal."
        )

    # -- solve ---------------------------------------------------------------
    try:
        result = optimise_budget(
            params, data, scaled_spend, scenario_cfg, total_budget=total_budget, max_lag=max_lag
        )
    except ValueError as exc:
        st.error(f"These constraints have no feasible solution: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total budget", f"{total_budget:,.0f}")
    col2.metric("Predicted uplift", f"{result.uplift_pct:.2f} percent")
    col3.metric("Budget moved", f"{np.abs(result.optimal_spend - scaled_spend).sum() / 2:,.0f}")

    frame = result.to_frame()
    st.subheader("Recommended allocation")
    st.dataframe(
        frame.style.format(
            {
                "current_spend": "{:,.0f}",
                "recommended_spend": "{:,.0f}",
                "change_absolute": "{:+,.0f}",
                "change_pct": "{:+.1f}",
                "current_share": "{:.1%}",
                "recommended_share": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    chart = frame.set_index("channel")[["current_spend", "recommended_spend"]]
    st.bar_chart(chart)

    if result.binding_constraints:
        st.info(
            "Binding constraints: "
            + "; ".join(result.binding_constraints)
            + ". The unconstrained optimum is more extreme than this. Loosening a binding "
            "constraint is where the remaining upside sits."
        )

    # -- returns -------------------------------------------------------------
    st.subheader("Average and marginal return")
    st.caption(
        "Reallocation should follow marginal return, the value of the next unit of spend, "
        "not average return across everything spent so far. On a saturating curve the two "
        "diverge, and following the average is how a reallocation that looked good on a "
        "spreadsheet underdelivers."
    )
    roi = roi_table(params, data, panel, cfg, max_lag)
    mroi = marginal_roi(params, data, panel, cfg, max_lag)
    merged = roi.merge(mroi, on="channel", how="left")
    st.dataframe(
        merged[
            [
                "channel",
                "assumed_spend",
                "roi_mean",
                "roi_hdi_lower",
                "roi_hdi_upper",
                "marginal_roi_mean",
                "prob_marginal_roi_above_1",
            ]
        ].style.format(
            {
                "assumed_spend": "{:,.0f}",
                "roi_mean": "{:.2f}",
                "roi_hdi_lower": "{:.2f}",
                "roi_hdi_upper": "{:.2f}",
                "marginal_roi_mean": "{:.2f}",
                "prob_marginal_roi_above_1": "{:.0%}",
            }
        ),
        use_container_width=True,
    )

    # -- what the constraint costs ------------------------------------------
    st.subheader("What the execution constraint costs")
    st.caption(
        "Uplift against how much budget is allowed to move. The curve flattens where the "
        "per channel caps take over from the total movement cap."
    )
    rows = []
    for cap in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        trial_cfg = DotDict(
            {**scenario_cfg, "optimizer": {**scenario_cfg["optimizer"],
                                           "max_total_reallocation_pct": cap,
                                           "n_restarts": 2}}
        )
        try:
            trial = optimise_budget(
                params, data, scaled_spend, trial_cfg, total_budget=total_budget, max_lag=max_lag
            )
            rows.append({"reallocation_cap": cap, "uplift_pct": trial.uplift_pct})
        except ValueError:
            continue
    if rows:
        st.line_chart(pd.DataFrame(rows).set_index("reallocation_cap"))

    with st.expander("Model and method caveats"):
        st.markdown(
            """
This dashboard reports what the model believes, and the model is a regression on
observational data. Three things to hold in mind before acting on it.

The currency figures rest on assumed cost per thousand impressions. The source data
contains impressions and views, not spend. Relative CPMs across channels drive the
ranking, so those matter more than the absolute levels.

The optimiser scales each channel's existing weekly and per division pattern by a single
multiplier. It reallocates between channels, it does not reschedule flighting within a
channel.

Media budgets were not randomly assigned, so the estimates are causal only to the extent
the validation checks in reports/tables support. See docs/limitations.md.
            """
        )


if __name__ == "__main__":
    main()
