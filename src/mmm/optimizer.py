"""Constrained budget reallocation.

The problem. Total media budget is fixed. Sales respond to each channel through a
saturating curve estimated by the model. Find the split across channels that maximises
expected sales, subject to constraints a media team would actually accept.

  maximise    sum over channels of contribution_c(spend_c)
  subject to  sum of spend_c = total budget
              floor_c <= spend_c <= cap_c
              sum of |spend_c - current_c| <= 2 * R * total budget

The third constraint is the one that makes this useful rather than academic. Without it
the optimiser happily proposes cutting a channel by 90 percent and tripling another,
which is not a plan anyone can execute inside a quarter. R is the share of total budget
allowed to move, and the factor of two is because under a fixed total, every dollar that
leaves one channel arrives at another, so the sum of absolute deviations counts it twice.

The absolute value is not differentiable, and SLSQP wants smooth constraints. Rather than
smoothing it and hoping, I introduce auxiliary variables u_c with u_c >= spend_c - current_c
and u_c >= current_c - spend_c, which is the standard exact linear reformulation. The
decision vector is therefore twice as long as the number of channels.

What this optimiser is not. It moves budget between channels while holding the weekly and
per division flighting pattern fixed, scaling each channel's impressions by a single
multiplier. Optimising the flighting as well is a much larger problem and I did not solve
it. That limitation is stated in the executive summary rather than buried here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from .config import DotDict, cpm_rates, load_config
from .models.bayesian_mmm import ModelData, PosteriorParams, media_contributions

logger = logging.getLogger(__name__)


@dataclass
class OptimisationResult:
    channels: list[str]
    current_spend: np.ndarray
    optimal_spend: np.ndarray
    total_budget: float
    predicted_sales_current: float
    predicted_sales_optimal: float
    success: bool
    message: str
    n_restarts_used: int
    binding_constraints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def uplift(self) -> float:
        return self.predicted_sales_optimal - self.predicted_sales_current

    @property
    def uplift_pct(self) -> float:
        if self.predicted_sales_current == 0:
            return float("nan")
        return 100.0 * self.uplift / self.predicted_sales_current

    def to_frame(self) -> pd.DataFrame:
        change = self.optimal_spend - self.current_spend
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(self.current_spend > 0, 100.0 * change / self.current_spend, np.nan)
        return pd.DataFrame(
            {
                "channel": self.channels,
                "current_spend": self.current_spend,
                "recommended_spend": self.optimal_spend,
                "change_absolute": change,
                "change_pct": pct,
                "current_share": self.current_spend / self.current_spend.sum(),
                "recommended_share": self.optimal_spend / self.optimal_spend.sum(),
            }
        )


def _multiplier_from_spend(spend: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Spend is translated into a multiplier on the observed impression pattern."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(current > 0, spend / current, 0.0)


def make_objective(
    params: PosteriorParams,
    data: ModelData,
    current_spend: np.ndarray,
    max_lag: int = 8,
) -> Callable[[np.ndarray], float]:
    """Return a function mapping a spend vector to expected total sales.

    Only the media block is evaluated. The baseline, trend, seasonality and controls do not
    depend on the decision variables, so they add a constant that would cancel out of every
    comparison. Leaving them out makes each evaluation cheaper and the objective easier to
    reason about, and the uplift figures are unaffected because they are differences.
    """

    def total_media_sales(spend: np.ndarray) -> float:
        multipliers = _multiplier_from_spend(np.asarray(spend, dtype=float), current_spend)
        X = data.X * multipliers[None, None, :]
        contrib = media_contributions(params, X, max_lag)
        return float(contrib.sum(axis=(1, 2, 3)).mean() * data.target_scale)

    return total_media_sales


def build_constraints(
    current_spend: np.ndarray,
    total_budget: float,
    min_pct: float,
    max_pct: float,
    max_realloc_pct: float,
) -> tuple[list[Any], list[tuple[float, float]]]:
    """Assemble the constraint set over the extended vector [spend, u]."""
    n = len(current_spend)

    # Budget equality: sum of spend equals the total, u unconstrained in this row.
    budget_row = np.concatenate([np.ones(n), np.zeros(n)])
    budget = LinearConstraint(budget_row, lb=total_budget, ub=total_budget)

    # u_c >= spend_c - current_c  becomes  u_c - spend_c >= -current_c
    a_upper = np.hstack([-np.eye(n), np.eye(n)])
    upper = LinearConstraint(a_upper, lb=-current_spend, ub=np.inf)

    # u_c >= current_c - spend_c  becomes  u_c + spend_c >= current_c
    a_lower = np.hstack([np.eye(n), np.eye(n)])
    lower = LinearConstraint(a_lower, lb=current_spend, ub=np.inf)

    # Total movement cap.
    move_row = np.concatenate([np.zeros(n), np.ones(n)])
    move = LinearConstraint(move_row, lb=-np.inf, ub=2.0 * max_realloc_pct * total_budget)

    bounds = [(float(min_pct * c), float(max_pct * c)) for c in current_spend]
    bounds += [(0.0, float(total_budget)) for _ in range(n)]

    return [budget, upper, lower, move], bounds


def optimise_budget(
    params: PosteriorParams,
    data: ModelData,
    current_spend: np.ndarray,
    cfg: DotDict | None = None,
    total_budget: float | None = None,
    max_lag: int = 8,
    seed: int = 42,
) -> OptimisationResult:
    """Solve the constrained reallocation with SLSQP and multiple restarts.

    The response surface is a sum of saturating curves, so it is concave in each channel
    individually, but the adstock normalisation and the joint constraint set mean I do not
    get to assume global concavity for free. Multiple random starts inside the feasible
    region are cheap insurance against a local optimum, and I report how many of them
    converged to the same value.
    """
    cfg = cfg or load_config()
    opt_cfg = cfg["optimizer"]
    channels = list(data.channels)
    current_spend = np.asarray(current_spend, dtype=float)
    n = len(channels)

    budget = float(total_budget if total_budget is not None else current_spend.sum())
    min_pct = float(opt_cfg["min_spend_pct_of_current"])
    max_pct = float(opt_cfg["max_spend_pct_of_current"])
    max_realloc = float(opt_cfg["max_total_reallocation_pct"])

    notes: list[str] = []
    if min_pct * current_spend.sum() > budget:
        raise ValueError(
            "The per channel floors already exceed the total budget. Lower "
            "min_spend_pct_of_current or raise the budget."
        )
    if max_pct * current_spend.sum() < budget:
        raise ValueError(
            "The per channel caps cannot absorb the total budget. Raise "
            "max_spend_pct_of_current."
        )

    objective = make_objective(params, data, current_spend, max_lag)
    constraints, bounds = build_constraints(
        current_spend, budget, min_pct, max_pct, max_realloc
    )

    def neg_objective(z: np.ndarray) -> float:
        return -objective(z[:n])

    rng = np.random.default_rng(seed)
    n_restarts = int(opt_cfg["n_restarts"])
    starts = [np.concatenate([current_spend, np.zeros(n)])]
    for _ in range(n_restarts - 1):
        # Random feasible start: perturb, clip to bounds, then rescale to hit the budget.
        perturbed = current_spend * rng.uniform(min_pct, max_pct, size=n)
        perturbed = np.clip(perturbed, min_pct * current_spend, max_pct * current_spend)
        total = perturbed.sum()
        perturbed = perturbed * (budget / total) if total > 0 else current_spend.copy()
        perturbed = np.clip(perturbed, min_pct * current_spend, max_pct * current_spend)
        u0 = np.abs(perturbed - current_spend)
        starts.append(np.concatenate([perturbed, u0]))

    best = None
    converged_values: list[float] = []
    for z0 in starts:
        res = minimize(
            neg_objective,
            z0,
            method=str(opt_cfg["method"]),
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-9},
        )
        if res.success:
            converged_values.append(-res.fun)
        if best is None or (res.fun < best.fun):
            best = res

    assert best is not None
    optimal = np.asarray(best.x[:n], dtype=float)

    # Renormalise against floating point drift on the equality constraint, then re-clip.
    drift = abs(optimal.sum() - budget) / budget if budget else 0.0
    if drift > 1e-6:
        optimal = optimal * (budget / optimal.sum())
        notes.append(f"Rescaled the solution to close a budget drift of {drift:.2e}.")

    binding = []
    tol = 1e-4
    for i, ch in enumerate(channels):
        if abs(optimal[i] - min_pct * current_spend[i]) < tol * max(current_spend[i], 1.0):
            binding.append(f"{ch} at its floor of {min_pct:.0%} of current spend")
        if abs(optimal[i] - max_pct * current_spend[i]) < tol * max(current_spend[i], 1.0):
            binding.append(f"{ch} at its cap of {max_pct:.0%} of current spend")
    moved = float(np.abs(optimal - current_spend).sum())
    if moved >= 2.0 * max_realloc * budget * (1 - 1e-4):
        binding.append(
            f"total reallocation cap of {max_realloc:.0%} of budget is binding"
        )
    if binding:
        notes.append(
            "Binding constraints mean the unconstrained optimum is more extreme than this "
            "recommendation. The gap between them is the cost of executability."
        )

    if converged_values:
        spread = float(np.max(converged_values) - np.min(converged_values))
        rel = spread / abs(np.mean(converged_values)) if np.mean(converged_values) else 0.0
        notes.append(
            f"{len(converged_values)} of {len(starts)} restarts converged, objective spread "
            f"{rel:.2e} in relative terms."
        )
        if rel > 1e-4:
            notes.append(
                "Restarts disagreed by more than a rounding error, so I treat the surface "
                "as multimodal here and report the best of the restarts."
            )

    return OptimisationResult(
        channels=channels,
        current_spend=current_spend,
        optimal_spend=optimal,
        total_budget=budget,
        predicted_sales_current=objective(current_spend),
        predicted_sales_optimal=objective(optimal),
        success=bool(best.success),
        message=str(best.message),
        n_restarts_used=len(starts),
        binding_constraints=binding,
        notes=notes,
    )


def naive_proportional_reallocation(
    roi: pd.DataFrame, current_spend: np.ndarray, channels: list[str]
) -> np.ndarray:
    """Reallocate in proportion to average ROI, which is the shortcut I want to beat.

    This is what a spreadsheet does. It ignores saturation entirely, so it pours budget
    into the channel with the best historical average return regardless of whether that
    channel still has headroom. I include it so the comparison in the write up is against
    the realistic alternative rather than against doing nothing.
    """
    roi_map = dict(zip(roi["channel"], roi["roi_mean"], strict=False))
    weights = np.array([max(roi_map.get(c, 0.0), 0.0) for c in channels], dtype=float)
    if weights.sum() <= 0:
        return current_spend.copy()
    return weights / weights.sum() * current_spend.sum()


def check_feasibility(
    spend: np.ndarray,
    current_spend: np.ndarray,
    total_budget: float,
    cfg: DotDict | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Test whether an arbitrary allocation satisfies the constraints.

    I added this after noticing that my first comparison table scored the naive
    proportional plan without applying any of the constraints the optimiser has to obey.
    That comparison flattered the naive approach: of course an unconstrained plan can beat
    a constrained one, it is solving an easier problem. Reporting it that way would have
    been wrong.
    """
    cfg = cfg or load_config()
    opt = cfg["optimizer"]
    min_pct = float(opt["min_spend_pct_of_current"])
    max_pct = float(opt["max_spend_pct_of_current"])
    max_realloc = float(opt["max_total_reallocation_pct"])

    violations: list[str] = []
    if abs(float(spend.sum()) - total_budget) > tol * max(total_budget, 1.0):
        violations.append("total budget not met")
    below = np.where(spend < min_pct * current_spend - tol * np.maximum(current_spend, 1.0))[0]
    above = np.where(spend > max_pct * current_spend + tol * np.maximum(current_spend, 1.0))[0]
    if below.size:
        violations.append(f"{below.size} channels below the per channel floor")
    if above.size:
        violations.append(f"{above.size} channels above the per channel cap")
    moved = float(np.abs(spend - current_spend).sum())
    cap = 2.0 * max_realloc * total_budget
    if moved > cap * (1 + 1e-4):
        violations.append(
            f"moves {moved / (2 * total_budget):.1%} of budget against a cap of {max_realloc:.0%}"
        )
    return {
        "feasible": len(violations) == 0,
        "violations": violations,
        "share_of_budget_moved": moved / (2.0 * total_budget) if total_budget else np.nan,
    }


def project_to_feasible(
    spend: np.ndarray,
    current_spend: np.ndarray,
    total_budget: float,
    cfg: DotDict | None = None,
    max_iter: int = 100,
) -> np.ndarray:
    """Pull an arbitrary allocation back into the feasible set.

    Alternating projection: clip to the per channel bounds, shrink toward current spend
    until the movement cap is satisfied, then rescale to the budget, and repeat. It is not
    the exact Euclidean projection onto the intersection, but it converges quickly here and
    it gives the naive plan the fairest possible treatment, which is what I want when the
    point of the comparison is that my method wins.
    """
    cfg = cfg or load_config()
    opt = cfg["optimizer"]
    lo = float(opt["min_spend_pct_of_current"]) * current_spend
    hi = float(opt["max_spend_pct_of_current"]) * current_spend
    cap = 2.0 * float(opt["max_total_reallocation_pct"]) * total_budget

    out = np.clip(np.asarray(spend, dtype=float), lo, hi)
    for _ in range(max_iter):
        moved = np.abs(out - current_spend).sum()
        if moved > cap and moved > 0:
            out = current_spend + (out - current_spend) * (cap / moved)
        out = np.clip(out, lo, hi)
        total = out.sum()
        if total > 0:
            out = out * (total_budget / total)
        out = np.clip(out, lo, hi)
        if (
            abs(out.sum() - total_budget) <= 1e-6 * max(total_budget, 1.0)
            and np.abs(out - current_spend).sum() <= cap * (1 + 1e-6)
        ):
            break
    return out


def optimise_with_uncertainty(
    params: PosteriorParams,
    data: ModelData,
    current_spend: np.ndarray,
    cfg: DotDict | None = None,
    n_draws: int | None = None,
    max_lag: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-solve the problem under individual posterior draws.

    A single optimal allocation computed at the posterior mean is a point estimate of a
    decision, and it hides how fragile that decision is. Solving the same problem under
    each of many draws gives a distribution over the recommended split, and the spread is
    the honest answer to "how confident are you in this reallocation".
    """
    cfg = cfg or load_config()
    n_draws = int(n_draws or cfg["optimizer"]["uncertainty"]["n_posterior_draws"])
    n_draws = min(n_draws, params.n_draws)
    rng = np.random.default_rng(seed)
    idx = rng.choice(params.n_draws, size=n_draws, replace=False)

    allocations = []
    uplifts = []
    for j, draw in enumerate(idx):
        single = PosteriorParams(
            decay=params.decay[[draw]],
            half_sat=params.half_sat[[draw]],
            slope=params.slope[[draw]],
            beta=params.beta[[draw]],
            alpha=params.alpha[[draw]],
            trend_coef=params.trend_coef[[draw]],
            seas_coef=params.seas_coef[[draw]],
            control_coef=params.control_coef[[draw]],
            sigma=params.sigma[[draw]],
        )
        # One restart per draw keeps this tractable. The multi restart check above already
        # established whether the surface is well behaved for this data.
        local_cfg = DotDict({**cfg, "optimizer": {**cfg["optimizer"], "n_restarts": 1}})
        try:
            res = optimise_budget(single, data, current_spend, local_cfg, max_lag=max_lag, seed=seed + j)
        except ValueError:
            continue
        allocations.append(res.optimal_spend)
        uplifts.append(res.uplift_pct)

    if not allocations:
        raise RuntimeError("No posterior draw produced a feasible solution.")

    alloc = np.vstack(allocations)
    share = alloc / alloc.sum(axis=1, keepdims=True)
    current_share = current_spend / current_spend.sum()

    summary = pd.DataFrame(
        {
            "channel": data.channels,
            "current_share": current_share,
            "recommended_share_mean": share.mean(axis=0),
            "recommended_share_p05": np.percentile(share, 5, axis=0),
            "recommended_share_p95": np.percentile(share, 95, axis=0),
            "prob_increase": (alloc > current_spend[None, :]).mean(axis=0),
        }
    )
    uplift_summary = pd.DataFrame(
        {
            "metric": ["uplift_pct"],
            "mean": [float(np.mean(uplifts))],
            "p05": [float(np.percentile(uplifts, 5))],
            "p50": [float(np.percentile(uplifts, 50))],
            "p95": [float(np.percentile(uplifts, 95))],
            "prob_positive": [float(np.mean(np.asarray(uplifts) > 0))],
            "n_draws": [len(uplifts)],
        }
    )
    return summary, uplift_summary


def cpm_sensitivity(
    params: PosteriorParams,
    data: ModelData,
    df: pd.DataFrame,
    cfg: DotDict | None = None,
    shocks: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
    max_lag: int = 8,
) -> pd.DataFrame:
    """Shock each channel's assumed CPM and re-solve, to see what survives.

    Since the CPM figures are assumptions rather than measurements, the useful question is
    not "what is the optimal split" but "which parts of the recommended split hold up when
    my cost assumption is wrong by a factor of two".
    """
    cfg = cfg or load_config()
    base_rates = cpm_rates(cfg)
    channels = list(data.channels)
    rows = []

    for channel in channels:
        for shock in shocks:
            rates = dict(base_rates)
            rates[channel] = base_rates[channel] * shock
            spend = np.array(
                [float(df[c].sum()) / 1000.0 * rates.get(c, 0.0) for c in channels]
            )
            try:
                res = optimise_budget(params, data, spend, cfg, max_lag=max_lag)
            except ValueError as exc:
                rows.append(
                    {
                        "shocked_channel": channel,
                        "cpm_multiplier": shock,
                        "status": f"infeasible: {exc}",
                    }
                )
                continue
            frame = res.to_frame().set_index("channel")
            row = {
                "shocked_channel": channel,
                "cpm_multiplier": shock,
                "status": "ok",
                "uplift_pct": res.uplift_pct,
            }
            for c in channels:
                row[f"{c}_recommended_share"] = float(frame.loc[c, "recommended_share"])
                row[f"{c}_direction"] = (
                    "increase" if frame.loc[c, "change_absolute"] > 0 else "decrease"
                )
            rows.append(row)
    return pd.DataFrame(rows)
