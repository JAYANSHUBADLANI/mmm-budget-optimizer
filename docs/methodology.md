# Methodology

## Model specification

For division `d` in week `t`:

```
sales[t, d] = alpha[d]
            + trend_coef * trend[t]
            + sum over k of seas_coef[k] * fourier[t, k]
            + sum over c of beta[c, d] * Hill(Adstock(media[t, d, c]; decay[c]); half_sat[c], slope[c])
            + sum over j of control_coef[j] * control[t, d, j]
            + epsilon[t, d]
```

with `epsilon ~ Normal(0, sigma)`. Sales and media are scaled to comparable ranges before
fitting, and the scaling is inverted when contributions are reported.

## Transforms

**Adstock.** Geometric with a finite lag window of eight weeks:

```
adstocked[t] = sum over l of w[l] * media[t - l],   w[l] proportional to decay ** l
```

Weights are normalised to sum to one. This matters more than it looks. Without
normalisation the transform's output scale changes as decay moves, so decay and the channel
coefficient trade off against each other along a ridge and the sampler explores that ridge
instead of converging. With normalisation the transform is a weighted average, its scale is
stable, and the two parameters are much better identified.

The lag is applied within division and never across the boundary between divisions. Applying
a lag to a stacked panel without grouping means the first weeks of one division inherit
carryover from the last weeks of another, which produces a model that looks fine and is
wrong.

**Saturation.** Hill curve:

```
Hill(x) = x ** slope / (half_sat ** slope + x ** slope)
```

At `x = half_sat` the response is exactly 0.5 regardless of slope, so `half_sat` reads as
"the volume at which this channel has delivered half of everything it can deliver". Slope
above 1 gives an S shape with a slow start; below 1 the channel saturates almost immediately.

## Priors and why each one

| Parameter | Prior | Reasoning |
| --- | --- | --- |
| `decay_raw` | `Beta(2, 2)`, rescaled to [0, 0.9] | Weakly informative, keeps mass away from 0 and 1. The upper bound stops carryover becoming an unidentified random walk. |
| `half_sat` | `Beta(2, 2)` on scaled media in [0, 1] | Says I expect the bend in the response curve inside the observed spend range. Without that the curve is not identifiable at all: data that only covers the linear part of a saturating curve cannot locate the bend. |
| `slope` | `TruncatedNormal(1, 0.5)` on [0.5, 3] | Centred on the plain Hill case, bounded away from extremes that make the curve a step function. |
| `mu_beta` | `Normal(-2, 1)` on the log scale | Prior mean channel effect of about 0.14 on scaled sales, with wide spread. |
| `sigma_beta` | `HalfNormal(0.5)` | Allows real between division variation without letting one division run away. |
| `beta` | `exp(mu_beta + sigma_beta * z)`, `z ~ Normal(0, 1)` | Lognormal, so media effects are strictly positive. Non centred, so a small between division variance does not create a funnel. |
| `sigma` | `HalfNormal(0.1)` | On scaled sales. |

**On the positivity constraint.** Forcing media effects positive is a substantive
assumption, not a convenience. The justification is that a negative media coefficient is
almost never a real finding. It is what collinearity looks like when the estimator has no
prior information to fall back on. The linear comparison model has no such constraint, which
is exactly why its coefficients are worth looking at: if it produces negative media
coefficients on the same data, that is direct evidence for the point.

## Estimation

NUTS via PyMC, four chains, 1000 tuning and 1000 draws, target acceptance 0.9. A smoke
configuration with a handful of draws exists for wiring checks and is labelled as unusable
for results everywhere it appears.

Convergence is assessed on r hat, effective sample size and divergent transitions, all
written to `reports/tables/model_diagnostics.csv`. The Airflow task refuses to publish a fit
with divergences above one percent of total draws or maximum r hat above 1.05.

## Evaluation protocol

Rolling origin, expanding window. Three folds, 13 week holdout, stepping back four weeks per
fold. Expanding rather than sliding because a real planning cycle refits on everything
available rather than discarding old weeks.

Every fold refits from scratch. Reusing one fit and re-slicing would let holdout weeks
influence the posterior through the shared seasonality basis and the hierarchy.

Adstock is applied over the full media series and then sliced, so carryover flows from
training into holdout as it would in production.

Metrics: MAPE as the headline, plus SMAPE, RMSE and R squared. The comparison set is the
linear marketing mix model through the identical harness, plus a seasonal naive baseline
predicting the same week last year.

## Contribution and ROI

Channel contribution is `beta[c, d] * Hill(Adstock(media))` summed over weeks and divisions,
rescaled to original sales units, computed per posterior draw so it carries an interval.

Average ROI is contribution divided by assumed spend. Marginal ROI is the derivative,
approximated by bumping a channel by one percent and dividing the change in contribution by
the change in spend.

The distinction matters for the recommendation. Average ROI includes the cheap early
impressions on the steep part of the curve. Marginal ROI is what the next dollar buys at the
current operating point, which on a saturating curve is a much smaller number. Reallocating
on average ROI is a well documented way to produce a plan that fails on execution.

## Optimisation

```
maximise    sum over c of contribution_c(spend_c)
subject to  sum of spend_c = B
            floor_c <= spend_c <= cap_c
            sum of |spend_c - current_c| <= 2 * R * B
```

The factor of two in the movement constraint is because under a fixed total, every dollar
leaving one channel arrives at another, so the sum of absolute deviations counts each moved
dollar twice.

The absolute value is not differentiable. Rather than smoothing it, I use the exact linear
reformulation: introduce `u_c` with `u_c >= spend_c - current_c` and
`u_c >= current_c - spend_c`, then constrain `sum(u_c)`. The decision vector is twice as long
and every constraint stays linear, which is what SLSQP wants.

Twelve random restarts from feasible points. The objective is a sum of concave functions in
each channel individually, but the adstock normalisation and the joint constraint set mean
global concavity is not free, so the restarts are cheap insurance and the spread of converged
objective values is reported.

Uncertainty on the decision comes from re-solving under individual posterior draws, which
gives a distribution over the recommended split rather than a single number.

## Causal validation

**Geo holdout.** Fit without a set of divisions chosen by media variability rank, not by
hand. Predict them using the population level channel parameters, `exp(mu_beta + sigma_beta^2 / 2)`,
which is the mean of the lognormal the division effects are drawn from and what a planner
would have to use for a region with no history. Compare against a media free baseline on the
same divisions. If the media terms carry real information they beat it.

**Difference in differences.** Detect step changes in a channel exceeding 1.5 division level
standard deviations between a 12 week pre window and a 12 week post window. Two way fixed
effects on log sales with division and week dummies, standard errors clustered at division
level because sales within a division are serially correlated. The estimator is not run below
a configured minimum number of treated units, because a difference in differences on two
units is an anecdote.

This estimates the effect of whatever happened at the shock, which includes the media change
and anything correlated with it. It is not the same estimand as the model's channel
coefficient and I would not expect the two to match numerically. What I want is a sign and an
order of magnitude that do not contradict the model.

**Parallel trends.** Regress log sales on treated interacted with a linear pre period trend,
using pre shock weeks only. A coefficient indistinguishable from zero is consistent with the
identifying assumption. A large one means the treated divisions were already diverging.

**Placebo.** Fifty random reassignments of treatment status and timing. The real estimate
should sit in the tail of the placebo distribution. This gives a randomisation inference p
value that does not depend on the regression standard errors being right.
