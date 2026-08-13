# Limitations

I would rather write these down than be caught out by them. Each section states the problem,
what I did about it, and what would actually fix it.

## 1. The data is observational

**The problem.** Media budgets were not randomly assigned. Planners set them, and planners
already expect certain weeks and certain divisions to sell well. So spend and sales are
correlated partly because spend causes sales and partly because expected sales cause spend.
A regression that ignores this attributes the planner's foresight to the media.

This is the fundamental criticism of marketing mix modelling and no amount of Bayesian
machinery removes it.

**What I did.** Three checks that would fail if the estimates were mostly reverse causality.
A geo holdout, where the model is fitted without a set of divisions and then predicts them
using only the pooled channel parameters. A difference in differences around media step
changes, which estimates an effect without relying on the marketing mix functional form at
all. A parallel trends test on pre period data, which can fail, and a placebo distribution
over randomly assigned fake treatments, which gives a p value that does not depend on the
regression standard errors being right.

**What would fix it.** A geo lift experiment. Hold media dark in matched divisions for six
to eight weeks. Nothing in this repository substitutes for that.

## 2. Two years is short for annual seasonality

**The problem.** 113 weeks is roughly two observations of any annual seasonal pattern. The
model has to separate an annual cycle from slow moving media effects and from the trend using
two passes through the calendar. That is not much.

**What I did.** Three Fourier harmonics rather than 52 weekly dummies. Weekly dummies would
fit each week's idiosyncrasy perfectly on two observations and generalise to nothing. Three
harmonics constrain seasonality to a smooth annual shape, which is a real restriction and the
right one at this sample size. Adstock decay is bounded away from 1 for the same reason: an
unbounded decay approaching 1 becomes indistinguishable from a random walk and would absorb
the trend.

**What would fix it.** More history. Four to five years is where seasonality separates
cleanly.

## 3. All spend figures are assumed

Covered in full in `cpm_assumptions.md`. The short version: the data has impressions, not
money, so every currency figure inherits an assumption. Relative CPMs across channels drive
the recommendation; absolute levels do not. The sensitivity sweep reports which conclusions
survive those assumptions being wrong by a factor of two.

## 4. Missing variables

**The problem.** The dataset has no price, no promotion, no distribution, no competitor
activity, no macro variable beyond the CPI series I joined on. Any of these that moves with
media will have its effect absorbed into the media coefficients or the baseline.

Promotion is the one that worries me most. Retailers usually run media and promotions
together, so the media terms are at risk of picking up promotional lift. If that is happening
here, media effects are overstated and the optimiser is too optimistic.

**What I did.** Nothing that fixes it, because the variables are not in the data. I joined
holidays, category search interest and CPI, which covers some of the demand side. The bad
control diagnostic in `enrichment.py` at least makes the risk visible for the controls I do
have.

**What would fix it.** A promotional calendar joined by week, and competitor share of voice
if it can be bought.

## 5. The Google Trends control is a bad control

**The problem.** Search interest for a product category is partly *caused* by the advertising
being measured. Controlling for a downstream consequence of the treatment biases the treatment
effect toward zero. This is a textbook bad control, and it appears in a great many marketing
mix implementations without comment.

**What I did.** Fit with and without it and report both, rather than including it silently.
The correlation between each control and each channel is written to
`reports/tables/enrichment_collinearity.csv` so the size of the risk is visible.

## 6. The optimiser holds flighting fixed

**The problem.** The optimiser scales each channel's existing weekly and per division pattern
by a single multiplier. It answers "how much should each channel get", not "when should it be
spent" or "which divisions should get more".

Given the model estimates saturation curves per channel and effects per division, there is
real value being left on the table. Concentrating spend into fewer weeks moves a channel
further up its saturation curve and gets less per impression; spreading it out does the
opposite. The model knows this and the optimiser does not use it.

**What I did.** Scoped it out and said so. Adding time and division to the decision variables
turns 5 unknowns into 5 times 113 times 26, which needs a different formulation and a
different solver.

**What would fix it.** A two stage approach: solve the channel split as here, then solve the
weekly schedule within each channel against its own saturation curve.

## 7. Additive model in level space

**The problem.** The model is additive in sales levels. A multiplicative or log linear
specification is often more natural for retail sales, where effects tend to be proportional,
and it would guarantee non negative predictions.

**What I did.** Kept the additive form because contribution decomposition is unambiguous in
it. In a log linear model, "how much did search contribute" has no unique answer, because
the contributions multiply rather than sum, and every allocation of the interaction is a
convention. Since the entire budget conversation depends on a contribution decomposition,
that unambiguity is worth the cost. This is a defensible choice rather than an obviously
correct one, and I would expect to be challenged on it.

## 8. Convergence, and the honest reporting of it

**The problem.** Hierarchical models with saturating transforms are hard to sample. Divergent
transitions and poor mixing are not edge cases here, they are the expected failure mode.

**What I did.** Non centred parameterisation throughout, bounded transform priors, and
diagnostics written to `reports/tables/model_diagnostics.csv` on every run. The Airflow fit
task refuses to publish a model with divergences above a threshold or maximum r hat above
1.05. `scripts/run_phase1.py` warns loudly rather than continuing quietly.

Anyone reading results from this repository should check the divergence count and r hat
before quoting anything. If either is bad, the estimates are not usable, however reasonable
the tables look.

## 9. The generalisation check is a software claim, not a scientific one

The secondary dataset has 300 rows and six channels. The hierarchy has nothing to pool over
and the saturation parameters are weakly identified. What that run demonstrates is that the
pipeline handles a different channel mix without special casing. It does not demonstrate that
the estimates from it mean anything, and the code says so in the output it emits. If the
intervals from that run came out narrow, I would treat it as evidence that my priors are
doing too much work, not as a strong result.

## 10. The benchmark dataset is simulated

Robyn's `dt_simulated_weekly` has a known data generating process, which is exactly why it is
useful as a correctness check on the adstock and saturation implementations. It is also why
it cannot validate any causal claim. It tests that the machinery works, not that the
machinery measures causation on real data.

## 11. The stand in dataset

If `scripts/make_replica_dataset.py` was used instead of the Kaggle download, the numbers in
`reports/` come from data I generated from a process I chose. The model will recover the
parameters I put in, which is a genuine test of the code and worthless as evidence about
advertising. `data/raw/REPLICA_NOTICE.txt` and `data/raw/replica_ground_truth.json` mark this
clearly, and the parameter recovery table exists precisely because that is the only thing
simulated data can legitimately be used to check.
