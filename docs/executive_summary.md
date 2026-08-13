# Executive summary

**Prepared for:** the marketing director of a mid sized multi division retailer with a fixed
annual media budget across five channels and no experimental measurement in place.

**Prepared by:** analytics.

---

## The problem

The business spends a fixed annual media budget across search, paid social, email, affiliate
and video, split across 26 sales divisions. The current split is inherited. It reflects
last year's split, adjusted by negotiation, and nobody in the room can say what would happen
if a fifth of it moved.

Three specific questions need answering.

1. What is each channel actually contributing to sales, separately from seasonality, trend
   and the divisions simply being different sizes.
2. Which channels still have headroom, and which are saturated so that the next dollar buys
   materially less than the last one.
3. If the budget is reallocated, how much gain is credible, and how much confidence should
   sit behind that number before anyone commits to it.

## The approach

I fitted a hierarchical Bayesian marketing mix model across all 26 divisions and 113 weeks,
with two features standard regression does not have.

**Carryover.** An impression served this week still works next week. The model estimates a
decay rate per channel rather than assuming one, and reports it as a half life in weeks,
which is the form a planner can act on.

**Diminishing returns.** Each channel gets a saturation curve, so the model distinguishes
between a channel that is underinvested and one that has run out of room. This is the
difference between a reallocation that works and one that does not.

**Pooling across divisions.** Each division gets its own channel effects, drawn from a
shared distribution. Divisions with limited media variation borrow strength from the rest
instead of producing an unstable estimate on their own data.

Three checks sit alongside the model, because the credibility of the recommendation depends
on them more than on the model's sophistication.

- A conventional linear marketing mix model, fitted through the same harness. It reports
  much narrower uncertainty, and the comparison table quantifies exactly how much of that
  confidence is real and how much comes from assuming the carryover and saturation
  parameters were known in advance.
- A rolling origin backtest with a seasonal naive baseline. The question answered is "had
  this been fitted a quarter ago, how wrong would it have been", not "how well does it
  describe the past".
- A causal validation layer: divisions held out entirely, plus a difference in differences
  analysis around media step changes, with a parallel trends test and a placebo distribution.

## The recommendation

*Results are produced by running the pipeline. See `reports/tables/optimal_allocation.csv`,
`reports/tables/uplift_uncertainty.csv` and `reports/tables/cpm_sensitivity.csv` after
`scripts/run_phase3.py`. I have deliberately left no numbers here that I have not verified.*

The recommendation takes the form of a reallocation under three constraints, chosen so the
plan is executable rather than theoretically optimal:

- no channel falls below 60 percent of its current spend, so no capability is dismantled
- no channel rises above 160 percent, reflecting inventory and team capacity limits
- at most 25 percent of the total budget moves in one cycle

The optimiser reports which of these constraints bind. Where one binds, the unconstrained
optimum is more aggressive than the recommendation, and the gap is the measurable price of
executability. That gap is the input to a separate conversation about whether the constraint
is real.

The recommendation follows **marginal** return, the value of the next dollar, rather than
average return across everything spent to date. On a saturating curve these diverge sharply,
and reallocating on average return is the most common way a media mix exercise produces a
plan that looks strong on a slide and underdelivers in market.

## Expected impact, and how to read it

The uplift figure is reported as an interval, not a point. It comes from re-solving the
optimisation under many draws from the model's posterior, so it reflects genuine uncertainty
about the channel effects rather than a single best guess dressed up as a forecast.

Three things it does not reflect, which I would say out loud in the room.

**Cost assumptions.** The source data records impressions, not spend. Every currency figure
rests on assumed cost per thousand rates documented in `docs/cpm_assumptions.md`. The
sensitivity analysis reports which parts of the recommendation survive those assumptions
being wrong by a factor of two. Only the parts that survive should be acted on before the
real rate card is checked.

**Observational data.** Budgets were not randomly assigned. If planners historically spent
more where they already expected strong sales, some of what the model attributes to media
belongs to that foresight. The validation layer constrains how large that problem can be,
but it cannot eliminate it.

**Held constant.** The optimiser moves budget between channels while holding the weekly and
per division flighting pattern fixed. Timing and regional weighting are separate levers and
are not optimised here.

## What I would do next

**Run a geo holdout experiment.** The single highest value follow up. Hold media dark in a
matched set of divisions for six to eight weeks and measure the difference. That converts
the strongest assumption in this analysis into a measurement, and it makes every future
refresh of this model more credible.

**Get actual spend data.** Replacing assumed CPM rates with the real rate card removes the
largest single source of error in the currency figures, and requires no modelling change at
all.

**Phase the reallocation.** Move a quarter of the recommended shift, hold for a quarter,
measure, then decide on the rest. The model is a decision aid, not a guarantee, and staging
the change keeps the downside small while the evidence accumulates.
