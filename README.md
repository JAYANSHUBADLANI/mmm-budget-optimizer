# Bayesian marketing mix model with a constrained budget optimiser

[![tests](https://github.com/JAYANSHUBADLANI/mmm-budget-optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/mmm-budget-optimizer/actions/workflows/tests.yml)

I built this to answer a question a media team actually has to answer every planning cycle:
given a fixed budget and five channels with different saturation curves, where should the
next dollar go, and how confident should anyone be in the answer.

The project fits a hierarchical Bayesian marketing mix model across 26 sales divisions and
113 weeks, with adstock carryover and Hill saturation estimated per channel rather than
assumed. It then solves a constrained reallocation problem with `scipy.optimize`, using
constraints a planner would recognise: per channel spend floors, per channel caps, and a
limit on how much of the total budget may move at once.

Alongside the model I run three checks that most portfolio marketing mix projects skip, and
that I would expect to be asked about:

- a linear regression marketing mix model fitted through the same harness, so the claim
  that the Bayesian approach reports more honest uncertainty is demonstrated rather than
  asserted
- a rolling origin backtest against a seasonal naive baseline, because in sample fit on a
  model with this much flexibility is close to meaningless
- a causal validation layer, geo holdout plus difference in differences with a parallel
  trends test and a placebo distribution, because the standard criticism of marketing mix
  modelling is that budgets are not randomly assigned

## Status

The code is complete across all four phases and the test suite passes. The model has not
yet been fitted on the real Kaggle data, so **this README quotes no results**. Every
results table in `reports/` is produced by running the pipeline yourself. I would rather
ship an empty results section than numbers I have not verified. `PROGRESS.md` lists exactly
what is done, what is pending, and what needs a decision.

## Quick start

```bash
pip install -r requirements.txt

# Option A, the real data. Needs a Kaggle API token at ~/.kaggle/kaggle.json
python scripts/fetch_data.py

# Option B, no credentials. Generates a schema faithful stand in with the same shape,
# date range and the same Division Z bug, so everything downstream runs immediately.
python scripts/make_replica_dataset.py

# Fast end to end wiring check, a few minutes
python scripts/run_phase1.py --smoke --no-backtest
python scripts/run_phase3.py --no-sensitivity

# Real run, expect tens of minutes for the sampling
python scripts/run_phase1.py
python scripts/run_phase2.py
python scripts/run_phase3.py

streamlit run app/streamlit_app.py
```

Everything also runs in Docker:

```bash
docker compose up app                    # dashboard
docker compose --profile pipeline up     # adds Airflow, dbt and Postgres
```

## The data

### Primary, and a real bug in it

The primary dataset is `yugagrawal95/sample-media-spends-data` from Kaggle: 3051 rows, 26
divisions labelled A to Z, 113 weeks running from 6 January 2018 to 29 February 2020, no
missing values.

I found two problems in it before modelling anything.

**Division Z appears twice.** Division Z ships 226 rows where every other division ships
113. The extra 113 rows are exact duplicates of the first 113: same weeks, same impressions,
same sales. This matters because the model pools information across divisions, so a division
that appears twice contributes twice as many likelihood terms and gets roughly double the
influence on the pooled channel coefficients and the group level hyperparameters. It also
breaks the balanced panel that the adstock lag operator depends on, because inside Division
Z the week sequence restarts partway through the frame.

I did not simply call `drop_duplicates` and move on. `src/mmm/cleaning.py` first checks
whether the duplicated keys carry identical measure values. If two rows share
`(Division, Calendar_Week)` but disagree on sales, that is a source conflict rather than a
duplicate, and the function refuses to pick a winner automatically. Only exact repeats are
removed. `tests/test_cleaning.py` locks in both behaviours, and the dbt test
`assert_no_duplicate_division_weeks.sql` catches the same class of problem in SQL.

**`Overall_Views` is a derived column.** It equals `Paid_Views + Organic_Views` exactly.
Leaving it in the design matrix would put a perfect linear combination of two existing
regressors alongside them, which makes the media block rank deficient. In the linear model
that shows up as unstable coefficients and inflated variance inflation factors; in the
Bayesian model it shows up as a ridge in the posterior that the sampler wanders along. The
pipeline tests the identity on the actual data rather than assuming it, and only drops the
column when it holds.

### The columns are impressions, not money

This is the single most important caveat in the project. The dataset contains impressions
and views. It contains no spend. Every currency figure downstream rests on an assumed cost
per thousand impressions per channel, set in `config/config.yaml` and justified in
[docs/cpm_assumptions.md](docs/cpm_assumptions.md).

What the assumption does not touch: the shape of the saturation curves, the ranking of
channels by incremental effect per impression, and the model fit. Those come out of the
data.

What it does touch: every ROI figure expressed per dollar, and therefore the optimiser's
recommendation. The *relative* CPMs across channels are what drive the reallocation, so
those carry the weight, not the absolute levels. `scripts/run_phase3.py` reruns the whole
optimisation across a range of CPM values and reports which parts of the recommendation
survive a factor of two error in either direction.

The pipeline also checks whether the implied total media spend is a plausible share of
revenue. Real advertisers sit somewhere between roughly 2 and 25 percent depending on
sector. A figure far outside that band means the CPM assumptions are wrong by an order of
magnitude, not that the business has found free advertising, and the run warns accordingly.

### Enrichment, joined by week

Three external series are joined on the week key and only on the week key: a holiday
calendar from the `holidays` package, category search interest from Google Trends via
`pytrends`, and CPI from FRED. These are national weekly series, so they broadcast to all 26
divisions within a week, which is correct for a public holiday.

One of them is a *bad control* in the technical sense. Search interest for a product
category is partly caused by the advertising being measured, so controlling for it absorbs
part of the media effect and biases the channel coefficients toward zero. I fit with and
without it and report the difference rather than including it silently.
`src/mmm/enrichment.py` also reports the correlation between every control and every channel
so the risk is visible rather than assumed away.

### Secondary and benchmark datasets, kept separate

Two further datasets are used, and neither is ever merged with the primary panel. Stacking
datasets with different products, channels, currencies and time ranges produces a wider
table that describes no real process, and any model fitted to it estimates a relationship
that does not exist.

**Generalisation check.** `singhnavjot2062001/product-advertising-data`, 300 rows, six
channels including TV, billboards and influencer marketing. Running the same pipeline on a
different channel mix tests whether the method travels or whether it was quietly tuned to
one file. The honest claim from 300 rows is narrow: the code runs end to end without special
casing, and the estimates are not absurd. That is a software generalisation claim, not a
scientific one, and `src/mmm/generalization.py` says so in the results it emits.

**Correctness check.** Meta Robyn's `dt_simulated_weekly`, read directly from `.RData` with
`pyreadr`, so no R installation is needed. This is deliberately not the centrepiece: it is
the most reproduced marketing mix dataset online and using it as the main analysis would be
unoriginal. It is simulated from a known process, which makes it the right place to test
whether my adstock, saturation and decomposition code recovers sensible effects. It cannot
validate any causal claim, because none of it is real.

## The model

```
sales[t, d] = base[d] + trend + seasonality
            + sum over channels of beta[c, d] * Hill(Adstock(media[t, d, c]))
            + controls + noise
```

**Pooling.** Every division gets its own coefficient per channel, drawn from a shared
distribution for that channel. Divisions with thin media variation are pulled toward the
channel mean; divisions with strong signal stay near their own data. This is why I did not
fit 26 separate models, which throws away the shared structure, and why I did not collapse
to one aggregate time series, which throws away the heterogeneity. The spread in
`reports/tables/eda_division_profile.csv` is the empirical case for the hierarchy.

**Transform parameters are estimated, not assumed.** Adstock decay and Hill half saturation
and slope are learned at the channel level. Fixing them in advance, which is what the grid
searched linear model has to do, means assuming an answer to one of the questions the model
exists to answer.

**Positivity.** Channel coefficients are lognormal, so media effects cannot come out
negative. A negative media coefficient is almost never a real finding, it is a symptom of
collinearity, and ruling it out on substantive grounds is more defensible than reporting it
and explaining it away.

**Non centred parameterisation.** The hierarchical offsets are sampled as standard normals
and rescaled. The centred version funnels badly when a group variance is small, which is
precisely the situation in the low variation divisions.

The NumPy and PyTensor implementations of adstock and Hill are tested against each other in
`tests/test_features.py`. The model is fitted with the PyTensor version and the optimiser
searches over the NumPy version, so a silent divergence between them would mean the
optimiser is maximising a function that was never fitted, and nothing else would catch it.

## Evaluation

The headline number is holdout MAPE from a rolling origin backtest, not in sample fit. Two
details that are easy to get wrong and that I handled explicitly:

- the split is always by time, never by row, because shuffling leaks future weeks into
  training through the adstock lag and the shared seasonality basis
- adstock is applied over the full media series and then sliced, so carryover flows from the
  training weeks into the holdout weeks as it would in production; transforming the holdout
  in isolation starts it with zero carryover and understates the error

Every fold refits from scratch. Reusing one fit and re-slicing would let holdout weeks
influence the posterior through the hierarchy.

The comparison set includes a seasonal naive baseline. A model that cannot beat "same week
last year" has not earned its complexity, and a MAPE quoted with nothing to compare it
against is the most common way a portfolio project overstates itself.

## The optimiser

```
maximise    sum over channels of contribution(spend)
subject to  sum of spend = total budget
            floor <= spend per channel <= cap
            sum of |spend - current| <= 2 * R * total budget
```

The third constraint is what makes this usable. Without it the optimiser proposes cutting a
channel by 90 percent and tripling another, which no team can execute in a quarter. The
absolute value is not differentiable, so rather than smoothing it I use the exact linear
reformulation with auxiliary variables, which SLSQP handles properly.

Three things the optimiser reports that a single recommended split would not:

- **binding constraints**, so the gap between the recommendation and the unconstrained
  optimum is visible; that gap is the cost of executability
- **uncertainty on the decision**, obtained by re-solving the same problem under individual
  posterior draws, which gives an interval on the recommended split rather than a point
- **the naive alternative**, reallocating in proportion to average ROI, which is what a
  spreadsheet does; it ignores saturation and pours budget into the best historical average
  performer regardless of headroom

On that last point, one correction I made to my own work is worth flagging, because it is
the kind of mistake that is easy to ship and hard to defend. My first version scored the
naive plan without applying any of the constraints the optimiser has to respect, and the
naive plan sometimes came out ahead. It was not better, it was solving an easier problem.
The comparison table now reports the naive plan three ways: unconstrained with a feasibility
flag, projected back onto the feasible set, and the constrained optimum, so the only
comparison a reader can draw is like for like. `tests/test_optimizer.py` locks that in.

Reallocation follows *marginal* ROI, the value of the next dollar, not average ROI. On a
saturating curve the two diverge, and following the average is a well known way to produce a
reallocation that looks good in a deck and underdelivers when executed.

What the optimiser does not do: it scales each channel's existing weekly and per division
pattern by a single multiplier. It reallocates between channels, it does not reschedule
flighting within a channel. That is a genuinely larger problem and I did not solve it.

## Engineering layer

Plumbing, not the contribution. It exists to show the modelling runs as a scheduled job
rather than a notebook.

- **dbt** builds Bronze, Silver and Gold marts on DuckDB. Bronze deliberately keeps the
  Division Z duplicates so Silver can be seen to remove them and so the tests have something
  to fail on. A bronze layer that quietly fixes problems destroys the audit trail that makes
  the layered design worth having. `silver_data_quality_log` records what was removed per
  division.
- **Airflow** orchestrates fetch, load, dbt run, dbt test, validate, fit, optimise, publish.
  `dbt test` is a hard gate: if the duplicate check fails the DAG stops before fitting. The
  fit task also refuses to publish a model with divergences or unmixed chains.
- **Docker Compose** runs the dashboard alone by default, with the full Airflow stack behind
  a profile, because standing up a scheduler to look at a chart is friction with no payoff.

## Repository layout

```
config/           single YAML holding every assumption and hyperparameter
src/mmm/          library code
  cleaning.py       the Division Z fix, with diagnosis separated from repair
  validation.py     data quality checks that block the pipeline
  features.py       adstock and Hill, in NumPy and PyTensor, tested against each other
  models/           hierarchical Bayesian model and the linear comparison
  backtest.py       rolling origin splits, metrics, seasonal naive baseline
  causal.py         geo holdout, difference in differences, parallel trends, placebo
  roi.py            CPM translation, contributions, average and marginal ROI
  optimizer.py      constrained reallocation with uncertainty and CPM sensitivity
  enrichment.py     holidays, Google Trends, FRED, with bad control diagnostics
scripts/          fetch_data, make_replica_dataset, run_phase1 to run_phase3
app/              Streamlit scenario dashboard
dbt/              Bronze, Silver, Gold models and data quality tests
airflow/dags/     the weekly refresh DAG
tests/            pytest suite
docs/             executive summary, methodology, CPM assumptions, limitations
```

## Reading order for a reviewer

1. [docs/executive_summary.md](docs/executive_summary.md), one page, problem to
   recommendation
2. [docs/cpm_assumptions.md](docs/cpm_assumptions.md), the assumption everything rests on
3. [docs/limitations.md](docs/limitations.md), what I would not claim from this
4. `src/mmm/cleaning.py` and `src/mmm/optimizer.py`, where the reasoning is densest

## Honest limitations

The full version is in [docs/limitations.md](docs/limitations.md). The short form:

- The data is observational. Media budgets were not randomly assigned. The causal checks
  constrain how wrong the estimates can be; a geo lift test would settle what they cannot.
- Two years of weekly data gives roughly two observations of any annual seasonal pattern.
  Seasonality and slow moving media effects are hard to separate over that span.
- All spend figures are assumed. See above.
- The optimiser holds flighting fixed.
- The dataset has no price, promotion, distribution or competitor variables. Anything they
  drive will be absorbed by the media terms or the baseline.

## Licence

MIT. See [LICENSE](LICENSE).
