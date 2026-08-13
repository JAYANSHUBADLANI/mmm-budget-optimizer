# Cost assumptions

## The problem

The primary dataset contains impressions and views. It contains no spend. To say anything
about budget I have to attach a cost to each impression, and nothing in the source data
tells me what that cost was.

So I state the assumption openly rather than burying it. Every currency figure in this
project is derived from the table below. If these rates are wrong, every ROI number is wrong
by the same factor, and I would rather a reader know that on page one than work it out for
themselves halfway through.

## The rates I assumed

All figures are cost per thousand impressions, in the same currency as the `Sales` column.

| Channel | Assumed CPM | How I arrived at it |
| --- | --- | --- |
| `Google_Impressions` | 52.50 | Search is bought on clicks, not impressions. At an assumed cost per click of 1.50 and a click through rate of 3.5 percent, the cost per impression is 0.0525, so 52.50 per thousand. The high figure relative to the others is real, not an error: a search impression is far more qualified than a display impression. |
| `Facebook_Impressions` | 7.20 | Mid range of published paid social feed benchmarks for the 2018 to 2020 window the data covers. |
| `Email_Impressions` | 1.50 | Owned channel. The cost is the email service provider send cost plus amortised creative, roughly 0.0015 per delivery. Not zero, because the list and the creative are not free. |
| `Affiliate_Impressions` | 4.00 | Affiliate is priced on acquisition, not impressions. I converted a typical commission structure to an effective CPM at assumed conversion rates. This is the least well grounded of the five. |
| `Paid_Views` | 25.00 | Video, bought on views. At an assumed cost per view of 0.025 that is 25.00 per thousand views, expressed per thousand for consistency with the others. |
| `Organic_Views` | 0.00 | No incremental media cost, and excluded from the optimiser budget. See below. |

These live in `config/config.yaml` under `cpm`, each with its basis recorded alongside the
number, so nothing here is only in a document.

## Why organic is not a budget lever

`Organic_Views` enters the model as a control, not as a channel. Organic volume is an
outcome of brand strength, content and prior paid activity. It is not a line item anyone can
move budget into. Treating it as a channel would let the optimiser recommend "spend more on
organic", which is not an action.

It is worth noting the cost of that decision. Because paid media partly drives organic
volume, controlling for organic absorbs some of the indirect effect of paid, so the paid
coefficients are conservative. I prefer that direction of bias to the alternative, which is
letting paid media take credit for a baseline it did not create.

## What the assumption changes and what it does not

**Unaffected by the CPM assumption**

- the fit of the model
- the shape of each channel's saturation curve
- the ranking of channels by incremental sales per impression
- the estimated adstock decay and half life per channel

**Determined by the CPM assumption**

- every ROI figure expressed per unit of currency
- the optimiser's recommended split, since it trades off channels on return per dollar
- the headline uplift number

**The part that carries the weight.** The optimiser compares channels against each other, so
what matters is the *relative* CPMs, not the absolute levels. Multiplying every rate by the
same factor rescales all spend and all ROI identically and leaves the recommended allocation
completely unchanged. Getting the overall level wrong is survivable. Getting the ratio
between search and social wrong is not.

## The plausibility check

`src/mmm/roi.py` includes `check_spend_plausibility`, which compares implied total media
spend against total revenue. Real advertisers sit somewhere between roughly 2 and 25 percent
of revenue depending on sector. A figure well outside that band means the assumed rates are
wrong by an order of magnitude.

I added this after an early run where the implied media budget came to well under one
percent of revenue, which produced ROI figures in the hundreds. A return of six hundred to
one is not a finding, it is an arithmetic consequence of a denominator that is far too
small, and anyone who has worked in media would dismiss it on sight. `suggested_cpm_scaling`
reports the uniform factor that would bring the total to a sensible share. I report the
factor rather than applying it silently, because a silent correction to an assumption is
just a different undocumented assumption.

## The sensitivity analysis

Since these are assumptions, the useful question is not "what is the optimal split" but
"which parts of the recommendation survive being wrong about cost".

`cpm_sensitivity` in `src/mmm/optimizer.py` shocks each channel's CPM by factors from 0.5 to
2.0 and re-solves the whole optimisation each time. `scripts/run_phase3.py` then reports,
for each channel, the share of scenarios in which the recommended direction stays the same.

That share is the number I would put in front of a stakeholder. A channel that is
recommended for an increase in every scenario is a decision I would defend. A channel whose
direction flips when a cost assumption moves by 25 percent is not a recommendation, it is
noise, and the correct action there is to go and find the real rate card before spending
anything.

## How to replace these with real numbers

If you have actual spend by channel by week, none of this is necessary. Add spend columns
alongside the impression columns, point `channels.paid` in `config/config.yaml` at the spend
columns instead, and set every CPM to 1.0 so the conversion becomes an identity. Nothing
else in the pipeline needs to change, because every downstream step reads from the config
rather than hardcoding the conversion.
