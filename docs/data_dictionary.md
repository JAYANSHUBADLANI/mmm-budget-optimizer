# Data dictionary

## Primary dataset

Source: Kaggle, `yugagrawal95/sample-media-spends-data`. 3051 rows as shipped, 2938 after
the duplicate fix. 26 divisions, 113 weeks, 6 January 2018 to 29 February 2020.

| Column | Type | Meaning | Role in the model |
| --- | --- | --- | --- |
| `Division` | string | Sales division, A to Z | Entity dimension of the hierarchy |
| `Calendar_Week` | date | Week start, `%m/%d/%Y` in the raw file | Time dimension |
| `Google_Impressions` | float | Search impressions | Paid channel, budget lever |
| `Facebook_Impressions` | float | Paid social impressions | Paid channel, budget lever |
| `Email_Impressions` | float | Emails delivered | Paid channel, budget lever |
| `Affiliate_Impressions` | float | Affiliate impressions | Paid channel, budget lever |
| `Paid_Views` | float | Paid video views | Paid channel, budget lever |
| `Organic_Views` | float | Organic video views | Control, not a budget lever |
| `Overall_Views` | float | Equals `Paid_Views + Organic_Views` | Dropped, perfectly collinear |
| `Sales` | float | Weekly sales in currency | Target |

### Derived columns added by the pipeline

| Column | Source | Notes |
| --- | --- | --- |
| `entity_idx`, `time_idx` | `cleaning.add_panel_index` | Integer indices for the model dimensions |
| `week_of_year`, `year` | `cleaning.add_panel_index` | Used in seasonality diagnostics |
| `n_holidays` | `holidays` package | Count of public holidays falling in the week |
| `is_major_holiday_week` | `holidays` package | Christmas, Thanksgiving, New Year, Independence Day, Easter |
| `is_black_friday_week` | derived from Thanksgiving | Not a public holiday, but drives retail volume |
| `trends_index` | Google Trends via pytrends | Category search interest. A bad control, see limitations |
| `cpi_index` | FRED `CPIAUCSL` | Monthly CPI forward filled to weekly, indexed to the series start |
| `<channel>_spend` | `roi.spend_frame` | Impressions divided by 1000 times the assumed CPM |

## Secondary dataset, generalisation check only

Source: Kaggle, `singhnavjot2062001/product-advertising-data`. 300 rows, six channels: TV,
billboards, Google Ads, social media, influencer marketing, affiliate marketing. Never merged
with the primary panel. Column names are resolved at runtime by
`generalization.resolve_secondary_columns` rather than hardcoded, because the Kaggle export
naming has varied between versions.

## Benchmark dataset, correctness check only

Source: Meta Robyn, `R/data/dt_simulated_weekly.RData`, read with `pyreadr`. Weekly,
December 2015 to November 2019. Revenue plus TV, out of home, print, Facebook and search
spend, competitor sales and event flags. Single geo, so the hierarchy collapses to one
entity, which also exercises the degenerate case in the array handling. Never merged with the
primary panel.

## Grain and keys

The primary panel grain is `(Division, Calendar_Week)`. This is enforced in three places:
`validation.check_primary_key` in Python, `dbt_utils.unique_combination_of_columns` in the
dbt schema tests, and the singular test `assert_no_duplicate_division_weeks.sql`. All three
exist because the raw file violated it.
