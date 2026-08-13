-- Division level rollup. The spread in sales scale across divisions is the empirical
-- argument for the hierarchical model, so it is worth materialising rather than
-- recomputing in a notebook every time someone asks.

{{ config(materialized='table') }}

select
    division,
    count(*)                                        as n_weeks,
    sum(sales)                                      as total_sales,
    avg(sales)                                      as mean_weekly_sales,
    stddev_samp(sales) / nullif(avg(sales), 0)      as sales_coefficient_of_variation,
    sum(total_assumed_spend)                        as total_assumed_spend,
    sum(total_assumed_spend) / nullif(sum(sales), 0) as spend_to_sales_ratio
from {{ ref('gold_media_spend_weekly') }}
group by 1
order by total_sales desc
