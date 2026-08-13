-- A row per data quality finding, so the cleaning is auditable from SQL rather than only
-- from the Python logs. This is the table I would point a stakeholder at when they ask
-- what happened to the extra Division Z rows.

{{ config(materialized='table') }}

with bronze_counts as (
    select
        division,
        count(*)                             as bronze_rows,
        count(distinct calendar_week)        as distinct_weeks
    from {{ ref('bronze_media_spend') }}
    group by 1
),

silver_counts as (
    select division, count(*) as silver_rows
    from {{ ref('silver_media_spend') }}
    group by 1
)

select
    b.division,
    b.bronze_rows,
    s.silver_rows,
    b.bronze_rows - s.silver_rows                        as rows_removed,
    b.distinct_weeks,
    case
        when b.bronze_rows > s.silver_rows then 'duplicate rows removed'
        when b.distinct_weeks != {{ var('expected_weeks') }} then 'unexpected week count'
        else 'clean'
    end                                                  as finding
from bronze_counts b
join silver_counts s using (division)
order by rows_removed desc, division
