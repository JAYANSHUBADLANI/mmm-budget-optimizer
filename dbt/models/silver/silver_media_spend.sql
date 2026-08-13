-- Silver: one row per division per week, duplicates removed, derived column dropped.
--
-- The deduplication is the Division Z fix expressed in SQL. I use row_number over the
-- grain rather than select distinct, because row_number lets the next model count what was
-- removed, and because it would still collapse to one row per key if a future load
-- introduced rows that differ in some trailing column I have not thought about.

{{ config(materialized='table') }}

with ranked as (
    select
        *,
        row_number() over (
            partition by division, calendar_week
            order by sales
        ) as row_in_key,
        count(*) over (partition by division, calendar_week) as rows_in_key
    from {{ ref('bronze_media_spend') }}
),

deduplicated as (
    select * from ranked where row_in_key = 1
)

select
    division,
    calendar_week,
    paid_views,
    organic_views,
    google_impressions,
    email_impressions,
    facebook_impressions,
    affiliate_impressions,
    sales,
    -- overall_views is excluded here. It equals paid_views plus organic_views exactly, so
    -- carrying it forward would put a perfect linear combination of two existing columns
    -- into every downstream model and into the design matrix.
    rows_in_key > 1 as was_duplicated,
    extract(week from calendar_week)  as week_of_year,
    extract(year from calendar_week)  as calendar_year
from deduplicated
