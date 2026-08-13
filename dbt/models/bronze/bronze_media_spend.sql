-- Bronze: the raw file, typed and renamed, nothing else.
-- No filtering and no deduplication here on purpose. Bronze has to keep the Division Z
-- duplicates so the silver layer can be seen to remove them, and so the test in
-- schema.yml has something to fail on. A bronze layer that quietly fixes problems
-- destroys the audit trail that makes the layered design worth having.

{{ config(materialized='view') }}

select
    cast(division as varchar)                       as division,
    cast(strptime(calendar_week, '%m/%d/%Y') as date) as calendar_week,
    cast(paid_views as double)                      as paid_views,
    cast(organic_views as double)                   as organic_views,
    cast(google_impressions as double)              as google_impressions,
    cast(email_impressions as double)               as email_impressions,
    cast(facebook_impressions as double)            as facebook_impressions,
    cast(affiliate_impressions as double)           as affiliate_impressions,
    cast(overall_views as double)                   as overall_views,
    cast(sales as double)                           as sales
from {{ source('raw', 'sample_media_spend') }}
