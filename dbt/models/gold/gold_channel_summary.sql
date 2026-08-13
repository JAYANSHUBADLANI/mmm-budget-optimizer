-- Channel level rollup used by the reporting layer and by the executive summary.

{{ config(materialized='table') }}

with unpivoted as (
    {% set channels = ['google_impressions', 'facebook_impressions', 'email_impressions',
                       'affiliate_impressions', 'paid_views'] %}
    {% for channel in channels %}
    select
        '{{ channel }}'                     as channel,
        sum({{ channel }})                  as total_impressions,
        sum({{ channel }}_spend)            as total_assumed_spend,
        count(distinct division)            as n_divisions,
        sum(case when {{ channel }} = 0 then 1 else 0 end) * 1.0 / count(*) as share_dark_weeks
    from {{ ref('gold_media_spend_weekly') }}
    {{ "union all" if not loop.last }}
    {% endfor %}
),

total as (
    select sum(total_assumed_spend) as budget from unpivoted
)

select
    u.channel,
    u.total_impressions,
    u.total_assumed_spend,
    u.total_assumed_spend / t.budget as share_of_budget,
    u.n_divisions,
    u.share_dark_weeks
from unpivoted u
cross join total t
order by u.total_assumed_spend desc
