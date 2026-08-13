-- Gold: the modelling mart. One row per division per week, impressions converted to
-- assumed spend, ready for the Python model to read without further transformation.
--
-- The CPM values are duplicated here from config/config.yaml. That duplication is a real
-- cost of splitting the pipeline across SQL and Python, and the dbt test
-- assert_cpm_matches_config.sql exists to make the two drift apart loudly rather than
-- silently.

{{ config(materialized='table') }}

{% set cpm = {
    'google_impressions': 52.50,
    'facebook_impressions': 7.20,
    'email_impressions': 1.50,
    'affiliate_impressions': 4.00,
    'paid_views': 25.00
} %}

select
    division,
    calendar_week,
    week_of_year,
    calendar_year,
    sales,
    organic_views,

    {% for channel, rate in cpm.items() %}
    {{ channel }},
    {{ channel }} / 1000.0 * {{ rate }} as {{ channel }}_spend{{ "," if not loop.last }}
    {% endfor %},

    (
        {% for channel, rate in cpm.items() %}
        {{ channel }} / 1000.0 * {{ rate }}{{ " + " if not loop.last }}
        {% endfor %}
    ) as total_assumed_spend

from {{ ref('silver_media_spend') }}
