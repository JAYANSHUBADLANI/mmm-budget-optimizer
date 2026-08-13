-- The CPM rates are defined in two places: config/config.yaml for the Python code and a
-- Jinja map in gold_media_spend_weekly.sql for the SQL. This test recomputes the implied
-- rate from the materialised columns and fails if it has drifted from the value the SQL
-- claims to use, which catches the case where someone edits one and forgets the other.

{% set cpm = {
    'google_impressions': 52.50,
    'facebook_impressions': 7.20,
    'email_impressions': 1.50,
    'affiliate_impressions': 4.00,
    'paid_views': 25.00
} %}

{% for channel, rate in cpm.items() %}
select
    '{{ channel }}' as channel,
    sum({{ channel }}_spend) / nullif(sum({{ channel }}) / 1000.0, 0) as implied_cpm,
    {{ rate }} as expected_cpm
from {{ ref('gold_media_spend_weekly') }}
having abs(
    sum({{ channel }}_spend) / nullif(sum({{ channel }}) / 1000.0, 0) - {{ rate }}
) > 0.01
{{ "union all" if not loop.last }}
{% endfor %}
