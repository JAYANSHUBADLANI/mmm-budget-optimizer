-- Every division must be observed in every week. The adstock transform is a lag over rows,
-- so a missing week silently spans a gap and corrupts the carryover estimate rather than
-- raising an error.

with expected as (
    select count(distinct calendar_week) as n_weeks from {{ ref('silver_media_spend') }}
),

per_division as (
    select division, count(distinct calendar_week) as n_weeks
    from {{ ref('silver_media_spend') }}
    group by 1
)

select p.division, p.n_weeks, e.n_weeks as expected_weeks
from per_division p
cross join expected e
where p.n_weeks != e.n_weeks
