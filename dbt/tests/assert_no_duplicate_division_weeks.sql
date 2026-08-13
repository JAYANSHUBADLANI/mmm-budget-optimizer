-- The Division Z regression test, written as a singular dbt test.
--
-- This is the check that would have caught the bug on the day the file landed. It returns
-- offending rows, and dbt fails the run when the result is non empty. I keep it as a
-- singular test in addition to the schema level uniqueness test because this version names
-- the offending divisions in the failure output, which is what a person actually needs when
-- the pipeline breaks at seven in the morning.

select
    division,
    calendar_week,
    count(*) as n_rows
from {{ ref('silver_media_spend') }}
group by 1, 2
having count(*) > 1
