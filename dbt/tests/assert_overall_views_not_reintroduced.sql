-- Guard against someone adding overall_views back into the silver or gold layer.
-- It equals paid_views plus organic_views exactly, so its presence in the modelling mart
-- would make the media block of the design matrix rank deficient.

select 1 as violation
from information_schema.columns
where lower(table_name) in ('silver_media_spend', 'gold_media_spend_weekly')
  and lower(column_name) = 'overall_views'
