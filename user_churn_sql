WITH beacons AS (
SELECT *, 
       CASE WHEN DENSE_RANK() OVER (PARTITION BY datepartition ORDER BY daily_brand_plays DESC) <= 20 THEN 1 END AS top20_brands_indicator
FROM 
(SELECT b.*,
       thing_custom_brand_id[1] brand,
       thing_custom_subcategory_id[1] subcat,
       COUNT(CASE WHEN action = 'play' THEN 1 END) OVER (PARTITION BY datepartition, thing_custom_brand_id[1]) AS daily_brand_plays,
       HOUR(CAST(SUBSTRING(REPLACE(b.timestamp_initiated,'T',' '),1,19) AS timestamp)) hour_timestamp
FROM beacon b
LEFT JOIN metadata m ON m.thing_id = b.thing_id
WHERE CAST(datepartition AS date) < DATE_ADD('month', -2, current_date))),

days AS (
SELECT datepartition 
FROM
(SELECT CAST(datepartition AS date) datepartition, ROW_NUMBER() OVER (ORDER BY CRC32(CAST(datepartition AS varbinary))) crc_row
FROM
(SELECT DISTINCT datepartition
FROM beacons
WHERE CAST(datepartition AS date) >= DATE_ADD('month', -16, current_date)))
WHERE crc_row <= 200
),

users_with_stats AS (
SELECT user_primaryid,
       CASE WHEN DENSE_RANK() OVER (ORDER BY first_brand_plays DESC) <= 10 THEN 1 ELSE 0 END first_brand_ranking_indicator
       FROM
        (SELECT f.user_primaryid,
                first_brand_played,        
                COUNT(*) OVER (PARTITION BY first_brand_played) first_brand_plays
         FROM user_table_firsts f
         LEFT JOIN user_table_sub_stats FOR TIMESTAMP AS OF (CURRENT_TIMESTAMP - INTERVAL '2' MONTH) s ON f.user_primaryid = s.user_primaryid
         WHERE CAST(SUBSTRING(timestamp_last_login, 1, 10) AS date) >= DATE_ADD('month', -16, current_date)
LIMIT 500)),

user_access AS (
SELECT a.user_primaryid, 
       first_brand_ranking_indicator,
       CAST(datepartition AS date) access_date
FROM users_with_stats a
LEFT JOIN beacons b ON b.user_primaryid = a.user_primaryid
GROUP BY 1, 2, 3),

users_last_actions AS ( 
SELECT user_primaryid,  
       DATE_DIFF('day', LAG(access_date) OVER (PARTITION BY user_primaryid ORDER BY access_date ASC), access_date) days_last_access,
       access_date,
       month_access_date,
       first_brand_ranking_indicator,
       plays_L60D, 
       t20_plays_L60D,
       recs_L60D,
       actions_L60D,
       brands_played_L60D, 
       subcats_played_L60D, 
       platforms_L60D,
       weeks_accessed_L60D,
       unq_recs_L60D,
       avg_hour_L60D,
       plays_L7D,
       t20_plays_L7D,
       recs_L7D,
       day_bounce_rate_L7D,
       brands_played_L7D,
       actions_L7D,
       days_accessed_L7D,
       plays_delta,
       t20_plays_delta,
       recs_delta,
       actions_delta,
       day_bounce_rate_delta,
       brands_played_delta,
       subcats_played_delta,
       CASE WHEN DATE_DIFF('day', access_date, LEAD(access_date) OVER (PARTITION BY user_primaryid ORDER BY access_date ASC)) IS NULL OR DATE_DIFF('day', access_date, LEAD(access_date) OVER (PARTITION BY user_primaryid ORDER BY access_date ASC)) >= 60 THEN 1 ELSE 0 END AS churn_status
FROM
(SELECT user_primaryid,  
        access_date,
        month_access_date,
        first_brand_ranking_indicator,

       -- Activity Last 60 Days
       plays_L60D, 
       CASE WHEN IS_NAN(t20_plays_L60D) THEN 0 ELSE t20_plays_L60D END AS t20_plays_L60D,
       recs_L60D,
       actions_L60D,
       CASE WHEN IS_NAN(day_bounce_rate_L60D) THEN 0 ELSE day_bounce_rate_L60D END AS day_bounce_rate_L60D, 
       brands_played_L60D, 
       subcats_played_L60D, 
       platforms_L60D,
       weeks_accessed_L60D,
       unq_recs_L60D,
       avg_hour_L60D,

       -- Activity Last 7 Days
       plays_L7D,
       CASE WHEN IS_NAN(t20_plays_L7D) THEN 0 ELSE t20_plays_L7D END AS t20_plays_L7D,
       recs_L7D,
       CASE WHEN IS_NAN(day_bounce_rate_L7D) THEN 0 ELSE day_bounce_rate_L7D END AS day_bounce_rate_L7D, 
       brands_played_L7D,
       actions_L7D,
       days_accessed_L7D,

       -- Activity Change Last 7 Days
       CASE WHEN IS_NAN(ROUND(CAST(plays_L7D AS double) / CAST(plays_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(plays_L7D AS double) / CAST(plays_L14D AS double) - 0.5, 3) END plays_delta,
       CASE WHEN IS_NAN(ROUND(CAST(t20_plays_L7D AS double) / CAST(t20_plays_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(t20_plays_L7D AS double) / CAST(t20_plays_L14D AS double) - 0.5, 3) END t20_plays_delta,
       CASE WHEN IS_NAN(ROUND(CAST(recs_L7D AS double) / CAST(recs_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(recs_L7D AS double) / CAST(recs_L14D AS double) - 0.5, 3) END recs_delta,
       CASE WHEN IS_NAN(ROUND(CAST(actions_L7D AS double) / CAST(actions_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(actions_L7D AS double) / CAST(actions_L14D AS double) - 0.5, 3) END actions_delta,
       CASE WHEN IS_NAN(ROUND(CAST(day_bounce_rate_L7D AS double) / CAST(day_bounce_rate_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(day_bounce_rate_L7D AS double) - CAST(day_bounce_rate_L14D AS double), 3) END day_bounce_rate_delta,
       CASE WHEN IS_NAN(ROUND(CAST(brands_played_L7D AS double) / CAST(brands_played_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(brands_played_L7D AS double) / CAST(brands_played_L14D AS double) - 0.5, 3) END brands_played_delta,
       CASE WHEN IS_NAN(ROUND(CAST(subcats_played_L7D AS double) / CAST(subcats_played_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(subcats_played_L7D AS double) / CAST(subcats_played_L14D AS double) - 0.5, 3) END subcats_played_delta
FROM
(SELECT a.user_primaryid,  
        access_date,
        MONTH(access_date) month_access_date,
        first_brand_ranking_indicator,
        COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 THEN 1 END) plays_L7D, 
        ROUND(CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 AND top20_brands_indicator = 1 THEN 1 END) AS double) / CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 THEN 1 END) AS double), 3) t20_plays_L7D, 
        COUNT(CASE WHEN custom_rule_id IS NOT NULL AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 THEN 1 END) recs_L7D, 
        COUNT(CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 THEN 1 END) actions_L7D,
        ROUND(CAST(COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 THEN b.datepartition END) AS double) / CAST(COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 THEN b.datepartition END) AS double), 3) day_bounce_rate_L7D, 
        COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 THEN brand END) brands_played_L7D, 
        COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 THEN subcat END) subcats_played_L7D, 
        COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 7 THEN b.datepartition END) days_accessed_L7D,
       
       COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 14 THEN 1 END) plays_L14D, 
       ROUND(CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 14 AND top20_brands_indicator = 1 THEN 1 END) AS double) / CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 14 THEN 1 END) AS double), 3) t20_plays_L14D, 
       COUNT(CASE WHEN custom_rule_id IS NOT NULL AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 14 THEN 1 END) recs_L14D, 
       COUNT(CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 14 THEN 1 END) actions_L14D,
       ROUND(CAST(COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 14 THEN b.datepartition END) AS double) / CAST(COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 14 THEN b.datepartition END) AS double), 3) day_bounce_rate_L14D, 
       COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 14 THEN brand END) brands_played_L14D, 
       COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 14 THEN subcat END) subcats_played_L14D,
    
       COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN 1 END) plays_L60D, 
       ROUND(CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 AND top20_brands_indicator = 1 THEN 1 END) AS double) / CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN 1 END) AS double), 3) t20_plays_L60D, 
       COUNT(CASE WHEN custom_rule_id IS NOT NULL AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN 1 END) recs_L60D, 
       COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN custom_rule_id END) unq_recs_L60D, 
       COUNT(CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN 1 END) actions_L60D,
       COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN custom_platform END) platforms_L60D, 
       ROUND(CAST(COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN b.datepartition END) AS double) / CAST(COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN b.datepartition END) AS double), 3) day_bounce_rate_L60D,
       COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN brand END) brands_played_L60D, 
       COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN subcat END) subcats_played_L60D, 
       COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN DATE_TRUNC('week',CAST(b.datepartition AS date)) END) weeks_accessed_L60D,
       ROUND(AVG(CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) < 60 THEN hour_timestamp END), 1) avg_hour_L60D
FROM user_access a
LEFT JOIN beacons b ON a.user_primaryid = b.user_primaryid AND CAST(b.datepartition AS date) < a.access_date 
WHERE DATE_DIFF('day', CAST(b.datepartition AS date), a.access_date) < 60 OR 
      CAST(b.datepartition AS date) IS NULL
GROUP BY 1, 2, 3, 4))),

training_set AS (
SELECT CASE WHEN days_last_access IS NULL THEN 0 ELSE days_last_access END days_last_access,
       month_access_date,
       first_brand_ranking_indicator,
       plays_L60D, 
       t20_plays_L60D,
       recs_L60D,
       actions_L60D,
       brands_played_L60D, 
       subcats_played_L60D, 
       platforms_L60D,
       weeks_accessed_L60D,
       unq_recs_L60D,
       CASE WHEN avg_hour_L60D IS NULL THEN 12 ELSE avg_hour_L60D END AS avg_hour_L60D,
       plays_L7D,
       t20_plays_L7D,
       recs_L7D,
       day_bounce_rate_L7D,
       brands_played_L7D,
       actions_L7D,
       days_accessed_L7D,
       plays_delta,
       t20_plays_delta,
       recs_delta,
       actions_delta,
       day_bounce_rate_delta,
       brands_played_delta,
       subcats_played_delta,
       churn_status,
       CASE WHEN days_last_access IS NULL THEN 1 ELSE 0 END AS new_user_stat
FROM users_last_actions a
LEFT JOIN days d ON d.datepartition = a.access_date
WHERE d.datepartition IS NOT NULL)

select *
from training_set
order by 1, 2
LIMIT 200
