WITH { keywords :Array(String) } AS keywords,
{ scores :Array(Float32) } AS scores
SELECT 
    q.keyword,
    q.score,
    d.website,
    d.average_position,

    -- count() AS item_count,
    -- avg(q.score) AS avg_score,
    -- pow(avg(q.score), -count()) AS weighted_score
FROM (
        SELECT arrayJoin(arrayEnumerate(keywords)) AS idx,
            keywords [idx] AS keyword,
            scores [idx] AS score
    ) AS q
    LEFT OUTER JOIN (
        SELECT keyword,
            website,
            average_position
        FROM ahrefs.keywords
    ) AS d ON q.keyword = d.keyword -- ORDER BY keyword
-- GROUP BY website 
-- ORDER BY weighted_score DESC
 -- LIMIT 20