WITH { keywords :Array(String) } AS keywords,
{ similarity :Array(Float32) } AS similarity
SELECT q.keyword,
    q.similarity,
    d.website,
    d.average_position,
    -- count() AS item_count,
    -- avg(q.similarity) AS avg_score,
    -- pow(avg(q.similarity), -count()) AS weighted_score
FROM (
        SELECT arrayJoin(arrayEnumerate(keywords)) AS idx,
            keywords [idx] AS keyword,
            similarity [idx] AS similarity
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