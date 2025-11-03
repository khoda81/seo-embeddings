SELECT website,
    COUNT(*) AS keyword_count
FROM ahrefs.words
GROUP BY website
ORDER BY keyword_count DESC;