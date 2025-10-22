SELECT keyword,
    average_position
FROM ahrefs.keywords
WHERE website = { website: String }