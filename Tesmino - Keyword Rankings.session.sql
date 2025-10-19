SELECT average_position, COUNT(*) as count
FROM keywords
GROUP BY average_position
ORDER BY average_position;

