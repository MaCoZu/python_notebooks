/*SELECT name, id, review_count, fans
 FROM user
 ORDER BY fans DESC*/


-- cities with the most reviews in descending order
SELECT city, SUM(review_count) as reviews
FROM business
GROUP BY city
ORDER BY reviews DESC

--Find the distribution of star ratings to the business in the following cities:
SELECT stars, review_count
FROM business
WHERE city='Avon'
ORDER BY stars

-- since there was multiple times stars = 5 etc. they need to be grouped and reviews summed up
SELECT stars,
  SUM(review_count) review_count
FROM business
WHERE city = 'Beachwood'
GROUP BY stars
ORDER BY stars

-- user with the most reviews
SELECT name, review_count
FROM user
GROUP BY id
ORDER BY review_count DESC


SELECT cov_xy*cov_xy /(var_x * var_y) as R_2
FROM (
    SELECT
      avg_x,
      avg_y,
      AVG((review_count * fans)) - (avg_x * avg_y) AS cov_xy,
      AVG((review_count - avg_x)*(review_count - avg_x)) AS var_x,
      AVG((fans - avg_y)*(fans - avg_y)) AS var_y
    FROM user,
      (
        SELECT AVG(review_count) avg_x, AVG(fans) avg_y
        FROM user
      )
  )

SELECT stars, hours, review_count, is_open, name, city, state, category
FROM business b JOIN category c ON b.id = c.business_id JOIN hours h ON b.id = h.business_id
WHERE c.category = 'Restaurants'
GROUP BY b.stars


-- count of restaurants with 2-3 stars and restaurants with 4-5 stars
SELECT
  COUNT(CASE WHEN stars >=2 AND stars <= 3 THEN 1 END ) AS [2-3],
  COUNT( CASE WHEN stars > 3 AND stars <= 5 THEN 1 END ) AS [4-5]
FROM business b JOIN category c ON b.id = c.business_id
WHERE c.category = 'Restaurants'


SELECT name, hours, stars, review_count, city, state, is_open
FROM business b JOIN category c ON b.id = c.business_id JOIN hours h ON b.id = h.business_id
WHERE c.category = 'Restaurants' AND stars >=2 AND stars <= 3


PRAGMA table_info(hours); -- get table info


SELECT SUBSTR(hours, -11)
FROM
(SELECT hours
FROM business b JOIN category c ON b.id = c.business_id JOIN hours h ON b.id = h.business_id
WHERE c.category = 'Restaurants' AND stars >=2 AND stars <= 3 )


SELECT stars, hours, name, review_count, address
FROM business b JOIN category c ON b.id = c.business_id JOIN hours h ON b.id = h.business_id
WHERE c.category = 'Restaurants' AND b.city = 'Phoenix'
ORDER BY stars


SELECT is_open, ROUND(AVG(stars), 2) as avg_stars, SUM(review_count) as reviews
FROM business
Group by is_open


SELECT category,
  avg_stars
FROM (
    SELECT category,
      AVG(stars) AS avg_stars
    FROM business b
      JOIN category c ON b.id = c.business_id
    GROUP BY category
    ORDER BY avg_stars DESC
    LIMIT 10
  )
UNION ALL
SELECT category,
  avg_stars
FROM (
    SELECT category,
      AVG(stars) AS avg_stars
    FROM business b
      JOIN category c ON b.id = c.business_id
    GROUP BY category
    ORDER BY avg_stars ASC
    LIMIT 10
  )
ORDER BY avg_stars DESC
