
// return path to all NATO countries
MATCH p=((c:Country)-[:IS_A]->(g:GovernmentType)-[:CONCERNS]->(s:PoliticalSubject))
WHERE c.nato_member = TRUE
RETURN p;

// return path to all EU countries
MATCH p=((c:Country)-[:IS_A]->(g:GovernmentType)-[:CONCERNS]->(s:PoliticalSubject))
WHERE c.eu_member = TRUE
RETURN p;

// Get all countries that have at least one form of government related to Federalism, and the other one - related to other political theme
MATCH p=((s2:PoliticalSubject)<-[:CONCERNS]-(g2:GovernmentType)-[:IS_A]-(c:Country)-[:IS_A]->(g:GovernmentType)-[:CONCERNS]->(s:PoliticalSubject))
WHERE s.political_subject = 'Federalism' AND s2.political_subject <> 'Federalism'
RETURN p;