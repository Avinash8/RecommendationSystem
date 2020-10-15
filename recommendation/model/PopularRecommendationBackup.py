from recommendation.db.DataConnection import conn
'''
PopularRecommendation strategy to return popular media objects in the database.
'''

def popularRecommendation(userId, numOfRecommendation):
    '''
    Parameters
    ----------
    userId : User ID.
    numOfRecommendation : Number of Recommendation.

    Returns
    -------
    recommendation : Popular Recommendation.
    '''
    recommendation = []
    cursor = conn.cursor()
    cursor.execute("SELECT COMPONENT_ID from portfolio where component_id IN (SELECT component_id FROM e_component WHERE language_id=(SELECT language_id FROM person where person_id = ?)) and person_id != (?) GROUP BY component_id ORDER BY COUNT(component_id)  DESC",(userId,userId))
    popularRec = cursor.fetchmany(numOfRecommendation)
    for row in popularRec:
        recommendation.append(int(row[0]))
    return recommendation
