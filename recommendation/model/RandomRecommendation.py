"""
Created on Thu Mar  5 14:54:40 2020

@author: Avinash.Kumar
"""

from recommendation.db.DataConnection import conn
'''
Importing the data and asking for random course reommendation for user.
'''
def randomRecommendation(userId):
    '''
    Parameters
    ----------
    userId : User ID.

    Returns
    -------
    recommendation : Popular Recommendation.
    '''
    recommendation = []
    cursor = conn.cursor()
    cursor.execute("SELECT component_id FROM portfolio WHERE person_id != (?) AND component_id IN (SELECT component_id FROM e_component WHERE language_id=(SELECT language_id FROM person where person_id = ?))",(userId,userId))
    randomRec = cursor.fetchmany(1000)
    for row in randomRec:
        recommendation.append(int(row[0]))
    return recommendation
