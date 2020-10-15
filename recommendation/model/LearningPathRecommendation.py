"""
Created on Mon Mar 16 09:01:46 2020

@author: Avinash.Kumar
"""

import pandas as pd
from recommendation.db.DataConnection import conn

'''
Recommendation strategy to find optimal learning path.
Learning path is defined as a set of courses which help person close skill gap between his current skills and job specific skills.
Markov model is used to construct learning path. At the time of implementation, data was insufficient to test its viability.
'''
def generateRecommendation(userId, numRec):
    '''
    Parameters
    ----------
    userId : Give the user ID of the user for whom the recommendation needs to considered.
    numRec : Number of recommendations needed.

    Returns
    -------
    recommendation : The recommended coursedID.
    '''
    # SQL to load the dataset for Model Building
    bookedComponents_dataset = """
    select component_id from portfolio where person_id = (%s)""" %userId + 'ORDER BY CONVERT(DateTime, lastupdated) DESC'
    bookedComponents_df = pd.io.sql.read_sql(bookedComponents_dataset, conn)    
    missingSkills = getMissingSkills(userId)
    courseSkills = getMemberCourses(userId, missingSkills)
    bookedComponents = bookedComponents_df.values.flatten().tolist()
    courseSkills = difference(courseSkills,bookedComponents)
    lastCourse = bookedComponents[0]
    recommendation = []
    while(len(courseSkills)!=0 and len(missingSkills)!=0):
        sucessorCourse = getNextMostLikelyCourse(lastCourse, courseSkills)
        if sucessorCourse != 0:
            missingSkills = difference(missingSkills,getCourseSkills(sucessorCourse))
            recommendation.append(sucessorCourse)
            courseSkills = difference(courseSkills,[sucessorCourse])
            lastCourse = sucessorCourse
        if len(recommendation) == numRec:
            break  
    return recommendation

def getMissingSkills(userId):
    '''
    Parameters
    ----------
    userId : User under consideration.

    Returns
    -------
    roleSkills : Missing skills.
    '''
    # Check the skills available for the User id
    userSkills_dataset = """
    select distinct skill_id from skill_value where object_id IN (SELECT role_id FROM  person_role WHERE person_id = (%s)""" %userId +')'
    userSkills_df = pd.io.sql.read_sql(userSkills_dataset, conn)
    userSkills = userSkills_df.values.flatten().tolist()
    # Check all the skills available    
    roleSkills_dataset = """
    select DISTINCT skill_id from skill_value where objecttype_id = 8
    """
    roleSkills_df = pd.io.sql.read_sql(roleSkills_dataset, conn)
    roleSkills = roleSkills_df.values.flatten().tolist()
    # Check for all the missing skills for the user
    roleSkills = difference(roleSkills,userSkills)
    return roleSkills

def getMemberCourses(userId, missingSkills):
    '''
    Parameters
    ----------
    userId : User under consideration.
    missingSkills : Misssing skills of user.
    
    Returns
    -------
    courseSkills : Set of courses which contain at least one required skill.
    '''
    # get the courses for the missing skill
    courseSkillsQuery_dataset = """
    select DISTINCT object_id from skill_value 
    where objecttype_id = 3 
    and skill_id in(""" + ','.join(map(str,missingSkills))+')'
    courseSkillsQuery_df = pd.io.sql.read_sql(courseSkillsQuery_dataset, conn)
    courseSkills = courseSkillsQuery_df.values.flatten().tolist()
    return courseSkills
    

def difference (list1, list2):  
    '''
    Parameters
    ----------
    list1 : A list of Ids.
    list2 : A list of Ids.

    Returns
    -------
    list_dif : The difference between first list and second list.
    '''
    list_dif = list(set(list1) - set(list2))
    return list_dif

def getNextMostLikelyCourse(rootNode, memberCourses):
    '''
    Parameters
    ----------
    rootNode : Current course / node in Markov model.
    memberCourses : Set of eligible courses which contain at least one required skill.
    
    Returns
    -------
    nextMostLikelyCourse : Set of courses which contain at least one required skill.
    '''
    #Markov model - compute probability
    highestProb = 0
    nextMostLikelyCourse = 0
    for course in memberCourses:
        jointCountQuery = """
        select count(person_id) from portfolio 
        where person_id IN (select person_id from portfolio where component_id = (%s)""" %rootNode + ') and component_id = (%s)' %course
        jointCount = pd.io.sql.read_sql(jointCountQuery, conn)
        prob = int(jointCount.values)
        if prob >= highestProb:
            highestProb = prob
            nextMostLikelyCourse = course
    return nextMostLikelyCourse

def getCourseSkills(courseId):
    '''
    Parameters
    ----------
    courseId : Id of the course / component.
    
    Returns
    -------
    Skills associated with the course.
    '''
    courseSkillsQuery = """
    SELECT DISTINCT skill_id FROM  skill_value WHERE object_id = (%s)""" %courseId + 'AND objecttype_id = 3'
    courseSkills = pd.io.sql.read_sql(courseSkillsQuery, conn)
    return courseSkills.values.flatten().tolist()
