# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:07:20 2020

@author: Avinash.Kumar
"""

from recommendation.db.DataConnection import conn
from recommendation.model.RandomRecommendation import randomRecommendation
from sklearn.preprocessing import LabelEncoder
from recommendation.model.PopularRecommendationBackup import popularRecommendation
from recommendation.model.LearningPathRecommendation import generateRecommendation
import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.externals import joblib


'''
RecommendationGenerator is the main file to initiate different strategies.
'''
W = torch.load(os.getcwd()+'/recommendation/model/learning.pt')
a = torch.load(os.getcwd()+'/recommendation/model/hidden_bias.pt')
b = torch.load(os.getcwd()+'/recommendation/model/visible_bias.pt')
labelencoderDataObject = LabelEncoder()
labelencoderY = LabelEncoder()
labelencoderDataObject.classes_ = np.load(os.getcwd()+'/recommendation/predictionattributes/labelEncoderObject.npy')
labelencoderY.classes_ = np.load(os.getcwd()+'/recommendation/predictionattributes/labelEncoderCourses.npy')
principleComponents = joblib.load(os.getcwd()+'/recommendation/predictionattributes/principleComponents.pkl')
standardScaler = joblib.load(os.getcwd()+'/recommendation/predictionattributes/standardScaler.bin')
classifierRandomForest = joblib.load(os.getcwd()+'/recommendation/model/joblibRfModel.pkl')
nbOfCourses = len(W[0])

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

def genCourseData(userId):
    '''
    Parameters
    ----------
    userId : User ID.

    Returns
    -------
    userCourseData : Courses taken by the user.
    '''
    bookedCoursesData = """
    select ec.OBJECT_ID from person p, portfolio pf, e_component ec where pf.person_id = p.person_id
    and pf.component_id = ec.component_id and pf.course_id = 0 and p.authentificationstatus_id = 1 and pf.status in (8,9,10,11,12) and p.person_id = (%s)""" %userId
    courseData = pd.io.sql.read_sql(bookedCoursesData, conn)  
    courseData = courseData.drop_duplicates()
    return courseData

def preprocess(userId):
    courseData = genCourseData(userId)
    courseData = courseData.values
    courseData[:,0] = labelencoderDataObject.transform(courseData[:,0])
    courseData = np.array(courseData, dtype = 'int')
    idCourses = courseData[:,0]
    courses = np.zeros(nbOfCourses)
    courses[idCourses] = idCourses+1
    userCourseData = list(courses)
    userCourseData = torch.FloatTensor(userCourseData)
    userCourseData[userCourseData >= 1] = 1
    return userCourseData, courseData
    
def predict(x):
    '''
    Parameters
    ----------
    x :Input data.

    Returns
    -------
    pv : Probability of Prediction.
    v : Prediction.
    '''
    x = x.view(1, nbOfCourses)
    x = torch.FloatTensor(x)
    wx = torch.mm(x, W.t())
    activation = wx + a.expand_as(wx)
    p_h_given_v = torch.sigmoid(activation)
    _, h = p_h_given_v, torch.bernoulli(p_h_given_v)
    wy = torch.mm(h, W)
    activation = wy + b.expand_as(wy)
    p_v_given_h = torch.sigmoid(activation)
    pv, v = p_v_given_h, torch.bernoulli(p_v_given_h)
    return pv,v
    
def predictTopK(userId):
    '''
    Parameters
    ----------
    userId : User ID.
       
    Returns
    -------
    idx : Course Id recommended.
    '''
    
    userData, courseData = preprocess(userId)
    prob, predictx = predict(userData)
    ''' numOfRec : Number of Recommendation '''
    numOfRec = len(courseData) + 10
    val,idx = torch.topk(prob, k=numOfRec, dim=-1, largest=True)
    idx = idx.numpy()
    for i in range(0,np.size(idx,0)):
        idx[i] = labelencoderDataObject.inverse_transform(idx[i])
    return idx, courseData
    
def genRecRbm(userId):
    '''
    Parameters
    ----------
    userId : User ID.

    Returns
    -------
    recommendation : Recommendations for the user.

    Asking the model to Generate 10 recommendation for the user.
    '''
    recommendedCourses, courseData = predictTopK(int(userId))
    recCourse = difference(recommendedCourses[0,:].tolist(), courseData[:,0].tolist())
    recommendation = {userId: recCourse[:10]}
    print('Recommendation from RBM')
    return recommendation

def genRecRf(userId):
    '''
    Parameters
    ----------
    userId : User ID.

    Returns
    -------
    recommendation : The recommendation for the User ID.
    '''
    userInfoData = """
    select p.PERSON_ID,p.LANGUAGE_ID,p.CLIENT_ID,p.COMPANY_ID,
    p.COUNTRY_ID,p.AUTHENTIFICATIONSTATUS_ID,p.SALUTATION_ID
    from person p
    where p.person_id =(%s)""" %userId
    userInfo = pd.io.sql.read_sql(userInfoData, conn)
    userInfo = userInfo.drop_duplicates()
    # filling the empty values with 0.   
    userInfo['languageId'] = userInfo['languageId'].fillna(0)
    userInfo['clientId'] = userInfo['clientId'].fillna(0)
    userInfo['companyId'] = userInfo['companyId'].fillna(0)
    userInfo['countryId'] = userInfo['countryId'].fillna(0)
    userInfo['authentificationId'] = userInfo['authentificationId'].fillna(0)
    userInfo['salutationId'] = userInfo['salutationId'].fillna(0)
    
    newPredictProb = classifierRandomForest.predict_proba(principleComponents.transform(standardScaler.transform(userInfo.values)))
    newPredictProb = np.asarray(newPredictProb)
    newPredictProb = newPredictProb.reshape(1,-2)
    newPredict = (-newPredictProb).argsort()[::1]
    for i in range(0,np.size(newPredict,1)):
        newPredict[:,i] = labelencoderY.inverse_transform(newPredict[:,i])
    
    '''
    Asking the model to Generate 10 recommendation for the user.
    '''
    recommendation = {userId: newPredict[0,:10].tolist()}
    print('Recommendation from Stack')
    return recommendation

def genPopularBackup(userId):
    '''
    Parameters
    ----------
    userId : User ID.

    Returns
    -------
    recommendation : Recommendation generated by user.
    
    Popular recommendation have a time frame of 3 and looking for 10 popular recommendation
    '''
    recCourses = popularRecommendation(userId, 10)
    recommendation = {userId: recCourses}
    print('Recommendation from popular')
    return recommendation

def genRandom(userId):
    '''
    Parameters
    ----------
    userId : User ID.

    Returns
    -------
    recommendation : Recommendations for the model.
    
    Asking the model to Generate 10 recommendation for the user.
    '''
    randRecommendation = randomRecommendation(userId)
    recCourses = random.choices(randRecommendation,k=10)
    recommendation = {userId: recCourses}
    print('Recommendation from Random')
    return recommendation

def genLearningPathRecommendation(userId):
    '''
    Parameters
    ----------
    user_id : User ID.

    Returns
    -------
    recommendation : Learning path recommendation by the user.
    '''
    recCourses = generateRecommendation(userId,10)
    recommendation = {userId: recCourses}
    print('Recommendation for Learning path')
    return recommendation
    

def genCourseRecommendation(userId):
    '''
    Parameters
    ----------
    userId : User ID.

    Returns
    -------
    recommendationId : Returns the recommendation for the user.

    '''
    try:
        recommendationId = genRecRbm(userId)
    except:
        try:
            recommendationId = genRecRf(userId)
        except:
            try:
                recommendationId = genPopularBackup(userId)
            except:
                recommendationId = genRandom(userId)
    return recommendationId
