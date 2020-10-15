"""
Created on Mon Feb 10 16:12:30 2020

@author: Avinash.Kumar
"""

'''
Generate the Score for the Model based on the train, test dataset and
create the model for the complete dataset.
'''

import datetime
print(datetime.datetime.now())
import os
import numpy as np
from sklearn.externals import joblib
from recommendation.modelbuilding.stackingrfxgb import TestMetrics
from recommendation.modelbuilding.stackingrfxgb import Preprocess
from recommendation.modelbuilding.stackingrfxgb import RecommenderEvaluator

def main_model_training(value):
    '''
    Getting the data after intial preprocessing
    '''
    Xdata,Ydata,XTraining,YTraining,XTesting,labelencoderY,standardScalar,principleComponent,rankingDict = Preprocess.preprocess(value)
    # Building the model on the training dataset 
    trainingClassifier = TestMetrics.model(XTraining,YTraining)
    # Preprocessing the test data
    XTesting = TestMetrics.testDataPreprocess(XTesting)
    # To select the top 100 predictions
    probPred, pred = TestMetrics.predict(XTesting,trainingClassifier,labelencoderY,standardScalar,principleComponent)
    topPrediction = pred[:,0:100]
    probPredict = np.asarray(probPred)
    topKValues = probPredict[:,0:100]
    print('Printing the scores')
    print('The value of Coverage: '+ str(RecommenderEvaluator.UserCoverage(topPrediction, topKValues, 0.1)))
    print('The value of Novelty: '+ str(RecommenderEvaluator.Novelty(topPrediction, rankingDict)))
    #print('The value of Personalization: ' + str(RecommenderEvaluator.Personalization(topPrediction)))
    # Building the model on the complete dataset
    classifier = TestMetrics.model(Xdata,Ydata)    
    '''
    Saving the model and other values for prediction
    '''
    np.save(os.getcwd()+'/recommendation/predictionattributes/labelEncoderCourses.npy', labelencoderY.classes_)
    joblib.dump(classifier, os.getcwd()+'/recommendation/model/joblibRfModel.pkl')
    joblib.dump(standardScalar, os.getcwd()+'/recommendation/predictionattributes/standardScaler.bin', compress=True)
    joblib.dump(principleComponent, os.getcwd()+'/recommendation/predictionattributes/principleComponents.pkl')

value = float(input("Enter the value of standard Deviation:\n"))
main_model_training(value)
print(datetime.datetime.now())