"""
Created on Mon Feb 17 13:02:43 2020

@author: Avinash.Kumar
"""

'''
Running the model in a spark session.
'''
import os
import numpy as np
from recommendation.modelbuilding.restrictedboltzmannmachine.DataProcessing import DataProcessing
from recommendation.modelbuilding.restrictedboltzmannmachine.TestMetrics import testing,test_score,training

class SparkSession():
    '''
    Preprocessing the data, training the model and printing the scores for the model.
    '''
    data = DataProcessing()
    dataset,trainingSet,testSet,hiddenNodes = data.dataPreprocess()
    dictOfRank = data.dictOfRank
    numberOfUsers = data.nb_users
    labelencoderDataPerson = data.labelencoder_data_person
    labelencoderDataObject = data.labelencoder_data_object
    '''
    Enter the model paramters obtained from parameter tuning below.
    '''
    print('Assigning the values')
    numberOfVisible = len(dataset[0])
    numberOfHidden = hiddenNodes
    batchSize = 20
    step = 100
    numEpoch = 100
    learningRate = 0.01
    
    topPrediction,topValues,bestEpoch = testing(trainingSet, testSet, numberOfVisible, numberOfHidden, step, batchSize, numEpoch, learningRate, numberOfUsers)
    topPrediction = topPrediction.numpy()
    for i in range(len(topPrediction)):
        topPrediction[i,:] = labelencoderDataObject.inverse_transform(topPrediction[i,:])
    test_score(topPrediction, topValues, dictOfRank)
    rbm = training(dataset, numberOfVisible, numberOfHidden, step, batchSize, bestEpoch, learningRate, numberOfUsers)
    print('Saving the labelencoder')
    np.save(os.getcwd()+'/recommendation/predictionattributes/labelEncoderUser.npy',labelencoderDataPerson.classes_)
    np.save(os.getcwd()+'/recommendation/predictionattributes/labelEncoderObject.npy',labelencoderDataObject.classes_)
