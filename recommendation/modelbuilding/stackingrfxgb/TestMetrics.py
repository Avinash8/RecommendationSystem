"""
Created on Tue Dec  3 16:07:29 2019

@author: Avinash.Kumar
"""

'''
Model Creation and Model prediction 
'''
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

'''
Model creation
'''
def model(X,Y):
    '''
    Parameters
    ----------
    X : Input feature.
    Y : Output feature.

    Returns
    -------
    Classifier after building
    '''
    print('Training the Model')
    '''
    Enter the model paramters obtained from parameter tuning below.
    '''
    clfRandomForest = MultiOutputClassifier(RandomForestClassifier(n_estimators=50,min_samples_leaf=10,max_features='sqrt'))
    clfRandomForest.fit(X,Y)
    return clfRandomForest

'''
Preprocessing of the test datset
'''
def testDataPreprocess(data):
    '''
    Parameters
    ----------
    data : data without preprocess

    Returns
    -------
    data : data after preprocess.
    '''
    print('Preprocessing the test data')
    sorted_idx = np.lexsort(data.T)
    sorted_data = data[sorted_idx,:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    data = sorted_data[row_mask]
    return data

'''
Generating the model predictions
'''
def predict(data,classifier,labelencoder_Y,sc,pca):
    '''
    Parameters
    ----------
    data : Preprcesses data.
    classifier : Model after building.
    labelencoder_Y : Encoder of the ouput feature.
    sc : Standard scaler.
    pca : Principle Components.

    Returns
    -------
    prob : Probability of the prediction.
    new_predict : Prediction value.
    '''
    print('Generating the Predictions')
    new_predict_prob = classifier.predict_proba(pca.transform(sc.transform(np.array(data))))
    new_predict_prob = np.asarray(new_predict_prob)
    new_predict_prob = new_predict_prob.squeeze(0)
    new_predict = (-new_predict_prob).argsort()[::1][:10]
    prob = []
    for i in range(0,np.size(new_predict,0)):
        value = new_predict[i,:]
        prob.append(new_predict_prob[i,value])    
    for i in range(0,np.size(new_predict,1)):
        # Contains the index value
        new_predict[:,i] = labelencoder_Y.inverse_transform(new_predict[:,i])   
    return prob, new_predict