"""
Created on Mon Feb 10 14:27:29 2020

@author: Avinash.Kumar
"""

from recommendation.db.DataConnection import dataDataframe, ranking_df
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

'''
Data preprocessing for Stacking of trees model.
'''
def preprocess(outlierStdDev):
    '''
    Returns
    -------
    X : Data processed for building the model(input features)
    Y : Data processed for building the model(output features)
    X_train : Data processed for training the model(input features)
    Y_train : Data processed for training the model(output features)
    X_test : Data processed for testing the model(input features)
    labelencoder_Y : value of the labels encoded
    sc : values of the standard scaler
    pca : values of the principle component analysis
    dictOfRank : Dictionary of ranking of courses in the dataset
    '''
    # Importing the dataset as datframe
    print('Convert to dataframe')
    dataset = dataDataframe
    print('Convert to ranking dataframe')
    ranking = ranking_df
    
    dataset['languageId'] = dataset['languageId'].fillna(0)
    dataset['clientId'] = dataset['clientId'].fillna(0)
    dataset['companyId'] = dataset['companyId'].fillna(0)
    dataset['countryId'] = dataset['countryId'].fillna(0)
    dataset['authentificationId'] = dataset['authentificationId'].fillna(0)
    dataset['salutationId'] = dataset['salutationId'].fillna(0)
    
    # Droping the duplicate values
    dataset = dataset.drop_duplicates()
    ranking = ranking.drop_duplicates()
    ranking = ranking.values
    
    '''
    Removing all the outliers in the dataset.
    '''
    print('The Raw Dataset')
    print(dataset.shape)

    coursesByUser = dataset.groupby('personId', as_index=False).agg({'objectId': "count"})
    print("Courses by user:")
    
    coursesByUser['outlier'] = (abs(coursesByUser.objectId - coursesByUser.objectId.mean()) > coursesByUser.objectId.std() * outlierStdDev)
    coursesByUser = coursesByUser.drop(columns=['objectId'])
    print("Users with outliers computed:")

    combined = dataset.merge(coursesByUser, on='personId', how='left')
    print("Merged dataframes:")
        
    filtered = combined.loc[combined['outlier'] == False]
    filtered = filtered.drop(columns=['outlier'])
    print("Filtered data:")
    print (filtered.shape)
    
    X = filtered.loc[:, ['personId', 'languageId', 'clientId', 'companyId', 'countryId','authentificationId','salutationId']].values
    Y = filtered.loc[:, ['objectId']].values
    
    # Encoding the output label
    labelencoder_Y = LabelEncoder()
    Y[:,0] = labelencoder_Y.fit_transform(Y[:,0])
    
    # Splitting the data for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    # Feature Scaling the input features
    print('Applying Standard scaler to data')
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train = sc.transform(X_train)
    
    '''
    Applying PCA to explain the direction in which the data is varies the most.
    Need to check the n_components first by passing n_components = None 
    and uncommenting the explained variance to check for the variance in the dataset.
    '''
    print('Applying pca to data')
    pca = PCA(n_components = 5)
    X = pca.fit_transform(X)
    X_train = pca.transform(X_train)
    #explained_variance = pca.explained_variance_ratio_
    
    '''
    Creating a Dictionay which has courses and the number of users taken that course.
    '''    
    zipbRank = zip(ranking[:,0],ranking[:,1])
    dictOfRank = dict(zipbRank)
    dictOfRank = defaultdict(lambda:0,dictOfRank)
    
    return X, Y, X_train, Y_train, X_test, labelencoder_Y, sc, pca, dictOfRank