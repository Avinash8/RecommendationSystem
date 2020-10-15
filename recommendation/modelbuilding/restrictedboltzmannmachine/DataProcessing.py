# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:58:26 2020

@author: Avinash.Kumar
"""

from recommendation.db.DataConnection import dataDataframe, ranking_df
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from collections import defaultdict
from scipy.special import entr

'''
Data preprocessing for the Restricted boltzmann machine(RBM) model.
'''
class DataProcessing():
    # Importing the dataset as datframe
    print('Convert to dataframe')
    dataset_df = dataDataframe
    print('Convert to ranking dataframe')
    ranking_df = ranking_df
    dataset = dataset_df.loc[:,['personId','objectId']]
    # Remove Duplicates
    dataset = dataset.drop_duplicates()
    # Importing the Ranking datset and removing duplicates
    ranking = ranking_df.drop_duplicates()
    dataset = dataset.values
    ranking = ranking.values
    # Encoding the dataset    
    labelencoder_data_person = LabelEncoder()
    labelencoder_data_object = LabelEncoder()
    dataset[:,0] = labelencoder_data_person.fit_transform(dataset[:,0])
    dataset[:,1] = labelencoder_data_object.fit_transform(dataset[:,1])
    
    '''
    Creating the Dictionary that has courses and the number of users taken that course.
    '''
    zipbRank = zip(ranking[:,0],ranking[:,1])
    dictOfRank = dict(zipbRank)
    dictOfRank = defaultdict(lambda:0,dictOfRank)   
    
    '''
    Data preprocessing
    '''
    def dataPreprocess(self):
        '''
         Returns
        -------
        data : Preprocessed complete data.
        train_data : Preprocessed training data.
        test_data : Preprocessed test data.
        '''
        np.random.shuffle(self.dataset)
        train_data, test_data = train_test_split(self.dataset,test_size=0.2)
        data = np.array(self.dataset, dtype = 'int')
        train_data = np.array(train_data, dtype = 'int')
        test_data = np.array(test_data, dtype = 'int')
        # Getting the number of users and courses
        self.nb_users = int(max(max(train_data[:,0]), max(test_data[:,0])))
        self.nb_courses = int(max(max(train_data[:,1]), max(test_data[:,1])))
        print('Using the Convert function')
        train_data = self.convert(train_data)
        print('Convert to Torch tensor')
        train_data = torch.FloatTensor(train_data)
        print('Convert to 1 0')
        train_data[train_data >= 1] = 1
        print('Using the Convert function')
        test_data = self.convert(test_data)
        print('Convert to Torch tensor')
        test_data = torch.FloatTensor(test_data)
        print('Convert to 1 0')
        test_data[test_data >= 1] = 1
        print('Using the Convert function')
        data = self.convert(data)
        print('Convert to Torch tensor')
        data = torch.FloatTensor(data)
        print('Convert to 1 0')
        data[data >= 1] = 1
        num_hidden = self.entropy(data)
        return data,train_data,test_data,num_hidden
    
    '''
    Convert function for converting the dataset into 1 and 0
    '''
    def convert(self,data):
        '''
        Parameters
        ----------
        data : dataset

        Returns
        -------
        new_data : Converted data into a array of 1 and 0.
        '''
        new_data = []
        for id_users in range(0, self.nb_users + 1): #(1, nb_users + 1)
            id_courses = data[:,1][data[:,0] == id_users]
            courses = np.zeros(self.nb_courses+1)
            courses[id_courses] = id_courses+1
            new_data.append(list(courses))
        return new_data
    
    '''
    Entropy function to calculate the entropy of the datset. 
    Entropy will give us the uncertainity in the dataset. We multiply 
    by 10 in order to fulfill the churn property of the recommender system. 
    The value of the entropy function has an upper bound of 500.
    '''
    def entropy(self, data):
        probability = data/data.sum()
        data_entropy = entr(probability).sum()/np.log(2)
        num_hidden = data_entropy * 10
        return int(num_hidden) if num_hidden < 500 else 500