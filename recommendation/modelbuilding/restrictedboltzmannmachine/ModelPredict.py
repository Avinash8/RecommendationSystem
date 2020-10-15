"""
Created on Mon Feb 17 11:02:11 2020

@author: Avinash.Kumar
"""
import datetime
print(datetime.datetime.now()) 
from recommendation.modelbuilding.restrictedboltzmannmachine.SparkSession import SparkSession

'''
Calling spark session
'''
modelRBM = SparkSession()
print(datetime.datetime.now())