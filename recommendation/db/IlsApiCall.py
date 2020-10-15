# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:11:04 2020

@author: Avinash.Kumar
"""

import requests
import pandas as pd
from pandas.io.json import json_normalize

data = {
        "client_secret": "802712e2-849d-11ea-bc55-0242ac130003",
        "client_id": "RECOMMENDATION",
        "grant_type": "client_credentials"
        }

headersAuth = {'Content-Type': 'application/x-www-form-urlencoded'}

req = requests.post('http://192.168.0.14:44900/ils/oauth/accesstoken', 
                    data=data, headers=headersAuth)

auth = req.json()
headersData = {'Accept': 'application/json', 'Authorization': 'Bearer '+auth['access_token']}
reqData = requests.get('http://192.168.0.14:44900/ils/restapi/systems/services/recommendation/userdetails', headers=headersData)
dataJson = reqData.json()
userData = dataJson['userDetails']
dataDataframe = json_normalize(userData)

ranking_df = dataDataframe.objectId.value_counts()
ranking_df = pd.DataFrame(ranking_df)
ranking_df = ranking_df.reset_index()