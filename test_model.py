# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 00:51:00 2022

@author: tp3
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

## "previous_hour_calls" is one of the generated feaute which tells number of the calls in previous hour
## test data does not contain this "previous_hour_calls" feature
## Therefore it has to be calculated on the go. 
## I used calls of the last training example to set "previous_hour_calls" for the first test example
## For later test example we used prediction of the previous test example to set "previous_hour_calls"

def test_model(model,X_test,y_test,y_train):
    predictions = []
    originals = []
    pred = int(y_train.iloc[-1]) # calls of the last training example

    for i in range(len(X_test)):
        x = X_test[i:i+1]
        x = x.assign(previous_hour_calls = pred) # update "previous_hour_calls" for current test example
        y = y_test[i:i+1]
        pred = model.predict(x)
        predictions.append(round(pred[0])) # model returns list of one element
        originals.append(int(y.values))
        
    return predictions, originals   