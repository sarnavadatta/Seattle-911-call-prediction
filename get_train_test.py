# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 00:57:07 2022

@author: tp3
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime

def get_train_test(df):
    
    df_train=df[df['Datetime']<'2022-08-01'] # training set
    df_test=df[df['Datetime']>='2022-08-01'] # test set (last 2 months)

    # add another column which holds the number of calls of previous the hour
    df_train = df_train.assign(previous_hour_calls =  df_train['calls_hour'].shift(1))
    df_train = df_train.dropna()

    X_train=df_train.drop(['Datetime','calls_hour','year'],axis=1)
    y_train=df_train['calls_hour']

    X_test=df_test.drop(['Datetime','calls_hour','year'],axis=1)
    y_test=df_test['calls_hour']
    
    return X_train, y_train, X_test, y_test