# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 00:47:30 2022

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

def train_model(X_train, y_train):
#     model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators = 1000)
#     model.fit(X_train, y_train)

    model=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    return model
