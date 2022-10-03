# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 00:55:57 2022

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

def plotting(predictions, originals):
    preds_day = np.add.reduceat(predictions, np.arange(0, len(predictions), 24)) # call prediction per day
    preds_week = np.add.reduceat(preds_day, np.arange(0, len(preds_day), 7)) # call prediction per week
    preds_month = np.add.reduceat(preds_day, np.arange(0, len(preds_day), 30)) # call prediction per month

    originals_day = np.add.reduceat(originals, np.arange(0, len(originals), 24)) # call per day 
    originals_week = np.add.reduceat(originals_day, np.arange(0, len(originals_day), 7)) # call per week
    originals_month = np.add.reduceat(originals_day, np.arange(0, len(originals_day), 30)) # call per month
    
    print("RMSE per hour",mean_squared_error(originals, predictions, squared=False))
    print("RMSE per day",mean_squared_error(preds_day, originals_day, squared=False))
    print("RMSE per week",mean_squared_error(preds_week, originals_week, squared=False))
    print("RMSE per month",mean_squared_error(preds_month, originals_month, squared=False))

    print("MAE per hour",mean_absolute_error(originals, predictions))
    print("MAE per day",mean_absolute_error(preds_day, originals_day))
    print("MAE per week",mean_absolute_error(preds_week, originals_week))
    print("MAE per month",mean_absolute_error(preds_month, originals_month))
    
    plt.figure(figsize=(15,5))
    plt.title("prediction vs target hourly")
    plt.plot(originals , label='target')
    plt.plot(predictions, label='prediction')
    plt.legend()

    plt.figure(figsize=(15,5))
    plt.title("prediction vs target daily")
    plt.plot(originals_day , label='target')
    plt.plot(preds_day, label='prediction')
    plt.legend()

    plt.figure(figsize=(15,5))
    plt.title("prediction vs target weekly")
    plt.plot(originals_week , label='target')
    plt.plot(preds_week, label='prediction')
    plt.legend()

    plt.figure(figsize=(15,5))
    plt.title("prediction vs target monthly")
    plt.plot(originals_month , label='target')
    plt.plot(preds_month, label='prediction')
    plt.legend()
