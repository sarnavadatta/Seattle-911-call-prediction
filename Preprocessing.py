# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 01:01:07 2022

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

# first data is the 911 call data and the second one is the weather data 
def Preprocessing(df1,df2):
    # drop non-relevant columns from df1
    df1=df1.drop(['Report Location','Incident Number','Latitude','Longitude','Address','Type'], axis=1)
    df1['Datetime']=pd.to_datetime(df1['Datetime'], format="%m/%d/%Y %I:%M:%S %p")
    df1['Datetime']=df1['Datetime'].dt.strftime('%Y-%m-%d %H')
    #consider data from 2016
    df1=df1[df1['Datetime']>='2016-01-01']
    df1.sort_values(by='Datetime', inplace=True)
    
    # calculate calls per hour
    Calls=pd.DataFrame(df1.groupby('Datetime')['Datetime'].count()).rename(columns = {'Datetime':'calls_hour'}).reset_index()
    #splitting datetime column into multiple column so that data information at individual level can be processed
    #by model
    Calls['year']=pd.DatetimeIndex(Calls['Datetime']).year
    Calls['month']=pd.DatetimeIndex(Calls['Datetime']).month
    Calls['dayofmonth']=pd.DatetimeIndex(Calls['Datetime']).day
    Calls['hour']=pd.DatetimeIndex(Calls['Datetime']).hour
    Calls['dayofyear']=pd.DatetimeIndex(Calls['Datetime']).dayofyear
    Calls['dayofweek']=pd.DatetimeIndex(Calls['Datetime']).dayofweek

    # imputation of missing values in the weather dataframe
    df2['pres'].interpolate(method='linear', inplace=True)
    df2['wdir'].interpolate(method='linear', inplace=True)
    df2['coco'].interpolate(method='bfill', inplace=True)
    df2=df2.reset_index()
    df2['Datetime']=df2['time'].dt.strftime('%Y-%m-%d %H')
    df2=df2.drop(['time','prcp','snow','wpgt','tsun'], axis=1)
    
    # merge two dataframe to have our cleaned final data
    df=pd.merge(Calls,df2, how="inner", on=["Datetime"])
    
    return df