# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 00:59:16 2022

@author: tp3
"""

from datetime import datetime
from meteostat import Point, Daily, Hourly

# Get Weather data from meteostat from 2016
def load_weather_data(latitude,longitude,elevation):
    station = Point(latitude,longitude,elevation) 
    start = datetime(2016, 1, 1)
    end = datetime(2022, 10, 1)
    df = Hourly(station, start, end)
    df = df.fetch()
    
    return df