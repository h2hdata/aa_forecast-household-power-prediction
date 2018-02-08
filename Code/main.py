"""
# -*- coding: utf-8 -*-

#POC Code  :  Power prediction :  H2H DATA
This code is for perdiction cumsumption of household power.
Data: household power cunsumption data
Models: many models are used to get comparesion between them.


#Copyright@ H2H DATA

#The entire prcess occurs in seven stages-
# 1. DATA INGESTION
# 2. DATA ANALYSIS 
# 3. DATA MUNGING
# 4. DATA EXPLORATION
# 5. DATA MODELING
# 6. HYPER-PARAMETERS OPTIMIZATION
# 7. PREDICTION
# 8. VISUAL ANALYSIS
# 9. RESULTS


Used library
1. pandas
2. numpy
3. time
4. sklearn
5. matplotlib
6. statsmodels
"""


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error as mse
import model
import helper


########################################## data ingestion ###############################
'''
Data Description
columns
Name:Date; 					value: date
Name:Time; 					value: time
Name:Global_active_power; 	value: real
Name:Global_reactive_power;	value: real
Name:Voltage;				value: real
Name:Global_intensity;		value: real
Name:Sub_metering_1; 		value: real
Name:Sub_metering_2; 		value: real
Name:Sub_metering_3; 		value: real
'''
def read_data():
	data = helper.read_data()
	return data
####################################### data ingestion  ends ###############################


####################################### data exploration ###################################
"""
returns pandas data frame after imputation of missing value
"""
def missing(data):
	data = helper.fill_missing(data)
	return data


'''
returns pandas data frame after outlier treatment
'''
def outlier(data):
	cols = list(data.columns.values)
	a,b = data.shape
	for num in cols:
		if data[num].describe()[0] != a:
			mean1 = data[num].mean()
			data[num] = data[num].fillna(mean1)
			data[num] = helper.outlier_treat(data[num],a)
		else:
			data[num] = helper.outlier_treat(data[num],a)
	return data

########################################## data exploration ends #############################



########################################## modeling ##########################################
'''
call different model to forecate household power consumption 
1st method is only using forecast variable
2nd model uses all feature and combination of 2 models 
'''
def model1(data):
	# model.simple_models(data)
	model.combined_model(data)
########################################## modeling ends ######################################


if __name__ == '__main__':
	print 'data read'
	data = read_data()
	# data = data.head(1000)
	data= data.drop(['Date','Time'],axis = 1)
	print 'col correction'
	for col in data.columns:
	  data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0).astype(float)
	print 'missing'
	data = missing(data)
	print 'outlier'
	data = outlier(data)
	print 'model'
	model1(data)