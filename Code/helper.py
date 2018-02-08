import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error as mse


#function to read data
def read_data():
	complete_data = pd.read_csv('../Data/household_power.csv')
	return complete_data


#functio to find fill missing values
def fill_missing(data,method = 'mean'):
	if method == 'ffill' or method == 'pad':
		data = data.fillna(method = 'pad')

	if method == 'bfill':
		data = data.fillna(method = 'bfill')

	if method == 'mean':
		data = data.fillna(data.mean())

	if method == 0:
		data = data.fillna(0)
	return data
 
#functio to treat outlier
def outlier_treat(df,a):

	mean1 = df.mean()
	std1 = df.std()
	upper_lim = mean1 + 3*std1
	lower_lim = mean1 - 3*std1
	data = df.values
	for num in range(0,a):
		if data[num] >= upper_lim:
			data[num] = upper_lim
		else:
			data[num] = lower_lim
	return pd.DataFrame(data)



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i + look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)