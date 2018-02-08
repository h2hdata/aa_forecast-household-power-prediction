
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error as mse
import model
import helper
# import sklearn.preprocessing.normalize as norm
 # rective power will be removed as its not used and bounce back



# function only using single model
def simple_models(data):
	'''
	uses arima only on 'Global_active_power'
	'''
	data = data['Global_active_power']
	data = scale(data.values)

	#####
	#rolling mean
	ma = pd.rolling_mean(data,12)
	ew_avg = pd.ewma(data,halflife= 12)
	#ARIMA
	model = ARIMA(data,order=(2,1,2))
	result = model.fit(disp=-1)
	plt.plot(data)
	plt.plot(result.fittedvalues, color = 'red')
	plt.show()
	result =  result.predict()
	print 'mean squared error',mse(data[1:],result)
	print 'rolling mean:',ma[10:20]
	print 'exponential weighted moving average',ew_avg[10:20]
	print 'ARIMA',result[10:20]

	pd.DataFrame(ma,columns = ['Forcast']).to_csv('../Output/rolling_mean',index = False)
	pd.DataFrame(ew_avg,columns = ['Forcast']).to_csv('../Output/exponential_weighted_moving_average',index = False)
	pd.DataFrame(result,columns = ['Forcast']).to_csv('../Output/ARIMA',index = False)
    




#function using many models
def combined_model(data):
	'''
	uses arima and regression  
	'''
	cols = list(data.columns.values)
	a,b = data.shape
	Y = data.pop("Global_active_power")
	X = data

	model = LinearRegression()
	data1 = data['Global_reactive_power'].values
	data2 = data['Voltage'].values
	data3 = data['Global_intensity'].values
	data4 = data['Sub_metering_1'].values
	data5 = data['Sub_metering_2'].values
	data6 = data['Sub_metering_3'].values
	look_back = 1440
	full_forecast = pd.DataFrame()
	indi_forecast = []
	dataframes = [data1,data2,data3,data4,data5,data6]

	i = 0 
	try:
		for num in dataframes:
			data = num
			print data
			fore = pd.DataFrame()
			for num in range(0,100):
				try:
					print 'arima'
					model = ARIMA(data, order=(2,1,2))
					model_fit = model.fit(disp=0)
					print 'forecast'
					output = model_fit.forecast()
					yhat = output[0]
					indi_forecast.append(yhat)
				except:
					break
			i= i+1
			fore = pd.DataFrame(indi_forecast)
			full_forecast = pd.concat([full_forecast, fore], axis=1)

		full_forecast = full_forecast.dropna()
		model.fit(X,Y)
		z = model.predict(full_forecast)
		print 'mse',mse(data[1:],z)

	except:
		print 'unable to predict long sequence'


