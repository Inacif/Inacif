# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:39:50 2022

@author: ignac
"""
#=======================
#Librerías

import pandas as pd
import matplotlib.pyplot as plt
from Generacion_dataset_a3 import *
from sktime.forecasting.model_selection import ForecastingGridSearchCV,ExpandingWindowSplitter
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA

#===========================
#Funciones
def train_test_data(data,n_val,n_test): 
#Se generan datos de entrenamiento y test
 data_train= data[:-(n_val+n_test)]
 data_val= data[-(n_val+n_test):-n_test]
 data_test= data[-n_test:]

 return data_train,data_val,data_test


#============================
# Obtención de hiperparametros

df12=pd.read_csv("C:\\Users\ignac\OneDrive\Escritorio\df12.csv")

#Obtener diccionarios
mini=df12.groupby('po_date').agg('mean').loc[:,['flag_cyber','flag_blackfriday']]
dic=mini.to_dict()
dic_cyber=dic['flag_cyber']
dic_blackfriday=dic['flag_blackfriday']

sublinea='J0907'
locales=[9949]
n_test=90
n_val=90
steps=30

data_ventas,pond_ventas= generacion_dataset(df12, sublinea, locales, dic_cyber, dic_blackfriday)

#Visualización demandas real y estimada   
fig, ax = plt.subplots(figsize=(20, 5))
pond_ventas[f'{locales[0]}'].plot(ax=ax, label='Demanda real clase')
plt.title(f' Gráfico {sublinea}-{locales[0]}')   
ax.legend(); 

pond_train,pond_val,pond_test=train_test_data(pond_ventas, n_val, n_test)


fh = list(range(1,31))
cv = ExpandingWindowSplitter(fh=fh, initial_window=len(pond_train), step_length=30)


# Hiperparametros para modelo Exponential Smoothing
print('Búsqueda de hiperparámetros para modelo Exponential Smoothing')
modelo_Exponential=ExponentialSmoothing()

forecaster = modelo_Exponential
param_grid_Exp = {"trend" : ["add"],
                  "seasonal" : ["add"],
                  "smoothing_level" : [0.05,0.2,0.5,0.9],
                  "smoothing_trend" : [0.01,0.03,0.05,0.1,0.5,0.9],
                  "smoothing_seasonal" : [0.01,0.03,0.05,0.1,0.5,0.9]
                 }

gscv_Exp = ForecastingGridSearchCV(
    forecaster=forecaster,
    param_grid=param_grid_Exp,
    cv=cv)
gscv_Exp.fit(pond_ventas[f'{locales[0]}'][:-90])

#y_pred = gscv.predict(fh)

best_params=gscv_Exp.best_params_
print('Los mejores hiperparametros son: ', best_params)

# Hiperparametros para modelo ARIMA
print('Búsqueda de hiperparámetros para modelo ARIMA')
modelo_arima=StatsForecastAutoARIMA()

forecaster = modelo_arima
param_grid_arima = {"d" : [1,2,3],
 #                   "max_p" : [2,5,8,10,20],
 #                   "max_q" : [2,10,20,50],
                    }

gscv_arima = ForecastingGridSearchCV(
    forecaster=forecaster,
    param_grid=param_grid_arima,
    cv=cv)
gscv_arima.fit(pond_ventas[f'{locales[0]}'][:-90])

best_params=gscv_arima.best_params_
print('Los mejores hiperparametros son: ', best_params)













