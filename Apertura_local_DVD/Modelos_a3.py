# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:20:15 2022

@author: ignac
"""
# ==============================================================================
#Librerias clásicas
import time
import pandas as pd
import numpy as np

# Librerias Forecasting
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from Post_procesamiento_a3 import *

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')


#=========================================================
#Modelos de apertura

# Modelo Exponential Smoothing
def Modelo_ExponentialSmoothing(data,params_es,n_test,steps,sublinea,locales,data_eval,data_test,ponds,lista_dic,model):
   
 
 len_fit=round(n_test/steps)
    
 for i in range(0,len_fit): 
  preds_es=pd.DataFrame()    
    
  print('------------------------------------------------------------------------------------')
  print(f'Resultados modelo de apertura Exponential Smoothing para el mes {i+1}') 
  inicio=time.time()
  
  for local in locales:
   data_array =np.array(data[f'{local}'].values).reshape(1,-1)
   cluster=model.predict(data_array)[0]   
   print(f'{local} es cluster {cluster}')    
      
   forecaster = ExponentialSmoothing(trend='add', seasonal='add',smoothing_level=params_es.get(cluster)[0],smoothing_trend=params_es.get(cluster)[1]) #,smoothing_seasonal=params_es.get(cluster)[2])  
   forecaster.fit(data[f'{local}'][:-(n_test-steps*i)])

   y_pred_expos = forecaster.predict(fh=list(range(1,31)))  

   preds_es[f'{local}']= y_pred_expos

  est_expos,lista_dic[i]=post_procesamiento(sublinea,locales,ponds,data_eval[-(n_test-steps*i):].reset_index(),data_test[-(n_test-steps*i):].reset_index(),preds_es,lista_dic[i])

  fin=time.time()
  duracion= fin-inicio
  print(f'El tiempo de ejecución fue de {duracion}')


 return lista_dic


# Modelo ARIMA
def Modelo_ARIMA(data,n_test,steps,sublinea,locales,data_eval,data_test,ponds,lista_dic):
    
 len_fit=round(n_test/steps)
    
 for i in range(0,len_fit): 
  print('------------------------------------------------------------------------------------')
  print(f'Resultados modelo de apertura ARIMA para el mes {i+1}') 
  inicio=time.time()
 
  forecaster = StatsForecastAutoARIMA(d=2, max_p=10, max_q=12) 
  forecaster.fit(data[:-(n_test-steps*i)])

  y_pred_expos = forecaster.predict(fh=list(range(1,31)))  

  est_expos,lista_dic[i]=post_procesamiento(sublinea,locales,ponds,data_eval[-(n_test-steps*i):].reset_index(),data_test[-(n_test-steps*i):].reset_index(),y_pred_expos.reset_index(),lista_dic[i])

  fin=time.time()
  duracion= fin-inicio
  print(f'El tiempo de ejecución fue de {duracion}')


 return lista_dic


# Modelo híbrido
def Modelo_Hibrido(len_test,steps,pond1,pond2,a,b,sublinea,locales,data_eval,data_test,ponds,lista_dic):
    
 len_fit=round(len_test/steps)
    
 for i in range(0,len_fit):
  print('------------------------------------------------------------------------------------')
  print(f'Resultados modelo de apertura híbrido para el mes {i+1}')    
  inicio=time.time()
  
  preds_hibrido=pd.DataFrame(index=range(30))

  for local in locales:
   preds_hibrido[f'{local}']= [0]*30
   for j in range(0,len(lista_dic[i][f'{local}'][a][1])):
    preds_hibrido[f'{local}'][j]=pond1*lista_dic[i][f'{local}'][a][1][j]+ pond2*lista_dic[i][f'{local}'][b][1][j]
   

  est_hibrido,lista_dic[i]=post_procesamiento(sublinea,locales,ponds,data_eval[-(len_test-steps*i):].reset_index(),data_test[-(len_test-steps*i):].reset_index(),preds_hibrido,lista_dic[i])

  fin=time.time()
  duracion= fin-inicio
  print(f'El tiempo de ejecución fue de {duracion}')
 
 return lista_dic

