# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:50:52 2022

@author: ignac
"""
#Librerias
#=====================================
import pandas as pd
import joblib
from Generacion_dataset_a3 import *
from Modelos_a3 import *
from Post_procesamiento_a3 import *

#Funciones
#=============================================
def train_test_data(data,n_val,n_test): 
#Se generan datos de entrenamiento y test
 data_train= data[:-(n_val+n_test)]
 data_val= data[-(n_val+n_test):-n_test]
 data_test= data[-n_test:]

 return data_train,data_val,data_test

#Obtiene las ponderacionesde de las clases en la sublinea
def ponds_subcategorias(data,n_test,sublinea,locales):
 
 ponds=[]
 data_sin_test=data[:-n_test]
 data_sin_ceros= data_sin_test[data_sin_test[sublinea]!=0]

 for local in locales:
  pond_ventas_sinceros=Data_pond_v2(data_sin_ceros,sublinea,local)
  ponds.append(pond_ventas_sinceros[f'{local}'].tolist())
  
 return ponds 


#Generación de diccionario para guardar resultados
def dic_resultados(locales,n_test,steps):
  len_fit= round(n_test/steps)

  lista_dic=[]
  for i in range(0,len_fit):
   dic_result={}
   for local in locales:
     dic_result[f'{local}']=[]
   lista_dic.append(dic_result)
   
  return lista_dic

#======================================

#Diccionario de hiperparámetros

params_es={0:[0.9,0.01,0.04],1:[0.05,0.07,0.5],2:[0.5,0.01,0.5],3:[0.05,0.01,0.01]}
params_arima={0:[2],1:[2],2:[2]}



#===============================================
df12=pd.read_csv("C:\\Users\ignac\OneDrive\Escritorio\df12.csv")

#Modelo cluster
modelo_cluster=joblib.load('C:\\Users\ignac\OneDrive\Escritorio\Cluster_local_etapa1.pkl')

#Obtener diccionarios
mini=df12.groupby('po_date').agg('mean').loc[:,['flag_cyber','flag_blackfriday']]
dic=mini.to_dict()
dic_cyber=dic['flag_cyber']
dic_blackfriday=dic['flag_blackfriday']

sublineas=['J0404'] #Seleccionar sublineas a evaluar
locales=[70,2000,3660,7200,9990] #Seleccionar a locales a considerar
n_test=90
n_val=90
steps=30

for sublinea in sublineas:
 data_ventas,pond_ventas= generacion_dataset(df12, sublinea, locales, dic_cyber, dic_blackfriday)

 lista_dic=dic_resultados(locales, n_test, steps)
 ponds=ponds_subcategorias(data_ventas, n_test, sublinea, locales)

 pond_train,pond_val,pond_test=train_test_data(pond_ventas, n_val, n_test)
 ventas_train,ventas_val,ventas_test=train_test_data(data_ventas, n_val, n_test)

 data_apertura= ventas_test

 lista_dic=Modelo_ExponentialSmoothing(pond_ventas,params_es, n_test, steps, sublinea, locales, data_apertura, ventas_test, ponds,lista_dic,modelo_cluster)
 lista_dic=Modelo_ARIMA(pond_ventas, n_test, steps, sublinea, locales, data_apertura, ventas_test, ponds,lista_dic)
 lista_dic=Modelo_Hibrido(n_test,steps,0.6,0.4,0,1,sublinea,locales,data_apertura,ventas_test,ponds,lista_dic)
