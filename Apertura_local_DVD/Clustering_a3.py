# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:44:40 2022

@author: ignac
"""

#==============================================================================
#Librerias clásicas
import pandas as pd
import numpy as np
import joblib
import collections
from Generacion_dataset_a3 import *

# Librerias Clustering
from tslearn.clustering import TimeSeriesKMeans, silhouette_score

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')

#================================================================================
#Funciones

#Búsqueda de los mejores parámetros para los cluster
def GridSearch_cluster(pond_ventas,clusters,metricas,iteraciones):
 for cluster in clusters:
  for metrica in metricas:
   for iteracion in iteraciones:   
    data_array = np.array(pond_ventas.T.drop('po_date').values)
    model = TimeSeriesKMeans(n_clusters=cluster, metric=metrica, max_iter=iteracion)
    model.fit(data_array)
    locales_list = pond_ventas.T.drop('po_date').index.tolist()
    labels=model.labels_
    print(f'El error para {cluster} clusters,{metrica} y {iteracion} es: {silhouette_score(data_array, labels, metric="euclidean") }')
 

#Función que agrupa series de tiempo en n clusters
def agrupacion_clusters(pond_ventas,cluster,metrica,iteracion):
    
 # Clustering etapa 1   
 print('Inicia proceso de entrenamiento de TimeSeriesKmeans para etapa 1')   
 data_array = np.array(pond_ventas.T.drop('po_date').values)
 model = TimeSeriesKMeans(n_clusters=cluster[0], metric=metrica[0], max_iter=iteracion)
 model.fit(data_array)
 locales_list = pond_ventas.T.drop('po_date').index.tolist()
 y=model.predict(data_array)
 joblib.dump(model, 'C:\\Users\ignac\OneDrive\Escritorio\Cluster_local_etapa1.pkl')
 conteo= collections.Counter(y)
 print(conteo)
 print('Fin de generación del modelos de agrupación para la etapa 1')
 
# max_value= max(dic_conteo)
# min_value= min(dic_conteo)
# 
# if (max_value - min_value) > 200: 
#  # Clustering etapa 2
#  lista=[]
#  for i in range(0,len(y)):
#    if y[i]==max_value:
#        lista.append(locales_list[i])
        
#  new_data=pond_ventas.loc[:,lista]

#  print('Inicia proceso de entrenamiento de TimeSeriesKmeans para etapa 2')   
#  data_array = np.array(new_data.T.drop('po_date').values)
#  model = TimeSeriesKMeans(n_clusters=cluster[1], metric=metrica[1], max_iter=iteracion)
#  model.fit(data_array)
#  locales_list = new_data.T.drop('po_date').index.tolist()
#  y=model.predict(data_array)
#  joblib.dump(model, 'C:\\Users\ignac\OneDrive\Escritorio\Cluster_local_etapa2.pkl')
#  conteo= collections.Counter(y)
#  print(conteo)
#  print('Fin de generación del modelos de agrupación para la etapa 2')
 
 return locales_list,y

#=============================================================================
#Generacion de dataset para entrenar el modelo (se debe definir la cantidad de sublineas y locales a usar para el entrenamiento)
#df12=pd.read_csv("C:\\Users\ignac\OneDrive\Escritorio\df12.csv")

#Obtener diccionarios
#mini=df12.groupby('po_date').agg('mean').loc[:,['flag_cyber','flag_blackfriday']]
#dic=mini.to_dict()
#dic_cyber=dic['flag_cyber']
#dic_blackfriday=dic['flag_blackfriday']

#sublineas=['J0404','J0401'] #Seleccionar sublineas a usar
#locales=sorted([2000,3660]) #Seleccionar locales a usar 


#Procesamiento y generación datasets
#data_ventas=pd.DataFrame()
#pond_ventas=pd.DataFrame()

#for sublinea in sublineas:
# for local in locales:
#  #Generacion dataset de demanda   
#  data= Data_ventas(df12,sublinea,local)
#  data_original=data.copy()
#  data_s1=rolling_mean_1(data,dic_cyber,dic_blackfriday,sublinea,local) #Se elimininan peaks por cybers day y BlackFriday
#  data_s2=round(data_s1.ewm(span=5,adjust=False).mean()) #Se suaviza la sublinea
#  data_ventas[f'{sublinea}-{local}']=data_s2[f'{sublinea}-{local}']
    
  #Generacion dataset ponderaciones
#  data_pond=Data_pond(data_s2,sublinea,local)
#  pond_ventas[f'{sublinea}-{local}']=data_pond[f'{sublinea}-{local}']

#pond_ventas.reset_index(inplace=True)

pond_ventas=pd.read_csv("C:\\Users\ignac\OneDrive\Escritorio\Pond_locales.csv")

#===================================================================
#Búsqueda de los mejores parametros
#clusters=[2,3,4,5,6,7,8,10]
#metricas=['euclidean']
#iteraciones=[30]

#GridSearch_cluster(pond_ventas, clusters, metricas, iteraciones)

#=============================================================
#Agrupamiento de series de tiempo en clusters
cluster=[4]
metrica=['euclidean']
iteracion=30

# Clustering etapa 1
locales_list,y=agrupacion_clusters(pond_ventas,cluster,metrica,iteracion)






