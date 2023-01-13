# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:33:54 2022

@author: ignac
"""

#==============================================================================
#Librerias clásicas
import pandas as pd
import matplotlib.pyplot as plt

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')


#==============================================================================
#Funciones

#Genera resultados de la métrica MAPE 
def mape(y_true,y_pred):
 n=len(y_true)
 
 suma=0
 for i in range(0,n):    
    if y_true[i] !=0:
     suma +=(abs(y_true[i]-y_pred[i]))/(abs(y_true[i]))
    else:
     suma +=(abs(1-(y_pred[i]+1)))/1
    
 error_mape=suma*100/n
 #print(f"Error de test (MAPE): {error_mape} para la {sublinea}" )
 #print('========================================================')
    
 return error_mape

#Genera un WMAPE para la sublinea considerando la ponderación según el peso de casa clase en la sublinea
def wmape(sublinea,locales,preds_m2,mape_m2,dic_result):

 wmape_error=0

 print(f'Error para la sublinea {sublinea}: \n')
# file.write(f'Error para la sublinea {sublinea}:' + os.linesep)
 for i in range(0,len(locales)):
#    datos=preds_m2[i]
    prom= sum(preds_m2[i])/len(preds_m2[i])
    error_clase=abs(prom)*mape_m2[i]
    print(f'Local {locales[i]}: Pond {abs(prom)}, mape {mape_m2[i]}, error: {error_clase} \n')
#    file.write(f'Local {locales[i]}: Ponderación->{abs(prom)}, MAPE->{mape_m2[i]}, Error->{error_clase}' + os.linesep)
    wmape_error+=error_clase
    
    dic_result[f'{locales[i]}'][-1][0]=error_clase
    
 return wmape_error,dic_result

#Función que obtiene la estimación de ventas a partir de las estimaciones de las ponderaciones
def post_procesamiento(sublinea,locales,ponds,data_eval,data_test,preds_ponderaciones,dic_result):
 
 mape_clase=[]

 data_estimaciones=pd.DataFrame()
 data_estimaciones[f'{sublinea}']=[0]*30


 for local in locales: 
  data_estimaciones[f'{local}']=round(data_eval[:30][f'{sublinea}']*preds_ponderaciones[:30].reset_index()[f'{local}'])   
    
  #Visualización demandas real y estimada   
#  fig, ax = plt.subplots(figsize=(20, 5))
#  data_test[:30][f'{local}'].plot(ax=ax, label='Demanda real clase')
#  data_estimaciones[f'{local}'].plot(ax=ax,label='Demanda estimada clase')
#  plt.title(f' Gráfico {local}')   
#  ax.legend();
 
  #Error para las clases
  y_true = data_test[:30][f'{local}']
  y_pred = data_estimaciones[f'{local}']
  dic_result[f'{local}'].append([0,list(preds_ponderaciones[:30][f'{local}'])])
  error_m= mape(y_true,y_pred) 
  mape_clase.append(error_m) 
 
  data_estimaciones[f'{sublinea}'] += data_estimaciones[f'{local}'] 
    
 mape_pond,dic_result=wmape(sublinea,locales,ponds,mape_clase,dic_result)

 print(f'MAPE ponderado es: {mape_pond}')
 #file.write(f'MAPE ponderado->{mape_pond}' + os.linesep)

 #Reconstrucción Sublinea   
 #fig, ax = plt.subplots(figsize=(20, 5))
 #data_test[:30][f'{sublinea}'].plot(ax=ax, label='Demanda real sublinea')
 #data_eval[:30][f'{sublinea}'].plot(ax=ax, label='Demanda estimada Forecast')
 #data_estimaciones[f'{sublinea}'].plot(ax=ax,label='Reconstrucción sublinea')
 #plt.title(f' Gráfico {sublinea}')   
 #ax.legend();

 y_true = data_test[:30][f'{sublinea}']
 y_pred = data_estimaciones[f'{sublinea}']
 error_m= mape(y_true,y_pred)
 print(f'Error reconstrucción sublinea es: {error_m}')
 #file.write(f'MAPE reconstrucción->{error_m}' + os.linesep)

 return data_estimaciones,dic_result