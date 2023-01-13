# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#==============================================================================
#Librerias clásicas
import pandas as pd

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')

#==============================================================================
#Funciones

#Función que genera un nuevo dataset, que posee las ventas por día para una sublinea en un local de abastecimiento
def Data_ventas(df12,sublinea,local):
    
 #Se obtiene las ventas por día para la sublinea
 group_f12_sublinea = df12.groupby(['po_date']+['sublinea']).agg({'units': 'sum'}).reset_index()
 subgroup_f12_sublinea=group_f12_sublinea[(group_f12_sublinea['sublinea']==sublinea)].loc[:,['po_date','units']]
 subgroup_f12_sublinea.rename(columns={'units':f'{sublinea}'},inplace=True)
    
 #Se obtiene las ventas por día para sublinea-local
 group_f12_ls = df12.groupby(['po_date']+['facility_alias_id']+['sublinea']).agg({'units': 'sum'}).reset_index()
 subgroup_f12_ls=group_f12_ls[(group_f12_ls['facility_alias_id']==local) & (group_f12_ls['sublinea']==sublinea)].loc[:,['po_date','units']]
 subgroup_f12_ls.rename(columns={'units':f'{sublinea}-{local}'},inplace=True)
    
 #Generación dataset con ventas por local y sublinea
 Data_ventas= pd.DataFrame()
 dias=df12['po_date'].unique()
 Data_ventas['po_date']=dias 
    
 Data_ventas = pd.merge(left=Data_ventas,right=subgroup_f12_ls,how='left' ,left_on='po_date', right_on='po_date')
 Data_ventas = pd.merge(left=Data_ventas,right=subgroup_f12_sublinea,how='left' ,left_on='po_date', right_on='po_date')
        
 Data_ventas.fillna(0,inplace=True)
 Data_ventas["po_date"] = pd.to_datetime(Data_ventas["po_date"])
 Data_ventas.set_index('po_date',inplace=True)
 Data_ventas= Data_ventas.loc['2020':'2022']
 Data_ventas=Data_ventas.asfreq('1D')
 Data_ventas.fillna(0,inplace=True) 
    
 return Data_ventas

#Función que modifica el dataset de entrada, generando una ponderación diaria para cada local de abastecimiento en la sublinea respectiva
def Data_pond(data,sublinea,local):
 
 #Se genera la tabla Ponderacion
 Ponderacion=pd.DataFrame()
 Ponderacion[f'{sublinea}-{local}']=data[f'{sublinea}-{local}']/data[f'{sublinea}']
 Ponderacion.fillna(0,inplace=True)
 return Ponderacion


def Data_pond_v2(data,sublinea,local):
 
 #Se genera la tabla Ponderacion
 Ponderacion=pd.DataFrame()
 Ponderacion[f'{local}']=data[f'{local}']/data[f'{sublinea}']
 Ponderacion.fillna(0,inplace=True)
 return Ponderacion


#Función que cambia el valor de venta para BlackFridays y CyberDays por un promedio de n días anteriores
def rolling_mean_1(ventas_ls,dic_cyber,dic_blackfriday,sublinea,local):
    
 ventas_mean = ventas_ls.ewm(span=12,adjust=False).mean()  #Suavizamiento exponencial
 dias=ventas_ls.index.strftime('%Y-%m-%d').tolist()

 for i in range(0,len(dias)):
    cyber= dic_cyber.get(dias[i])
    blackfriday= dic_blackfriday.get(dias[i])
    if cyber== 1.0:
        ventas_ls[f'{sublinea}-{local}'][i] = round(ventas_mean[f'{sublinea}-{local}'][i])
        ventas_ls[f'{sublinea}'][i]= round(ventas_mean[f'{sublinea}'][i])
    if blackfriday==1.0:     
        ventas_ls[f'{sublinea}-{local}'][i] = round(ventas_mean[f'{sublinea}-{local}'][i])
        ventas_ls[f'{sublinea}'][i]= round(ventas_mean[f'{sublinea}'][i])
        
 return ventas_ls

#Se genera el dataset preprocesado
def generacion_dataset(df12,sublinea,locales,dic_cyber,dic_blackfriday):
 #Procesamiento y generación datasets
 data_ventas=pd.DataFrame()
 pond_ventas=pd.DataFrame()
    
 for local in locales:
  #Generacion dataset de demanda   
  data= Data_ventas(df12,sublinea,local)
 # data_original=data.copy
  data_s1=rolling_mean_1(data,dic_cyber,dic_blackfriday,sublinea,local) #Se elimininan peaks por cybers day y BlackFriday
  data_s2=round(data_s1.ewm(span=5,adjust=False).mean()) #Se suaviza la sublinea
  data_ventas[f'{local}']=data_s2[f'{sublinea}-{local}']
    
  #Visualizacion
#  fig, ax = plt.subplots(figsize=(20, 5))
#  data_original[f'{sublinea}-{local}'].plot(ax=ax, label='Demanda real para sublinea-local')
#  data_ventas[f'{local}'].plot(ax=ax,label='Demanda suavizada para sublinea-local')
#  plt.title(f' Gráfico {sublinea}-{local}')   
#  ax.legend();
    
  #Generacion dataset ponderaciones
  data_pond=Data_pond(data_s2,sublinea,local)
  pond_ventas[f'{local}']=data_pond[f'{sublinea}-{local}']

 #Se añade demanda sublinea
 data_ventas[f'{sublinea}']=data_s2[f'{sublinea}']
 #data_ventas['total']=data_ventas[f'{sublinea}']
 
 return data_ventas,pond_ventas

 #Visualizacion Sublinea
# fig, ax = plt.subplots(figsize=(20, 5))
# data_original[f'{sublinea}'].plot(ax=ax, label='Demanda real para sublinea')
# data_ventas[f'{sublinea}'].plot(ax=ax,label='Demanda suavizada para sublinea')
# plt.title(f' Gráfico {sublinea}')   
# ax.legend();
