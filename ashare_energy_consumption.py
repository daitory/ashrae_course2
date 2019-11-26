import pandas as pd 
import numpy as np 


building_data=pd.read_csv('D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae-energy-prediction\\building_metadata.csv')
building_data.head(5)
building_data.shape
building_data.columns


weather_train=pd.read_csv('D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae-energy-prediction\\weather_train.csv')
weather_train.head(5)

weather_train.shape
weather_train.columns


train=pd.read_csv('D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae-energy-prediction\\train.csv')
train.head(5)

train.shape


train.columns

train.iloc[:,0].unique

prueba_train1=train[train.iloc[:,0]==0]

prueba_train1


building_data.columns

building_data['primary_use'].unique()
building_data['year_built'].unique()

data_meassured=pd.merge(pd.merge(prueba_train1,building_data,how='left',on='building_id'),weather_train,how='left',on=['site_id','timestamp'])

data_meassured['meter_reading']

building_data['site_id'].unique()
(building_data['primary_use'].unique()).size
building_data['year_built'].unique()

data_meassured.tail(5)