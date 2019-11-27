import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

df_buildings=pd.read_csv('building_metadata.csv',header=0
                        #,skiprows=1,header=None
                        #,names=['passenger_id','is_survivor','ticket_class','name','sex','age','nb_siblings_spouses','nb_parents_children','ticket_no', 'fare', 'cabin_no','embarkation_port']

                        ,index_col=['building_id'])

weather_train=pd.read_csv('weather_train.csv')

building_data=pd.read_csv('building_metadata.csv')
building_data.head(5)
building_data.shape
building_data.columns


weather_train=pd.read_csv('weather_train.csv')
weather_train.head(5)

weather_train.shape
weather_train.columns


train=pd.read_csv('train.csv')
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

print(data_meassured['meter_reading'].head())

building_data['site_id'].unique()
(building_data['primary_use'].unique()).size
building_data['year_built'].unique()

data_meassured.tail(5)


#remove any dups
print(len(data_meassured.index))
df_buildings = data_meassured.drop_duplicates()
print(len(data_meassured.index))

print('Determine % of useful data per column')
for col in data_meassured.columns:
    print(col + ' : ' + (data_meassured[col].count() * 100 / len(data_meassured.index)).astype(str))


print('Floor count has 0% presence so we should drop')
data_meassured.drop(columns=['floor_count'], axis=1, inplace=True)

print('Any missing values ?')
print(data_meassured.isna().any())


print('Precipitation depth is negative sometimes. I would normalize this to 0')
data_meassured.loc[data_meassured['precip_depth_1_hr'] < 0, 'precip_depth_1_hr'] = 0

print('Convert year_built to int')
data_meassured = data_meassured.astype({'year_built': int})

plt.figure()

print('Visually examine for outliers. I have a function that can clean them out later')
boxplot = data_meassured.boxplot(['meter'])

print('Something seems off with meter value. It is always zero')

plt.show()
boxplot = data_meassured.boxplot(['air_temperature'])

plt.show()
exit(0)


data_meassured.to_csv('data_meassured_cleaned.csv')
# Can we guess at the cloud coverage from the precipitation ?

exit(0)

# Thoughts

# engineer a building_age date from year_built

