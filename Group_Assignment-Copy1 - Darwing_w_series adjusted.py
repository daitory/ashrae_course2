#!/usr/bin/env python
# coding: utf-8

# # GROUP ASSIGNMENT - STATISTICS FOR DATA SCIENCE - FALL 2019 #

# ### Group Members: Darwing Cara, David Kobayashi, Eric Koritko, Stefan Lazarevic, Patrick McDonnell ###

# We have selected to work on a data set from ASHRAE (a technical society for heating, ventilation, and air conditioning in buildings). The dataset can be found here: https://www.kaggle.com/c/ashrae-energy-prediction
# 
# The goal of this exercise is to use the dataset to build a model that can predict energy use in a building based on that building's features (building use type, outdoor air conditions, etc).

# ## 1.0 - IMPORT PACKAGES ##



import pandas as pd
import numpy  as np
import math as math
import datetime
import time
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
#import statsmodels as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import statsmodels

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# ## 1.1 - DEFINE USEFUL FUNCTIONS




def compress_dataframe(dataframe):

    start = time.time()
    
    print('Dataframe was {:.2f} MB'.format(startingMemoryUsage))

    for column in dataframe.columns:
        columnDataType = dataframe[column].dtype
        
        if column == 'timestamp':
            dataframe[column] = pd.to_datetime(dataframe[column], format='%Y-%m-%d %H:%M:%S')
        
        elif columnDataType != object:
            columnMin = dataframe[column].min()
            columnMax = dataframe[column].max()

            if str(columnDataType)[:3] == 'int':
                if columnMin > np.iinfo(np.int8).min and columnMax < np.iinfo(np.int8).max:
                    dataframe[column] = dataframe[column].astype(np.int8)
                elif columnMin > np.iinfo(np.int16).min and columnMax < np.iinfo(np.int16).max:
                    dataframe[column] = dataframe[column].astype(np.int16)
                elif columnMin > np.iinfo(np.int32).min and columnMax < np.iinfo(np.int32).max:
                    dataframe[column] = dataframe[column].astype(np.int32)
                elif columnMin > np.iinfo(np.int64).min and columnMax < np.iinfo(np.int64).max:
                    dataframe[column] = dataframe[column].astype(np.int64)  
            else:
                if columnMin > np.finfo(np.float16).min and columnMax < np.finfo(np.float16).max:
                    dataframe[column] = dataframe[column].astype(np.float16)
                elif columnMin > np.finfo(np.float32).min and columnMax < np.finfo(np.float32).max:
                    dataframe[column] = dataframe[column].astype(np.float32)
                else:
                    dataframe[column] = dataframe[column].astype(np.float64)
        else:
            dataframe[column] = dataframe[column].astype('category')

    endingingMemoryUsage = dataframe.memory_usage().sum() / 1024**2

    print('Dataframe is now: {:.2f} MB'.format(endingingMemoryUsage))
    
    print("Time to reduce dataframe size:",round((time.time()-start)/60, 2), 'minutes.')





# Given a building_id, a meter type, and a date range, plot the energy use

def plotData(df, building_id, meter, name, start_day, end_day):
  
    df_plt = df[(df['building_id'] == building_id) & (df['meter'] == meter)]
    df_plt = df_plt[(df_plt['timestamp'] > start_day) & (df_plt['timestamp'] < end_day)]
    
    plt.figure(figsize=(20,2))
    plt.plot(df_plt['timestamp'], df_plt['meter_reading'])
    plt.xlabel('Time')
    plt.ylabel('Meter Reading')
    plt.title(name)
    plt.show()




# Given a zone_id and a date range, plot the weather

def plotWeather(df, zone_id, name, start_day, end_day, parameter):

    df_plt = df[df['site_id'] == zone_id]
    df_plt = df_plt[(df_plt['timestamp'] > start_day) & (df_plt['timestamp'] < end_day)]
    
    plt.figure(figsize=(20,2))
    plt.plot(df_plt['timestamp'], df_plt[parameter])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(name)
    plt.show()




# Given a building_id, plot energy use (all three meters, if available) and weather data

def display_energy_use(df, building_id, start_day, end_day):

    print("Displaying data for Building", building_id, "between:", start_day, "and", end_day, '\n')
    
    print("Building's primary use:", df.loc[df['building_id'] == building_id, 'primary_use'].iloc[0])
    
    print("Building size:", df.loc[df['building_id'] == building_id, 'square_feet'].iloc[0], 'square feet')
      
    for meter_type in df[df['building_id'] == building_id].groupby('meter').meter.unique():
        if(meter_type == elec_meter):
            plotData(df, building_id, elec_meter, "Electricity Meter Data", start_day, end_day)
        if(meter_type == chw_meter):
            plotData(df, building_id, chw_meter,"Chilled Water Meter Data", start_day, end_day)
        if(meter_type == steam_meter):
            plotData(df, building_id, steam_meter, "Steam Meter Data", start_day, end_day)
        if(meter_type == hw_meter):
            plotData(df, building_id,hw_meter, "Hot Water Meter Data", start_day, end_day)            
            
    plotWeather(df, df.loc[df['building_id'] == building, 'site_id'].iloc[0], "Weather Data - Air Temperature", start_day, end_day, 'air_temperature')
    plotWeather(df, df.loc[df['building_id'] == building, 'site_id'].iloc[0], "Weather Data - Cloud Coverage", start_day, end_day, 'cloud_coverage')
    plotWeather(df, df.loc[df['building_id'] == building, 'site_id'].iloc[0], "Weather Data - Dewpoint Temperature", start_day, end_day, 'dew_temperature')
    plotWeather(df, df.loc[df['building_id'] == building, 'site_id'].iloc[0], "Weather Data - Precipitation Depth", start_day, end_day, 'precip_depth_1_hr')
    plotWeather(df, df.loc[df['building_id'] == building, 'site_id'].iloc[0], "Weather Data - Sea Level Pressure", start_day, end_day, 'sea_level_pressure')
    plotWeather(df, df.loc[df['building_id'] == building, 'site_id'].iloc[0], "Weather Data - Wind Direction", start_day, end_day, 'wind_direction')
    plotWeather(df, df.loc[df['building_id'] == building, 'site_id'].iloc[0], "Weather Data - Wind Speed", start_day, end_day, 'wind_speed')


# ## 1.2 - DEFINE USEFUL CONSTANTS

def forecast_performance(original_data,predicted):
    MAE=np.mean(abs(original_data.values-predicted.values))
    MAPE=np.mean(abs((predicted.values/original_data.values)-1))
    RMSLE=(np.sum(np.log(predicted.values+1)-np.log(original_data.values+1))**2)/len(original_data.values)

    return([MAE,MAPE,RMSLE])




elec_meter = 0
chw_meter = 1
steam_meter = 2
hw_meter = 3


# # 2.0 - LOAD DATASET INTO MEMORY




start = time.time()

df = pd.read_csv(filepath_or_buffer='D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae original\\train.csv', sep=',', low_memory=False)

print('Time to read the CSV file into a dataframe: ',round((time.time()-start), 2), 'seconds \n')

startingMemoryUsage = df.memory_usage().sum() / 1024**2

print('Dataframe size is {:.2f} MB'.format(startingMemoryUsage))





df.info()





df.head()





#compress_dataframe(df)





start = time.time()

df_weather = pd.read_csv(filepath_or_buffer='D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae original\\weather_train.csv', sep=',', low_memory=False)

print('Time to read the CSV file into a dataframe: ',round((time.time()-start), 2), 'seconds \n')

startingMemoryUsage = df_weather.memory_usage().sum() / 1024**2

print('Dataframe size is {:.2f} MB'.format(startingMemoryUsage))





df_weather.info()





df_weather.head()





#compress_dataframe(df_weather)





start = time.time()

df_building_metadata = pd.read_csv(filepath_or_buffer='D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae original\\building_metadata.csv', sep=',', low_memory=False)

print('Time to read the CSV file into a dataframe: ',round((time.time()-start), 2), 'seconds \n')

startingMemoryUsage = df_weather.memory_usage().sum() / 1024**2

print('Dataframe size is {:.2f} MB'.format(startingMemoryUsage))





df_building_metadata.info()





df_building_metadata.head()





#compress_dataframe(df_building_metadata)


# # 3.0 - CLEAN THE DATASET

# ## 3.1 - CHECK FOR DUPLICATE ROWS




df[df.duplicated()].info()


# ## 3.2 - CHECK FOR NULL VALUES IN THE MAIN DATASET AND WEATHER DATASET




# Show how many nulls there are in the main dataset (expressed as a percentage)
print(round(df.isna().sum().sort_values(ascending=False) / len(df.index) * 100, 3))





# Show how many nulls there are in the weather dataset (expressed as a percentage)
print(round(df_weather.isna().sum().sort_values(ascending=False) / len(df_weather.index) * 100, 3))






# we have to be careful with this step because 2 close registers  could have very distinct behaviors

# Fill nulls with linear interpolation
df_weather = df_weather.interpolate()
df_weather = df_weather.fillna(0)

# Double check how many nulls are in the feature (should be 0 now)
print(round(df_weather.isna().sum().sort_values(ascending=False) / len(df_weather.index) * 100, 3))





# Show how many nulls there are in the metadata dataset (expressed as a percentage)
print(round(df_building_metadata.isna().sum().sort_values(ascending=False) / len(df_building_metadata.index) * 100, 3))


# ## 3.3 - FILTER OUT OFFICE BUILDINGS AND MERGE DATASETS




# count the number of different building types in the dataset
# df_building_metadata['primary_use'].value_counts()

# filter out the data that relates to office buildings

df_building_metadata.columns

df_building_metadata = df_building_metadata[df_building_metadata['primary_use'] == 'Office']

# merge this data with the main dataframe, so that we only have meter readings for office buildings
df = df.merge(df_building_metadata, on='building_id', how='inner')

# merge weather data with the main dataframe, so that all the information is in one place
df = df.merge(df_weather, on=['site_id', 'timestamp'], how='inner')

# check dataframe
df.head()


# ## 3.4 - CHECK FOR NULL VALUES IN THE METADATA DATASET




print(round(df.isna().sum().sort_values(ascending=False) / len(df.index) * 100, 3))





# There's no great way to fill the floor_count and year_built columns
# Since there are a large number of nulls in the floor_count and year_built features, we will drop these columns

df = df.drop('floor_count', axis=1)
df = df.drop('year_built', axis=1)

df.head()


# ## 3.5 - USE THIS SECTION TO VIEW METER DATA AND WEATHER DATA TOGETHER




# Change the building_id, the start date, and the end date. This will plot everything, in increments of days

# --------------------------#
building = 9
start_day = "2016-07-01"
end_day = "2016-07-02"
# --------------------------#

if building in df['building_id'].unique():
    display_energy_use(df, building, start_day, end_day)  


# ## 4 - PERFORM VARIABLE SELECTION

# ## 4.1 - EXAMINE PRIMARY USE COLUMN:





print(df['primary_use'].describe())

# We can see that the Primary Use column, has only a single unique value (Office).

# This is the result of the data cleaning phase, that resulted in extracting the results for the office buildings only.

# To that end, this feature is not very useful for us, since all rows represent the Office Buildings data, and we can remove it
# from the Data Set:

#df = df.drop(['primary_use'], axis=1)


# ## 4.2 - EXAMINE ENERGY CONSUMPTION OVER THE YEAR:




# First, let's transform the meter column s.t. it has actual energy type name as opposed to different numbers. This will make
# the Data Set more clear.

# Every row in the column has a different number representing the different energy type:

# 0 - electricity
# 1 - chilledwater
# 2 - steam
# 3 - hotwater

df['meter'].replace( { 0: 'Electricity', 1: 'ChilledWater' ,2: 'Steam', 3: 'HotWater' }, inplace=True )


# Now, examine the timestamp column:

print(df['timestamp'].head(10))

# Extract the year from each timestamp:

df['year'] = pd.to_datetime(df['timestamp']).dt.year

df['timestamp']=pd.to_datetime(df['timestamp'])
#.dt.year.astype('uint8')

# Examine the different years for which the data was collected:

print('The number of different years the data was collected for: ' + str(df['year'].nunique()))

# Based on our findings, we can see that the data was collected for the year of 2016 only. Thus, we will extract the different
# month values from each timestamp, and drop the 'year' column, as it is not very useful for us:

df = df.drop(['year'], axis=1)

# Extract the month column:

df['month'] = df['timestamp'].dt.month#.astype('uint8')

# Examine the month column:

print('The number of different months the data was collected for: ' + str(df['month'].nunique()))





# Count the frequency of different types of energy in the Data Set using the countplot graph:

sns.set()
sns.countplot(df['meter'],order=df['meter'].value_counts().sort_values().index)
plt.title("Count Distribution of different Energey Types")
plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.show()
# Based on our countplot, we can see that Electricity has the highest frequency, followed by ChilledWater.
# Steam has lower frequency than the two above, and HotWater has the lowest.





# Check how the energy consumption is distributed for the different energy types:

revision=df.groupby('meter')['meter_reading'].agg(['min','max','mean','median','count','std','sum'])

# Based on our findings, we can see that Steam has a much higher consumption as opposed to the other energy types.


df.columns


revision1=df.groupby(['site_id','meter'])['meter_reading'].agg(['min','max','mean','median','count','std','sum'])

df.columns

rev=df[['meter_reading','site_id','meter','month']]

check1=rev.groupby(['site_id','meter','month'])['meter_reading'].sum()
plt.plot(check1[0]['ChilledWater'])
plt.show()

plt.plot(check1[2]['ChilledWater'])
plt.show()

plt.plot(check1[14]['ChilledWater'])
plt.plot(check1[14]['Electricity'])
plt.plot(check1[14]['HotWater'])
plt.plot(check1[14]['Steam'])
plt.legend(['ChilledWater','Electricity','HotWater','Steam'])
plt.show()


#########################################################################################################


#PROCEDURE:
#  I WILL CONSTRUCT A DISSIMILARITY MATRIX
# TO PONDERATE K1*ELECTRICITY + K2* CHILLEDWATER+K3* STEAM+K4* HOTWATER KI WILL BE THE TOTAL_METERED_ENERGY_FROM_A_CITY_OF_FACTORX/TOTAL_METERED_ENERGY_FROM_FACTORX
#  AFTER KNOW HOW THE CITY ARE DISTRIBUTED BY THE SIMILARITY MATRIX , I WILL CONSTRUCT THE REGRESSIONS (INSIDE CLUSTERS)


k1=check1.unstack(0).loc['Electricity'].sum()/(check1.unstack(1)).sum().loc['Electricity']
k2=check1.unstack(0).loc['ChilledWater'].sum()/(check1.unstack(1)).sum().loc['ChilledWater']
k3=check1.unstack(0).loc['HotWater'].sum()/(check1.unstack(1)).sum().loc['HotWater']
k4=check1.unstack(0).loc['Steam'].sum()/(check1.unstack(1)).sum().loc['Steam']
similarity_constructor=k1+k2+k3+k4

len(similarity_constructor)

import scipy.spatial.distance as ssd

dist=np.zeros((13,13))
dist[:,0]=similarity_constructor.values


dist1=ssd.cdist(dist,dist,'euclidean')

dist1

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
clusters_study=sch.fcluster(sch.single(dist1),0.2,criterion='distance') # Here I create 4 groups


# Just left to create the 4 regressions

clusters_study

# Buildings with similar characteristics 
similarity_constructor[clusters_study==4]
similarity_constructor[clusters_study==3]
similarity_constructor[clusters_study==2]
similarity_constructor[clusters_study==1]



df.columns
grp1_df=df[df['site_id'].isin(similarity_constructor[clusters_study==1].index.values)]
grp2_df=df[df['site_id'].isin(similarity_constructor[clusters_study==2].index.values)]
grp3_df=df[df['site_id'].isin(similarity_constructor[clusters_study==3].index.values)]
grp4_df=df[df['site_id'].isin(similarity_constructor[clusters_study==4].index.values)]

grp2_df.columns



grp2_df[['meter_reading','month','timestamp','site_id']]

grp2_df['date']=grp2_df['timestamp'].dt.date
grp2_df_rev=grp2_df.groupby(['date'])['meter_reading','square_feet','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed'].agg(['mean'])

grp2_df_rev.columns

plt.plot(grp2_df_rev['meter_reading'])
plt.show()

from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

#import johansen

adfuller(grp2_df_rev['meter_reading']['mean']) # NO TREND 




plt.plot(grp2_df_rev['meter_reading']['mean'])
plt.show()

plot_acf(grp2_df_rev['meter_reading'])
plt.show()

len(grp2_df_rev['meter_reading'])
plot_pacf(grp2_df_rev['meter_reading'].diff(periods=200).dropna())
plt.show()

plot_pacf(grp2_df_rev['meter_reading'])
plt.show()


adfuller(grp2_df_rev['square_feet']['mean'])
adfuller(grp2_df_rev['air_temperature']['mean'])
adfuller(grp2_df_rev['dew_temperature']['mean'])
adfuller(grp2_df_rev['cloud_coverage']['mean']) 
adfuller(grp2_df_rev['precip_depth_1_hr']['mean']) #we can not differentiate this
adfuller(grp2_df_rev['sea_level_pressure']['mean']) #we can not differentiate this
adfuller(grp2_df_rev['wind_direction']['mean']) #we can not differentiate this
adfuller(grp2_df_rev['wind_speed']['mean']) #we can not differentiate this NO TREND 



plt.plot(grp2_df_rev['meter_reading'])
plt.plot(grp2_df_rev['cloud_coverage'])
plt.plot(grp2_df_rev['precip_depth_1_hr'])
plt.plot(grp2_df_rev['sea_level_pressure'])
plt.plot(grp2_df_rev['wind_direction'])
plt.plot(grp2_df_rev['wind_speed'])
#plt.plot(grp2_df_rev['square_feet'])
plt.legend(['meter_reading','cloud_coverage','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed','square_feet'])
plt.show()

(grp2_df_rev['meter_reading']['mean'],grp2_df_rev['wind_direction']['mean'])
(grp2_df_rev.corr()).iloc[:,0]


plt.plot(grp2_df_rev['meter_reading'])
plt.plot(grp2_df_rev['wind_direction'])
plt.legend(['meter_reading','wind_direction'])
plt.show()



import statsmodels.api as sm
import statsmodels


# OK THERE IS NO TREND HERE  (ON BOTH SERIES )
mod1_grp2=sm.OLS(np.array(grp2_df_rev['meter_reading']['mean']),statsmodels.tools.tools.add_constant(np.array(grp2_df_rev['wind_direction']['mean'])),hasconst=False)
mod1_grp2.fit().summary()
results=mod1_grp2.fit()
plt.hist(results.resid)
plt.show()


plt.plot(results.resid)
plt.show()

adfuller(results.resid)
plot_acf(results.resid)
plt.show()

plot_pacf(results.resid)
plt.show()

# The series has apparently an AR seasonal component, that we must clean,  to get a better fit


# OBVIOUSLY RESIDUALS WITHOUT TREND
# THE RESIDUALS SHOWED A PATTERN VERY SIMILAR TO THE ORIGINAL DATA , THAT OCCURS BECAUSE OF THE SEASONAL AR PATTERN THAT ORIGNAL EXHIBIT
# WE SHOULD TEST A VAR MODEL


mod2_grp2=sm.OLS(np.array(grp2_df_rev['meter_reading']['mean']),np.array(grp2_df_rev['wind_direction']['mean']),hasconst=False)
mod2_grp2.fit().summary()
results2=mod2_grp2.fit()
plt.hist(results2.resid)
plt.show()


#ADJUSTING WITH VAR (SARIMAX)

model_group2=sm.tsa.statespace.SARIMAX((grp2_df_rev['meter_reading']['mean']),(grp2_df_rev['wind_direction']),trend='c',order=(np.concatenate([np.zeros(180),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp2=model_group2.fit()
res_grp2.summary()

plt.plot(res_grp2.resid)
plt.show()

plt.hist(res_grp2.resid)
plt.show()

plot_acf(res_grp2.resid)
plt.show()

plot_pacf(res_grp2.resid)
plt.show()




grp4_df_rev.columns



grp4_df['date']=grp4_df['timestamp'].dt.date
grp4_df_rev=grp4_df.groupby(['date'])['meter_reading','square_feet','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed'].agg(['mean'])


grp4_df_rev.columns

plt.plot(grp4_df_rev['meter_reading'])
plt.show()


grp4_df_rev.corr()

plt.plot(grp4_df_rev['meter_reading'])
plt.plot(grp4_df_rev['air_temperature'])
plt.show()



grp4_df_rev['meter_reading']


adfuller(grp4_df_rev['meter_reading']['mean']) # NO TREND HERE 
adfuller(grp4_df_rev['square_feet']['mean'])
adfuller(grp4_df_rev['air_temperature']['mean']) # THIS SERIE HAS TREND (IT IS NOT POSSSIBLE TO USE LIKE THAT i HAVE TO DIFF THIS SERIE)
adfuller(grp4_df_rev['dew_temperature']['mean'])
adfuller(grp4_df_rev['cloud_coverage']['mean']) 
adfuller(grp4_df_rev['precip_depth_1_hr']['mean']) 
adfuller(grp4_df_rev['sea_level_pressure']['mean'])
adfuller(grp4_df_rev['wind_direction']['mean']) 
adfuller(grp4_df_rev['wind_speed']['mean']) 



plt.plot(grp4_df_rev['air_temperature']['mean'])
plt.show()

plt.plot(grp4_df_rev['meter_reading']['mean'])
plt.show()




plt.plot(grp4_df_rev['air_temperature']['mean'].diff())
#plt.plot(grp4_df_rev['meter_reading']['mean'])
plt.show()



grp4_df_rev.columns
grp4_df_rev.corr().iloc[:,0]

plt.plot(grp4_df_rev['meter_reading'].diff())
plt.plot(grp4_df_rev['sea_level_pressure'].diff())
plt.plot(grp4_df_rev['air_temperature'].diff())
# plt.plot(grp4_df_rev['square_feet'].diff())
# plt.plot(grp4_df_rev['sea_level_pressure'].diff())
# plt.plot(grp4_df_rev['cloud_coverage'].diff())
# plt.plot(grp4_df_rev['precip_depth_1_hr'].diff())
# plt.plot(grp4_df_rev['wind_speed'].diff())
plt.legend(['meter_reading','square_feet'])#,'square_feet','sea_level_pressure','cloud_coverage','precip_depth_1_hr','wind_speed'])
plt.show()

X=grp4_df_rev[['square_feet', 'air_temperature','sea_level_pressure','cloud_coverage', 'dew_temperature','precip_depth_1_hr','wind_direction', 'wind_speed']].reset_index() # check the entry

mod1_grp4=sm.OLS(np.array((grp4_df_rev['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X.iloc[:,range(1,X.shape[1])])),hasconst=True)
mod1_grp4.fit().summary()
results=mod1_grp4.fit()
plt.hist(results.resid)
plt.show()






# X.iloc[:,6]
# X1=grp4_df_rev[['square_feet', 'air_temperature','sea_level_pressure','cloud_coverage', 'dew_temperature','wind_direction', 'wind_speed']].reset_index()
# mod2_grp4=sm.OLS(np.array((grp4_df_rev['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X1.iloc[:,range(1,X1.shape[1])])),hasconst=True)
# mod2_grp4.fit().summary()
# results2=mod2_grp4.fit()
# plt.hist(results2.resid)
# plt.show()


# X2=grp4_df_rev[['square_feet', 'air_temperature','sea_level_pressure','cloud_coverage', 'dew_temperature', 'wind_speed']].reset_index()
# mod3_grp4=sm.OLS(np.array((grp4_df_rev['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X2.iloc[:,range(1,X2.shape[1])])),hasconst=True)
# mod3_grp4.fit().summary()
# results3=mod3_grp4.fit()
# plt.hist(results3.resid)
# plt.show()

# X3=grp4_df_rev[['square_feet', 'air_temperature','sea_level_pressure','cloud_coverage', 'dew_temperature']].reset_index()
# mod4_grp4=sm.OLS(np.array((grp4_df_rev['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X3.iloc[:,range(1,X3.shape[1])])),hasconst=True)
# mod4_grp4.fit().summary()
# results4=mod4_grp4.fit()
# plt.hist(results4.resid)
# plt.show()

# X4=grp4_df_rev[['square_feet', 'air_temperature','sea_level_pressure','cloud_coverage']].reset_index()
# mod5_grp4=sm.OLS(np.array((grp4_df_rev['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X4.iloc[:,range(1,X4.shape[1])])),hasconst=True)
# mod5_grp4.fit().summary()
# results5=mod5_grp4.fit()
# plt.hist(results5.resid)
# plt.show()

# X5=grp4_df_rev[['square_feet', 'air_temperature','cloud_coverage']].reset_index()
# X5.corr().iloc[:,0]
# mod6_grp4=sm.OLS(np.array((grp4_df_rev['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X5.iloc[:,range(1,X5.shape[1])])),hasconst=True)
# mod6_grp4.fit().summary()
# results6=mod6_grp4.fit()
# plt.hist(results6.resid)
# plt.show()

 #'air_temperature'
X6=grp4_df_rev[['square_feet','air_temperature']].reset_index()
X6.corr().iloc[:,0]
mod7_grp4=sm.OLS(np.array((grp4_df_rev['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X6.iloc[:,range(1,X6.shape[1])])),hasconst=True)
mod7_grp4.fit().summary()
results7=mod7_grp4.fit()
plt.hist(results7.resid)
plt.show()


#_______________________________________________________________________________________________________

#LIKE THE DATA HAS SOME IMPORTANT VARIABLES of order 1 and order order 0 , we proceed to differentiate everything

grp4_df_rev_differentiated=grp4_df_rev.diff().dropna()

grp4_df_rev_differentiated.corr().iloc[:,0]



plt.plot(grp4_df_rev_differentiated['meter_reading'])
plt.plot(grp4_df_rev_differentiated['sea_level_pressure'])
plt.plot(grp4_df_rev_differentiated['air_temperature'])
# plt.plot(grp4_df_rev['square_feet'].diff())
# plt.plot(grp4_df_rev['sea_level_pressure'].diff())
# plt.plot(grp4_df_rev['cloud_coverage'])
plt.plot(grp4_df_rev_differentiated['dew_temperature'])
plt.plot(grp4_df_rev_differentiated['precip_depth_1_hr'])
plt.plot(grp4_df_rev_differentiated['wind_speed'])

plt.legend(['meter_reading','sea_level_pressure','air_temperature','dew_temperature','precip_depth_1_hr','wind_speed'])#,'square_feet','sea_level_pressure','cloud_coverage','precip_depth_1_hr','wind_speed'])
plt.show()

plt.plot(grp4_df_rev_differentiated['meter_reading'])
plt.show()

plot_acf(grp4_df_rev_differentiated['meter_reading'])
plt.show()

plot_pacf(grp4_df_rev_differentiated['meter_reading'])
plt.show()

# The differentiated serie shows a break in the 220-240 lag (It is a signal of possible seasonality)

plot_pacf(grp4_df_rev_differentiated['meter_reading'].diff(periods=360).dropna())   
plt.show()




X_diff=grp4_df_rev_differentiated[['square_feet', 'air_temperature','sea_level_pressure','cloud_coverage', 'dew_temperature','precip_depth_1_hr','wind_direction', 'wind_speed']].reset_index()
X_diff.corr().iloc[:,0]
mod_dif_grp4=sm.OLS(np.array((grp4_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff.iloc[:,range(1,X_diff.shape[1])])),hasconst=True)
mod_dif_grp4.fit().summary()
results_mod_dif=mod_dif_grp4.fit()
plt.hist(results_mod_dif.resid)
plt.show()

plt.plot(results_mod_dif.resid)
plt.show()


plot_acf(results_mod_dif.resid)
plt.show()

plot_pacf(results_mod_dif.resid)
plt.show()


plt.plot(np.log(grp4_df_rev['meter_reading']))
plt.show()

plot_acf(np.log(grp4_df_rev['meter_reading']))
plt.show()

plot_pacf(np.log(grp4_df_rev['meter_reading']))
plt.show()




X_diff1=grp4_df_rev_differentiated[['square_feet', 'air_temperature','sea_level_pressure','cloud_coverage', 'dew_temperature','precip_depth_1_hr','wind_direction']].reset_index()
X_diff1.corr().iloc[:,0]
mod_dif1_grp4=sm.OLS(np.array((grp4_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff1.iloc[:,range(1,X_diff1.shape[1])])),hasconst=True)
mod_dif1_grp4.fit().summary()
results_mod_dif1=mod_dif1_grp4.fit()
plt.hist(results_mod_dif1.resid)
plt.show()

plt.plot(results_mod_dif.resid)
plt.show()


plot_acf(results_mod_dif.resid)
plt.show()

plot_pacf(results_mod_dif.resid)
plt.show()


plt.plot(np.log(grp4_df_rev['meter_reading']))
plt.show()

plot_acf(np.log(grp4_df_rev['meter_reading']))
plt.show()

plot_pacf(np.log(grp4_df_rev['meter_reading']))
plt.show()



X_diff2=grp4_df_rev_differentiated[['square_feet','sea_level_pressure','cloud_coverage', 'dew_temperature','precip_depth_1_hr','wind_direction']].reset_index()
X_diff2.corr().iloc[:,0]
mod_dif2_grp4=sm.OLS(np.array((grp4_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff2.iloc[:,range(1,X_diff2.shape[1])])),hasconst=True)
mod_dif2_grp4.fit().summary()
results_mod_dif2=mod_dif2_grp4.fit()
plt.hist(results_mod_dif2.resid)
plt.show()

plt.plot(results_mod_dif2.resid)
plt.show()


plot_acf(results_mod_dif2.resid)
plt.show()

plot_pacf(results_mod_dif2.resid)
plt.show()


plt.plot(np.log(grp4_df_rev['meter_reading']))
plt.show()

plot_acf(np.log(grp4_df_rev['meter_reading']))
plt.show()

plot_pacf(np.log(grp4_df_rev['meter_reading']))
plt.show()







model_group4=sm.tsa.statespace.SARIMAX(np.log(grp4_df_rev['meter_reading']),np.log(grp4_df_rev['air_temperature']),trend='ct',order=(np.concatenate([np.zeros(270),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp4=model_group4.fit()

res_grp4.summary()

plt.plot(res_grp4.resid)
plt.show()


plt.hist(res_grp4.resid)
plt.show()



plot_acf(res_grp4.resid)
plt.show()


plot_pacf(res_grp4.resid)
plt.show()
plt.boxplot(np.log(grp4_df_rev['meter_reading']['mean']))
plt.show()

model_group4=sm.tsa.statespace.SARIMAX((grp4_df_rev['meter_reading']),(grp4_df_rev[['square_feet','sea_level_pressure','cloud_coverage', 'dew_temperature','precip_depth_1_hr','wind_direction']]),trend='ct',order=(np.concatenate([np.zeros(300),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp4=model_group4.fit()

res_grp4.summary()

plt.plot(res_grp4.resid)
plt.show()


plt.hist(res_grp4.resid)
plt.show()



plot_acf(res_grp4.resid)
plt.show()


plot_pacf(res_grp4.resid)
plt.show()
plt.boxplot((grp4_df_rev['meter_reading']['mean']))
plt.show()



model_group4_1=sm.tsa.statespace.SARIMAX((grp4_df_rev['meter_reading']),(grp4_df_rev[['square_feet','sea_level_pressure','cloud_coverage','precip_depth_1_hr','wind_direction']]),trend='ct',order=(np.concatenate([np.zeros(300),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp4_1=model_group4_1.fit()

res_grp4_1.summary()


plt.plot(res_grp4_1.resid)
plt.show()


plt.hist(res_grp4_1.resid)
plt.show()



plot_acf(res_grp4_1.resid)
plt.show()


plot_pacf(res_grp4_1.resid)
plt.show()
plt.boxplot((grp4_1_df_rev['meter_reading']['mean']))
plt.show()



model_group4_2=sm.tsa.statespace.SARIMAX((grp4_df_rev['meter_reading']),(grp4_df_rev[['square_feet','sea_level_pressure','cloud_coverage','precip_depth_1_hr','wind_direction']]),trend='t',order=(np.concatenate([np.zeros(300),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp4_2=model_group4_2.fit()
res_grp4_2.summary()
plt.plot(res_grp4_2.resid)
plt.show()



model_group4_3=sm.tsa.statespace.SARIMAX((grp4_df_rev['meter_reading']),(grp4_df_rev[['square_feet','sea_level_pressure','precip_depth_1_hr','wind_direction']]),trend='t',order=(np.concatenate([np.zeros(300),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp4_3=model_group4_3.fit()
res_grp4_3.summary()
plt.plot(res_grp4_3.resid)
plt.show()



model_group4_4=sm.tsa.statespace.SARIMAX((grp4_df_rev['meter_reading']),(grp4_df_rev[['square_feet','sea_level_pressure','wind_direction']]),trend='t',order=(np.concatenate([np.zeros(300),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp4_4=model_group4_4.fit()
res_grp4_4.summary()
plt.plot(res_grp4_4.resid)
plt.show()


model_group4_5=sm.tsa.statespace.SARIMAX((grp4_df_rev['meter_reading']),(grp4_df_rev[['sea_level_pressure','wind_direction']]),trend='t',order=(np.concatenate([np.zeros(300),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp4_5=model_group4_5.fit()
res_grp4_5.summary()
plt.plot(res_grp4_5.resid)
plt.show()

plot_acf(res_grp4_5.resid)
plt.show()

plot_pacf(res_grp4_5.resid)
plt.show()

plt.hist(res_grp4_5.resid)
plt.show()

# FINAL BEST FIT GROUP4 IS GROUP4_5



#_____________________________________________________________________________________________________




grp1_df['date']=grp1_df['timestamp'].dt.date
grp1_df_rev=grp1_df.groupby(['date'])['meter_reading','square_feet','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed'].agg(['mean'])
grp1_df_rev.columns
plt.plot(grp1_df_rev['meter_reading'])
plt.show()

adfuller(grp1_df_rev['meter_reading']['mean'])#WE have to differentiate the series (IT HAS TREND - we cannot reject H0)
adfuller(grp1_df_rev['square_feet']['mean'])
adfuller(grp1_df_rev['air_temperature']['mean'])
adfuller(grp1_df_rev['dew_temperature']['mean'])
adfuller(grp1_df_rev['cloud_coverage']['mean']) 
adfuller(grp1_df_rev['precip_depth_1_hr']['mean']) 
adfuller(grp1_df_rev['sea_level_pressure']['mean']) 
adfuller(grp1_df_rev['wind_direction']['mean']) 
adfuller(grp1_df_rev['wind_speed']['mean']) 





plt.plot((grp1_df_rev['meter_reading']))
plt.show()

plot_acf((grp1_df_rev['meter_reading']))
plt.show()

plot_pacf((grp1_df_rev['meter_reading']))
plt.show()



grp1_df_rev_differentiated=grp1_df_rev.diff().dropna()


plt.plot((grp1_df_rev_differentiated['meter_reading']))
plt.show()

X_diff=grp1_df_rev_differentiated[['square_feet', 'air_temperature','sea_level_pressure','cloud_coverage', 'dew_temperature','precip_depth_1_hr','wind_direction', 'wind_speed']].reset_index()
X_diff.corr().iloc[:,0]
mod_dif_grp1=statsmodels.regression.linear_model.OLS(np.array((grp1_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff.iloc[:,range(1,X_diff.shape[1])])),hasconst=True)
mod_dif_grp1.fit().summary()
results_mod_dif=mod_dif_grp1.fit()
plt.hist(results_mod_dif.resid)
plt.show()



X_diff2=grp1_df_rev_differentiated[['square_feet', 'air_temperature','sea_level_pressure', 'dew_temperature','precip_depth_1_hr','wind_direction']].reset_index()
X_diff2.corr().iloc[:,0]
mod_dif2_grp1=statsmodels.regression.linear_model.OLS(np.array((grp1_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff2.iloc[:,range(1,X_diff2.shape[1])])),hasconst=True)
mod_dif2_grp1.fit().summary()
results_mod_dif2=mod_dif2_grp1.fit()
plt.hist(results_mod_dif2.resid)
plt.show()




X_diff3=grp1_df_rev_differentiated[['square_feet', 'air_temperature','sea_level_pressure', 'dew_temperature','wind_direction']].reset_index()
X_diff3.corr().iloc[:,0]
mod_dif3_grp1=statsmodels.regression.linear_model.OLS(np.array((grp1_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff3.iloc[:,range(1,X_diff3.shape[1])])),hasconst=True)
mod_dif3_grp1.fit().summary()
results_mod_dif3=mod_dif3_grp1.fit()
plt.hist(results_mod_dif3.resid)
plt.show()


# AS THERE IS A STRONG CORRELATION AMONG AIR_TEMP AND DEW_TEMP
X_diff3=grp1_df_rev_differentiated[['square_feet', 'air_temperature','sea_level_pressure', 'dew_temperature','wind_direction']].reset_index()
X_diff3.corr().iloc[:,0]
mod_dif3_grp1=statsmodels.regression.linear_model.OLS(np.array((grp1_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff3.iloc[:,range(1,X_diff3.shape[1])])),hasconst=True)
mod_dif3_grp1.fit().summary()
results_mod_dif3=mod_dif3_grp1.fit()
plt.hist(results_mod_dif3.resid)
plt.show()


X_diff4=grp1_df_rev_differentiated[['square_feet', 'air_temperature','sea_level_pressure','wind_direction']].reset_index()
X_diff4.corr().iloc[:,0]
mod_dif4_grp1=statsmodels.regression.linear_model.OLS(np.array((grp1_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff4.iloc[:,range(1,X_diff4.shape[1])])),hasconst=True)
mod_dif4_grp1.fit().summary()
results_mod_dif4=mod_dif4_grp1.fit()
plt.hist(results_mod_dif4.resid)
plt.show()


X_diff5=grp1_df_rev_differentiated[['air_temperature','sea_level_pressure','wind_direction']].reset_index()
X_diff5.corr().iloc[:,0]
mod_dif5_grp1=statsmodels.regression.linear_model.OLS(np.array((grp1_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff5.iloc[:,range(1,X_diff5.shape[1])])),hasconst=True)
mod_dif5_grp1.fit().summary()
results_mod_dif5=mod_dif5_grp1.fit()
plt.hist(results_mod_dif5.resid)
plt.show()


X_diff6=grp1_df_rev_differentiated[['air_temperature']].reset_index()
X_diff6.corr().iloc[:,0]
mod_dif6_grp1=statsmodels.regression.linear_model.OLS(np.array((grp1_df_rev_differentiated['meter_reading']['mean'].reset_index()).iloc[:,1]),statsmodels.tools.tools.add_constant(np.array(X_diff6.iloc[:,range(1,X_diff6.shape[1])])),hasconst=True)
mod_dif6_grp1.fit().summary()
results_mod_dif6=mod_dif6_grp1.fit()
plt.hist(results_mod_dif6.resid)
plt.show()




model_group1=sm.tsa.statespace.SARIMAX((grp1_df_rev['meter_reading']),(grp1_df_rev[['air_temperature']]),trend='t',order=(np.concatenate([np.zeros(180),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp1=model_group1.fit()
res_grp1.summary()




plt.plot(res_grp1.resid)
plt.show()

plt.hist(res_grp1.resid)
plt.show()


plot_acf(res_grp1.resid)
plt.show()

plot_pacf(res_grp1.resid)
plt.show()


###########################################################
#previous one is the last model fit for the group1





grp3_df['date']=grp3_df['timestamp'].dt.date
grp3_df_rev=grp3_df.groupby(['date'])['meter_reading','square_feet','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed'].agg(['mean'])

grp3_df_rev.columns

plt.plot(grp3_df_rev['meter_reading'])
plt.show()


plot_acf(grp3_df_rev['meter_reading'])
plt.show()

plot_pacf(grp3_df_rev['meter_reading'])
plt.show()


adfuller(grp3_df_rev['meter_reading']['mean']) 
adfuller(grp3_df_rev['square_feet']['mean'])
adfuller(grp3_df_rev['air_temperature']['mean']) # THIS SERIE HAS TREND (IT IS NOT POSSSIBLE TO USE LIKE THAT i HAVE TO DIFF THIS SERIE)
adfuller(grp3_df_rev['dew_temperature']['mean'])
adfuller(grp3_df_rev['cloud_coverage']['mean']) 
adfuller(grp3_df_rev['precip_depth_1_hr']['mean']) 
adfuller(grp3_df_rev['sea_level_pressure']['mean'])
adfuller(grp3_df_rev['wind_direction']['mean']) 
adfuller(grp3_df_rev['wind_speed']['mean']) 





model_group3=sm.tsa.statespace.SARIMAX((grp3_df_rev['meter_reading']),(grp3_df_rev[['square_feet','cloud_coverage','dew_temperature','precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed']]),trend='t',order=(np.concatenate([np.zeros(180),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp3=model_group3.fit()
res_grp3.summary()


model_group3_1=sm.tsa.statespace.SARIMAX((grp3_df_rev['meter_reading']),(grp3_df_rev[['square_feet','cloud_coverage','dew_temperature', 'sea_level_pressure', 'wind_direction','wind_speed']]),trend='t',order=(np.concatenate([np.zeros(180),np.ones(1)]),1,0),enforce_stationarity=False)    
res_grp3_1=model_group3_1.fit()
res_grp3_1.summary()


model_group3_2=sm.tsa.statespace.SARIMAX((grp3_df_rev['meter_reading']),(grp3_df_rev[['square_feet','cloud_coverage', 'sea_level_pressure', 'wind_direction','wind_speed']]),trend='t',order=(np.concatenate([np.zeros(180),np.ones(1)]),1,0),enforce_stationarity=True)    
res_grp3_2=model_group3_2.fit()
res_grp3_2.summary()


model_group3_3=sm.tsa.statespace.SARIMAX((grp3_df_rev['meter_reading']),(grp3_df_rev[['square_feet', 'sea_level_pressure', 'wind_direction','wind_speed']]),trend='t',order=(np.concatenate([np.zeros(300),np.ones(1)]),1,0),enforce_stationarity=True)    
res_grp3_3=model_group3_3.fit()
res_grp3_3.summary()

plt.plot(res_grp3_3.resid)
plt.show()


plot_acf(res_grp3_3.resid)
plt.show()

plot_pacf(res_grp3_3.resid)
plt.show()


plt.hist(res_grp3_3.resid)
plt.show()



###########################################################################################################
# TEST


start = time.time()

df_test = pd.read_csv(filepath_or_buffer='D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae original\\test.csv', sep=',', low_memory=False)

print('Time to read the CSV file into a dataframe: ',round((time.time()-start), 2), 'seconds \n')

startingMemoryUsage = df_test.memory_usage().sum() / 1024**2

print('Dataframe size is {:.2f} MB'.format(startingMemoryUsage))


df_test.columns

df_test.shape

df_test['date']=pd.to_datetime(df_test['timestamp']).dt.date

test_data=df_test[['date','building_id']].drop_duplicates()
#df_test=df_test.groupby(['date','building_id'])[].agg(['mean'])
test_data['date'].unique()

test_data['date'].unique()

df_test[['date']].drop_duplicates()



##########################################################################################################
df['date']=pd.to_datetime(df['timestamp']).dt.date
df.columns



data_to_test_set=df[['building_id','site_id']].drop_duplicates()

data_to_test_set['site_id'][data_to_test_set['site_id'].isna()]



test_data_complete=pd.merge(test_data,data_to_test_set,how="left",on='building_id')

test_data_complete[test_data_complete['building_id']==15]

data_to_test_set[data_to_test_set['site_id']==15]

test_data_complete[test_data_complete['site_id']==15]

test_data_filtered=test_data_complete[(test_data_complete['site_id'].isna())==False]

to_serie_1=test_data_filtered[test_data_filtered['site_id'].isin(similarity_constructor[clusters_study==1].index.values)]
to_serie_1

#test_data_filtered=test_data_filtered.drop('cluster',axis=1)

df[df['site_id'].isin(similarity_constructor[clusters_study==1].index.values)]

##############################################################################################################
# FORECAST GROUP 1
forecast_30_days_grp1=res_grp1.forecast(steps=30,exog=grp1_df_rev[['air_temperature','sea_level_pressure']].iloc[range(0,30),range(0,2)])

plt.plot(forecast_30_days_grp1)
plt.plot(grp1_df_rev['meter_reading'].iloc[range(grp1_df_rev['meter_reading'].shape[0]-30,grp1_df_rev['meter_reading'].shape[0])])
plt.legend(['Predicted','Actual'])
plt.show()

salida=forecast_performance(grp1_df_rev['meter_reading'].iloc[range(grp1_df_rev['meter_reading'].shape[0]-30,grp1_df_rev['meter_reading'].shape[0])],forecast_30_days_grp1)



########################################################################################
# FORECAST GROUP 2

res_grp2

grp2_df_rev.tail()
forecast_30_days_grp2=res_grp2.forecast(steps=30,exog=grp2_df_rev[['wind_direction']].iloc[range(grp2_df_rev[['wind_direction']].shape[0]-30,grp2_df_rev[['wind_direction']].shape[0]),range(0,1)])


forecast_30_days_grp2.index=grp2_df_rev['meter_reading'].iloc[range(grp2_df_rev['meter_reading'].shape[0]-30,grp2_df_rev['meter_reading'].shape[0])].index.values
plt.plot(forecast_30_days_grp2)
plt.plot(grp2_df_rev['meter_reading'].iloc[range(grp2_df_rev['meter_reading'].shape[0]-30,grp2_df_rev['meter_reading'].shape[0])])
plt.legend(['Predicted','Actual'])
plt.show()

salida2=forecast_performance(grp2_df_rev['meter_reading'].iloc[range(grp2_df_rev['meter_reading'].shape[0]-30,grp2_df_rev['meter_reading'].shape[0])],forecast_30_days_grp2)




##############################################################################################################
# FORECAST GROUP 3

forecast_30_days_grp3=res_grp3_3.forecast(steps=30,exog=grp3_df_rev[['square_feet', 'sea_level_pressure', 'wind_direction','wind_speed']].iloc[range(grp3_df_rev[['wind_direction']].shape[0]-30,grp3_df_rev[['wind_direction']].shape[0]),range(0,4)])


forecast_30_days_grp3.index=grp3_df_rev['meter_reading'].iloc[range(grp3_df_rev['meter_reading'].shape[0]-30,grp3_df_rev['meter_reading'].shape[0])].index.values
plt.plot(forecast_30_days_grp3)
plt.plot(grp3_df_rev['meter_reading'].iloc[range(grp3_df_rev['meter_reading'].shape[0]-30,grp3_df_rev['meter_reading'].shape[0])])
plt.legend(['Predicted','Actual'])
plt.show()

salida3=forecast_performance(grp3_df_rev['meter_reading'].iloc[range(grp3_df_rev['meter_reading'].shape[0]-30,grp3_df_rev['meter_reading'].shape[0])],forecast_30_days_grp3)




########################################################################################
# FORECAST GROUP 4


forecast_30_days_grp4=res_grp4_5.forecast(steps=30,exog=grp4_df_rev[['sea_level_pressure','wind_direction']].iloc[range(grp4_df_rev[['wind_direction']].shape[0]-30,grp4_df_rev[['wind_direction']].shape[0]),range(0,2)])


forecast_30_days_grp4.index=grp4_df_rev['meter_reading'].iloc[range(grp4_df_rev['meter_reading'].shape[0]-30,grp4_df_rev['meter_reading'].shape[0])].index.values
plt.plot(forecast_30_days_grp4)
plt.plot(grp4_df_rev['meter_reading'].iloc[range(grp4_df_rev['meter_reading'].shape[0]-30,grp4_df_rev['meter_reading'].shape[0])])
plt.legend(['Predicted','Actual'])
plt.show()

salida4=forecast_performance(grp4_df_rev['meter_reading'].iloc[range(grp4_df_rev['meter_reading'].shape[0]-30,grp3_df_rev['meter_reading'].shape[0])],forecast_30_days_grp4)



# THIS PART WAS REMOVED 




# # Draw a plot to see any trends in energy consumption throughout the year:

# df[['timestamp','meter_reading']].set_index('timestamp').resample('H')['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Average Energy Consumption throughout the Year:')
# plt.legend()
# plt.xlabel('Timestamp')
# plt.ylabel('Average Energy Consumption')
# plt.title('Graph of Averagy Energy Consumption throughout the year: ')

# # Based on our graph, we can see that the energy consumption is the highest in the months of January-February, July-August,
# # and on the month of December.


# 


# # Examine the energy consumption throughout the year for each energy type:

# # Define a function to plot the average energy consumption for individual energent:
# def consumption_grapher(meter_value):
#     energy_type = df[df['meter'] == meter_value]
#     energy_type[['timestamp','meter_reading']].set_index('timestamp').resample('H')['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Average Meter Reading')
#     plt.legend()
#     plt.xlabel('Timestamp')
#     plt.ylabel('Average Meter Reading')
#     plt.title('Graph of Average Meter Reading for ' + meter_value)
    


# 


# # Examine the Electricity Consumption throughout the year:

# consumption_grapher('Electricity')

# # We can see that the Electricity consumption generally increases throughout the year until the month of October, when it
# # starts decreasing for the rest of the year.


# 


# # Examine the ChilledWater consumption throughout the year:

# consumption_grapher('ChilledWater')

# # We can see a large increase in ChilledWater consumption from the months of January-August when it is the highest.
# # The consumption then starts dropping, and records the lowest values in the month of December.


# 


# # Examine the Steam consumption throughout the year:

# consumption_grapher('Steam')

# # We can see that the Steam consumption generally follows the similar trend as the trend of all energy types combined.


# 


# # Examine the HotWater consumption throughout the year:

# consumption_grapher('HotWater')

# # We can see that the HotWater consumption also follows the similar trend to the trend of all energy types combined.


# 


# # Use groupby to calculate the min, max, median, count, and std values for each energent for each different month:

# df.groupby(['meter','month'])['meter_reading'].agg(['max','mean','median','count','std'])

# # Based on our findings, we can see that the Steam consumption is generally a lot higher than that of the other energy types
# # for the different months, and has lower counts.


# 

# df['meter_reading','meter','month'].groupby['meter','month'].agg(['max','mean','median','count','std'])






# 


# # Since we cannot see any clear distribution in the energy consumption, we will transform the meter_reading using
# # log transformation:

# df['meter_reading'] = np.log1p(df['meter_reading'])

# # Use the distribution plot to see how the log-transformed energy consumption is distributed:

# sns.distplot(df['meter_reading'])
# plt.title("Distribution of Log-transformed of Meter Reading Variable")

# # Based on the graph we can see that the log-transformed energy consumption follows a nearly normal distribution, however
# # it seems to have a lot of 0-value outliers.


# 


# # Examine the outliers using the Box Plot graphs:

# energy_types = [ 'Electricity', 'ChilledWater', 'HotWater', 'Steam' ]

# def boxplot_grapher(meter_type):
#     sns.boxplot(df[df['meter'] == meter_type]['meter_reading'])
#     plt.title('Boxplot of Meter Reading Variable for the Meter Type: ' + meter_type)

# # Based on our Box Plots, we can see that there are a lot of 0-valued outliers, and that Electricity has the outliers in
# # values greater than 8.


# 


# boxplot_grapher('Electricity')


# 


# boxplot_grapher('ChilledWater')


# 


# boxplot_grapher('HotWater')


# 


# boxplot_grapher('Steam')


# 


# # Based on our findings, we can determine the following:

# # 1) Despite a relatively low count, the Energy Consumption of Steam is much higher than that of the other Energy Types,
# #    to the extend where it dictates the trends in the energy consumption. To that end we will remove it from the Data Set.

# # 2) The Data Set appears to have a lot of 0-valued outliers, and we will be removing these:

# # Remove the 0-valued outliers:
# df.drop(df.loc[(df['meter'] == 'Electricity') & (df['meter_reading'] == 0)].index, inplace=True)
# df.drop(df.loc[(df['meter'] == 'HotWater') & (df['meter_reading'] == 0)].index, inplace=True)
# df.drop(df.loc[(df['meter'] == 'ChilledWater') & (df['meter_reading'] == 0)].index, inplace=True)

# # Remove the outliers from Electricity with values greater than 8:
# df.drop(df.loc[(df['meter'] == 'Electricity') & (df['meter_reading'] > 8)].index, inplace=True)


# 


# # Remove Steam from the Data Set:
# df.drop(df.loc[df['meter']=='Steam'].index, inplace=True)


# 


# # Examine the Energy Distribution throughout the year after implementing the transformation:

# sns.distplot(df['meter_reading'])
# plt.title("Distribution of Log-transformed of Meter Reading Variable")

# # Based on the graph below, the meter reading now follows the normal distribution.


# # ## 4.3 EXAMINE THE WEATHER FEATURES:

# 


# # Define an array to hold the column names of all weather features:
# weather_columns = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_speed', 'wind_direction']

# # Examine the weather data:
# df[weather_columns].describe()


# 


# # Use the Distribution Plot to see how the different Weather Features are distributed:

# for x,column in enumerate(df[weather_columns]):
#     plt.figure(x)
#     sns.distplot(df[column])
    
# # Based on our findings, we can determine the following:

# # air_temperature appears to be positively-skewed.
# # cloud_coverage is unique in that it is composed of a number of distinct values.
# # dew_temperature appears to be positively-skewed.
# # percip_depth_1_hr has a lot of 0 values in the Data Set.
# # sea_level_pressure follows the normal distribution.
# # wind_speed seems to be negatively skewed.


# 


# # Now, we will examine the columns to see if they have a very high level of coordination:

# # Threshold for removing correlated variables
# threshold = 0.9

# # Absolute value correlation matrix
# corr_matrix = df.corr().abs()
# corr_matrix.head()


# 


# # Select the columns with very high levels of correlation to be removed
# columns = [column for column in corr_matrix[weather_columns] if any(corr_matrix[column] > threshold)]

# print ("Columns with high levels of correlation to be removed {}".format(columns))

# # We will keep the sea_level_pressure, as it follows the normal distribution, and can be used for energy prediction:

# columns.remove('sea_level_pressure')


# 


# # Based on our findings, the below columns have a very high levels of correlation, and thus can be removed:

# # air_temperature
# # cloud_coverage
# # dew_temperature
# # precip_depth_1_hr
# # wind_speed


# df = df.drop(columns, axis=1)


# # ## 4.4 EXAMINE THE SURFACE AREA:

# 


# df['square_feet'].describe()


# 


# # Use the Distribution Plot to see how the Surface Area is distributed:

# sns.distplot(df['square_feet'])
# plt.title("Distribution of Building's Surface Area: ")


# 


# # The Surface Area seems to be negatively-skewed.

# # Let's transform it using the log based 10 transformation:

# df['square_feet'] = np.log10(df['square_feet'])

# # We can see that after the log-transformation the Surface Area follows an almost Normal Distribution.


# 


# sns.distplot(df['square_feet'])
# plt.title("Distribution of Building's Surface Area: ")


# # ## 4.5 REMOVE FEATURES THAT ARE NO LONGER NEEDED

# 


# # Now, we can remove the features that are no longer needed:

# # The site_id and building_id are used as unique identifiers for buildings and sites they are located, and are thus not very
# # useful for predicting energy.

# # Likewise, we no longer need the timestamp and time features, as they were used for transforming data.

# df = df.drop(['month', 'timestamp', 'site_id', 'building_id'], axis=1)


# 


# df.head(10)


# 


# df.columns


# 




