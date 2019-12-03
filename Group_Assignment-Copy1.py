#!/usr/bin/env python
# coding: utf-8

# # GROUP ASSIGNMENT - STATISTICS FOR DATA SCIENCE - FALL 2019 #

# ### Group Members: Darwing Cara, David Kobayashi, Eric Koritko, Stefan Lazarevic, Patrick McDonnell ###

# We have selected to work on a data set from ASHRAE (a technical society for heating, ventilation, and air conditioning in buildings). The dataset can be found here: https://www.kaggle.com/c/ashrae-energy-prediction
# 
# The goal of this exercise is to use the dataset to build a model that can predict energy use in a building based on that building's features (building use type, outdoor air conditions, etc).

# ## 1.0 - IMPORT PACKAGES ##
# In[1]:


import pandas as pd
import numpy  as np
import math as math
import datetime
import time
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# ## 1.1 - DEFINE USEFUL FUNCTIONS

# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


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

# In[6]:


elec_meter = 0
chw_meter = 1
steam_meter = 2
hw_meter = 3


# # 2.0 - LOAD DATASET INTO MEMORY

# In[7]:


start = time.time()

df = pd.read_csv(filepath_or_buffer='D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae original\\train.csv', sep=',', low_memory=False)

print('Time to read the CSV file into a dataframe: ',round((time.time()-start), 2), 'seconds \n')

startingMemoryUsage = df.memory_usage().sum() / 1024**2

print('Dataframe size is {:.2f} MB'.format(startingMemoryUsage))


# In[8]:


df.info()


# In[9]:


df.head()


# In[10]:


compress_dataframe(df)


# In[11]:


start = time.time()

df_weather = pd.read_csv(filepath_or_buffer='D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae original\\weather_train.csv', sep=',', low_memory=False)

print('Time to read the CSV file into a dataframe: ',round((time.time()-start), 2), 'seconds \n')

startingMemoryUsage = df_weather.memory_usage().sum() / 1024**2

print('Dataframe size is {:.2f} MB'.format(startingMemoryUsage))


# In[12]:


df_weather.info()


# In[13]:


df_weather.head()


# In[14]:


compress_dataframe(df_weather)


# In[15]:


start = time.time()

df_building_metadata = pd.read_csv(filepath_or_buffer='D:\\Documents\\Data Science Certificate\\Course 2\\Group Assignment\\ashrae original\\building_metadata.csv', sep=',', low_memory=False)

print('Time to read the CSV file into a dataframe: ',round((time.time()-start), 2), 'seconds \n')

startingMemoryUsage = df_weather.memory_usage().sum() / 1024**2

print('Dataframe size is {:.2f} MB'.format(startingMemoryUsage))


# In[16]:


df_building_metadata.info()


# In[17]:


df_building_metadata.head()


# In[18]:


compress_dataframe(df_building_metadata)


# # 3.0 - CLEAN THE DATASET

# ## 3.1 - CHECK FOR DUPLICATE ROWS

# In[19]:


df[df.duplicated()].info()


# ## 3.2 - CHECK FOR NULL VALUES IN THE MAIN DATASET AND WEATHER DATASET

# In[20]:


# Show how many nulls there are in the main dataset (expressed as a percentage)
print(round(df.isna().sum().sort_values(ascending=False) / len(df.index) * 100, 3))


# In[21]:


# Show how many nulls there are in the weather dataset (expressed as a percentage)
print(round(df_weather.isna().sum().sort_values(ascending=False) / len(df_weather.index) * 100, 3))


# In[22]:


# we have to be careful with this step because 2 close registers  could have very distinct behaviors

# Fill nulls with linear interpolation
df_weather = df_weather.interpolate()
df_weather = df_weather.fillna(0)

# Double check how many nulls are in the feature (should be 0 now)
print(round(df_weather.isna().sum().sort_values(ascending=False) / len(df_weather.index) * 100, 3))


# In[23]:


# Show how many nulls there are in the metadata dataset (expressed as a percentage)
print(round(df_building_metadata.isna().sum().sort_values(ascending=False) / len(df_building_metadata.index) * 100, 3))


# ## 3.3 - FILTER OUT OFFICE BUILDINGS AND MERGE DATASETS

# In[24]:


# count the number of different building types in the dataset
# df_building_metadata['primary_use'].value_counts()

# filter out the data that relates to office buildings



df_building_metadata = df_building_metadata[df_building_metadata['primary_use'] == 'Office']

# merge this data with the main dataframe, so that we only have meter readings for office buildings
df = df.merge(df_building_metadata, on='building_id', how='inner')

# merge weather data with the main dataframe, so that all the information is in one place
df = df.merge(df_weather, on=['site_id', 'timestamp'], how='inner')

# check dataframe
df.head()


# ## 3.4 - CHECK FOR NULL VALUES IN THE METADATA DATASET

# In[25]:


print(round(df.isna().sum().sort_values(ascending=False) / len(df.index) * 100, 3))


# In[26]:


# There's no great way to fill the floor_count and year_built columns
# Since there are a large number of nulls in the floor_count and year_built features, we will drop these columns

df = df.drop('floor_count', axis=1)
df = df.drop('year_built', axis=1)

df.head()


# ## 3.5 - USE THIS SECTION TO VIEW METER DATA AND WEATHER DATA TOGETHER

# In[27]:


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

# In[28]:


print(df['primary_use'].describe())

# We can see that the Primary Use column, has only a single unique value (Office).

# This is the result of the data cleaning phase, that resulted in extracting the results for the office buildings only.

# To that end, this feature is not very useful for us, since all rows represent the Office Buildings data, and we can remove it
# from the Data Set:

df = df.drop(['primary_use'], axis=1)


# ## 4.2 - EXAMINE ENERGY CONSUMPTION OVER THE YEAR:

# In[29]:


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

df['year'] = df['timestamp'].dt.year.astype('uint8')

# Examine the different years for which the data was collected:

print('The number of different years the data was collected for: ' + str(df['year'].nunique()))

# Based on our findings, we can see that the data was collected for the year of 2016 only. Thus, we will extract the different
# month values from each timestamp, and drop the 'year' column, as it is not very useful for us:

df = df.drop(['year'], axis=1)

# Extract the month column:

df['month'] = df['timestamp'].dt.month.astype('uint8')

# Examine the month column:

print('The number of different months the data was collected for: ' + str(df['month'].nunique()))


# In[30]:


# Count the frequency of different types of energy in the Data Set using the countplot graph:

sns.set()
sns.countplot(df['meter'],order=df['meter'].value_counts().sort_values().index)
plt.title("Count Distribution of different Energey Types")
plt.xlabel("Energy")
plt.ylabel("Frequency")

# Based on our countplot, we can see that Electricity has the highest frequency, followed by ChilledWater.
# Steam has lower frequency than the two above, and HotWater has the lowest.


# In[31]:


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
plt.plot(grp2_df['meter_reading'])
plt.show()

grp2_df[['meter_reading','month','timestamp','site_id']]

grp2_df_rev=grp2_df.groupby(['timestamp'])['meter_reading','square_feet','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed'].agg(['mean'])

grp2_df_rev.columns

plt.plot(grp2_df_rev['meter_reading'])
plt.show()


grp4_df_rev=grp4_df.groupby(['timestamp'])['meter_reading','square_feet','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed'].agg(['mean'])

grp4_df_rev.columns

plt.plot(grp4_df_rev['meter_reading'])
plt.show()



grp1_df_rev=grp1_df.groupby(['timestamp'])['meter_reading','square_feet','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed'].agg(['mean'])

grp1_df_rev.columns

plt.plot(grp1_df_rev['meter_reading'])
plt.show()



grp3_df_rev=grp3_df.groupby(['timestamp'])['meter_reading','square_feet','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr', 'sea_level_pressure', 'wind_direction','wind_speed'].agg(['mean'])

grp3_df_rev.columns

plt.plot(grp3_df_rev['meter_reading'])
plt.show()

# # In[32]:


# # Draw a plot to see any trends in energy consumption throughout the year:

# df[['timestamp','meter_reading']].set_index('timestamp').resample('H')['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Average Energy Consumption throughout the Year:')
# plt.legend()
# plt.xlabel('Timestamp')
# plt.ylabel('Average Energy Consumption')
# plt.title('Graph of Averagy Energy Consumption throughout the year: ')

# # Based on our graph, we can see that the energy consumption is the highest in the months of January-February, July-August,
# # and on the month of December.


# # In[33]:


# # Examine the energy consumption throughout the year for each energy type:

# # Define a function to plot the average energy consumption for individual energent:
# def consumption_grapher(meter_value):
#     energy_type = df[df['meter'] == meter_value]
#     energy_type[['timestamp','meter_reading']].set_index('timestamp').resample('H')['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Average Meter Reading')
#     plt.legend()
#     plt.xlabel('Timestamp')
#     plt.ylabel('Average Meter Reading')
#     plt.title('Graph of Average Meter Reading for ' + meter_value)
    


# # In[34]:


# # Examine the Electricity Consumption throughout the year:

# consumption_grapher('Electricity')

# # We can see that the Electricity consumption generally increases throughout the year until the month of October, when it
# # starts decreasing for the rest of the year.


# # In[ ]:


# # Examine the ChilledWater consumption throughout the year:

# consumption_grapher('ChilledWater')

# # We can see a large increase in ChilledWater consumption from the months of January-August when it is the highest.
# # The consumption then starts dropping, and records the lowest values in the month of December.


# # In[ ]:


# # Examine the Steam consumption throughout the year:

# consumption_grapher('Steam')

# # We can see that the Steam consumption generally follows the similar trend as the trend of all energy types combined.


# # In[ ]:


# # Examine the HotWater consumption throughout the year:

# consumption_grapher('HotWater')

# # We can see that the HotWater consumption also follows the similar trend to the trend of all energy types combined.


# # In[ ]:


# # Use groupby to calculate the min, max, median, count, and std values for each energent for each different month:

# df.groupby(['meter','month'])['meter_reading'].agg(['max','mean','median','count','std'])

# # Based on our findings, we can see that the Steam consumption is generally a lot higher than that of the other energy types
# # for the different months, and has lower counts.


# # In[ ]:

# df['meter_reading','meter','month'].groupby['meter','month'].agg(['max','mean','median','count','std'])






# # In[ ]:


# # Since we cannot see any clear distribution in the energy consumption, we will transform the meter_reading using
# # log transformation:

# df['meter_reading'] = np.log1p(df['meter_reading'])

# # Use the distribution plot to see how the log-transformed energy consumption is distributed:

# sns.distplot(df['meter_reading'])
# plt.title("Distribution of Log-transformed of Meter Reading Variable")

# # Based on the graph we can see that the log-transformed energy consumption follows a nearly normal distribution, however
# # it seems to have a lot of 0-value outliers.


# # In[ ]:


# # Examine the outliers using the Box Plot graphs:

# energy_types = [ 'Electricity', 'ChilledWater', 'HotWater', 'Steam' ]

# def boxplot_grapher(meter_type):
#     sns.boxplot(df[df['meter'] == meter_type]['meter_reading'])
#     plt.title('Boxplot of Meter Reading Variable for the Meter Type: ' + meter_type)

# # Based on our Box Plots, we can see that there are a lot of 0-valued outliers, and that Electricity has the outliers in
# # values greater than 8.


# # In[ ]:


# boxplot_grapher('Electricity')


# # In[ ]:


# boxplot_grapher('ChilledWater')


# # In[ ]:


# boxplot_grapher('HotWater')


# # In[ ]:


# boxplot_grapher('Steam')


# # In[ ]:


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


# # In[ ]:


# # Remove Steam from the Data Set:
# df.drop(df.loc[df['meter']=='Steam'].index, inplace=True)


# # In[ ]:


# # Examine the Energy Distribution throughout the year after implementing the transformation:

# sns.distplot(df['meter_reading'])
# plt.title("Distribution of Log-transformed of Meter Reading Variable")

# # Based on the graph below, the meter reading now follows the normal distribution.


# # ## 4.3 EXAMINE THE WEATHER FEATURES:

# # In[ ]:


# # Define an array to hold the column names of all weather features:
# weather_columns = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_speed', 'wind_direction']

# # Examine the weather data:
# df[weather_columns].describe()


# # In[ ]:


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


# # In[ ]:


# # Now, we will examine the columns to see if they have a very high level of coordination:

# # Threshold for removing correlated variables
# threshold = 0.9

# # Absolute value correlation matrix
# corr_matrix = df.corr().abs()
# corr_matrix.head()


# # In[ ]:


# # Select the columns with very high levels of correlation to be removed
# columns = [column for column in corr_matrix[weather_columns] if any(corr_matrix[column] > threshold)]

# print ("Columns with high levels of correlation to be removed {}".format(columns))

# # We will keep the sea_level_pressure, as it follows the normal distribution, and can be used for energy prediction:

# columns.remove('sea_level_pressure')


# # In[ ]:


# # Based on our findings, the below columns have a very high levels of correlation, and thus can be removed:

# # air_temperature
# # cloud_coverage
# # dew_temperature
# # precip_depth_1_hr
# # wind_speed


# df = df.drop(columns, axis=1)


# # ## 4.4 EXAMINE THE SURFACE AREA:

# # In[ ]:


# df['square_feet'].describe()


# # In[ ]:


# # Use the Distribution Plot to see how the Surface Area is distributed:

# sns.distplot(df['square_feet'])
# plt.title("Distribution of Building's Surface Area: ")


# # In[ ]:


# # The Surface Area seems to be negatively-skewed.

# # Let's transform it using the log based 10 transformation:

# df['square_feet'] = np.log10(df['square_feet'])

# # We can see that after the log-transformation the Surface Area follows an almost Normal Distribution.


# # In[ ]:


# sns.distplot(df['square_feet'])
# plt.title("Distribution of Building's Surface Area: ")


# # ## 4.5 REMOVE FEATURES THAT ARE NO LONGER NEEDED

# # In[ ]:


# # Now, we can remove the features that are no longer needed:

# # The site_id and building_id are used as unique identifiers for buildings and sites they are located, and are thus not very
# # useful for predicting energy.

# # Likewise, we no longer need the timestamp and time features, as they were used for transforming data.

# df = df.drop(['month', 'timestamp', 'site_id', 'building_id'], axis=1)


# # In[ ]:


# df.head(10)


# # In[ ]:


# df.columns


# # In[ ]:




