import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



# test our ashrae data
df_ashrae_data = pd.read_csv('full_transform_dec5_8pm.csv')

#convert meter back to numeric for scikit
df_ashrae_data['meter'].replace( {'Electricity':0, 'ChilledWater':1, 'HotWater':3 }, inplace=True )

df_ashrae_data =df_ashrae_data.iloc[0:2000]

# cols ,meter,meter_reading,square_feet,sea_level_pressure
X = df_ashrae_data[['meter','square_feet','sea_level_pressure']]

target = df_ashrae_data['meter_reading']

lab_enc = preprocessing.LabelEncoder()
target_encoded = lab_enc.fit_transform(target)



grid = GridSearchCV(RandomForestClassifier(), cv=5,
                    param_grid = {'max_depth': [2, 16], #[2, 4, 8, 16]
                                  'n_estimators': [10]}) #[10, 50, 100]
grid.fit(X, target_encoded)
print('best params: ', grid.best_params_)
print('best score: ', grid.best_score_)
