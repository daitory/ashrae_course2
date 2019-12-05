#Paste this in the group notebook
for i in list(weather_columns):
    for j in list(weather_columns):
        print(i, ' vs ', j, corr_matrix.loc[i,j] , ' is greater than 0.9 ? : ', corr_matrix.loc[i,j] > threshold)