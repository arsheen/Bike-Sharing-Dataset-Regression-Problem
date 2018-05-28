import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error
from sklearn.svm import LinearSVR
import numpy as np
from scipy import stats

import warnings
warnings.filterwarnings("ignore")


# data cleaning i.e. removing outliers
def remove_outliers(data, type):
    print("Shape of {} Data frame before removing Outliers: {}".format(type, data.shape))
    no_outliers = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
    print("Shape of {} Data frame before removing Outliers: {}".format(type, no_outliers.shape))
    return no_outliers

def mse(x,y,model,Y_test,predictions):
    mse = mean_squared_error(Y_test, predictions)
    print("\n\t\tMSE without cross validation            : {}".format(mse))
    reg = cross_val_score(estimator=model, X=x, y=y, scoring='mean_squared_error', cv=10)
    score = reg.mean() * -1
    print("\t\tMSE with cross validation               : {}".format(score))

def mae(x,y,model,Y_test,predictions):
    mae = mean_absolute_error(Y_test, predictions)
    print("\n\t\tMAE without cross validation            : {}".format(mae))
    reg = cross_val_score(estimator=model, X=x, y=y, scoring='mean_absolute_error', cv=10)
    score = reg.mean() * -1
    print("\t\tMAE with cross validation               : {}".format(score))

def r2_scoring(x,y,model,Y_test,predictions):
    r2 = r2_score(Y_test, predictions)
    print("\n\t\tR squared score without cross validation: {}".format(r2))
    reg = cross_val_score(estimator=model, X=x, y=y, scoring='r2', cv=10)
    score = reg.mean()
    print("\t\tR squared score with cross validation   : {}".format(score))

def hour_data(model):
    print("\n\t Hour Dataset")
    model.fit(X_hour_train, Y_hour_train)
    predictions = model.predict(X_hour_test)
    mse(x_hour, y_hour, model, Y_hour_test, predictions)
    mae(x_hour, y_hour, model, Y_hour_test, predictions)
    r2_scoring(x_hour, y_hour, model, Y_hour_test, predictions)

def day_data(model):
    print("\n\t Day Dataset")
    model.fit(X_day_train, Y_day_train)
    predictions = model.predict(X_day_test)
    mse(x_day, y_day, model, Y_day_test, predictions)
    mae(x_day, y_day, model, Y_day_test, predictions)
    r2_scoring(x_day, y_day, model, Y_day_test, predictions)



#Read and drop missing values if any
hourly_data = pd.read_csv('/Users/arsheenkhatib/Downloads/Bike-Sharing-Dataset/hour.csv', na_values='?').dropna()
daily_data = pd.read_csv('/Users/arsheenkhatib/Downloads/Bike-Sharing-Dataset/day.csv', na_values='?').dropna()

#change date to int
list_dh=[]
for i in hourly_data['dteday']:
    list1 = i.split('-')
    list_dh.append(int(list1[2]))
dfh = pd.DataFrame(list_dh, columns=['dteday'])
hourly_data[['dteday']]=dfh[['dteday']]

list_dd=[]
for i in daily_data['dteday']:
    list2 = i.split('-')
    list_dd.append(int(list2[2]))
dfd = pd.DataFrame(list_dd, columns=['dteday'])
daily_data[['dteday']]=dfd[['dteday']]


no_outliers=remove_outliers(hourly_data,'Hourly')
y_hour = no_outliers.cnt
x_hour = no_outliers.drop(['cnt','instant','registered','casual'],axis=1)


no_outliers=remove_outliers(daily_data,'Daily')
y_day = no_outliers.cnt
x_day = no_outliers.drop(['cnt','instant','registered','casual'],axis=1)


# Using Lasso for feature selection

'''for k in range(1,10):
    clf = Lasso(alpha = k/10)
    clf.fit(x_hour,y_hour)
    print('alpha', k/10)
    print(clf.coef_)
    print('\n')

for k in range(1,10):
    clf = Lasso(alpha = k/10)
    clf.fit(x_day,y_day)
    print('alpha', k/10)
    print(clf.coef_)
    print('\n')
'''

# choosing alpha as 0.8 where coefficients of holiday, atemp and windspeed are becoming zero.
x_hour = x_hour.drop(['holiday','atemp','windspeed'],axis=1)

# choosing alpha as 0.3 where coefficients of holiday is becoming zero.
x_day = x_day.drop(['holiday','dteday','mnth'],axis=1)


#Splitting data
X_hour_train, X_hour_test, Y_hour_train, Y_hour_test = train_test_split(x_hour, y_hour, test_size=0.3, random_state=5)
X_day_train, X_day_test, Y_day_train, Y_day_test = train_test_split(x_day, y_day, test_size=0.3, random_state=5)


#Random Forest
rgr = RandomForestRegressor(n_estimators=50)
print('\n\nRandom Forest')
hour_data(rgr)
day_data(rgr)


#Linear Regression
lr = LinearRegression(fit_intercept=True)
print("\n\nLinear Regression")
hour_data(lr)
day_data(lr)


#Support Vector Regression
svr = LinearSVR()
print("\n\nSupport Vector Regression")
hour_data(svr)
day_data(svr)

