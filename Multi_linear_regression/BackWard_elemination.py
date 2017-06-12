#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 23:00:26 2017

@author: risingh
"""
# remove independent value which has less
# significance 
#we use backward elimination selection 
#.05 or 5% as P value , remove all columns 
# until we have P value less then .05
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

dataset = pd.read_csv("Company_Profit.csv");
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder = LabelEncoder()
X[:,3]=labelEncoder.fit_transform(X[:,3])

#create dummy variables
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

# removing one dummy variable from model
X = X[:,1:]

# backward elemanition 
# remove highest P value
import statsmodels.formula.api as sm
rows = np.size(X,0)
X_temp = np.append(arr = np.ones((rows,1)).astype(int),values = X ,axis = 1)
#orderly square
OLS_reg = sm.OLS(endog = Y,exog = X_temp).fit()
OLS_reg.summary()

X_temp = X_temp[:,[0,1,3,4,5]]
OLS_reg = sm.OLS(endog = Y,exog = X_temp).fit()
OLS_reg.summary()

X_temp = X_temp[:,[0,2,3,4]]
OLS_reg = sm.OLS(endog = Y,exog = X_temp).fit()
OLS_reg.summary()

X_temp = X_temp[:,[0,1,3]]
OLS_reg = sm.OLS(endog = Y,exog = X_temp).fit()
OLS_reg.summary()

# now we no longer remove X2 column as it has value > 0.05
# but Adj R-squared will further decrease.
# Adj R-squared will be needed close to 1


X = X_temp
#train and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state=0)

#linear regression train the model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,Y_train)
value = lr.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(Y_test, value)
# score of 92 




