#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 23:00:26 2017

@author: risingh
"""
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

#X  =np.delete(X, np.s_[3], 1)
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
# score 87








