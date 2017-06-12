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

#importing data_set

dataset = pd.read_csv("Salaries.csv");
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#python svm does not support scaling by default
# so we are doing of our own 
from sklearn.preprocessing import StandardScaler
st_X = StandardScaler()
st_Y = StandardScaler()
X= st_X.fit_transform(X)
Y= st_Y.fit_transform(Y)

from sklearn.svm import SVR

lr = SVR(kernel ='rbf') #default kernel
lr.fit(X,Y)

# predict salary
st_Y.inverse_transform(lr.predict(st_X.inverse_transform(np.array([[7.3]]))))


# plot train
plt.scatter(X,Y,color='red')
plt.plot(X,lr.predict(X),color = 'green')
plt.show()
