#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 23:00:26 2017

@author: ricky
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data_set

dataset = pd.read_csv("Salary_Data.csv");
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#train and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state=0)

#linear regression train the model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,Y_train)


# plot train
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,lr.predict(X_train),color = 'green')
plt.show()

# plot test
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,lr.predict(X_test),color = 'green')
plt.show()



