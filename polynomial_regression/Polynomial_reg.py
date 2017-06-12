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
# cmd + i inspect
#importing data_set

dataset = pd.read_csv("Salaries.csv");
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X,Y)
value = lr.predict(X)

# predict salary
lr.predict(7.3)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3);
X_poly = poly_reg.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(X_poly,Y)


# predict salary polynomial
lr2.predict(poly_reg.fit_transform(7.3))


# plot train
plt.scatter(X,Y,color='red')
plt.plot(X,lr.predict(X),color = 'green')
plt.show()

# plot train
plt.scatter(X,Y,color='red')
plt.plot(X,lr2.predict(X_poly),color = 'green')
plt.show()

