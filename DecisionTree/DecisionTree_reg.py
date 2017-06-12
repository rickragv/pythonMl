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


from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(criterion="mse",random_state=0);
reg.fit(X,Y)

#predection
yPred = reg.predict(7.3)

#plot
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'black')
plt.plot(X_grid, reg.predict(X_grid), color = 'red')
plt.show()
