#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 4 00:22:19 2017

@author: risingh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
# cmd + i inspect
#importing data_set

dataset = pd.read_csv("product.csv");
X = dataset.iloc[:,2:4].values
Y = dataset.iloc[:,4].values

#train and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state=0)


#column age & salary ecludean distance is 
# more, to bridge the gap
#feature scaling for more accurate result
from sklearn.preprocessing import StandardScaler
st_X = StandardScaler()
X_test= st_X.fit_transform(X_test)
X_train= st_X.fit_transform(X_train)

# linear SVC model
from sklearn.svm import SVC
classify = SVC(kernel='rbf',random_state=0)
classify.fit(X_train,Y_train)

Y_pred = classify.predict(X_test)

# confution matrix, to judge the predection
# we have 4 + 3 = 7 wrong predection out of 100(Y_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)


# Visualising 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classify.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('white', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.legend()
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()



