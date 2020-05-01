# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:15:02 2020

@author: Amit
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #independent matrix
y = dataset.iloc[:, 1].values #dependent vector

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# fitting a straight line to our model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # model is trained on these values;

#testing
y_pred = regressor.predict(X_test)

# visualization
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs experience train data")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()
"""
the result of plt.plot is the same because the equation of line is decided in the fit() funtion alredy
the only difference is the start and end of the line
"""
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_pred, color="blue")
plt.title("Salary vs experience test data")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()


plt.scatter(X, y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("Salary vs experience full data")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()
