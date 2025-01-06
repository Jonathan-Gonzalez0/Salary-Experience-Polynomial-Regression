# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:27:28 2025

@author: jonat
"""

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

SalaryData = pd.read_csv("Employee_Salary.csv")

# Prints first 5 data points
print(SalaryData.head())
print("\n")

# Prints last 5 data points
print(SalaryData.tail())
print("\n")

# Prints first 10 data points
print(SalaryData.head(10))
print("\n")

# Prints last 10 data points
print(SalaryData.tail(10))

print(SalaryData.describe())

print(SalaryData.info())

plt.close('all')

# Plots Salary vs Years of Experience
sns.jointplot(x = "Years of Experience", y = "Salary", data = SalaryData)

# Plots how a linear regression would look like
sns.lmplot(x = "Years of Experience", y = "Salary", data = SalaryData)

# Plots Years of Experience vs Salary
sns.jointplot(x = "Salary", y = "Years of Experience", data = SalaryData)

sns.pairplot (data = SalaryData)

x = SalaryData[["Years of Experience"]]

y = SalaryData["Salary"]

x_train = x

y_train = y

from sklearn.linear_model import LinearRegression

regressor = LinearRegression(fit_intercept=True)

regressor.fit(x_train,y_train)

m = regressor.coef_

b = regressor.intercept_

plt.figure()
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train,regressor.predict(x_train), color = "blue")
plt.xlabel("Years of Experience")
plt.ylabel("Salary/Year [$]")
plt.title("Salary vs. Years of Experience (Linear)")

from sklearn.preprocessing import PolynomialFeatures

poly_regressor = PolynomialFeatures(degree = 2)

x_columns = poly_regressor.fit_transform(x_train)

regressor = LinearRegression()

regressor.fit(x_columns, y_train)

print(f"\nModel coefficients: {regressor.coef_}")

y_predict = regressor.predict( x_columns )

plt.figure()
plt.scatter(x_train, y_train, color = 'gray')
plt.plot(x_train, y_predict, color ='r')
plt.xlabel("Years of Experience")
plt.ylabel("Salary/Year [$]")
plt.title("Salary vs. Years of Experience (Poly order = 2)")

poly_regressor = PolynomialFeatures(degree = 5)

x_columns = poly_regressor.fit_transform(x_train)

regressor = LinearRegression()

regressor.fit(x_columns, y_train)

print(f"\nModel coefficients: {regressor.coef_}")

y_predict = regressor.predict( x_columns )

plt.figure()
plt.scatter(x_train, y_train, color = 'gray')
plt.plot(x_train, y_predict, color ='r')
plt.xlabel("Years of Experience")
plt.ylabel("Salary/Year [$]")
plt.title("Salary vs. Years of Experience (Poly order = 5)")