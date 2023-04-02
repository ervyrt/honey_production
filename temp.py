# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv(r"C:\Users\yurtb\OneDrive\Desktop\honey_production\US_honey_dataset_updated.csv")
print(df.head())
prod_per_year=df.groupby('year').production.mean().reset_index()
print(prod_per_year.head())
X=prod_per_year['year']
X = X.values.reshape(-1, 1)
y=prod_per_year["production"]
plt.scatter(X,y)
regr=linear_model.LinearRegression()
regr.fit(X,y)
print("m=",regr.coef_,"b=",regr.intercept_)
y_predicted=regr.predict(X)
plt.plot(X,y_predicted)
X_future = np.array(range(2021, 2050))
X_future = X_future.reshape(-1, 1)
y_future=regr.predict(X_future)
plt.plot(X_future,y_future)
plt.show()