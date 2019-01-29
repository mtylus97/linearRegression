import inline as inline
import matplotlib

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np

dane = pd.read_csv('Pokemon.csv')

x = dane[['Attack', 'Defense', 'HP']].values
y = dane['Total'].values

model = linear_model.LinearRegression()

model.fit(x, y)

ypred = model.predict(x)

print("Absolute error: ", mean_absolute_error(y, ypred))
print("Squared error: ", mean_squared_error(y, ypred))

matplotlib
inline
plt.xlabel('Numer pokemona', fontsize=20)
plt.ylabel('Total', fontsize=20)

plt.scatter(dane.X, dane.Total, color='red', marker='.')
plt.scatter(dane.X, ypred, color='blue', marker='.')

plt.show()
