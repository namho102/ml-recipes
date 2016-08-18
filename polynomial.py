from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')

cloumns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
for i in range(14):
	df = df.rename(columns = {i : cloumns[i]})

# sample data
# X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
# y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])[:, np.newaxis]

X = df[['RM']].values
y = df['MEDV'].values
 	
lr = linear_model.LinearRegression()
pr = linear_model.LinearRegression()
quadratic = PolynomialFeatures(degree = 2)
X_quad = quadratic.fit_transform(X)

# Fit a simple linear regression model for comparison
lr.fit(X, y)
X_fit = np.arange(np.amin(X) ,np.amax(X), .1)[:, np.newaxis]
# print X_fit
y_lin_fit = lr.predict(X_fit)

# Fit a multiple regression model on the transformed features for polynomial regression
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# plot the result
plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--', color='red')
plt.plot(X_fit, y_quad_fit, label='quadratic fit', color='green')
plt.legend(loc='upper left')
plt.show()