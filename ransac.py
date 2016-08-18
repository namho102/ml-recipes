import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')

cloumns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
for i in range(14):
	df = df.rename(columns = {i : cloumns[i]})

# print df.head()

X = df[['RM']].values
y = df['MEDV'].values

# print X

from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np

model = linear_model.LinearRegression()	
ransac = linear_model.RANSACRegressor(model,
	max_trials=100,
	min_samples=50,
	residual_metric=lambda x: np.sum(np.abs(x), axis=1),
	residual_threshold=5.0,
	random_state=0)

# model.fit(X, y)
ransac.fit(X, y)

def lin_regplot(X, y, model):
	plt.scatter(X, y, c='blue')
	plt.plot(X, ransac.predict(X), color='red')
	return None

# lin_regplot(X, y, model)
# plt.xlabel('Average number of rooms [RM] (standardized)')
# plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
# plt.show()

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()