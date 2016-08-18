import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
# df = pd.DataFrame(df)

cloumns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
for i in range(14):
	df = df.rename(columns = {i : cloumns[i]})

print df.head()

X = df[['RM']].values
y = df['MEDV'].values

# print X

from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

model = linear_model.LinearRegression()
model.fit(X, y)

def lin_regplot(X, y, model):
	plt.scatter(X, y, c='blue')
	plt.plot(X, model.predict(X), color='red')
	return None

lin_regplot(X, y, model)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()