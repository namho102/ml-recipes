
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

N = 50
X = [[2], [3], [4], [7]]
y = [2, 4, 3, 10]

X_test = [[1], [10]]
X_test2 = [[5], [9]]
# y_test = [[6]]
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, y)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean square error
# print("Residual sum of squares: %.2f"
#       % np.mean((regr.predict(X_test) - y_test) ** 2))

print regr.predict(X_test)
plt.scatter(X, y, color='black')
plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)
# plt.plot(X_test2, regr.predict(X_test2), color='yellow', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()