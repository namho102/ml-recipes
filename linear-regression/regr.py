
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

N = 50
X = [[886], [911], [930], [947], [958], [972]]
# y = [2875.96, 2968.28, 3048.882, 3207.66, 3364.82, 4633.92]
# y = [1576.3, 1670.16, 1764.61, 1927.93, 1737.53, 1837.11]
y= [2435.85, 2544.62, 2660.2569, 2735.4054, 2816.9952, 2709.5]
# Create linear regression object
regr = linear_model.LinearRegression()
X_test = [[850], [1000]]

# Train the model using the training sets
regr.fit(X, y)

# The coefficients
print('Coefficients: \n', regr.coef_)

# Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

print regr.predict(X_test)

fig = plt.figure()
fig.suptitle('', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.set_ylabel('Dien tich chuyen dung')
ax.set_xlabel('Mat do dan so')

plt.scatter(X, y, color='black')

plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)


plt.show()