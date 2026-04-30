import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Коефіцієнти регресії (regr.coef_): \n", regr.coef_)
print("\nТочка перетину (regr.intercept_): ", regr.intercept_)
print("\nКоефіцієнт детермінації (R2 score): ", round(r2_score(ytest, ypred), 2))
print("Середня абсолютна похибка (MAE): ", round(mean_absolute_error(ytest, ypred), 2))
print("Середньоквадратична похибка (MSE): ", round(mean_squared_error(ytest, ypred), 2))

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно (Реальні значення)')
ax.set_ylabel('Передбачено (Прогноз моделі)')
plt.title("Прогнозування розвитку діабету")
plt.show()