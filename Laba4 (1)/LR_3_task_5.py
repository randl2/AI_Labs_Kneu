import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

m = 100
np.random.seed(42)
X = 6 * np.random.rand(m, 1) - 3
y = 0.8 * X**2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

print("Справжня (модельна) формула: y = 0.80 * x^2 + 1.00 * x + 2.00")
print(f"Отримана моделлю формула:    y = {lin_reg.coef_[0][1]:.2f} * x^2 + {lin_reg.coef_[0][0]:.2f} * x + {lin_reg.intercept_[0]:.2f}")

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new_predict = lin_reg.predict(X_new_poly)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', s=15, label='Випадкові дані (Варіант 2)')
plt.plot(X_new, y_new_predict, color='red', linewidth=3, label='Прогнози (Поліном 2-го ступеня)')
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$y$', fontsize=14, rotation=0)
plt.title("Поліноміальна регресія для індивідуальних даних")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()