import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

m = 100
np.random.seed(42)
X = 6 * np.random.rand(m, 1) - 3
y = 0.8 * X**2 + X + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y, title):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []
    
    for m_val in range(1, len(X_train)):
        model.fit(X_train[:m_val], y_train[:m_val])
        y_train_predict = model.predict(X_train[:m_val])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m_val]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір")
    plt.xlabel("Розмір навчального набору", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True)
    plt.ylim([0, 3.5])
    plt.show()

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання: Лінійна регресія (Недонавчання)")

polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_10, X, y, "Криві навчання: Поліном 10-го ступеня (Перенавчання)")

polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_2, X, y, "Криві навчання: Поліном 2-го ступеня (Оптимальна модель)")