import numpy as np
import matplotlib.pyplot as plt

# Дані Варіанта 7
x = np.array([-12, 29, 0, 4, 6, 8])
y = np.array([-3, 0, 1, 2, 9, 5])

beta1, beta0 = np.polyfit(x, y, 1)

print(f"Рівняння прямої: y = {beta0:.2f} + {beta1:.2f}x")

# Візуалізація
plt.scatter(x, y, color='red', label='Точки (вар. 7)')
plt.plot(x, beta1*x + beta0, color='blue', label='Лінія регресії')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Лінійна апроксимація (МНК)')
plt.legend()
plt.grid(True)
plt.show()
