import numpy as np
import matplotlib.pyplot as plt

# Вектори даних згідно із завданням
x_points = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y_points = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

coeffs = np.polyfit(x_points, y_points, 4)
p = np.poly1d(coeffs)

val_02 = p(0.2)
val_05 = p(0.5)

print(f"Значення в точці x=0.2: {val_02:.4f}")
print(f"Значення в точці x=0.5: {val_05:.4f}")

x_axis = np.linspace(0.1, 0.7, 100)
plt.plot(x_axis, p(x_axis), 'g-', label='Інтерполяційний поліном')
plt.scatter(x_points, y_points, color='black', label='Вузли інтерполяції')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Інтерполяція поліномом 4-го ступеня')
plt.legend()
plt.grid(True)
plt.show()