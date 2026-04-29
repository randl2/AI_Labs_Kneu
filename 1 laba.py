def compute_neuron(x1, x2, bias):
    # Синаптичні ваги за умовою
    w1, w2 = 1, 1
    # Обчислення входу функції активації
    v = x1 * w1 + x2 * w2 + bias
    # Порогова функція активації
    return 1 if v >= 0 else 0

def logic_or(x1, x2):
    return compute_neuron(x1, x2, -0.5)

def logic_and(x1, x2):
    return compute_neuron(x1, x2, -1.5)

def logic_xor(x1, x2):
    # Реалізація XOR через комбінацію нейронів OR та AND
    y1 = logic_or(x1, x2)
    y2 = logic_and(x1, x2)
    return 1 if y1 != y2 else 0

# Вивід результатів
print("x1 x2 | XOR")
print("-" * 12)
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"{x1}  {x2}  |  {logic_xor(x1, x2)}")