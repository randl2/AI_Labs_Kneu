import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

X = np.array([
    [2, 0, 1], [2, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1],
    [1, 1, 0], [0, 1, 0], [2, 0, 1], [2, 1, 1], [1, 1, 1],
    [2, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]
])

y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

model = GaussianNB()
model.fit(X, y)

query = np.array([[0, 0, 0]])

predicted = model.predict(query)
probabilities = model.predict_proba(query)

print(f"Прогноз для умов [Overcast, High, Strong]: {'ГРА ВІДБУДЕТЬСЯ (Yes)' if predicted[0] == 1 else 'ГРИ НЕ БУДЕ (No)'}")
print(f"Ймовірність 'No': {round(probabilities[0][0], 4)}")
print(f"Ймовірність 'Yes': {round(probabilities[0][1], 4)}")
