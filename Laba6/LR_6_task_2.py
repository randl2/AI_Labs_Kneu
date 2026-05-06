import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv'
df = pd.read_csv(url)

df = df.drop(['insert_date', 'start_date', 'end_date'], axis=1)
df = df.dropna()

le = LabelEncoder()
cols = ['origin', 'destination', 'train_type', 'train_class', 'fare']
for col in cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('fare', axis=1)
y = df['fare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("РЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ RENFE")
print(f"Точність моделі (Accuracy): {round(accuracy_score(y_test, y_pred), 4)}")
print("\nДетальний звіт:")
print(classification_report(y_test, y_pred))