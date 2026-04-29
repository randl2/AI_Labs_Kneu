import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print("Форма датасету:", dataset.shape)
print("\nПерші 10 рядків:\n", dataset.head(10))
print("\nСтатистичне резюме:\n", dataset.describe())
print("\nРозподіл за класами:\n", dataset.groupby('class').size())

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.suptitle("Діаграма розмаху (Boxplot)")
pyplot.show()

dataset.hist()
pyplot.suptitle("Гістограми розподілу")
pyplot.show()

scatter_matrix(dataset)
pyplot.suptitle("Матриця діаграм розсіювання")
pyplot.show()

array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names_models = []
print("\nРезультати порівняння алгоритмів (середня точність та стандартне відхилення):")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names_models.append(name)
    print(f'{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')

pyplot.boxplot(results, tick_labels=names_models)
pyplot.title('Algorithm Comparison')
pyplot.show()

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\nОцінка якості моделі SVM на контрольній вибірці:")
print("Accuracy Score:", accuracy_score(Y_validation, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(Y_validation, predictions))
print("\nClassification Report:\n", classification_report(Y_validation, predictions))

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = model.predict(X_new)
print("\nПрогноз для нової квітки [5, 2.9, 1, 0.2]:")
print("Результат передбачення (клас):", prediction[0])