import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

input_file = 'income_data.txt'
X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 5000  # Обмежимо вибірку для швидкості обчислень

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = [item.strip() for item in line.split(',')]
        if len(data) < 15:
            continue
        label = data[-1].strip('.')
        if label == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif label == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        label_encoder.append(le)
        X_encoded[:, i] = le.fit_transform(X[:, i])

X_data = X_encoded[:, :-1].astype(int)
y_data = X_encoded[:, -1].astype(int)

X_train, X_validation, Y_train, Y_validation = train_test_split(X_data, y_data, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
print("Результати порівняння алгоритмів для набору даних про доходи:")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')

pyplot.boxplot(results, tick_labels=names)
pyplot.title('Algorithm Comparison (Income Dataset)')
pyplot.show()