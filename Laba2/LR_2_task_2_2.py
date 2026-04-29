import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score

input_file = 'income_data.txt' 

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

print("Читаємо файл і збираємо дані...")
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

if len(X) == 0:
    print("ПОМИЛКА: Масив даних порожній! Перевір, чи лежить файл adult.data у тій самій папці, що і скрипт.")
    exit()

print(f"Зібрано {len(X)} записів. Кодуємо текст у цифри...")
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

print("Навчаємо SVM-класифікатор (Гаусове ядро / RBF)...")
classifier = SVC(kernel='rbf', max_iter=10000)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=5)
classifier.fit(X_train, y_train)

f1 = cross_val_score(classifier, X_data, y_data, scoring='f1_weighted', cv=3)
accuracy = cross_val_score(classifier, X_data, y_data, scoring='accuracy', cv=3)
precision = cross_val_score(classifier, X_data, y_data, scoring='precision_weighted', cv=3)
recall = cross_val_score(classifier, X_data, y_data, scoring='recall_weighted', cv=3)

print("F1-міра: " + str(round(100 * f1.mean(), 2)) + "%")
print("Акуратність: " + str(round(100 * accuracy.mean(), 2)) + "%")
print("Точність: " + str(round(100 * precision.mean(), 2)) + "%")
print("Повнота: " + str(round(100 * recall.mean(), 2)) + "%")

# 5. ПЕРЕДБАЧЕННЯ ДЛЯ ТЕСТОВОЇ ТОЧКИ
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', 
              '0', '0', '40', 'United-States']

input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array([input_data_encoded])
predicted_class = classifier.predict(input_data_encoded)

print("Прогнозований клас для тестової точки:", label_encoder[-1].inverse_transform(predicted_class)[0])
