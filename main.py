from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import accuracy_score
import keras.datasets.mnist as mnist

# from hw2 #
def cross_validation_error(X, y, model, folder):
    sum_train_error = 0
    sum_val_error = 0
    kf = KFold(n_splits=folder)
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        fit_model = model.fit(X_train, y_train)
        X_train_pred = fit_model.predict(X_train)
        X_val_pred = fit_model.predict(X_val)
        sum_train_error += 1 - accuracy_score(X_train_pred, y_train)
        sum_val_error += 1 - accuracy_score(X_val_pred, y_val)
    avg_train_error = sum_train_error / folder
    avg_val_error = sum_val_error / folder
    return [avg_train_error, avg_val_error]

filterwarnings('ignore')

# A
def SVM_results(X_train, X_test, y_train, y_test):
    results_dict = {}
    d = [2,4,6,8,10]
    g = [0.001, 0.01, 0.1, 1, 10]
    model = svm.SVC(kernel='linear')
    linear_v = cross_validation_error(X_train, y_train, model, 5)
    fit_m = model.fit(X_train, y_train)
    test_pred = fit_m.predict(X_test)
    results_dict['svm_linear'] = [linear_v[0], linear_v[1], 1 - accuracy_score(test_pred, y_test)]
    for i in d:
        model = svm.SVC(kernel='poly', degree=i)
        poly_v = cross_validation_error(X_train, y_train, model, 5)
        fit_m = model.fit(X_train, y_train)
        test_pred = fit_m.predict(X_test)
        results_dict['svm_poly_degree_' + str(i)] = [poly_v[0], poly_v[1], 1 - accuracy_score(test_pred, y_test)]
    for i in g:
        model = svm.SVC(kernel='rbf', gamma=i)
        rbf_v = cross_validation_error(X_train, y_train, model, 5)
        fit_m = model.fit(X_train, y_train)
        test_pred = fit_m.predict(X_test)
        results_dict['svm_rbf_gamma_' + str(i)] = [rbf_v[0], rbf_v[1], 1 - accuracy_score(test_pred, y_test)]
    print(results_dict)
    return results_dict


# B
def load_mnist():
    np.random.seed(2)
    (X, y), (_, _) = mnist.load_data()
    indexes = np.random.choice(len(X), 8000, replace=False)
    X = X[indexes]
    y = y[indexes]
    X = X.reshape(len(X), -1)
    return X, y


# C
x, y = load_mnist()
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=98)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# D
model = svm.SVC(kernel='linear')
fit_model = model.fit(X_train, y_train)
y_pred = fit_model.predict(X_test)
my_confusion_matrix = confusion_matrix(y_test, y_pred)
my_confusion_matrix_norm = my_confusion_matrix.astype('float') / my_confusion_matrix.sum(axis=1)[:, np.newaxis]

sn.heatmap(my_confusion_matrix_norm, annot=True, cmap="tab20b",fmt=".2f")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Normalized Confusion Matrix')
plt.show()

# E
train_e = []
val_e = []
test_e = []
results_dict = SVM_results(X_train, X_test, y_train, y_test)

for model in results_dict.values():
    train_e.append(round(model[0], 3))
    val_e.append(round(model[1], 3))
    test_e.append(round(model[2], 3))

w = 0.25
x = np.arange(len(train_e))
f, a = plt.subplots(figsize=(12,6))
r1 = a.bar(x - w, train_e, w, label='Train Error')
r2 = a.bar(x, val_e, w, label='Validation Error')
r3 = a.bar(x + w, test_e, w, label='Test Error')

for i in range(len(train_e)):
    plt.text(x = x[i]-0.38, y =train_e[i]+0.01, s = np.around(train_e[i] ,decimals=3), size = 6)
    plt.text(x = x[i]-0.1 , y=val_e[i] + 0.01, s=np.around(val_e[i], decimals=3), size=6)
    plt.text(x = x[i]+0.15, y=test_e[i] + 0.01, s=np.around(test_e[i], decimals=3), size=6)

a.set_ylabel('Errors')
a.set_title('Errors by different models')
x = np.arange(len(train_e))
a.set_xticks(x)
a.set_xticklabels(tuple(results_dict.keys()), fontsize=6)
a.legend()

f.tight_layout()
plt.show()