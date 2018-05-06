from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# print(iris_X)
# print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print(knn.predict(X_test))
print(y_test)
print(knn.get_params())
print(knn.score(X_test, y_test))

import pickle
#
# with open('clf.pickle', 'wb') as f:
#     pickle.dump(knn, f)
# with open('clf.pickle', 'rb') as f:
#     knn1 = pickle.load(f)
#     print(knn1.score(X_test, y_test))

# from sklearn.externals import joblib
# joblib.dump(knn, 'knn.pkl')
# knn1 = joblib.load('knn.pkl')
# print(knn1.score(X_test, y_test))



# X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
# plt.scatter(X, y)
# plt.show()


# X, y = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
#                                     random_state=22, n_clusters_per_class=1, scale=100)
# X = preprocessing.scale(X)      # normalization
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# svc = SVC()
# svc.fit(X_train, y_train)
#
# print(svc.score(X_test, y_test))


# from sklearn.cross_validation import cross_val_score
# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn, iris_X, iris_y, cv=5, scoring='accuracy')     # 交叉验证
# # loss = cross_val_score(knn, iris_X, iris_y, cv=5, scoring='mean_squared_error')
# print(scores)
# print(scores.mean())


# digits = datasets.load_digits()
# X = digits.data
# y = digits.target
#
# train_sizes, train_loss, test_loss = learning_curve(
#     SVC(gamma=0.001), X, y, cv=10, scoring='mean_squared_error',
#     train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
# )
#
# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_loss, axis=1)
#
# print(train_loss_mean)
# print(test_loss_mean)


# digits = datasets.load_digits()
# X = digits.data
# y = digits.target
# param_range = np.logspace(-6, -2.3, 5)
#
# train_loss, test_loss = validation_curve(
#     SVC(), X, y, param_name='gamma', param_range=param_range,
#     cv=10, scoring='mean_squared_error'
# )
#
# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_loss, axis=1)
#
# print(train_loss_mean)
# print(test_loss_mean)
#
# # 可视化
# plt.plot(param_range, train_loss_mean, 'o-', color='r', label='Training')
# plt.plot(param_range, test_loss_mean, 'o-', color='g', label='Cross-validation')
#
# plt.xlabel('gamma')
# plt.ylabel('loss')
# plt.legend(loc='best')
# plt.show()

