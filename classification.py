from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.learning_curve
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
file = open('E:/Data/Abalone/data.txt', 'r')
data = []
for line in file:
    line = line.replace('\n', '')
    features = line.split(',')
    features[1:] = [float(x) for x in features[1:]]
    data.append(features)

# 使用OneHot表示标称型属性
np_data = np.array(data)
np_data[:, 0] = preprocessing.LabelEncoder().fit_transform(np_data[:, 0])
np_data = preprocessing.OneHotEncoder(categorical_features=[0]).fit_transform(np_data).toarray()

X = np_data[:, :-1]
y = np_data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 定义、训练模型
# model = SVC(gamma=2.2, C=2, decision_function_shape='ovo')
model = DecisionTreeClassifier()
# model = KNeighborsClassifier()
model.fit(X_train, y_train)

# 评估模型
print(model.score(X_test, y_test))
print('p:', model.predict(X_test[:10]))
print('y:', y_test[:10])

# 保存模型
joblib.dump(model, 'model.pkl')
# 读取模型
model1 = joblib.load('model.pkl')

# 调参
# gamma_range = np.linspace(0.5, 1.5, 10)
# train_loss, test_loss = sklearn.learning_curve.validation_curve(
#     SVC(), data, target, param_name='gamma', param_range=gamma_range,
#     cv=10, scoring='mean_squared_error'
# )
#
# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_loss, axis=1)

# 可视化
# plt.plot(gamma_range, train_loss_mean, 'o-', color='r', label='Training')
# plt.plot(gamma_range, test_loss_mean, 'o-', color='g', label='Cross-validation')
#
# plt.xlabel('gamma')
# plt.ylabel('loss')
# plt.legend(loc='best')
# plt.show()
