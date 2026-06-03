# 데이터셋 불러오기
from sklearn import datasets
linnerud = datasets.load_linnerud()
print(linnerud)

# 데이터셋 분리(train/test)
from sklearn.model_selection import train_test_split
X = linnerud.data
y = linnerud.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
print(X_train.shape)
print(X_test.shape)

# 선형회귀 모델 학습
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)

# 모델 평가
from sklearn import metrics
y_pred = LR.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("MSE =", mse)

# 새로운 데이터 예측
x_new = [[10, 50, 100], [9, 120, 150]]
y_predict = LR.predict(x_new)
print(y_predict)

import matplotlib.pyplot as plt

target_names = linnerud.target_names

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):
    axes[i].scatter(y_test[:, i], y_pred[:, i])

    # 이상적인 예측선
    min_val = min(y_test[:, i].min(), y_pred[:, i].min())
    max_val = max(y_test[:, i].max(), y_pred[:, i].max())

    axes[i].plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--'
    )

    axes[i].set_title(target_names[i])
    axes[i].set_xlabel("Actual")
    axes[i].set_ylabel("Predicted")

plt.tight_layout()
plt.show()