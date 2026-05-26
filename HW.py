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