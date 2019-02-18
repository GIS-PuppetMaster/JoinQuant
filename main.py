from jqdatasdk import *
from sklearn import *
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, explained_variance_score
from sklearn.preprocessing import  Normalizer

# 登录
auth('13074581737', 'trustno1')
# 获取原始资金流数据
raw = get_money_flow("600779.XSHG", start_date="2018-01-01", end_date="2019-02-14")
# 训练用X和Y
X = []
Y = []
# 测试用testX和testY（真实值）
testX = []
testY = []
# 储存整个数据集的原始Y值
orgY = []
# 储存实际日期
dateList = []

# 总数据组数(实际数据组数-1，因为抛弃了一组数据)
dataNum = raw.shape[0] - 1
# 训练数据组数
trainNum = (2 * dataNum) // 3
# 总测试数据组数
testNum = dataNum - trainNum

# 声明归一化对象
scaler = preprocessing.MinMaxScaler()
# raw.to_csv("data.csv")
# 处理原数数据X，丢弃最后一个X
for i in range(0, dataNum):
    roll = raw[i:i + 1]
    dateList.append(str(roll['date'].values[0])[:10])
    dataX = roll.drop(
        ["date", "sec_code", "change_pct", "net_amount_main", "net_pct_main", "net_amount_xl", "net_amount_l",
         "net_amount_m", "net_amount_s"], axis=1)
    if i < trainNum:
        X.append(dataX.values.tolist()[0])
    else:
        testX.append(dataX.values.tolist()[0])
# 处理原始数据Y，相对于X后错一天，意味着丢弃最后一个X和第0个Y
for i in range(1, dataNum + 1):
    roll = raw[i:i + 1]
    dataY = roll.loc[:, ["change_pct"]]
    if i < trainNum + 1:
        Y.append(dataY.values.tolist()[0])
    else:
        testY.append(dataY.values.tolist()[0])
    orgY.append(dataY.values.tolist()[0])
# 归一化
'''缩放至[0,1]'''
'''
X= Normalizer().fit(X).transform(X)
# Y= Normalizer().fit(Y).transform(Y)
testX= Normalizer().fit(testX).transform(testX)
# testY= Normalizer().fit(testY).transform(testY)
# orgY= Normalizer().fit(orgY).transform(orgY)
'''

X = scaler.fit_transform(X)
testX = scaler.transform(testX)
Y = scaler.fit_transform(Y)
testY = scaler.transform(testY)
orgY = scaler.transform(orgY)

# 数据输出
print("X:")
print(X)
print("Y:")
print(Y)
print("testX:")
print(testX)
print("testY")
print(testY)
# 参数设置
C = 100000
# 模型训练
clf = svm.SVR(kernel='rbf', gamma='auto',C=C)
# clf = GridSearchCV(svm.SVR(kernel="rbf"), param_grid={"C": np.logspace(-5, 5, 2), "gamma": np.logspace(-5, 5, 2)})
clf.fit(X, Y)
# 进行预测
result = clf.predict(testX)

# 结果输出
print("result:")
print(result)
'''
realY = []
for i in testY:
    realY.append(i[0])
'''
# 结果评分并输出
score = explained_variance_score(result,testY,multioutput="raw_values")
print(score)

# 绘图
plt.figure(figsize=(2 * 19.2, 2 * 10.8))
# plt.plot(range(0+trainNum,testY.shape[0]+trainNum),testY,marker=".",linewidth=3,linestyle="-",color="blue")
plt.plot(range(0, dataNum), orgY, marker=".", linewidth=1, linestyle="-", color="blue")
plt.plot(range(trainNum, dataNum), result, marker="x", linewidth=1, linestyle="--", color="orange")
plt.xticks(range(0, dataNum), dateList, rotation=45)
plt.grid(True)
# plt.title("params=" + str(clf.best_params_) + "    Score=" + str(score))
plt.title("params=" + str(C) + "    Score=" + str(score))
print(clf.gamma)
plt.legend(["Real", "Predict"], loc="upper right")
plt.show()
