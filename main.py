from jqdatasdk import *
from sklearn import *
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, explained_variance_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import Normalizer


def login(user, password):
    auth(user, password)


def getData(security, startDate, endDate, frequency, skipPaused):
    global dataNum
    global trainNum
    global testNum
    raw = get_price(security=security, start_date=startDate, end_date=endDate, frequency=frequency,
                    skip_paused=skipPaused)
    # 总数据组数(实际数据组数-1，因为抛弃了一组数据)
    dataNum = raw.shape[0] - 1
    # 训练数据组数
    trainNum = (2 * dataNum) // 3
    # 总测试数据组数
    testNum = dataNum - trainNum
    return raw


def cleanData(dataNum, trainData, mode="train"):
    global X
    global testX
    global testY
    global Y
    global orgY
    # 处理原数数据X，丢弃最后一个X
    for i in range(0, dataNum):
        roll = raw[i:i + 1]
        dateList.append(str(roll.index.values[0])[:10])
        dataX = roll.drop(["open", "volume", "money"], axis=1)
        # dataX = dataX.drop(dataX.columns[0:1],axis=1)
        if mode == "train":
            if i < trainNum:
                X.append(dataX.values.tolist()[0])
            else:
                testX.append(dataX.values.tolist()[0])
        elif mode == "test":
            testX.append(dataX.values.tolist()[0])
    # 处理原始数据Y，相对于X后错一天，意味着丢弃最后一个X和第0个Y
    for i in range(1, dataNum + 1):
        roll = raw[i:i + 1]
        # 股价
        dataY = []
        dataY.append(roll.loc[:, ["money"]].values[0][0] / roll.loc[:, ["volume"]].values[0][0])
        if mode == "train":
            if i < trainNum + 1:
                Y.append(dataY)
            else:
                testY.append(dataY)
        elif mode == "test":
            testY.append(dataY)
        orgY.append(dataY)
    # 归一化
    '''缩放至[0,1]'''
    '''
    X= Normalizer().fit(X).transform(X)
    # Y= Normalizer().fit(Y).transform(Y)
    testX= Normalizer().fit(testX).transform(testX)
    # testY= Normalizer().fit(testY).transform(testY)
    # orgY= Normalizer().fit(orgY).transform(orgY)
    '''
    if mode == "train":
        X = scaler.fit_transform(X)
        Y = scaler.fit_transform(Y)
    testX = scaler.fit_transform(testX)
    testY = scaler.fit_transform(testY)
    orgY = scaler.fit_transform(orgY)


def outputData():
    global X
    global Y
    global testX
    global testY
    global result
    global score
    # 数据输出
    print("X:")
    print(X)
    print("Y:")
    print(Y)
    print("testX:")
    print(testX)
    print("testY")
    print(testY)
    # 结果输出
    print("result:")
    print(result)
    print(score)


def drawPlot(mode="train"):
    # 绘图
    plt.figure(figsize=(2 * 19.2, 2 * 10.8))
    # plt.plot(range(0+trainNum,testY.shape[0]+trainNum),testY,marker=".",linewidth=3,linestyle="-",color="blue")
    '''前面的数据处理过程中Y被提前了一天，此处画图时是时间轴对应的某天当天的股票，所以应延迟一天'''
    plt.plot(range(0 + 1, dataNum + 1), orgY, marker=".", linewidth=1, linestyle="-", color="blue")
    if mode=="train":
        plt.plot(range(trainNum, dataNum), result, marker="x", linewidth=1, linestyle="--", color="orange")
    elif mode=="test":
        plt.plot(range(0, dataNum), result, marker="x", linewidth=1, linestyle="--", color="orange")
    plt.xticks(range(0, dataNum), dateList, rotation=45)
    plt.grid(True)
    # plt.title("params=" + str(clf.best_params_) + "    Score=" + str(score))
    plt.title("params=" + str(clf.best_params_) + "    Score=" + str(score))
    # print(clf.gamma)
    plt.legend(["Real", "Predict"], loc="upper right")
    plt.show()


def _init_():
    global X
    global Y
    global testX
    global testY
    global result
    global score
    global orgY
    global dateList
    global dataNum
    global trainNum
    global testNum
    X = []
    Y = []
    testX = []
    testY = []
    orgY = []
    result = []
    dateList = []
    dataNum = 0
    trainNum = 0
    testNum = 0


user = '13074581737'
password = 'trustno1'
security = "600779.XSHG"
startDate = "2015-01-01"
endDate = "2017-02-14"
frequency = "1d"
skipPaused = True
mode="test"
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
dataNum = 0
# 训练数据组数
trainNum = 0
# 总测试数据组数
testNum = 0
# 声明归一化对象
scaler = preprocessing.MinMaxScaler()

login(user, password)
raw = getData(security, startDate, endDate, frequency, skipPaused)
# raw.to_csv("data.csv")
cleanData(dataNum, trainNum)
# 参数设置
C = 50000
# 参数择优
clf = GridSearchCV(svm.SVR(kernel="rbf"), param_grid={"C": np.logspace(-5, 5, 2), "gamma": np.logspace(-5, 5, 2)})
# 模型训练
clf.fit(X, Y)
# 进行预测
result = clf.predict(testX)
# 结果评分并输出
score = mean_absolute_error(result, testY)
outputData()
drawPlot()

_init_()
getData(security, "2018-01-01", "2019-01-01", frequency, skipPaused)
cleanData(dataNum, trainNum, mode)
# 进行预测
result = clf.predict(testX)
# 结果评分并输出
score = mean_absolute_error(result, testY)
outputData()
drawPlot(mode)
