import os
from itertools import groupby

import PySide2
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler    # 导入标准化库
import random
from sklearn.neighbors import NearestNeighbors    # k近邻算法
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LR    # 逻辑回归
from sklearn.svm import SVC    # SVM
from sklearn.ensemble import RandomForestClassifier as RF    # 随机森林
from sklearn.ensemble import AdaBoostClassifier as Adaboost    # AdaBoost
from xgboost import XGBClassifier as XGB    # XGBoost
from sklearn.metrics import precision_score, recall_score, f1_score    # 导入精确率、召回率、F1值等评价指标


#---------------------------配置---------------------------------
#      忽略弹出的warnings信息
warnings.filterwarnings('ignore')
#不然会有qt5错误
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


#---------------------------数据集---------------------------------
data = pd.read_csv('./datasets/Telco-Customer-Churn.csv')
pd.set_option('display.max_columns', None)    # 显示所有列

#---------------------------数据集缺失值处理---------------------------------
#转换成float数据类型
#convert_numeric如果为True，则尝试强制转换为数字，不可转换的变为NaN
data['TotalCharges'] = data['TotalCharges'].apply(pd.to_numeric, errors='coerce')
# 固定值填充，用0
fnDf = data['TotalCharges'].fillna(0).to_frame()
# 用MonthlyCharges的数值填充TotalCharges的缺失值
data['TotalCharges'] = data['TotalCharges'].fillna(data['MonthlyCharges'])
#对有数字的列进行统计
print(data.describe())

#---------------------------客户流失图---------------------------------
# 观察是否存在类别不平衡现象
p = data['Churn'].value_counts()  # 目标变量正负样本的分布
print(p)
plt.figure(figsize=(10, 6))  # 构建图像
# 绘制饼图并调整字体大小
patches, l_text, p_text = plt.pie(p, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
# l_text是饼图对着文字大小，p_text是饼图内文字大小
for t in p_text:
    t.set_size(15)
for t in l_text:
    t.set_size(15)
plt.show()  # 展示图像



#---------------------------特征工程---------------------------------
#数值特征标准化,使数据集方差为1，均值为0     x=(x-mean)/std
scaler = StandardScaler()
data[['tenure']] = scaler.fit_transform(data[['tenure']])
data[['MonthlyCharges']] = scaler.fit_transform(data[['MonthlyCharges']])
data[['TotalCharges']] = scaler.fit_transform(data[['TotalCharges']])

#修改内容使标准化，只有No和Yes
data.loc[data['MultipleLines']=='No phone service', 'MultipleLines'] = 'No'
internetCols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in internetCols:
    data.loc[data[i]=='No internet service', i] = 'No'


#修改内容使标准化，用1代替'Yes’，0代替 'No'
encodeCols = list(data.columns[3: 17].drop(['tenure', 'PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies', 'Contract']))
for i in encodeCols:
    data[i] = data[i].map({'Yes': 1, 'No': 0})
# 顺便把目标变量也进行编码
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})


# 其他无序的类别特征采用独热编码,独立成列
onehotCols = ['InternetService', 'Contract', 'PaymentMethod']
churnDf = data['Churn'].to_frame()    # 取出目标变量列，以便后续进行合并
featureDf = data.drop(['Churn'], axis=1)    # 所有特征列
for i in onehotCols:
    onehotDf = pd.get_dummies(featureDf[i],prefix=i)
    featureDf = pd.concat([featureDf, onehotDf],axis=1)    # 编码后特征拼接到去除目标变量的数据集中
data = pd.concat([featureDf, churnDf],axis=1)    # 拼回目标变量，确保目标变量在最后一列
data = data.drop(onehotCols, axis=1)    # 删除原特征列


# 删去无用特征 'customerID'、'gender'、 'PhoneService'、'StreamingTV'和'StreamingMovies'
data = data.drop(['customerID', 'gender', 'PhoneService', 'StreamingTV', 'StreamingMovies'], axis=1)
data = data.drop(['TotalCharges'],axis=1)

#---------------------------模型训练与预测---------------------------------
# 预测客户流失的概率值
def prob_cv(X, y, classifier, **kwargs):
    """
    :param X: 特征
    :param y: 目标变量
    :param classifier: 分类器
    :param **kwargs: 参数
    :return: 预测结果
    """
    #划分成五分，一份测试，四份训练
    kf = KFold(n_splits=5, random_state=0,shuffle=True)
    y_pred = np.zeros(len(y))

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        clf = classifier(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict_proba(X_test)[:, 1]  # 注：此处预测的是概率值

    return y_pred

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

prob = prob_cv(X, y, RF)    # 预测概率值
prob = np.round(prob, 1)    # 对预测出的概率值保留一位小数，便于分组观察

#合并预测值和真实值
probDf = pd.DataFrame(prob)
churnDf = pd.DataFrame(y)
df1 = pd.concat([probDf, churnDf], axis=1)
df1.columns = ['prob', 'churn']
df1 = df1[:6000]    # 只取原始数据集的6000条样本进行决策
print(df1)


#分组计算每种预测概率子所对应的真实流失率
group=df1.groupby(['prob'])
cnt=group.count()
true_prob=group.sum()/group.count()
df2=pd.concat([cnt,true_prob],axis=1).reset_index()
df2.columns=['prob','cnt','true_prob']
print(df2)

