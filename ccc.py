import pandas as pd
import numpy as np
from itertools import groupby
import seaborn as sns
import json
import math
import paddle
from paddle.io import DataLoader, Dataset
import torch
from tqdm import tqdm
import torchvision
from sklearn.model_selection import StratifiedKFold
#---------------------------------预处理---------------------
#忽视报错
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
#如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#--------------------------加载数据---------------------------
PATH = './'
data_dir = "./wsdm_model_data/"
user_interaction = pd.read_csv(PATH + 'user_interaction_data.csv')#用户行为信息
user_portrait = pd.read_csv(PATH + 'user_portrait_data.csv')#用户基础信息
user_playback = pd.read_csv(PATH + 'user_playback_data.csv')#用户观看信息
app_launch = pd.read_csv(PATH + 'app_launch_logs.csv')#用户登录时间和方式
video_related = pd.read_csv(PATH + 'video_related_data.csv')#视频信息
data = pd.read_csv(data_dir + "www.txt", sep="\t")
test_data = pd.read_csv(data_dir + "www.txt", sep="\t")
#----------------------------------------定义模型数据集-------------------------------------
class CoggleDataset(Dataset):
    #构造函数
    def __init__(self, df):
        #继承父类构造函数
        super(CoggleDataset, self).__init__()
        self.df = df
        #定义其他几列组合在一起
        self.feat_col = list(set(self.df.columns) - set(['user_id', 'end_date', 'label', 'launch_seq', 'playtime_seq',
                                                         'duration_prefer', 'interact_prefer']))
        self.df_feat = self.df[self.feat_col]

    # 定义需要参与训练的字段
    def __getitem__(self, index):
        #按行索引
        launch_seq = self.df['launch_seq'].iloc[index]
        playtime_seq = self.df['playtime_seq'].iloc[index]
        duration_prefer = self.df['duration_prefer'].iloc[index]
        interact_prefer = self.df['interact_prefer'].iloc[index]
        label=self.df['label'].iloc[index]
        feat = self.df_feat.iloc[index].values.astype(np.float32)
        # 字符串类型转换成list
        data["launch_seq"] = json.loads(data["launch_seq"])
        data["playtime_seq"] = json.loads(data["playtime_seq"])
        data["duration_prefer"] = json.loads(data["duration_prefer"])
        data["interact_prefer"] = json.loads(data["interact_prefer"])
        #转换成张量,并且转换数据类型
        launch_seq = paddle.to_tensor(launch_seq).astype(paddle.float32)
        playtime_seq = paddle.to_tensor(playtime_seq).astype(paddle.float32)
        duration_prefer = paddle.to_tensor(duration_prefer).astype(paddle.float32)
        interact_prefer = paddle.to_tensor(interact_prefer).astype(paddle.float32)
        label = paddle.to_tensor(label).astype(paddle.float32)
        feat = paddle.to_tensor(feat).astype(paddle.float32)
        return launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label
    #长度
    def __len__(self):
        return len(self.df)

#----------------------------------------定义模型（LSTM + FC）-------------------------------------
class CoggleModel(paddle.nn.Layer):
    #构造函数
    def __init__(self):
        #继承父类
        super(CoggleModel, self).__init__()
        # 循环神经网络层，门控循环单元网络
        self.launch_seq_gru = paddle.nn.GRU(1, 32)
        self.playtime_seq_gru = paddle.nn.GRU(1, 32)
        # 线性层，线性变换层，全连接层
        self.fc1 = paddle.nn.Linear(102, 64)
        self.fc2 = paddle.nn.Linear(64, 1)

    def forward(self, launch_seq, playtime_seq, duration_prefer, interact_prefer, feat):
        #改变数组形状，相当于转置
        launch_seq = launch_seq.reshape((-1, 32, 1))
        playtime_seq = playtime_seq.reshape((-1, 32, 1))

        launch_seq_feat = self.launch_seq_gru(launch_seq)[0][:, :, 0]
        playtime_seq_feat = self.playtime_seq_gru(playtime_seq)[0][:, :, 0]

        all_feat = paddle.concat([launch_seq_feat, playtime_seq_feat, duration_prefer, interact_prefer, feat], 1)
        all_feat_fc1 = self.fc1(all_feat)
        all_feat_fc2 = self.fc2(all_feat_fc1)

        return all_feat_fc2

#----------------------------------------模型训练--------------------------------
# 模型训练函数
def train(model, train_loader, optimizer, criterion):#模型，数据集，优化，标准
    # 启用BatchNormalization和Dropout，泛化能力更强
    model.train()
    train_loss = []
    #进度条
    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label in tqdm(train_loader):
        #预测
        pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)
        #损失
        loss = criterion(pred, label)
        loss.backward()
        #优化
        optimizer.step()
        optimizer.clear_grad()
        train_loss.append(loss.item())
    return np.mean(train_loss)

# 模型验证函数
def validate(model, val_loader, optimizer, criterion):#模型，数据集，优化，标准
    #不启用BatchNormalization和Dropout。
    model.eval()
    val_loss = []
    #进度条
    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label in tqdm(val_loader):
        #预测
        pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)
        #损失
        loss = criterion(pred, label)
        loss.backward()
        #优化
        optimizer.step()
        optimizer.clear_grad()
        val_loss.append(loss.item())
    return np.mean(val_loss)
# 模型预测函数
def predict(model, test_loader):#模型，数据集
    # 不启用BatchNormalization和Dropout
    model.eval()
    test_pred = []
    #进度条
    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label in tqdm(test_loader):
        #预测
        pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)
        test_pred.append(pred.numpy())
    return test_pred

#------------------------------------模型训练----------------------------------
# 字符串类型转换成list
test_data["launch_seq"] = test_data.launch_seq.apply(lambda x: json.loads(x))
test_data["playtime_seq"] = test_data.playtime_seq.apply(lambda x: json.loads(x))
test_data["duration_prefer"] = test_data.duration_prefer.apply(lambda x: json.loads(x))
test_data["interact_prefer"] = test_data.interact_prefer.apply(lambda x: json.loads(x))
test_data['label'] = 0
#k折交叉验证
skf = StratifiedKFold(n_splits=7)
fold = 0
for tr_idx, val_idx in skf.split(data, data['label']):
    train_dataset = data.iloc[tr_idx]
    val_dataset = data.iloc[val_idx]
    # 定义模型、损失函数和优化器
    model = CoggleModel()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
    criterion = paddle.nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 每个epoch训练
    for epoch in range(3):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, optimizer, criterion)

        print(fold, epoch, train_loss, val_loss)

        paddle.save(model.state_dict(), f"model_{fold}.pdparams")
    fold += 1
#----------------------------------模型预测------------------------------
test_data = pd.read_csv(data_dir + "test_data.txt", sep="\t")
# 字符串类型转换成list
test_data["launch_seq"] = test_data.launch_seq.apply(lambda x: json.loads(x))
test_data["playtime_seq"] = test_data.playtime_seq.apply(lambda x: json.loads(x))
test_data["duration_prefer"] = test_data.duration_prefer.apply(lambda x: json.loads(x))
test_data["interact_prefer"] = test_data.interact_prefer.apply(lambda x: json.loads(x))
test_data['label'] = 0

test_dataset = CoggleDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
test_pred_fold = np.zeros(test_data.shape[0])

#k折交叉验证
for idx in range(7):
    model = CoggleModel()
    layer_state_dict = paddle.load(f"model_{idx}.pdparams")
    model.set_state_dict(layer_state_dict)
    #不启用BatchNormalization和Dropout。
    model.eval()
    test_pred = predict(model, test_loader)
    test_pred = np.vstack(test_pred)
    test_pred_fold += test_pred[:, 0]

test_pred_fold /= 7
test_data["prediction"] = test_pred[:, 0]
test_data = test_data[["user_id", "prediction"]]
test_data.to_csv("./baseline_submission.csv", index=False, header=False, float_format="%.2f")





