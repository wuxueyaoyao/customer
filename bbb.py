import pandas as pd
import numpy as np
from itertools import groupby
import seaborn as sns
import json
import math
import paddle
from paddle.io import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

PATH = './'
data_dir = "./wsdm_model_data/"
#--------------------------加载数据---------------------------
user_interaction = pd.read_csv(PATH + 'user_interaction_data.csv')#用户行为信息
user_portrait = pd.read_csv(PATH + 'user_portrait_data.csv')#用户基础信息
user_playback = pd.read_csv(PATH + 'user_playback_data.csv')#用户观看信息
app_launch = pd.read_csv(PATH + 'app_launch_logs.csv')#用户登录时间和方式
video_related = pd.read_csv(PATH + 'video_related_data.csv')#视频信息
#test_a = pd.read_csv('test-a.csv')#test_a
data = pd.read_csv(data_dir + "train_data.txt", sep="\t")
#
#---------------------------（一）用户基础信息分析-------------------------------
print(user_portrait.shape)
#基本列展示
user_portrait.head(2)
#获取字段类型和有唯一值的数量
for col in user_portrait.columns:
  print(f'{col} \t {user_portrait.dtypes[col]} {user_portrait[col].nunique()}')
#删除重复值
user_portrait = user_portrait.drop_duplicates()
#整理每一列重复值
print(user_portrait['user_id'].value_counts())#用户Id
print(user_portrait['device_type'].value_counts())#手机型号 1=>IOS  2=>安卓   3=>wp    4=>其他
print(user_portrait['sex'].value_counts())#性别
#图片展示信息
user_portrait['device_ram'] = user_portrait['device_ram'].apply(lambda x: str(x).split(';')[0])
user_portrait['device_rom'] = user_portrait['device_rom'].apply(lambda x: str(x).split(';')[0])
sns.displot(user_portrait['device_ram'])#手机运行内存
sns.displot(user_portrait['device_rom'])#手机内存
sns.displot(user_portrait['age'])#性别
sns.displot(user_portrait['education'])#受教育程度
sns.displot(user_portrait['occupation_status'])#工作状态
sns.displot(user_portrait['territory_code'])#用户常住地编号
#---------------------------（二）用户登录时间和方式分析-------------------------------
print(app_launch.shape)
#基本列展示
app_launch.head(2)
#获取字段类型和有唯一值的数量
for col in app_launch.columns:
   print(f'{col} \t {app_launch.dtypes[col]} {app_launch[col].nunique()}')
#删除重复值
app_launch = app_launch.drop_duplicates()
#整理每一列重复值
print(app_launch['launch_type'].value_counts())#登录方式
app_launch.groupby('user_id')['launch_type'].mean()
app_launch = app_launch.sort_values(by=['user_id', 'date'])
#---------------------------（三）视频信息分析-------------------------------
print(video_related.shape)
#基本列展示
video_related.head(2)
#删除重复值
video_related = video_related.drop_duplicates()
# 图片展示信息
sns.distplot(video_related['duration'])
#---------------------------（四）用户观看信息分析-------------------------------
print(user_playback.shape)
#基本列展示
user_playback.head(2)
#---------------------------（五）用户行为信息分析-------------------------------
print(user_interaction.shape)
#基本列展示
user_interaction.head(2)

#------------------------------------处理训练集数据----------------------------------
data["launch_seq"] = data.launch_seq.apply(lambda x: json.loads(x))
data["playtime_seq"] = data.playtime_seq.apply(lambda x: json.loads(x))
data["duration_prefer"] = data.duration_prefer.apply(lambda x: json.loads(x))
data["interact_prefer"] = data.interact_prefer.apply(lambda x: json.loads(x))
# shuffle data
data = data.sample(frac=1).reset_index(drop=True)
# print(data.columns)
# print(data)
print(11)


#----------------------------------------定义模型-------------------------------------
class CoggleDataset(Dataset):
    def __init__(self, df):
        super(CoggleDataset, self).__init__()
        self.df = df
        self.feat_col = list(set(self.df.columns) - set(['user_id', 'end_date', 'label', 'launch_seq', 'playtime_seq',
                                                         'duration_prefer', 'interact_prefer']))
        self.df_feat = self.df[self.feat_col]

    # 定义需要参与训练的字段
    def __getitem__(self, index):
        launch_seq = self.df['launch_seq'].iloc[index]
        playtime_seq = self.df['playtime_seq'].iloc[index]
        duration_prefer = self.df['duration_prefer'].iloc[index]
        interact_prefer = self.df['interact_prefer'].iloc[index]

        feat = self.df_feat.iloc[index].values.astype(np.float32)

        launch_seq = paddle.to_tensor(launch_seq).astype(paddle.float32)
        playtime_seq = paddle.to_tensor(playtime_seq).astype(paddle.float32)
        duration_prefer = paddle.to_tensor(duration_prefer).astype(paddle.float32)
        interact_prefer = paddle.to_tensor(interact_prefer).astype(paddle.float32)
        feat = paddle.to_tensor(feat).astype(paddle.float32)

        label = paddle.to_tensor(self.df['label'].iloc[index]).astype(paddle.float32)
        return launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label

    def __len__(self):
        return len(self.df)


# 定义模型，这里是LSTM + FC
class CoggleModel(paddle.nn.Layer):
    def __init__(self):
        super(CoggleModel, self).__init__()

        # 序列建模
        self.launch_seq_gru = paddle.nn.GRU(1, 32)
        self.playtime_seq_gru = paddle.nn.GRU(1, 32)

        # 全连接层
        self.fc1 = paddle.nn.Linear(102, 64)
        self.fc2 = paddle.nn.Linear(64, 1)

    def forward(self, launch_seq, playtime_seq, duration_prefer, interact_prefer, feat):
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
def train(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = []
    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label in tqdm(train_loader):
        pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        train_loss.append(loss.item())
    return np.mean(train_loss)

# 模型验证函数
def validate(model, val_loader, optimizer, criterion):
    model.eval()
    val_loss = []
    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label in tqdm(val_loader):
        pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        val_loss.append(loss.item())

    return np.mean(val_loss)

# 模型预测函数
def predict(model, test_loader):
    model.eval()
    test_pred = []
    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label in tqdm(test_loader):
        pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)
        test_pred.append(pred.numpy())

    return test_pred



# 模型多折训练
skf = StratifiedKFold(n_splits=7)
fold = 0
for tr_idx, val_idx in skf.split(data, data['label']):
    train_dataset = CoggleDataset(data.iloc[tr_idx])
    val_dataset = CoggleDataset(data.iloc[val_idx])

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
test_data["launch_seq"] = test_data.launch_seq.apply(lambda x: json.loads(x))
test_data["playtime_seq"] = test_data.playtime_seq.apply(lambda x: json.loads(x))
test_data["duration_prefer"] = test_data.duration_prefer.apply(lambda x: json.loads(x))
test_data["interact_prefer"] = test_data.interact_prefer.apply(lambda x: json.loads(x))
test_data['label'] = 0

test_dataset = CoggleDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
test_pred_fold = np.zeros(test_data.shape[0])

# 模型多折预测
for idx in range(7):
    model = CoggleModel()
    layer_state_dict = paddle.load(f"model_{idx}.pdparams")
    model.set_state_dict(layer_state_dict)

    model.eval()
    test_pred = predict(model, test_loader)
    test_pred = np.vstack(test_pred)
    test_pred_fold += test_pred[:, 0]

test_pred_fold /= 7
test_data["prediction"] = test_pred[:, 0]
test_data = test_data[["user_id", "prediction"]]
# can clip outputs to [0, 7] or use other tricks
test_data.to_csv("./baseline_submission.csv", index=False, header=False, float_format="%.2f")



