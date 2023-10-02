import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

# 定义激活函数（sigmoid）


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y))
# the way we use this y is already sigmoided


def dsigmoid(y):
    return y * (1.0 - y)


# 调用sklearn的MLP


class skMlpRegression:
    def __init__(self, hidden_layer_sizes=(100, 50, 25), solver='adam', random_state=1, max_iter=500):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver=solver,
                                  random_state=random_state, max_iter=max_iter)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2_score = self.model.score(X_test, y_test)
        return mse, r2_score


# 定义自实现MLP类


# # 导入数据
# house_data = fetch_california_housing()
# # 将数据切分成训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(
#     house_data.data,
#     house_data.target,
#     test_size=0.3,
#     random_state=42)
# # 数据标准化处理
# scale = StandardScaler()
# X_train_s = scale.fit_transform(X_train)
# X_test_s = scale.transform(X_test)

# 将数据集转化为张量
# 数据处理的过程
# train_xt = torch.from_numpy(X_train_s.astype(np.float32))
# train_yt = torch.from_numpy(y_train.astype(np.float32))
# test_xt = torch.from_numpy(X_test_s.astype(np.float32))
# test_yt = torch.from_numpy(y_test.astype(np.float32))
# # 将训练数据处理为数据加载器
# train_data = Data.TensorDataset(train_xt, train_yt)
# test_data = Data.TensorDataset(test_xt, test_yt)
# train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)


"""搭建全连接神经网络回归模型"""


class myMlpRegression(nn.Module):
    def __init__(self):
        super(myMlpRegression, self).__init__()
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=7, out_features=100, bias=True)
        # 定义第二个隐藏层
        self.hidden2 = nn.Linear(100, 100)
        # 定义第三个隐藏层
        self.hidden3 = nn.Linear(100, 50)
        # 回归预测层
        self.predict = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        y = self.predict(x)
        # 返回一个一维向量
        return y[:, 0]

    def train(self, train_loader, num_epochs=30, learning_rate=0.01):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        loss_func = nn.MSELoss()  # 均方根误差损失函数
        train_loss_all = []

        # 对模型进行迭代训练，对所有的数据训练epoch轮
        for epoch in range(num_epochs):
            train_loss = 0
            train_num = 0
            # 对训练数据的加载器进行迭代计算
            for step, (b_x, b_y) in enumerate(train_loader):
                output = self.forward(b_x)  # mlp在训练batch上的输出
                loss = loss_func(output, b_y)  # 均方根误差损失函数
                optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
                loss.backward()  # 损失的后向传播，计算梯度
                optimizer.step()  # 使用梯度进行优化
                train_loss += loss.item() * b_x.size(0)
                train_num += b_x.size(0)

            train_loss_all.append(train_loss / train_num)

        """可视化损失函数变化情况"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_all, "ro-", label="Train loss")
        plt.legend()
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.show()

    def test(self, test_x, test_y):
        pre_y = self.forward(test_x)
        pre_y = pre_y.data.numpy()
        mae = mean_absolute_error(test_y, pre_y)
        print(mae)
        mse = mean_squared_error(test_y, pre_y)
        print(mse)
        # r2_score = self.model.score(X_test, y_test)
        # return mse, r2_score
        # Visualize the predictions
        index = np.argsort(test_y)
        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(len(test_y)), test_y[index], "r", label="Original Y")
        plt.scatter(np.arange(len(pre_y)), pre_y[index], s=3, c="b", label="Prediction")
        plt.legend(loc="upper left")
        plt.grid()
        plt.xlabel("Index")
        plt.ylabel("Y")
        plt.show()
