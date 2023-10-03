import torch.nn as nn
import torch.nn.functional as F

# 定义自实现MLP类

"""搭建全连接神经网络回归模型"""
class MlpRegression(nn.Module):
    def __init__(self):
        super(MlpRegression, self).__init__()
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

    def train(self, train_loader, optimizer, num_epochs=500):
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

            train_loss_epoch = train_loss / train_num
            train_loss_all.append(train_loss_epoch)
            # 输出每个epoch的损失和当前使用的优化器
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss_epoch:.4f}, Optimizer: {optimizer}')
        return train_loss_all

    def test(self, test_x, test_y):
        pre_y = self.forward(test_x)
        pre_y = pre_y.data.numpy()
        return pre_y
