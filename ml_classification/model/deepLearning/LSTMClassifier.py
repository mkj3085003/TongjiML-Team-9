import torch
import torch.nn as nn

from ml_classification.model.deepLearning.BasicBlock_adjust import BasicBlock2


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, drop_rate=0):
        super(LSTMClassifier, self).__init__()

        # Create BiLSTM
        self.input_size = 39  # 这一项是RNN的"input_dim"，RNN需要对"单"个数据进行处理
        self.hidden_size = 512  # 这一项是RNN的"hidden_dim"
        self.num_layers = 6  # 这一项是RNN的"hidden_layers"


        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                   batch_first=True, dropout=drop_rate, bidirectional=True)

        # 后接全连接层
        self.fc = nn.Sequential(
            # 修改成 2 * self.hidden_size 的原因是因为LSTM()中的bidirectional设置为了True，这表示使用Bi（双向）LSTM模型，所以需要修改输入维度以匹配
            BasicBlock2(2 * self.hidden_size, hidden_dim),
            # 在函数的调用中，一个 * 表示将一个序列展开为单独的位置参数，这一行代码是列表推导，最终的表现是重复生成多个 hidden layer
            # （原来的整段代码实际上生成了 hidden_layers+1 个隐藏层，所以我修改了一下代码，让其符合定义）
            *[BasicBlock2(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # 通过RNN层，得到输出和最后一个隐藏状态，注意输出的shape
        # x.shape: (batch_size, seq_len, RNN_input_size)
        x, _ = self.rnn(x)  # => (batch_size, seq_len, RNN_hidden_size)

        # 取最后一个时间步的输出作为分类的输入
        x = x[:, -1]  # => (batch_size, RNN_hidden_size)

        # 通过线性层，得到最终的分类结果
        x = self.fc(x)  # => (batch_size, labels)

        return x
