import torch
import torch.nn as nn


class BidirectionalGRUClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, drop_rate=0):
        super(BidirectionalGRUClassifier, self).__init__()
        self.lstm = nn.GRU(input_size=39, hidden_size=hidden_dim, num_layers=hidden_layers, bidirectional=True,
                            batch_first=True, dropout=drop_rate)
        self.fc = nn.Linear(hidden_dim*2, output_dim)


    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1, 39)  # b,l*hin ==>b,l,hin
        x, h_n = self.lstm(x, None)  # x:b,l,h  h_n:d*num_layer,b,h
        # x = x[:, -1, :]  # final state of final layer  ==>  x:b,h
        x_fd = h_n[-2, :, :]  # forward final state of final layer  ==>  x:b,h
        x_bd = h_n[-1, :, :]  # backward final state of final layer  ==>  x:b,h
        out = self.fc(torch.cat([x_fd, x_bd], dim=-1))
        return out


