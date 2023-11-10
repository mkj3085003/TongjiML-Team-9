import torch.nn as nn

class BasicBlock2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock2, self).__init__()

        # TODO: apply batch normalization and dropout for strong baseline.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html (batch normalization)
        #       https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html (dropout)
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            # 在此处增加 nn.Dropout()
            nn.Dropout(p=0.15)
        )

    def forward(self, x):
        x = self.block(x)
        return x