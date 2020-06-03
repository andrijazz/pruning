import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        in_dim = 28 * 28
        layers = []
        prev_dim = in_dim
        for i in range(len(self.config.ARCH)):
            layers.append(nn.Linear(prev_dim, self.config.ARCH[i], bias=False))
            if i != len(self.config.ARCH) - 1:
                layers.append(nn.ReLU())
            prev_dim = self.config.ARCH[i]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
