from abc import ABC
import torch.nn as nn


class NN(nn.Module, ABC):
    """Defines NN architecture"""

    def __init__(self, input_shape=500, output_size=1):
        super(NN, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=100), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.1)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100), nn.Tanh())
        self.drop2 = nn.Dropout(p=0.1)

        # Number of outputs depends on the method
        self.out = nn.Linear(100, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.drop1(x)
        x = self.hidden_layer2(x)
        x = self.drop2(x)
        return self.out(x)
