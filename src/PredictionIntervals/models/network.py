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


class NN2(nn.Module, ABC):
    """Defines deep NN architecture (3 hidden layers)"""

    def __init__(self, input_shape: int = 10, output_size: int = 1):
        """
        Initialize NN
        :param input_shape: Input shape of the network.
        :param output_size: Output shape of the network.
        """
        super(NN2, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=500), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.01)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.drop2 = nn.Dropout(p=0.01)
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU())
        self.drop3 = nn.Dropout(p=0.01)

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.drop1(x)
        x = self.hidden_layer2(x)
        x = self.drop2(x)
        x = self.hidden_layer3(x)
        x = self.drop3(x)
        return self.out(x)


class NN3(nn.Module, ABC):
    """Defines deeper NN architecture (5 hidden layers)"""

    def __init__(self, input_shape: int = 10, output_size: int = 1):
        """
        Initialize NN
        :param input_shape: Input shape of the network.
        :param output_size: Output shape of the network.
        """
        super(NN3, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=200), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.01)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=500), nn.ReLU())
        self.drop2 = nn.Dropout(p=0.01)
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=500), nn.ReLU())
        self.drop3 = nn.Dropout(p=0.01)
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU())

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.drop1(x)
        x = self.hidden_layer2(x)
        x = self.drop2(x)
        x = self.hidden_layer3(x)
        x = self.drop3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        return self.out(x)
