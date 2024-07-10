import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        # L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
        # nn.init.xavier_normal_(L.weight.data)
        nn.init.kaiming_normal_(L.weight.data)

    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class SimCLRNet(nn.Module):
    def __init__(self, name):
        super(SimCLRNet, self).__init__()

        self.name = name
        self.dataset_size = 0

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding='same')

        # self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3))

        # self.pool1 = nn.AvgPool2d(kernel_size=(3, 3))

        # self.fc1 = nn.Linear(6240, 128)
        self.fc1 = nn.Linear(5952, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 128)

        self.parametrized_layers = [self.conv1, self.conv2, self.conv3]

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        # x = F.normalize(x)

        return x
