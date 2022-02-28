import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, kernel_size=3, hidden_layers=3):
        super(SimpleCNN, self).__init__()

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='same'))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(torch.nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            layers.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same'))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='same'))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
