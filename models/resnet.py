import torch
import torch.nn as nn


class ResNetBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=3):
        super(ResNetBlock, self).__init__()

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
        layers.append(torch.nn.BatchNorm2d(out_channels)),
        layers.append(torch.nn.LeakyReLU(inplace=True))
        self.resnet_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_block(x) + x


class ResNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, kernel_size=3,
                 hidden_layers=3):
        super(ResNet, self).__init__()

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='same'))
        for i in range(hidden_layers):
            layers.append(ResNetBlock(hidden_channels, hidden_channels, kernel_size=kernel_size))

        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='same'))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
