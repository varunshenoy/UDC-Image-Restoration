from dataset import UDCDataset
import utils
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import ResNet
from scipy.ndimage.filters import gaussian_filter

# set random seeds
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

def train_conv(dataset):
    net = Net()
    print(net)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    crit = nn.MSELoss()
    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            inputs = torch.reshape(torch.squeeze(inputs, 0).float().to(device), (1,3,inputs.shape[1],inputs.shape[2]))
            labels = torch.reshape(torch.squeeze(labels, 0).float().to(device), (1,3,labels.shape[1],labels.shape[2]))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return net.conv1.weight

def train_test_split(dataset):
    train_size = int(0.8 * len(dataset))
    print(len(dataset))
    test_size = int((len(dataset) - train_size)/2)
    print(test_size)
    train_dataset, test_dataset, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, test_size])
    return train_dataset, test_dataset, val_set

def train_simple(dataset):
    net = ResNet(hidden_layers=9).to(device)
    print(net)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    crit = nn.L1Loss()
    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            inputs = inputs.permute(0, 3, 1, 2).float().to(device)
            labels = labels.permute(0, 3, 1, 2).float().to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 50 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.6f}')
                running_loss = 0.0
            if i % 50 == 0:
                plt.imshow(np.moveaxis(inputs[0].cpu().detach().numpy(), 0, 2))
                plt.show()
                plt.imshow(np.moveaxis(outputs[0].cpu().detach().numpy(), 0, 2))
                plt.show()
                plt.imshow(np.moveaxis(labels[0].cpu().detach().numpy(), 0, 2))
                plt.show()

    print('Finished Training')

    return net, test, val

def main():

    # Load dataset
    data = UDCDataset()
    print(data.lq_images.shape)

    # split data
    train, test, val = train_test_split(data)

    # Set up model for inference
    model = ResNet(hidden_layers=9)
    model.load_state_dict(torch.load("./final_model.pth", map_location=torch.device('cpu')))

    # Test random image from val set
    point = val[0]
    tensor = torch.from_numpy(point[0]).float()
    res = model(torch.unsqueeze(tensor, 0).permute(0, 3, 1, 2))[0].permute(1, 2, 0).cpu().detach().numpy()
    utils.display_three(point[0], point[1], res)

if __name__ == "__main__":
    main()