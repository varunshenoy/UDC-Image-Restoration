from dataset import UDCDataset
import utils
import numpy as np
import scipy
import matplotlib.pyplot as plt

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 3 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 3, 3, padding='same', groups=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x

def train(dataset, batch_size=1):
    net = Net()
    print(net)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    crit = nn.L1Loss()
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        print(f"running epoch {epoch}...")
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

def main():
    data = UDCDataset()
    print(data.lq_images.shape)

    #res = np.moveaxis(train(data).cpu().detach().numpy(), 0, -1).squeeze()
    #np.save("./psf.npy", res)
    res = np.load("./psf.npy")
    print(res.shape)
    point = data[231]
    
    #res = utils.find_filter(point[0], point[1], window=5)
    #print(res)
    # print(res.shape)
    # plt.imshow(res)
    # plt.show()
    rc = utils.reconstruct(res, point[0])
    # print(rc)
    utils.display_three(point[0], point[1], rc)

    point2 = data[3]
    rc = utils.reconstruct(res, point2[0])
    # # print(rc)
    utils.display_three(point2[0], point2[1], rc)


if __name__ == "__main__":
    main()