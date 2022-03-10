from dataset import UDCDataset
import utils
import numpy as np
import scipy
import matplotlib.pyplot as plt

import time
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import ResNet

# set random seeds
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

# pass training data to this method to train the model
def train_resnet(dataset):
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

                # plot images to witness training process
                plt.imshow(np.moveaxis(inputs[0].cpu().detach().numpy(), 0, 2))
                plt.show()
                plt.imshow(np.moveaxis(outputs[0].cpu().detach().numpy(), 0, 2))
                plt.show()
                plt.imshow(np.moveaxis(labels[0].cpu().detach().numpy(), 0, 2))
                plt.show()

    print('Finished Training')

    return net

def example_train():
    # Load dataset
    data = UDCDataset()

    # split data
    train, test, val = utils.train_test_split(data)

    # train and save model
    res = train_resnet(train)
    torch.save(res.state_dict(), "./my_model.pth")

    
def example_inference():

    # Load dataset
    data = UDCDataset()

    # split data
    train, test, val = utils.train_test_split(data)

    # Set up model for inference
    model = ResNet(hidden_layers=9)
    model.load_state_dict(torch.load("./my_model.pth", map_location=torch.device('cpu')))

    # Test random image from val set
    s = time.time()
    point = val[12]
    tensor = torch.from_numpy(point[0]).float()
    res = model(torch.unsqueeze(tensor, 0).permute(0, 3, 1, 2))[0].permute(1, 2, 0).cpu().detach().numpy()

    # document inference time
    print(time.time()- s)

    # display images for qualitative comparison
    utils.display_three(point[0], point[1], res)

if __name__ == "__main__":
    example_inference()