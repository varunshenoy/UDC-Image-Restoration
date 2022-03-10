from dataset import UDCDataset
from utils import train_test_split, psnr
import numpy as np
import matplotlib.pyplot as plt
from models import ResNet

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plotResults(originalPSNRS, reconstructedPSNRs):
    originalPSNRS = np.asarray(originalPSNRS)
    reconstructedPSNRs = np.asarray(reconstructedPSNRs)

    # N_points = 100000
    # n_bins = 20
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

    # Histogram generation
    axs[0].hist(originalPSNRS)
    axs[1].hist(reconstructedPSNRs)
    axs[2].hist(reconstructedPSNRs - originalPSNRS)

    axs[0].title.set_text('Original PSNRs')
    axs[1].title.set_text('Reconstructed PSNRs')
    axs[2].title.set_text('Difference in PSNRs')
    plt.show()
    plt.savefig('psnrs.png')

def main():
    data = UDCDataset()
    print(data.lq_images.shape)
    
    # split data
    train, test, val = train_test_split(data)

    # Set up model for inference
    model = ResNet(hidden_layers=9)
    model.load_state_dict(torch.load("./my_model.pth", map_location=torch.device('cpu')))

    print("Length: ", len(val))

    # Iterate over the test dataset and calculate PSNR
    iterationNum = 0
    originalPSNRs = []
    reconstructedPSNRs = []
    for point in test: 
        tensor = torch.from_numpy(point[0]).float()

        res = model(torch.unsqueeze(tensor, 0).permute(0, 3, 1, 2))[0].permute(1, 2, 0).cpu().detach().numpy()

        originalPSNR = psnr(point[1], point[0])
        reconstructedPSNR = psnr(point[1], res)
        print("Iteration Number: ", iterationNum)
        print(f"PSNRS: noisy={originalPSNR}, clean={reconstructedPSNR}")
        originalPSNRs.append(originalPSNR)
        reconstructedPSNRs.append(reconstructedPSNR)
        iterationNum += 1

    plotResults(originalPSNRs, reconstructedPSNRs)
    print("Original PSNR Average: ", np.average(originalPSNRs))
    print("Reconstructed PSNR Average: ", np.average(reconstructedPSNRs))
    print("Difference in PSNR Average: ", np.average(np.subtract(reconstructedPSNRs, originalPSNRs)))

if __name__ == "__main__":
    main()