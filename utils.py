import skimage.io
import os
import numpy as np
import matplotlib.pyplot as plt
from models import Unet
import scipy

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_test_split(dataset):
    train_size = int(0.8 * len(dataset))
    print(len(dataset))
    test_size = int((len(dataset) - train_size)/2)
    print(test_size)
    train_dataset, test_dataset, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, test_size])
    return train_dataset, test_dataset, val_set

def display_two(img1, img2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(img1)
    f.add_subplot(1,2, 2)
    plt.imshow(img2)
    plt.show(block=True)

def display_three(img1, img2, img3):
    # img1: low quality image
    # img2: high quality image
    # img3: reconstructed image

    print(f"PSNRs: ground truth={psnr(img2, img1)}, reconstructed={psnr(img2, img3)}")
    f = plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(img1)
    f.add_subplot(1,3,2)
    plt.imshow(img2)
    f.add_subplot(1,3,3)
    plt.imshow(img3)
    plt.show(block=True)

def mse(true, recon):
    return np.concatenate((1/(3 * true.shape[0] * true.shape[1])) * pow(true - recon, 2)).sum()

def psnr(true, recon):
    return 10 * np.log10(pow(true.max(), 2)/mse(true, recon))

def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)