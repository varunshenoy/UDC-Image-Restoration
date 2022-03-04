import skimage.io
import os
import numpy as np
import matplotlib.pyplot as plt
from models import Unet
import scipy

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def display_two(img1, img2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(img1)
    f.add_subplot(1,2, 2)
    plt.imshow(img2)
    plt.show(block=True)

def display_three(img1, img2, img3):
    print(f"PSNRS: noisy={psnr(img2, img1)}, clean={psnr(img2, img3)}")
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

def dncnn(img):
    model = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')        
    model.load_state_dict(torch.load('dncnn_25.pth'), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    img = np.moveaxis(img, -1, 0)
    for i, channel in enumerate(img):
        x_tensor = torch.reshape(torch.from_numpy(channel).float().to(device), (1,1,channel.shape[0],channel.shape[1]))
        x_tensor_denoised = model(x_tensor)
        img[i] = x_tensor_denoised
    
    return np.moveaxis(img, 0, 2)

from scipy.signal import fftconvolve

def richardson_lucy_blind(image, num_iter=100):    
    im_deconv = np.full(image.shape, 0.1, dtype='float')    # init output
    psf = np.ones(image.shape) * 0.5
    for i in range(num_iter):
        print(i)
        psf_mirror = np.flip(psf)
        conv = fftconvolve(im_deconv, psf, mode='same')
        relative_blur = image / conv
        im_deconv *= fftconvolve(relative_blur, psf_mirror, mode='same')
        im_deconv_mirror = np.flip(im_deconv)
        psf *= fftconvolve(relative_blur, im_deconv_mirror, mode='same')   
        print(psf) 
    return im_deconv, psf

def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)

def deblur_denoise(img):
    
    model_deblur_denoise = Unet().to(device)
    model_deblur_denoise.load_state_dict(torch.load('pretrained/deblur_denoise.pth', map_location=device))

    model_deblur_denoise.eval()

    print(img.shape)
    img_tensor = torch.reshape(torch.from_numpy(img).float().to(device), (1,3,img.shape[0],img.shape[1]))

    deblur_denoise = model_deblur_denoise(img_tensor)
    return  (img_to_numpy(deblur_denoise)*255).astype(np.uint8)

def convolution_matrix(img, window=5):
    plt.imshow(img)
    plt.show()
    pad = int(window/2)
    padded = np.pad(img, ((pad,pad),(pad,pad)), 'constant')
    matrix = []
    for i in range(pad, pad + img.shape[0]):
        for j in range(pad, pad + img.shape[1]):
            patch = padded[i-pad:i+pad+1, j-pad:j+pad+1]
            matrix.append(patch.flatten())
    print(len(matrix[0]))
    return np.array(matrix)

def find_filter(noisy, true, window=5):
    noisy = np.moveaxis(noisy, -1, 0)
    true = np.moveaxis(true, -1, 0)

    filt = []
    for channel in range(3):
        A = convolution_matrix(true[channel], window=window)
        b = noisy[channel].flatten()

        print(A.shape)
        print(b.shape)

        res = np.linalg.lstsq(A, b, rcond=None)[0]
        res = res.reshape((window, window))
        filt.append(res/np.sum(res))

    return np.moveaxis(np.array(filt), 0, -1)

def reconstruct(filt, img):
    out = []
    filt = np.moveaxis(filt, -1, 0)
    img = np.moveaxis(img, -1, 0)
    print(filt)
    for channel in range(3):
        print(filt[channel].shape)
        print(img[channel].shape)
        out.append(scipy.signal.convolve2d(img[channel], filt[channel], mode="same"))
    out = np.moveaxis(np.array(out), 0, -1).astype(int)
    print(out)
    return out