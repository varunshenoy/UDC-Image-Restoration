import skimage.io
import os
import numpy as np
import matplotlib.pyplot as plt

def display_two(self, img1, img2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(img1)
    f.add_subplot(1,2, 2)
    plt.imshow(img2)
    plt.show(block=True)