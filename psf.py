from dataset import UDCDataset
import utils
import numpy as np
import scipy
import matplotlib.pyplot as plt

def __main__():
    data = UDCDataset()
    print(data.lq_images.shape)

    point = data[231]
    
    data.display(231)

__main__()