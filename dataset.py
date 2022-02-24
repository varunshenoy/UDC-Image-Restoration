import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
import skimage.io
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# set random seeds
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

class UDCDataset():
    def __init__(self, type="Poled"):
        LQ_fnames = sorted(glob(os.path.join("./images", f"train/{type}/", "LQ", '*')))
        HQ_fnames = sorted(glob(os.path.join("./images", f"train/{type}/", "HQ", '*')))
        
        self.lq_images = self.load_images(LQ_fnames)
        self.hq_images = self.load_images(HQ_fnames)

    def load_images(self, files):
        print("Importing images...")
        pbar = tqdm(total=len(files))
        out = []
        for fname in files:
            img = skimage.io.imread(fname)
            out.append(torch.from_numpy(img))
            pbar.update(1)
        return torch.stack(out)

    def __len__(self):
        return len(lq_images)

    def __getitem__(self, idx):
        return (self.lq_images[idx], self.hq_images[idx])

    def display(self, idx):
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(self.lq_images[idx])
        f.add_subplot(1,2, 2)
        plt.imshow(self.hq_images[idx])
        plt.show(block=True)

# ds = UDCDataset()
# ds.display(3)
# ds.display(10)
# ds.display(100)
# ds.display(231)