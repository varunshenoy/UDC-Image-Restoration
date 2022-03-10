# A Deep Learning Approach for Image Reconstruction from Smartphone Under Display Camera Technology

Authors: Varun Shenoy (vnshenoy@stanford.edu) and Arjun Dhawan (akdhawan@stanford.edu)
Affiliation: Dept. of Electrical Engineering, Stanford University

Email is the best way to reach out to either of us.

## Project Files Overview

- `requirements.txt` contains all the dependencies necessary to run this project. You can use conda or pip to install the exact versions of dependencies which we used from this file.
- `main.py` contains all of the training and inference code necessary to understand the project at a high level. A quick look at `example_train` and `example_infenrence` would give the reader a quick idea at how to train and evaluate the model. This is the place to start if you want to build this project for yourself.
- `dataset.py` houses the `UDCDataset` object that is in charge of loading/caching the dataset to disk and integrating with Pytorch so that the training loop can be kept simple.
- `test_harness.py` generates histograms and PSNR calculations over the entire test set. These can be seen in the project report.
- `utils.py` has basic code for splitting datasets, displaying images in matplotlib, and calculating PSNR.

After running `main.py`, several new files will be created:
- `hq.npy` and `lq.npy` are cached copies of the dataset to quicken experimentation.
- `my_model.pth` is the state dictionary for the final trained model.
