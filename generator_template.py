import glob
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
torch.backends.cudnn.benchmark = True
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
    RandAffined,
)
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
from generator import Dataset2hD
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt


root_dir="MODELS" # why do I need that?
train_transforms = Compose( [ RandAffined(keys=["img","seg"],mode = ["bilinear", "nearest"], prob=0.9, shear_range= [(0.1),(0.1),(0.1)]),
ToTensord(keys=["img", "seg"]) ])
# train_transforms = Compose( [ ToTensord(keys=["img", "seg"]) ]) # why twice?
val_transforms = Compose( [ToTensord(keys=["img", "seg"])])

# data from hard drive
folder = "/Users/sophieostmeier/Desktop/NCCT_project_ncctROI"
ncct_list = sorted(glob.glob(folder + "/*ncct.nii"))
roi_list = sorted(glob.glob(folder + "/*ROI.nii"))

training_cases = list(zip(ncct_list[:10],roi_list[:10])) # only 10 for reduced memory

train_dataset = Dataset2hD(training_cases,train_transforms) # make sure enough memory available
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

# how can I check the Dataset2hD() is correctly set up and spits out the img and seg correctly?

for train_dataset[:,:] in train_dataset:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_title('IMAGE')
    ax1.imshow(train_dataset.img)

    ax2.set_title('GROUND TRUTH')
    ax2.imshow(train_dataset.seg, cmap='gray')