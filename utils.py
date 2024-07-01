import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import glob
import cv2
import numpy as np
import torch


class MyDataset(Dataset):
    def __init__(self, img_path, device):
        super(MyDataset, self).__init__()
        self.device = device
        self.fnames = glob.glob(os.path.join(img_path+"*.jpg"))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img = self.transforms(img)
        img = img.to(self.device)
        return img

    def __len__(self):
        return len(self.fnames)


class AnimeDataset(Dataset):
    def __init__(self, img_pths, device="cpu"):
        self.device = device
        self.img_pths = glob.glob(img_pths+"*.jpg")

    def __len__(self):
        return len(self.img_pths)

    def __getitem__(self, id):
        img = cv2.imread(self.img_pths[id], cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (64, 64))
        img = np.moveaxis(img, 2, 0)
        img = torch.tensor(img, dtype=torch.float).to(self.device)
        return img / 255.0
