from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
import skimage
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

Img_folder='./images/'

class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    #  def __init__(self, df, _transform=None):

    def __init__(self, df, mode):
        self.read_csv = df
        # self.root_dir = root_dir
        if mode=="val":
            self._transform = transforms_val
        else:
            self._transform=transforms_train


    def __len__(self):
        return len(self.read_csv)



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.read_csv.iloc[idx, 0]

        image = imread(img_name)
        image=skimage.color.gray2rgb(image)
        if self._transform:
            image = self._transform(image)
        label = self.read_csv.iloc[idx, 1:]
        # label_ = np.array([label])
        label = torch.tensor(label).float()
        sample = {'image': image, 'label': label}

        # print(type(sample["image"]))
        # print(sample["label"].shape)

        return sample["image"], sample['label']


transforms_train = tv.transforms.Compose([
    tv.transforms.ToPILImage(),
    # tv.transforms.Resize((224, 224)),
    # tv.transforms.Pad(15, padding_mode='symmetric'),
    # tv.transforms.RandomHorizontalFlip(),
    # tv.transforms.RandomRotation(2),
    # tv.transforms.RandomCrop(140),
    # tv.transforms.ColorJitter(brightness=2),
    # tv.transforms.ColorJitter(contrast=2),
    # tv.transforms.ColorJitter(saturation=2),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(train_mean, train_std),
    # tv.transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
])



transforms_val = tv.transforms.Compose([
    tv.transforms.ToPILImage(),
    # tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(train_mean, train_std)
])
