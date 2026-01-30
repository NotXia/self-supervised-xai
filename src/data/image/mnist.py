
import os
import json
import pathlib
import random

import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import PIL

from typing import Literal



class MNISTDataset(Dataset):
    num_classes = 10
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42
    ):
        rng = random.Random(42)
        self.data_dir = data_dir

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        if split in ["train", "validation"]:
            self.data = MNIST(data_dir, train=True, download=True)

            self.train_idxs = list(range(len(self.data)))
            rng.shuffle(self.train_idxs)
            train_size = int(len(self.train_idxs) * 0.8)
            val_size = len(self.data) - train_size

            match split:
                case "train":
                    self.data_idxs = self.train_idxs[:train_size]
                    assert len(self.data_idxs) == train_size
                case "validation":
                    self.data_idxs = self.train_idxs[train_size:train_size+val_size]
                    assert len(self.data_idxs) == val_size
        else:
            self.data = MNIST(data_dir, train=False, download=True)
            self.data_idxs = list(range(len(self.data)))


    def __len__(self):
        return len(self.data_idxs)


    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        image = self.transforms(data[0].convert("RGB"))
        label = data[1]

        return image, label