
import os
import json
import pathlib
import random

from datasets import load_dataset

import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import PIL

from typing import Literal



class SWAGDataset(Dataset):
    num_classes = 4
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42,
    ):
        ds = load_dataset("allenai/swag", "regular", cache_dir=data_dir)

        match split:
            case "train": self.data = ds["train"]
            case "validation": 
                self.data = ds["validation"].train_test_split(test_size=0.2)["train"]
            case "test":
                self.data = ds["validation"].train_test_split(test_size=0.2)["test"]
        self.data_idxs = list(range(len(self.data)))

    def __len__(self):
        return len(self.data_idxs)


    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        context = data["sent1"]
        question = data["sent2"]
        options = [ data[f"ending{i}"] for i in range(4) ]
        answer = data["label"]

        return context, question, options, answer