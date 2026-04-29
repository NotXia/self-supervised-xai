
import os
import json
import pathlib
import random

from datasets import load_dataset

import torch
import torchvision
from torch.utils.data import Dataset
import PIL
import io

from typing import Literal



class Flickr8kDataset(Dataset):
    num_classes = 2
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42,
    ):
        self.rng = random.Random(seed)
        self.data = load_dataset("jxie/flickr8k")
        self.id2label = {
            0: "incorrect",
            1: "correct"
        }

        match split:
            case "train": self.data = self.data["train"]
            case "validation": self.data = self.data["validation"]
            case "test": self.data = self.data["test"]
        self.data_idxs = list( range(len(self.data)*2) )
        self.rng.shuffle(self.data_idxs)
        self.data_idxs = self.data_idxs
            
        self.image_tsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] % len(self.data) ]

        image = self.image_tsfm( data["image"].convert('RGB') )
        if idx < len(self.data):
            text = data["caption_0"]
            label = 1
        else:
            negative_idx = self.rng.randint(0, len(self.data))
            if negative_idx == (idx % len(self.data)): (negative_idx+1) % len(self.data)
            text = self.data[negative_idx]["caption_0"]
            label = 0

        return image, text, label