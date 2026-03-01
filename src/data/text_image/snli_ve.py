
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



class SNLIVEDataset(Dataset):
    num_classes = 3
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42,
    ):
        self.rng = random.Random(seed)
        self.data = load_dataset("Multimodal-Fatima/SNLI-VE_train")
        self.data = self.data["train"]
        self.data_idxs = list( range(len(self.data)) )
        self.rng.shuffle(self.data_idxs)
        train_size = int(len(self.data_idxs) * 0.6)
        val_size = int(len(self.data_idxs) * 0.2)
        match split:
            case "train":       self.data_idxs = self.data_idxs[:train_size]
            case "validation":  self.data_idxs = self.data_idxs[train_size:train_size+val_size]
            case "test":        self.data_idxs = self.data_idxs[train_size+val_size:]
            
        self.image_tsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        image = self.image_tsfm( data["image"].convert('RGB') )
        text = f"[PREMISE] {data['premise']} [HYPOTHESIS] {data['hypothesis']}"
        label = data["label"]

        return image, text, label