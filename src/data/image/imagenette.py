
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


class ImagenetteDataset(Dataset):
    num_classes = 10
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42,
    ):
        self.rng = random.Random(seed)
        self.data = load_dataset("frgfm/imagenette", "320px")

        self.id2label = {
            0: "fish",
            1: "dog",
            2: "cassette player",
            3: "chain saw",
            4: "church",
            5: "french horn",
            6: "garbage truck",
            7: "gas pump",
            8: "golf ball",
            9: "parachute",
        }

        match split:
            case "train": 
                self.data = self.data["train"]
                self.data_idxs = list( range(len(self.data)) )
            case "validation" | "test": 
                self.data = self.data["validation"]
                self.data_idxs = list( range(len(self.data)) )
                self.rng.shuffle(self.data_idxs)
                thr = int(len(self.data_idxs)*0.3)
                if split == "validation":
                    self.data_idxs = self.data_idxs[:thr]
                else:
                    self.data_idxs = self.data_idxs[thr:]
                
        self.image_tsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((320, 320)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        image = self.image_tsfm( data["image"].convert('RGB') )
        label = data["label"]

        return image, label