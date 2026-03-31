
import os
import json
import pathlib
import random

from datasets import load_dataset

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import PIL
import io

from typing import Literal


class OxfordPetDataset(Dataset):
    num_classes = 2
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42,
        return_xai_label = False
    ):
        self.return_xai_label = return_xai_label
        self.rng = random.Random(seed)
        self.data = load_dataset("dpdl-benchmark/oxford_iiit_pet", cache_dir=data_dir)
        match split:
            case "train" | "validation": 
                self.data = self.data["train"]
                self.data_idxs = list( range(len(self.data)) )
                self.rng.shuffle(self.data_idxs)
                thr = int(len(self.data_idxs)*0.2)
                if split == "train":
                    self.data_idxs = self.data_idxs[:thr]
                else:
                    self.data_idxs = self.data_idxs[thr:]
            case "test": 
                self.data = self.data["test"]
                self.data_idxs = list( range(len(self.data)) )
                
        self.image_tsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((320, 320)),
            torchvision.transforms.ToTensor()
        ])
        self.mask_tsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((320, 320), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        image = self.image_tsfm( data["image"].convert("RGB") )
        label = data["species"]
        segmentation_mask = torchvision.transforms.functional.pil_to_tensor( self.mask_tsfm(data["segmentation_mask"]) )
        segmentation_mask = (segmentation_mask == 1).type(torch.int8) # Ignore border class

        if self.return_xai_label:
            return image, label, segmentation_mask
        else:
            return image, label