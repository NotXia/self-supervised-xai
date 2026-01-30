import os
import json
import pathlib
import random
import pandas as pd

from datasets import load_dataset

import torch
import torchvision
from torch.utils.data import Dataset
import PIL
import io

from typing import Literal



class HatefulMemesDataset(Dataset):
    num_classes = 2
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42,
    ):
        self.rng = random.Random(seed)
        self.dataset_dir = os.path.join(data_dir, "hateful_memes")

        match split:
            case "train": 
                self.data = pd.read_json(path_or_buf=os.path.join(self.dataset_dir, "train.jsonl"), lines=True)
            case "validation" | "test": 
                self.data = pd.read_json(path_or_buf=os.path.join(self.dataset_dir, "dev.jsonl"), lines=True)
                self.data = self.data.sample(frac=1).reset_index(drop=True)
                thr = int(len(self.data) * 0.6)
                if split == "validation":
                    self.data = self.data.iloc[:thr]
                else:
                    self.data = self.data.iloc[thr:]
        self.data_idxs = list( range(len(self.data)) )
            
        self.image_tsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((500, 500)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data.iloc[ self.data_idxs[idx] ]

        image_path = os.path.join(self.dataset_dir, data["img"])
        image = self.image_tsfm( PIL.Image.open(image_path).convert('RGB') )
        text = data["text"]
        label = data["label"]

        return image, text, label

