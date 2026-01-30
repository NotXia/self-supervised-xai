
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
import io

from typing import Literal



class FigureQADataset(Dataset):
    num_classes = 2
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42,
    ):
        rng = random.Random(seed)
        self.data = load_dataset("vikhyatk/figureqa", cache_dir=data_dir)["train"]
        self.data_idxs = list( range(len(self.data)) )
        rng.shuffle(self.data_idxs)
            
        train_th = int(len(self.data_idxs) * 0.6)
        val_th = int(len(self.data_idxs) * 0.2)

        match split:
            case "train": 
                self.data_idxs = self.data_idxs[:train_th]
            case "validation": 
                self.data_idxs = self.data_idxs[train_th : train_th+val_th]
            case "test":
                self.data_idxs = self.data_idxs[train_th+val_th:]

        self.image_tsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        image = self.image_tsfm( PIL.Image.open(io.BytesIO(data["image"]["bytes"])).convert('RGB') )

        # Default to first question
        question = data["qa"][0]["question"]
        answer = 1 if data["qa"][0]["answer"] == "Yes." else 0

        for i in range(len(data["qa"])):
            # For balanced labels
            if ((idx % 2 == 0) and (data["qa"][i]["answer"] == "Yes.")) or ((idx % 2 != 0) and (data["qa"][i]["answer"] == "No.")):
                question = data["qa"][i]["question"]
                answer = 1 if data["qa"][i]["answer"] == "Yes." else 0
                break
            

        return image, question, answer