
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



class RACEDataset(Dataset):
    num_classes = 4
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42,
        subset: Literal["all", "high", "middle"] = "all"
    ):
        ds = load_dataset("ehovy/race", subset, cache_dir=data_dir)

        match split:
            case "train": self.data = ds["train"]
            case "validation": self.data = ds["validation"]
            case "test": self.data = ds["test"]
        self.data_idxs = list(range(len(self.data)))

    def __len__(self):
        return len(self.data_idxs)


    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        answer_map = { "a": 0, "b": 1, "c": 2, "d": 3 }

        context = data["article"]
        question = data["question"]
        options = data["options"]
        answer = data["answer"].lower()

        return context, question, options, answer_map[answer]