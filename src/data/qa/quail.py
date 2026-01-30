
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



class QuAILDataset(Dataset):
    num_classes = 4
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42,
    ):
        ds = load_dataset("textmachinelab/quail", cache_dir=data_dir)

        match split:
            case "train": self.data = ds["train"]
            case "validation": self.data = ds["validation"]
            case "test": self.data = ds["challenge"]

        self.data = self.data.filter(lambda x: x["question_type"] == "Factual")
        
        self.data_idxs = list(range(len(self.data)))

    def __len__(self):
        return len(self.data_idxs)


    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        context = data["context"]
        question = data["question"]
        options = data["answers"]
        answer = data["correct_answer_id"]

        return context, question, options, answer