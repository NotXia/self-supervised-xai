import os
import json
import random
import nltk.data
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import numpy as np

from typing import Literal, Optional



class HateXplainDataset(Dataset):
    num_classes = 3
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: Optional[str] = None,
        seed = 42,
        return_xai_label = False
    ):
        self.return_xai_label = return_xai_label
        rng = random.Random(seed)
        ds = load_dataset("Hate-speech-CNERG/hatexplain", cache_dir=data_dir, trust_remote_code=True)

        self.data = ds[split]
        if split == "test":
            self.data = self.data.filter(lambda x: len(x["rationales"]) > 0)
        self.data_idxs = list( range(len(self.data)) )

    def __len__(self):
        return len(self.data_idxs)

    def __get_label(self, data):
        votes = [0, 0, 0]
        for l in data["annotators"]["label"]: 
            votes[l] += 1
        return int(np.argmax(votes))

    def __get_rationale(self, data):
        max_len = 0
        for r in data["rationales"]:
            max_len = max(max_len, len(r))
        # Sum rationales (pad if necessary)
        rationale = sum( np.array(r + [0]*(max_len-len(r))) for r in data["rationales"] )
        return np.clip(rationale, 0, 1)
            
    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        text = " ".join(data["post_tokens"])
        label = self.__get_label(data)
        rationale = self.__get_rationale(data)

        if self.return_xai_label:
            return text, label, (rationale, data["post_tokens"])
        else:
            return text, label