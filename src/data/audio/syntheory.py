import os
import json
import random
from datasets import load_dataset, Audio
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from typing import Literal, Optional



class SynTheoryDataset(Dataset):
    num_classes = 4
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: Optional[str] = None,
        seed = 42,
        sampling_rate = 16_000,
    ):
        rng = random.Random(seed)
        self.sampling_rate = sampling_rate
        ds = load_dataset("meganwei/syntheory", "chords", cache_dir=data_dir)
        ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

        self.label2id = {
            'major': 0,
            'minor': 1,
            'aug': 2,
            'dim': 3,
        }

        ds_train_test = ds["train"].train_test_split(train_size=0.8, shuffle=True, seed=seed)
        ds_train, ds_test = ds_train_test["train"], ds_train_test["test"]
        ds_train_val = ds_train.train_test_split(train_size=0.8, shuffle=True, seed=seed)
        ds_train, ds_val = ds_train_val["train"], ds_train_val["test"]
        match split:
            case "train": self.data = ds_train
            case "validation": self.data = ds_val
            case "test": self.data = ds_test
        self.data_idxs = list( range(len(self.data)) )

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]
        
        audio = torch.as_tensor(data["audio"]["array"])
        label = self.label2id[data["chord_type"]]

        return audio, label