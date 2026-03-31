import os
import json
import random
from datasets import load_dataset, Audio
import torch
from torch.utils.data import Dataset

from typing import Literal, Optional



class TUTUrbanDataset(Dataset):
    num_classes = 10
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: Optional[str] = None,
        seed = 42,
        sampling_rate = 16_000,
    ):
        rng = random.Random(seed)
        self.sampling_rate = sampling_rate
        ds = load_dataset("wetdog/TUT-urban-acoustic-scenes-2018-development-16bit", cache_dir=data_dir)

        self.label2id = {
            'airport': 0,
            'bus': 1,
            'metro': 2,
            'metro_station': 3,
            'park': 4,
            'public_square': 5,
            'shopping_mall': 6,
            'street_pedestrian': 7,
            'street_traffic': 8,
            'tram': 9
        }

        match split:
            case "train" | "validation":
                self.data = ds["train"]
                self.data_idxs = list( range(len(self.data)) )
                rng.shuffle(self.data_idxs)
                
                match split:
                    case "train":
                        self.data_idxs = self.data_idxs[:int(len(self.data_idxs)*0.8)]
                    case "validation":
                        self.data_idxs = self.data_idxs[int(len(self.data_idxs)*0.8):]
            case "test":
                self.data = ds["test"]
                self.data_idxs = list( range(len(self.data)) )
                rng.shuffle(self.data_idxs)

        self.data = self.data.cast_column("audio", Audio(sampling_rate=sampling_rate))

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]
        
        audio = torch.as_tensor(data["audio"]["array"])
        label = self.label2id[data["label"]]

        return audio, label