import os
import json
import random

import torch
from torch.utils.data import Dataset
import torchvision
import PIL

from typing import Literal



class SciVerSimpleDataset(Dataset):
    num_classes = 2

    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: str, 
        seed = 42
    ):
        rng = random.Random(42)
        self.data_dir = data_dir

        self.image_tsfm = torchvision.transforms.Compose([
            torchvision.transforms.v2.RGB(),
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.ToTensor()
        ])

        match split:
            case "train" | "validation":
                self.data = json.load(open(os.path.join(data_dir, "SciVer/valset.json"), "r"))
                self.data = self.__filter(self.data)
                self.data_idxs = list(range(len(self.data)))
                rng.shuffle(self.data_idxs)
                if split == "train":
                    self.data_idxs = self.data_idxs[:int(len(self.data)*0.8)]
                else:
                    self.data_idxs = self.data_idxs[int(len(self.data)*0.8):]
            case "test":
                self.data = json.load(open(os.path.join(data_dir, "SciVer/testset.json"), "r"))
                self.data = self.__filter(self.data)
                self.data_idxs = list(range(len(self.data)))

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]
        image = self.image_tsfm( PIL.Image.open(os.path.join(self.data_dir, data["image_path"])) )

        return (
            {
                "claim": data["claim"],
                "image": image
            },
            (1 if data["label"] else 0)
        )


    def __filter(self, data):
        data = [d for d in data if d["claim_type"] in ["direct", "analytical"]]
        return data