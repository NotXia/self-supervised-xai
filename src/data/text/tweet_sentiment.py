import os
import json
import random
import nltk.data
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

from typing import Literal, Optional



class TweetSentimentDataset(Dataset):
    num_classes = 3
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: Optional[str] = None,
        seed = 42
    ):
        rng = random.Random(seed)
        ds = load_dataset("mteb/tweet_sentiment_extraction", cache_dir=data_dir)
        self.id2label = {
            0: "negative",
            1: "neutral",
            2: "positive"
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

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]
        return data["text"], data["label"]