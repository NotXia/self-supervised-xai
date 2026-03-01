import os
import json
import random
import nltk.data
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

from typing import Literal, Optional



class FEVERDataset(Dataset):
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        join_seq: str,
        data_dir: Optional[str] = None,
        seed = 42
    ):
        rng = random.Random(42)
        self.data = load_dataset("bigbio/pubhealth", "pubhealth_source", split=split, cache_dir=data_dir)
        self.join_seq = join_seq
        self.sentence_splitter = nltk.data.load("tokenizers/punkt/english.pickle")

    def __len__(self):
        # return 1000
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return (
            (self.join_seq).join(
                [ data["claim"] ] 
                + self.__split_into_sentences(data["main_text"])
            ) + self.join_seq, 
            (1 if data["label"] else 0)
        )

    def __split_into_sentences(self, text):
        sents = self.sentence_splitter.tokenize(text)
        return [ s.strip() for s in sents ]