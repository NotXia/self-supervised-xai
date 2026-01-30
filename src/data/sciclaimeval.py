import os
import json
import pathlib
import random
import nltk.data

import torch
import torchvision
from torch.utils.data import Dataset
import PIL

from typing import Literal



class SciClaimEvalDataset(Dataset):
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        join_seq: str,
        data_dir: str, 
        seed = 42
    ):
        rng = random.Random(42)
        self.data_dir = data_dir
        self.join_seq = join_seq
        self.sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.label2id = {
            "Supported": 0,
            "Refuted": 1
        }

        self.data = (
            json.load(open(os.path.join(data_dir, "ml_figure_final.json"), "r"))
            + json.load(open(os.path.join(data_dir, "nlp_figure_final.json"), "r"))
            + json.load(open(os.path.join(data_dir, "peerj_figure_final.json"), "r"))
        )
        self.data_idxs = list(range(len(self.data)))
        rng.shuffle(self.data_idxs)
        train_size = int(len(self.data_idxs) * 0.8)
        val_size = int(len(self.data_idxs) * 0.1)
        test_size = len(self.data_idxs) - train_size - val_size

        match split:
            case "train":
                self.data_idxs = self.data_idxs[:train_size]
                assert len(self.data_idxs) == train_size
            case "validation":
                self.data_idxs = self.data_idxs[train_size:train_size+val_size]
                assert len(self.data_idxs) == val_size
            case "test":
                self.data_idxs = self.data_idxs[train_size+val_size:]
                assert len(self.data_idxs) == test_size

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512), interpolation=PIL.Image.BILINEAR),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        try:
            data = self.data[ self.data_idxs[idx] ]

            label = self.label2id[data["label"]]
            claim = data["claim"]
            caption = data["caption"]
            context = data["context"]
            figure_path = pathlib.Path( *pathlib.Path(data["evi_path"]).parts[1:] ) # Strip root directory
            figure_path = os.path.join(self.data_dir, figure_path)

            text_data = (self.join_seq).join(
                [ claim ] 
                # + self.__split_into_sentences(caption)
                # + self.__split_into_sentences(context)
            ) + self.join_seq

            image_data = self.transforms(PIL.Image.open(figure_path).convert("RGB"))

            return ( text_data, image_data, label )
        except:
            return self.__getitem__(0)

    def __split_into_sentences(self, text):
        sents = self.sentence_splitter.tokenize(text)
        return [ s.strip() for s in sents ]