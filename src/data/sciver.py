import os
import json
import random
import nltk.data
import torch
from torch.utils.data import Dataset

from typing import Literal



class SciVerDataset(Dataset):
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

        match split:
            case "train":
                self.data = json.load(open(os.path.join(data_dir, "valset.json"), "r"))
                self.data_idxs = list(range(1000))
                rng.shuffle(self.data_idxs)
                self.data_idxs = self.data_idxs[:800]
            case "validation":
                self.data = json.load(open(os.path.join(data_dir, "valset.json"), "r"))
                self.data_idxs = list(range(1000))
                rng.shuffle(self.data_idxs)
                self.data_idxs = self.data_idxs[800:]
            case "test":
                self.data = json.load(open(os.path.join(data_dir, "testset.json"), "r"))
                self.data_idxs = list(range(len(self.data)))

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        context = self.__prepare_context(os.path.join(self.data_dir, data["paper_path"][9:]), data["section"])
        # caption = self.__prepare_caption(os.path.join(self.data_dir, data["paper_path"][9:]), data["type"], data["item"])
        caption = ""

        return (
            (self.join_seq).join(
                [ data["claim"] ] 
                + self.__split_into_sentences(context)
                + self.__split_into_sentences(caption)
            ) + self.join_seq, 
            (1 if data["label"] else 0)
        )


    def __prepare_context(self, paper_path, section_list):
        with open(paper_path,'r',encoding='utf-8')as f:
            paper = json.load(f)
        top_sections = []
        for item in section_list:
            if item.split('.')[0] not in top_sections:
                top_sections.append(item.split('.')[0])
        context = ""
        for sec in paper['sections']:
            if sec['section_id'].split('.')[0] in top_sections:
                context += sec['section_name'] + ':\n' + sec['text'] + '\n'
        return context

    def __prepare_caption(self, paper_path, item_type, item_id):
        with open(paper_path,'r',encoding='utf-8')as f:
            paper = json.load(f)
        if item_type=='chart':
            caption = paper['image_paths'][item_id]['caption']
        else:
            caption = paper['tables'][item_id]['capture']
        return caption


    def __split_into_sentences(self, text):
        sents = self.sentence_splitter.tokenize(text)
        return [ s.strip() for s in sents ]