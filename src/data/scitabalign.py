import os
import json
import random
import itertools

import nltk.data
import torch
from torch.utils.data import Dataset

from typing import Literal



class SciTabAlignDataset(Dataset):
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        join_seq: str,
        data_dir: str, 
        seed = 42,
        return_explanation = False
    ):
        rng = random.Random(42)
        self.data_dir = data_dir
        self.join_seq = join_seq
        self.sentence_splitter = nltk.data.load("tokenizers/punkt/english.pickle")
        self.split = split
        self.return_explanation = return_explanation
        self.label2id = {
            "supports": 0,
            "Supported": 0,

            "refutes": 1,
            "Refuted": 1,

            "not enough info": 2
        }

        with open(os.path.join(data_dir, "data.json"), "r") as f:
            self.scitabalign = json.load(f)
        with open(os.path.join(data_dir, "sci_tab.json"), "r") as f:
            self.scitab = json.load(f)

        self.scitab_ids = [ x["id"] for x in self.scitab ]
        self.scitabalign_ids = [ x["id"] for x in self.scitabalign ]

        match split:
            case "train":
                self.data = self.scitab
                self.data_idxs = [ idx for idx, id in enumerate(self.scitab_ids) if id not in self.scitabalign_ids ]
                rng.shuffle(self.data_idxs)
                self.data_idxs = self.data_idxs[:700]
                # self.data_idxs = self.data_idxs[:100]
            case "validation":
                self.data = self.scitab
                self.data_idxs = [ idx for idx, id in enumerate(self.scitab_ids) if id not in self.scitabalign_ids ]
                rng.shuffle(self.data_idxs)
                self.data_idxs = self.data_idxs[700:]
            case "test":
                self.data = self.scitabalign
                self.data_idxs = list(range(len(self.scitabalign_ids)))

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        data = self.data[ self.data_idxs[idx] ]

        if (self.split == "test") and (self.return_explanation):
            return (
                (self.join_seq).join(
                    [ data["claim"] ] 
                    + [ data["table_caption"] ]
                    + data["table_column_names"]
                    + [str(x) for x in list( itertools.chain(*data["table_content_values"]) )]
                    # + [ self.__toMarkdown(data["table_column_names"], data["table_content_values"]) ]
                ) + self.join_seq, 
                self.label2id[ data["label"] ],
                len(data["table_column_names"]),
                data["explanation_cells"]
            )

        # if (self.split == "test"):
        #     return (
        #         data["claim"],
        #         data["table_caption"],
        #         data["table_column_names"],
        #         data["table_content_values"],
        #         self.label2id[ data["label"] ],
        #     )
        else:
            return (
                (self.join_seq).join(
                    [ data["claim"] ] 
                    + [ data["table_caption"] ]
                    + data["table_column_names"]
                    + [str(x) for x in list( itertools.chain(*data["table_content_values"]) )]
                    # + [ self.__toMarkdown(data["table_column_names"], data["table_content_values"]) ]
                ) + self.join_seq, 
                self.label2id[ data["label"] ]
            )
            # return (
            #     data["claim"],
            #     data["table_caption"],
            #     data["table_column_names"],
            #     data["table_content_values"],
            #     self.label2id[ data["label"] ],
            # )


    def __split_into_sentences(self, text):
        sents = self.sentence_splitter.tokenize(text)
        return [ s.strip() for s in sents ]

    
    def __toMarkdown(self, header, rows):
        table_md = ""
        table_md += "| " + " | ".join(header) + " |\n"
        table_md += "".join(["| ---- " for _ in range(len(header))]) + " |\n"
        for r in rows:
            table_md += "| " + " | ".join([str(v) for v in r]) + " |\n"
        return table_md
        