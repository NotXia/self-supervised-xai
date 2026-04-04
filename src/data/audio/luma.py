import os
import json
import random
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf

from typing import Literal, Optional



class LUMADataset(Dataset):
    num_classes = 50
    
    def __init__(self, 
        split: Literal["train", "validation", "test"], 
        data_dir: Optional[str] = None,
        seed = 42,
        sampling_rate = 16_000,
        return_xai_label = False

    ):
        self.rng = random.Random(seed)
        self.sampling_rate = sampling_rate
        self.return_xai_label = return_xai_label

        # snapshot_download(repo_id="bezirganyan/LUMA", repo_type="dataset", local_dir=os.path.join(data_dir, "LUMA"))
        df = pd.read_csv(os.path.join(data_dir, "LUMA/audio/datalist.csv"))

        df_train, df_test = train_test_split(df, train_size=0.8, shuffle=True, random_state=seed)
        df_train, df_val = train_test_split(df_train, train_size=0.8, shuffle=True, random_state=seed)

        match split:
            case "train": self.data = df_train
            case "validation": self.data = df_val
            case "test": self.data = df_test
        self.data["path"] = os.path.join(data_dir, "LUMA/audio/") + self.data["path"]
        self.data_idxs = list( range(len(self.data)) )


    def __len__(self):
        return len(self.data_idxs)

    def __inject_noise(self, audio):
        noise = 0.01 * torch.normal(mean=0, std=torch.ones(5 * self.sampling_rate)) # Total 5 seconds of audio
        true_mask = torch.zeros(len(noise))
        
        # Randomly place audio into noise
        audio_pos = self.rng.randint(0, len(noise)-len(audio))
        noise[audio_pos:audio_pos+len(audio)] = audio
        true_mask[audio_pos:audio_pos+len(audio)] = 1 # Mark portion with real audio

        return noise, true_mask

    def __getitem__(self, idx):
        data = self.data.iloc[ self.data_idxs[idx] ]
        
        audio_path = data["path"]
        audio, in_sampling_rate = sf.read(data["path"])
        audio = torchaudio.functional.resample(torch.as_tensor(audio), in_sampling_rate, self.sampling_rate)
        audio, xai_label = self.__inject_noise(audio)
        label = int(data["class"])

        if self.return_xai_label:
            return audio, label, xai_label
        else:
            return audio, label