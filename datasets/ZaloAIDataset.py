import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.audio import Audio

def create_dataset(hp, dataset_name, base_dir="."):
    return SpeakerVerifyDataset(hp, dataset=dataset_name, base_dir=base_dir)
    

class SpeakerVerifyDataset(Dataset):
    def __init__(self, hp, dataset, base_dir):
        self.sr = hp.audio.sample_rate
        self.hp = hp
        if dataset == "test":
            self.data = pd.read_csv(os.path.join(base_dir, 'datasets/ZaloAI2020/private-test-fixed.csv'))
        else:
            raise NotImplementedError

        self.audio_len = int(self.sr * self.hp.data.audio_len)
        self.audio = Audio(hp)        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s1_dvec = self.data["audio_1"].iloc[idx]
        s2_dvec = self.data["audio_2"].iloc[idx]
        label = self.data["label"].iloc[idx]
        w1, _ = librosa.load(s1_dvec, sr=self.sr)
        w2, _ = librosa.load(s2_dvec, sr=self.sr)
        assert len(w1.shape) == len(w2.shape) == 1, \
            'wav files must be mono, not stereo'

        # w1, _ = librosa.effects.trim(w1, top_db=20)
        # w2, _ = librosa.effects.trim(w2, top_db=20)

        # if reference for d-vector is too short, discard it
        # if w1.shape[0] < 1.1 * self.hp.embedder.window * self.hp.audio.hop_length:
        #     return None
        # if w2.shape[0] < 1.1 * self.hp.embedder.window * self.hp.audio.hop_length:
        #     return None
        w1_p = w1.shape[0]-self.hp.embedder.window * self.hp.audio.hop_length
        w2_p = w2.shape[0]-self.hp.embedder.window * self.hp.audio.hop_length
        w1_p = 0 if w1_p > 0 else w1_p*-1
        w2_p = 0 if w2_p > 0 else w2_p*-1
        w1 = np.pad(w1, (0, w1_p), constant_values=0)
        w2 = np.pad(w2, (0, w2_p), constant_values=0)

        w1_mel = self.audio.get_mel(w1)
        w1_mel = torch.from_numpy(w1_mel).float()
        w2_mel = self.audio.get_mel(w2)
        w2_mel = torch.from_numpy(w2_mel).float()

        return {
            "s1_path": s1_dvec, 
            "s2_path": s2_dvec,
            "s1_mel": w1_mel,
            "s2_mel": w2_mel,
            "s1_wav": w2,
            "s2_wav": w1,
            "label": label
        }