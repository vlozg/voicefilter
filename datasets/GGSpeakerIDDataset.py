import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.audio import Audio

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

def create_dataset(hp, dataset_name, base_dir="."):
    return VFGGDataset(hp, dataset=dataset_name, base_dir=base_dir)


class VFGGDataset(Dataset):
    def __init__(self, hp, dataset, base_dir):
        self.sr = hp.audio.sample_rate
        self.hp = hp
        if dataset == "libri_train":
            raise NotImplementedError
        elif dataset == "libri_dev":
            self.data = pd.read_csv(os.path.join(base_dir, 'datasets/LibriSpeech/dev_tuples_path.csv'))
        elif dataset == "vctk_train":
            raise NotImplementedError
        elif dataset == "vctk_test":
            self.data = pd.read_csv(os.path.join(base_dir, 'datasets/VCTK-Corpus/test_tuples_path.csv'))
        elif dataset == "train":
            raise NotImplementedError
        elif dataset == "test":
            df1 = pd.read_csv(os.path.join(base_dir, 'datasets/LibriSpeech/dev_tuples_path.csv'))
            df2 = pd.read_csv(os.path.join(base_dir, 'datasets/VCTK-Corpus/test_tuples_path.csv'))
            self.data = pd.concat([df1, df2], ignore_index=True)
        else:
            raise NotImplementedError
        self.audio_len = int(self.sr * self.hp.data.audio_len)
        self.audio = Audio(hp)        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s1_dvec = self.data["embedding_utterance_path"].iloc[idx]
        s1_target = self.data["clean_utterance_path"].iloc[idx]
        s2 = self.data["interference_utterance_path"].iloc[idx]

        d, _ = librosa.load(s1_dvec, sr=self.sr)
        w1, _ = librosa.load(s1_target, sr=self.sr)
        w2, _ = librosa.load(s2, sr=self.sr)
        assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
            'wav files must be mono, not stereo'

        # Do not trimp dvec since it will cause dvec too short
        # d, _ = librosa.effects.trim(d, top_db=20)
        w1, _ = librosa.effects.trim(w1, top_db=20)
        w2, _ = librosa.effects.trim(w2, top_db=20)

        # Fix length to reserve istft
        # w1 = librosa.util.fix_length(w1, w1.shape[0] + self.hp.audio.n_fft // 2)

        # LibriSpeech dataset have many silent interval, so let's vad-merge them
        # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
        # if vad == 1:
        #     w1, w2 = vad_merge(w1), vad_merge(w2)

        # Mix 2 audio, then trim to match clean audio length
        mixed = w1 + np.pad(w2[:w1.shape[0]], (0, max(0, w1.shape[0]-w2.shape[0])), constant_values=0)

        norm = np.max(np.abs(mixed)) * 1.1
        w1, w2, mixed = w1/norm, w2/norm, mixed/norm

        dvec_mel = self.audio.get_mel(d)
        dvec_mel = torch.from_numpy(dvec_mel).float()

        # magnitude spectrograms (old)
        target_mag, target_phase = self.audio.wav2spec(w1)
        mixed_mag, mixed_phase = self.audio.wav2spec(mixed)
        # STFT, must transpose to get [time, freq] format
        target_stft = self.audio.stft(w1).T
        mixed_stft = self.audio.stft(mixed).T
        
        return s1_dvec, s1_target, s2, \
            dvec_mel, w1, mixed, \
            torch.from_numpy(target_mag), torch.from_numpy(target_phase), \
            torch.from_numpy(mixed_mag), torch.from_numpy(mixed_phase), \
            torch.from_numpy(target_stft), \
            torch.from_numpy(mixed_stft)