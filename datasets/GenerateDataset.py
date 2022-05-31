import warnings
import random

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

class VFDataset(Dataset):
    def __init__(self, config, dataset_path):
        self.sr = config.audio.sample_rate
        self.audio = Audio(config)
        self.data = pd.read_csv(dataset_path)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_item(idx, self.audio)

    def get_item(self, idx, audio):
        if audio is None:
            warnings.warn("Only use this function when you want to get audio with custom audio pre-processing")
            audio = self.audio

        s1_dvec = self.data["embedding_utterance_path"].iloc[idx]
        s1_target = self.data["clean_utterance_path"].iloc[idx]
        s2 = self.data["interference_utterance_path"].iloc[idx]
        l1 = self.data["clean_segment_start"].iloc[idx]
        l2 = self.data["interference_segment_start"].iloc[idx]
        audio_len = self.data["segment_length"].iloc[idx]

        if not np.isnan(audio_len):
            audio_len = int(audio_len*self.sr)

        d, _ = librosa.load(s1_dvec, sr=self.sr)
        w1, _ = librosa.load(s1_target, sr=self.sr)
        w2, _ = librosa.load(s2, sr=self.sr)
        assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
            'wav files must be mono, not stereo'

        if not np.isnan(audio_len):
            w1, w2 = w1[l1:l1+audio_len], w2[l2:l2+audio_len]
        else:
            w1, w2 = w1[l1:], w2[l2:]
            max_len = max(w1.shape[0], w2.shape[0])
            w1 = np.pad(w1, (0, max(0, max_len - w1.shape[0])), constant_values = 0)
            w2 = np.pad(w2, (0, max(0, max_len - w2.shape[0])), constant_values = 0)
        mixed = w1 + w2

        norm = np.max(np.abs(mixed)) * 1.1
        w1, w2, mixed = w1/norm, w2/norm, mixed/norm

        dvec_mel = audio.get_mel(d)
        dvec_mel = torch.from_numpy(dvec_mel).float()

        # magnitude spectrograms (old)
        target_mag, target_phase = audio.wav2spec(w1)
        mixed_mag, mixed_phase = audio.wav2spec(mixed)
        # STFT, must transpose to get [time, freq] format
        target_stft = audio.stft(w1).T
        mixed_stft = audio.stft(mixed).T
        
        return {
            "dvec_path": s1_dvec, 
            "target_path": s1_target, 
            "interf_path": s2,
            "dvec_mel": dvec_mel,
            "dvec_wav": d,
            "target_wav": torch.from_numpy(w1),
            "interf_wav": torch.from_numpy(w2),
            "mixed_wav": torch.from_numpy(mixed),
            "target_mag": torch.from_numpy(target_mag),
            "target_phase": torch.from_numpy(target_phase),
            "mixed_mag": torch.from_numpy(mixed_mag),
            "mixed_phase": torch.from_numpy(mixed_phase),
            "target_stft": torch.from_numpy(target_stft),
            "mixed_stft": torch.from_numpy(mixed_stft),
            "target_segment_start": l1,
            "interf_segment_start": l2,
            "segment_length": audio_len
        }

def generate_dataset_df(exp_config, dataset_config, speakers):
    config = exp_config
    sr = config.audio.sample_rate

    if dataset_config.audio_len is not None:
        audio_len = int(sr * dataset_config.audio_len)
    else:
        audio_len = 0

    s1_targets = list()
    s1_dvecs = list()
    s2s = list()
    l1s = list()
    l2s = list()

    i=0

    while i<dataset_config.size:
        # Random 2 speaker
        spk1, spk2 = random.sample(speakers, 2)
        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)

        d, _ = librosa.load(s1_dvec, sr=sr)
        w1, _ = librosa.load(s1_target, sr=sr)
        w2, _ = librosa.load(s2, sr=sr)
        assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
            'wav files must be mono, not stereo'


        # if reference for d-vector is too short, discard it
        if d.shape[0] < 1.1 * config.embedder.window * config.audio.hop_length:
            continue

        # I think random segment length will be better, but let's follow the paper first
        # fit audio to `hp.data.audio_len` seconds.
        # if merged audio is shorter than `L`, discard it
        if w1.shape[0] < audio_len or w2.shape[0] < audio_len:
            continue

        # Random 2 segment
        l1 = random.randint(0, w1.shape[0]-audio_len)
        l2 = random.randint(0, w2.shape[0]-audio_len)

        # Post check if data sample is qualify
        if audio_len > 0:
            w1, w2 = w1[l1:l1+audio_len], w2[l2:l2+audio_len]
        else:
            w1, w2 = w1[l1:], w2[l2:]

        # Discard almost silent target audio sample (since it will cause error in torch_mir_eval)
        if w1.sum() < 10e-5:
            continue

        s1_targets.append(s1_target)
        s1_dvecs.append(s1_dvec)
        s2s.append(s2)
        l1s.append(l1)
        l2s.append(l2)

        i += 1

    df=pd.DataFrame(
        {'clean_utterance_path': s1_targets
        ,'embedding_utterance_path': s1_dvecs
        ,'interference_utterance_path':s2s
        ,'clean_segment_start':l1s
        ,'interference_segment_start':l2s})

    df['segment_length'] = dataset_config.audio_len
    return df