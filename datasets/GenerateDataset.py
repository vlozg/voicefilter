import os
import glob
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

def create_dataset(config, dataset_config):
    speaker_folders=[]
    
    # Get all file paths
    dataset_detail = dataset_config.detail

    if "librispeech-train" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.libri_dir, 'train-clean-100', '*'))
                        if os.path.isdir(x)] + \
                        [x for x in glob.glob(os.path.join(config.env.data.libri_dir, 'train-clean-360', '*'))
                        if os.path.isdir(x)]

    if "librispeech-test" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.libri_dir, 'dev-clean', '*'))] + \
                        [x for x in glob.glob(os.path.join(config.env.data.libri_dir, 'test-clean', '*'))]

    if "vctk" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.vctk_dir, 'wav48', '*')) if os.path.isdir(x)]
    
    if "vivos-train" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.vivos_dir, 'train/waves', '*')) if os.path.isdir(x)]
    if "vivos-test" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.vivos_dir, 'test/waves', '*')) if os.path.isdir(x)]

    if "voxceleb1-train" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.voxceleb1_dir, 'dev/wav', '*')) if os.path.isdir(x)]
    if "voxceleb1-test" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.voxceleb1_dir, 'test/wav', '*')) if os.path.isdir(x)]

    if "voxceleb2-train" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.voxceleb2_dir, 'dev/aac', '*')) if os.path.isdir(x)]
    if "voxceleb1-test" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.voxceleb2_dir, 'aac', '*')) if os.path.isdir(x)]

    ##### START OF PRIVATE CODE #####

    if "zalo-train" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.zalo_dir, 'dataset', '*')) if os.path.isdir(x)]
    if "zalo-test" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.zalo_dir, 'private-test', '*')) if os.path.isdir(x)]

    if "vin" in dataset_detail:
        speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.vin_dir, 'speakers', '*')) if os.path.isdir(x)]

    ##### END OF PRIVATE CODE #####

    speaker_sets = [glob.glob(os.path.join(spk, '**', "*.flac"), recursive=True) + \
        glob.glob(os.path.join(spk, '**', "*.wav"), recursive=True)
                    for spk in speaker_folders]
    speaker_sets = [x for x in speaker_sets if len(x) >= 2]
    
    dataset_path = os.path.join(config.env.base_dir, dataset_config.file_path)
    if not os.path.exists(dataset_path):
        dataset_df = generate_dataset_df(config.experiment, dataset_config, speakers=speaker_sets)
        dataset_df.to_csv(dataset_path,index=False)

    return VFDataset(config.experiment, dataset_path=dataset_path)
    

class VFDataset(Dataset):
    def __init__(self, config, dataset_path):
        self.sr = config.audio.sample_rate
        self.audio = Audio(config)
        self.data = pd.read_csv(dataset_path)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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


        # LibriSpeech dataset have many silent interval, so let's vad-merge them
        # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
        # if vad == 1:
        #     w1, w2 = vad_merge(w1), vad_merge(w2)

        # I think random segment length will be better, but let's follow the paper first
        # fit audio to `hp.data.audio_len` seconds.

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

        dvec_mel = self.audio.get_mel(d)
        dvec_mel = torch.from_numpy(dvec_mel).float()

        # magnitude spectrograms (old)
        target_mag, target_phase = self.audio.wav2spec(w1)
        mixed_mag, mixed_phase = self.audio.wav2spec(mixed)
        # STFT, must transpose to get [time, freq] format
        target_stft = self.audio.stft(w1).T
        mixed_stft = self.audio.stft(mixed).T
        
        return {
            "dvec_path": s1_dvec, 
            "target_path": s1_target, 
            "interf_path": s2,
            "dvec_mel": dvec_mel,
            "target_wav": w1,
            "mixed_wav": mixed,
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