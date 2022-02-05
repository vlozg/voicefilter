import os
import glob
import torch
import librosa
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

from utils.audio import Audio

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

def create_dataloader(hp, args, train):
    def train_collate_fn(batch):
        dvec_list = list()
        target_stft_list = list()
        mixed_stft_list = list()
        mixed_mag_list = list()
        target_mag_list = list()
        mixed_phase_list = list()
        target_phase_list = list()
        

        for dvec_mel, _, _, mixed_mag, mixed_phase, target_mag, target_phase, target_stft, mixed_stft in batch:
            dvec_list.append(dvec_mel)
            target_stft_list.append(target_stft)
            mixed_stft_list.append(mixed_stft)
            mixed_mag_list.append(mixed_mag)
            mixed_phase_list.append(mixed_phase)
            target_mag_list.append(target_mag)
            target_phase_list.append(target_phase)
        target_stft_list = torch.stack(target_stft_list, dim=0)
        mixed_stft_list = torch.stack(mixed_stft_list, dim=0)
        mixed_mag_list = torch.stack(mixed_mag_list, dim=0)
        mixed_phase_list = torch.stack(mixed_phase_list, dim=0)
        target_mag_list = torch.stack(target_mag_list, dim=0)
        target_phase_list = torch.stack(target_phase_list, dim=0)

        return dvec_list, mixed_mag_list, mixed_phase_list, \
            target_mag_list, target_phase_list, \
            target_stft_list, mixed_stft_list

    def test_collate_fn(batch):
        return batch

    args = {
        "libri_dir": "datasets/LibriSpeech",
        "voxceleb_dir": None,
        "out_dir": "tmp",
        "vad": 0
    }

    args = SimpleNamespace(**args)

    if args.libri_dir is None and args.voxceleb_dir is None:
        raise Exception("Please provide directory of data")

    # Get all file paths
    if args.libri_dir is not None:
        train_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-100', '*'))
                            if os.path.isdir(x)] + \
                        [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-360', '*'))
                            if os.path.isdir(x)]
        test_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'dev-clean', '*'))]

    elif args.voxceleb_dir is not None:
        all_folders = [x for x in glob.glob(os.path.join(args.voxceleb_dir, '*'))
                            if os.path.isdir(x)]
        train_folders = all_folders[:-20]
        test_folders = all_folders[-20:]

    train_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in train_folders]
    train_spk = [x for x in train_spk if len(x) >= 2]

    test_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in test_folders]
    test_spk = [x for x in test_spk if len(x) >= 2]
    
    if train:
        return DataLoader(dataset=VFDataset(hp, args, speakers=train_spk, train=True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=VFDataset(hp, args, speakers=test_spk, train=False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)


class VFDataset(Dataset):
    def __init__(self, hp, args, speakers, train):
        self.sr = hp.audio.sample_rate
        self.hp = hp
        self.args = args
        self.speakers = speakers
        self.train = train
        self.data_dir = hp.data.train_dir if train else hp.data.test_dir
        self.audio_len = int(self.sr * self.hp.data.audio_len)
        self.audio = Audio(hp)        

        # Constant file for visualize on tensorboard
        if train==False:
            random.seed(10)
            self.const_item = self.get_suitable_set(None)
            while self.const_item is None:
                self.const_item = self.get_suitable_set(None)


    def __len__(self):
        return 10**5 if self.train else 100

    def __getitem__(self, idx):
        item = self.get_suitable_set(idx)
        while item is None:
            item = self.get_suitable_set(idx)

        return item

    def get_suitable_set(self, idx):
        if idx==0 and self.train==False:
            return self.const_item
        else:
            # Random 2 speaker
            spk1, spk2 = random.sample(self.speakers, 2)
            s1_dvec, s1_target = random.sample(spk1, 2)
            s2 = random.choice(spk2)

        d, _ = librosa.load(s1_dvec, sr=self.sr)
        w1, _ = librosa.load(s1_target, sr=self.sr)
        w2, _ = librosa.load(s2, sr=self.sr)
        assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
            'wav files must be mono, not stereo'

        d, _ = librosa.effects.trim(d, top_db=20)
        w1, _ = librosa.effects.trim(w1, top_db=20)
        w2, _ = librosa.effects.trim(w2, top_db=20)

        # if reference for d-vector is too short, discard it
        if d.shape[0] < 1.1 * self.hp.embedder.window * self.hp.audio.hop_length:
            return None

        # LibriSpeech dataset have many silent interval, so let's vad-merge them
        # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
        # if vad == 1:
        #     w1, w2 = vad_merge(w1), vad_merge(w2)

        # I think random segment length will be better, but let's follow the paper first
        # fit audio to `hp.data.audio_len` seconds.
        # if merged audio is shorter than `L`, discard it
        if w1.shape[0] < self.audio_len or w2.shape[0] < self.audio_len:
            return None

        # Random 2 segment
        l1 = random.randint(0, w1.shape[0]-self.audio_len-1)
        l2 = random.randint(0, w2.shape[0]-self.audio_len-1)
        w1, w2 = w1[l1:l1+self.audio_len], w2[l2:l2+self.audio_len]
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
        
        return dvec_mel, w1, mixed, \
            torch.from_numpy(target_mag), torch.from_numpy(target_phase), \
            torch.from_numpy(mixed_mag), torch.from_numpy(mixed_phase), \
            torch.from_numpy(target_stft), \
            torch.from_numpy(mixed_stft)