import os
import glob
import torch
import librosa
import random
import numpy as np
from torch.utils.data import Dataset
from utils.audio import Audio

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

def create_dataset(hp, dataset_name, train, size=None):
    speaker_folders=[]
    
    # Get all file paths
    if type(dataset_name) is not list:
        dataset_name = [dataset_name]

    if "librispeech-train" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.libri_dir, 'train-clean-100', '*'))
                        if os.path.isdir(x)] + \
                        [x for x in glob.glob(os.path.join(hp.data.libri_dir, 'train-clean-360', '*'))
                        if os.path.isdir(x)]

    if "librispeech-test" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.libri_dir, 'dev-clean', '*'))] + \
                        [x for x in glob.glob(os.path.join(hp.data.libri_dir, 'test-clean', '*'))]

    if "vctk" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.vctk_dir, 'wav48', '*')) if os.path.isdir(x)]

    if "vin" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.vin_dir, 'dataset', '*')) if os.path.isdir(x)]

    if "voxceleb1-train" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.voxceleb1_dir, 'dev/wav', '*')) if os.path.isdir(x)]
    if "voxceleb1-test" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.voxceleb1_dir, 'test/wav', '*')) if os.path.isdir(x)]

    if "voxceleb2-train" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.voxceleb2_dir, 'dev/aac', '*')) if os.path.isdir(x)]
    if "voxceleb1-test" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.voxceleb2_dir, 'aac', '*')) if os.path.isdir(x)]

    if "zalo-train" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.zalo_dir, 'dataset', '*')) if os.path.isdir(x)]
    if "zalo-test" in dataset_name:
        speaker_folders += [x for x in glob.glob(os.path.join(hp.data.zalo_dir, 'private-test', '*')) if os.path.isdir(x)]


    speaker_sets = [glob.glob(os.path.join(spk, '**', "*.flac"), recursive=True) + \
        glob.glob(os.path.join(spk, '**', "*.wav"), recursive=True)
                    for spk in speaker_folders]
    speaker_sets = [x for x in speaker_sets if len(x) >= 2]
    
    return VFDataset(hp, speakers=speaker_sets, train=train, size=size)
    

class VFDataset(Dataset):
    def __init__(self, hp, speakers, train, size=None):
        self.sr = hp.audio.sample_rate
        self.hp = hp
        self.speakers = speakers
        self.train = train
        self.audio_len = int(self.sr * self.hp.data.audio_len)
        self.audio = Audio(hp)     
        self.size=size

        # Constant file for visualize on tensorboard
        if train==False:
            random.seed(10)
            self.const_item = self.get_suitable_set(None)
            while self.const_item is None:
                self.const_item = self.get_suitable_set(None)
            random.seed(11)


    def __len__(self):
        if self.size:
            return self.size
        else:
            return 10**5 if self.train else 100

    def __getitem__(self, idx):
        item = self.get_suitable_set(idx)
        while item is None:
            item = self.get_suitable_set(idx)

        return item

    def get_suitable_set(self, idx):
        if idx==0 and self.train==False:
            random.seed(11)
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
        
        return s1_dvec, s1_target, s2, \
            dvec_mel, w1, mixed, \
            torch.from_numpy(target_mag), torch.from_numpy(target_phase), \
            torch.from_numpy(mixed_mag), torch.from_numpy(mixed_phase), \
            torch.from_numpy(target_stft), \
            torch.from_numpy(mixed_stft), \
            l1, l2