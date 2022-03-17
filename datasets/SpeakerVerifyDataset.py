import os
import glob
import torch
import librosa
import random
import numpy as np
from torch.utils.data import Dataset
from utils.audio import Audio

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
    
    return SpeakerVerifyDataset(hp, speakers=speaker_sets, train=train, size=size)
    

class SpeakerVerifyDataset(Dataset):
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
            # Random label (0 = not match, 1 = match)
            label = random.randint(0, 1)
            if label == 0:
                spk1, spk2 = random.sample(self.speakers, 2)
                s1_dvec = random.choice(spk1)
                s2_dvec = random.choice(spk2)
            else:
                spk = random.choice(self.speakers)
                s1_dvec, s2_dvec = random.sample(spk, 2)

        w1, _ = librosa.load(s1_dvec, sr=self.sr)
        w2, _ = librosa.load(s2_dvec, sr=self.sr)
        assert len(w1.shape) == len(w2.shape) == 1, \
            'wav files must be mono, not stereo'

        # w1, _ = librosa.effects.trim(w1, top_db=20)
        # w2, _ = librosa.effects.trim(w2, top_db=20)

        # if reference for d-vector is too short, discard it
        if w1.shape[0] < 1.1 * self.hp.embedder.window * self.hp.audio.hop_length:
            return None
        if w2.shape[0] < 1.1 * self.hp.embedder.window * self.hp.audio.hop_length:
            return None

        w1_mel = self.audio.get_mel(w1)
        w1_mel = torch.from_numpy(w1_mel).float()
        w2_mel = self.audio.get_mel(w2)
        w2_mel = torch.from_numpy(w2_mel).float()

        return s1_dvec, s2_dvec, \
            w1, w2, \
            w1_mel, w2_mel, \
            label