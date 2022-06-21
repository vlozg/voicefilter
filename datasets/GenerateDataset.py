import warnings
import random
import concurrent.futures

import torch
import librosa
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils.audio import Audio
import tqdm as tqdm

def vad_merge(w, top_db=20, ref=None):
    if ref:
        intervals = librosa.effects.split(w, top_db=top_db)
    else:
        intervals = librosa.effects.split(w, top_db=top_db, ref=ref)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

class VFDataset(Dataset):
    def __init__(self, config, dataset_path, features="all"):
        self.sr = config.audio.sample_rate
        self.audio = Audio(config)
        self.data = pd.read_csv(dataset_path)
        if type(features) is list:
            self.features = features
        elif features == "all":
            self.features = ["stft", "dvec_mel", "spec"]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_item(idx, self.audio)

    def get_item(self, idx, audio):
        if audio is None:
            warnings.warn("Only use this function when you want to get audio with custom audio pre-processing")
            audio = self.audio

        meta = self.data.iloc[idx].to_dict()
        s1_dvec = meta["embedding_utterance_path"]
        s1_target = meta["clean_utterance_path"]
        s2 = meta["interference_utterance_path"]
        l1 = meta["clean_segment_start"]
        l2 = meta["interference_segment_start"]
        audio_len = meta["segment_length"]

        if not np.isnan(audio_len):
            audio_len = int(audio_len*self.sr)

        features_dict = {
            "dvec_path": s1_dvec, 
            "target_path": s1_target, 
            "interf_path": s2,
            "target_segment_start": l1,
            "interf_segment_start": l2,
            "segment_length": audio_len
        }


        d, _ = librosa.load(s1_dvec, sr=self.sr)
        w1, _ = librosa.load(s1_target, sr=self.sr)
        w2, _ = librosa.load(s2, sr=self.sr)
        assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
            'wav files must be mono, not stereo'

        if not np.isnan(audio_len):
            w1, w2 = w1[l1:l1+audio_len], w2[l2:l2+audio_len]
            w1_len, w2_len = w1.shape[0], w2.shape[0]
        else:
            w1, w2 = w1[l1:], w2[l2:]
            w1_len, w2_len = w1.shape[0], w2.shape[0]
            max_len = max(w1_len, w2_len)
            w1 = np.pad(w1, (0, max(0, max_len - w1.shape[0])), constant_values = 0)
            w2 = np.pad(w2, (0, max(0, max_len - w2.shape[0])), constant_values = 0)
        mixed = w1 + w2

        norm = np.max(np.abs(mixed)) * 1.1
        w1, w2, mixed = w1/norm, w2/norm, mixed/norm

        features_dict.update({
            "dvec_wav": d,
            "target_wav": torch.from_numpy(w1),
            "target_len": w1_len,
            "interf_wav": torch.from_numpy(w2),
            "interf_len": w1_len,
            "mixed_wav": torch.from_numpy(mixed),
        })

        if "dvec_mel" in self.features:
            if meta.get("dvec_tensor_path"):
                dvec = torch.load(meta["dvec_tensor_path"], "cpu")
                dvec_mel = None
                features_dict.update({"dvec_tensor": dvec})
            else:
                dvec_mel = audio.get_mel(d)
                dvec_mel = torch.from_numpy(dvec_mel).float()
            
            features_dict.update({"dvec_mel": dvec_mel})

        # magnitude spectrograms (old)
        if "spec" in self.features:
            target_mag, target_phase = audio.wav2spec(w1)
            mixed_mag, mixed_phase = audio.wav2spec(mixed)
            features_dict.update({
                "target_mag": torch.from_numpy(target_mag),
                "target_phase": torch.from_numpy(target_phase),
                "mixed_mag": torch.from_numpy(mixed_mag),
                "mixed_phase": torch.from_numpy(mixed_phase),
            })

        # STFT, must transpose to get [time, freq] format
        if "stft" in self.features:
            target_stft = audio.stft(w1).T
            mixed_stft = audio.stft(mixed).T
            features_dict.update({
                "target_stft": torch.from_numpy(target_stft),
                "mixed_stft": torch.from_numpy(mixed_stft),
            })
        
        return features_dict


def generate_sample(s1_target, s2, s1_dvec, sr, audio_len, min_dvec_len):
    d, _ = librosa.load(s1_dvec, sr=sr)
    w1, _ = librosa.load(s1_target, sr=sr)
    w2, _ = librosa.load(s2, sr=sr)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
        'wav files must be mono, not stereo'

    # if reference for d-vector is too short, discard it
    if d.shape[0] < min_dvec_len:
        return

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    if w1.shape[0] < audio_len or w2.shape[0] < audio_len:
        return

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
        return

    return {
        'clean_utterance_path': s1_target
        ,'embedding_utterance_path': s1_dvec
        ,'interference_utterance_path':s2
        ,'clean_segment_start':l1
        ,'interference_segment_start':l2
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

    print('Generating ',dataset_config.detail[0],' dataset (save in file',dataset_config.file_path,'):')
    i=0
    pbar = tqdm.tqdm(total = dataset_config.size)

    dataset_detail = dataset_config.detail
    dataset_weight = dataset_config.get("weight")

    min_dvec_len = 1.1 * config.embedder.window * config.audio.hop_length

    ###
    # Start multiprocess generate
    ###
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_samples = []
        counter = 0

        while i<dataset_config.size:
            # Random datasets to get speaker
            if dataset_config.get("type") == "mixed" or dataset_config.get("type") is None:
                dataset_choice = np.random.choice(dataset_detail, 2, replace=True, p=dataset_weight)
            elif dataset_config.type == "cross":
                dataset_choice = np.random.choice(dataset_detail, 1, replace=True, p=dataset_weight)
                dataset_choice = np.repeat(dataset_choice, 2)

            # Random 2 speakers from selected datasets
            if dataset_choice[0] == dataset_choice[1]:
                spk1, spk2 = random.sample(speakers[dataset_choice[0]], 2)
            else:
                spk1 = random.choice(speakers[dataset_choice[0]])
                spk2 = random.choice(speakers[dataset_choice[1]])
            s1_dvec, s1_target = random.sample(spk1, 2)
            s2 = random.choice(spk2)

            future_samples.append(executor.submit(generate_sample, s1_target, s2, s1_dvec, sr, audio_len, min_dvec_len))
            counter += 1

            if counter == 100:
                for future in concurrent.futures.as_completed(future_samples):
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('Generated an exception: %s' % (exc))
                    else:
                        if data is None: continue

                        s1_targets.append(data["clean_utterance_path"])
                        s1_dvecs.append(data["embedding_utterance_path"])
                        s2s.append(data["interference_utterance_path"])
                        l1s.append(data["clean_segment_start"])
                        l2s.append(data["interference_segment_start"])
                        i += 1
                        pbar.update(1)

                        if i==dataset_config.size: break

                counter = 0

        pbar.close()

    df=pd.DataFrame(
        {'clean_utterance_path': s1_targets
        ,'embedding_utterance_path': s1_dvecs
        ,'interference_utterance_path':s2s
        ,'clean_segment_start':l1s
        ,'interference_segment_start':l2s})

    df['segment_length'] = dataset_config.audio_len
    return df