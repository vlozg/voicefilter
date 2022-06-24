import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.audio import Audio


class VFDataset(Dataset):
    def __init__(self, config, dataset_path, features="all"):
        self.sr = config.audio.sample_rate
        self.audio = Audio(config)
        self.data = pd.read_csv(dataset_path)
        if type(features) is list:
            self.features = features
        elif features == "all":
            self.features = ["stft", "dvec_mel", "spec", "asr"]
        else:
            self.features = None


    def __len__(self):
        return len(self.data)


    def __getaudio__(self, idx, audio):
        meta = self.data.iloc[idx].to_dict()
        s1_dvec = meta["embedding_utterance_path"]
        s1_target = meta["clean_utterance_path"]
        mixed = meta["mixed_utterance_path"]

        features_dict = {
            "dvec_path": s1_dvec, 
            "target_path": s1_target,
            "mixed_path": mixed
        }

        if self.features is not None:
            d, _ = librosa.load(s1_dvec, sr=self.sr)
            w1, _ = librosa.load(s1_target, sr=self.sr)
            mixed, _ = librosa.load(mixed, sr=self.sr)
            assert len(d.shape) == len(w1.shape) == len(mixed.shape) == 1, \
                'wav files must be mono, not stereo'

            features_dict.update({
                "dvec_wav": d,
                "target_wav": w1,
                "target_len": len(w1),
                "mixed_wav": mixed,
            })

        return meta, features_dict


    def __getitem__(self, idx):
        return self.get_item(idx, self.audio)


    def get_item(self, idx, audio):
        if audio is None:
            warnings.warn("Only use this function when you want to get audio with custom audio pre-processing")
            audio = self.audio

        meta, features_dict = self.__getaudio__(idx, audio)    

        features_dict.update({**meta})

        if meta.get("index") is None:
            features_dict.update({ "index": idx })
        else:
            features_dict.update({ "index": meta["index"] })

        if self.features is None:
            return features_dict

        if "dvec_mel" in self.features:
            if meta.get("dvec_tensor_path"):
                dvec = torch.load(meta["dvec_tensor_path"], "cpu")
                dvec_mel = None
                features_dict.update({"dvec_tensor": dvec})
            else:
                dvec_mel = audio.get_mel(features_dict["dvec_wav"])
                dvec_mel = torch.from_numpy(dvec_mel).float()
            
            features_dict.update({"dvec_mel": dvec_mel})

        # magnitude spectrograms (old)
        if "spec" in self.features:
            if features_dict.get("target_wav") is not None:
                target_mag, target_phase = audio.wav2spec(features_dict["target_wav"])
                features_dict.update({
                    "target_mag": torch.from_numpy(target_mag),
                    "target_phase": torch.from_numpy(target_phase)
                })

            mixed_mag, mixed_phase = audio.wav2spec(features_dict["mixed_wav"])
            features_dict.update({
                "mixed_mag": torch.from_numpy(mixed_mag),
                "mixed_phase": torch.from_numpy(mixed_phase),
            })

        # STFT, must transpose to get [time, freq] format
        if "stft" in self.features:
            if features_dict.get("target_wav") is not None:
                features_dict.update({"target_stft": torch.from_numpy(audio.stft(features_dict["target_wav"]).T)})
            features_dict.update({"mixed_stft": torch.from_numpy(audio.stft(features_dict["mixed_wav"]).T)})

        if "asr" in self.features:
            asr_label = None
            if meta.get("clean_utterance_text"):
                asr_label = meta.get("clean_utterance_text")
            else:
                if meta.get("clean_utterance_text_path"):
                    f_path = meta.get("clean_utterance_text_path")
                elif meta.get("clean_utterance_path"):
                    f_path = Path(meta.get("clean_utterance_path")).parent / (Path(meta.get("clean_utterance_path")).stem + ".txt")
                
                if f_path.exists():
                    with open(f_path, "r") as f:
                        asr_label = f.read().strip()
            
            features_dict.update({"target_text": asr_label})

        # Finally, convert wav from numpy to torch.tensor
        for k in ["target_wav", "mixed_wav"]:
            if features_dict.get(k) is not None:
                features_dict[k] = torch.from_numpy(features_dict[k])

        return features_dict