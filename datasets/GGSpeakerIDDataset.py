import librosa
import numpy as np

from .GeneralDataset import VFDataset


class VFGGDataset(VFDataset):
    def __init__(self, config, dataset_path, features="all"):
        super().__init__(config, dataset_path, features)


    def __getaudio__(self, idx, audio):
        meta = self.data.iloc[idx].to_dict()
        s1_dvec = meta["embedding_utterance_path"]
        s1_target = meta["clean_utterance_path"]
        s2 = meta["interference_utterance_path"]

        features_dict = {
            "dvec_path": s1_dvec, 
            "target_path": s1_target, 
            "interf_path": s2
        }

        d, _ = librosa.load(s1_dvec, sr=self.sr)
        w1, _ = librosa.load(s1_target, sr=self.sr)
        w2, _ = librosa.load(s2, sr=self.sr)
        assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
            'wav files must be mono, not stereo'

        mixed = w1 + np.pad(w2[:w1.shape[0]], (0, max(0, w1.shape[0]-w2.shape[0])), constant_values=0)

        norm = np.max(np.abs(mixed))
        w1, w2, mixed = w1/norm, w2/norm, mixed/norm

        features_dict.update({
            "dvec_wav": d,
            "target_wav": w1,
            "target_len": len(w1),
            "interf_wav": w2,
            "interf_len": len(w2),
            "mixed_wav": mixed,
        })

        return meta, features_dict
