import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .get_dataset import get_dataset


###
#   Test collate function doesn't need to combine these tensor into 1 
#   since we must left these audios in their original length (or else it would affect metric result)
###
def test_collate_fn(batch):
    # List of dict to dict of list
    features = {k: [dic[k] for dic in batch] for k in batch[0].keys()}
    
    if features.get("dvec_mel") is not None:
        features["dvec"] = features.pop("dvec_mel")
    
    if features.get("dvec_tensor"):
        features["dvec_tensor"] = torch.stack(features["dvec_tensor"], dim=0)
    
    # Add batch dimension to these features
    to_be_unsqueezed = ["target_stft", "target_wav", "mixed_wav", "mixed_stft", "mixed_mag", "mixed_phase", "target_mag", "target_phase", "dvec_tensor"]
    if features.get("dvec") is not None:
        if features["dvec"][0] is not None:
            to_be_unsqueezed.append("dvec")

    for k in to_be_unsqueezed:
        if features.get(k) is not None:
            try:
                features[k] = [f.unsqueeze(0) for f in features[k]]
            except Exception as exc:
                # print("Cannot unsqueeze ", exc)
                pass

    return features


def create_dataloader(config, scheme, features="all"):

    def train_collate_fn(batch):
        dvecs = list()
        dvec_wavs = list()
        dvec_tensors = list()
        target_wavs = list()
        mixed_wavs = list()
        target_stfts = list()
        mixed_stfts = list()
        mixed_mags = list()
        target_mags = list()
        mixed_phases = list()
        target_phases = list()
        
        for sample in batch:
            dvecs.append(sample["dvec_mel"])
            dvec_wavs.append(sample["dvec_wav"])
            if sample.get("dvec_tensor") is not None:
                dvec_tensors.append(sample["dvec_tensor"])
            target_wavs.append(sample["target_wav"])
            mixed_wavs.append(sample["mixed_wav"])
            target_stfts.append(sample["target_stft"])
            mixed_stfts.append(sample["mixed_stft"])
            mixed_mags.append(sample["mixed_mag"])
            mixed_phases.append(sample["mixed_phase"])
            target_mags.append(sample["target_mag"])
            target_phases.append(sample["target_phase"])
        
        target_stfts = pad_sequence(target_stfts, batch_first=True)
        target_wavs = pad_sequence(target_wavs, batch_first=True)
        mixed_wavs = pad_sequence(mixed_wavs, batch_first=True)
        mixed_stfts = pad_sequence(mixed_stfts, batch_first=True)
        mixed_mags = pad_sequence(mixed_mags, batch_first=True)
        mixed_phases = pad_sequence(mixed_phases, batch_first=True)
        target_mags = pad_sequence(target_mags, batch_first=True)
        target_phases = pad_sequence(target_phases, batch_first=True)

        features = {
            "dvec": dvecs, 
            "dvec_wav": dvec_wavs,
            "target_wav": target_wavs,
            "mixed_wav": mixed_wavs,
            "target_stft": target_stfts,
            "target_mag": target_mags, 
            "tagret_phase": target_phases,
            "mixed_stft": mixed_stfts,
            "mixed_mag": mixed_mags,
            "mixed_phase": mixed_phases
        }

        if len(dvec_tensors) > 0:
            dvec_tensors = torch.stack(dvec_tensors, dim=0)
            features.update({"dvec_tensor": dvec_tensors})
        
        return features

    def eval_collate_fn(batch):
        dvecs = list()
        dvec_wavs = list()
        dvec_tensors = list()
        target_wavs = list()
        mixed_wavs = list()
        target_stfts = list()
        mixed_stfts = list()
        mixed_mags = list()
        target_mags = list()
        mixed_phases = list()
        target_phases = list()
        
        for sample in batch:
            dvecs.append(sample["dvec_mel"])
            dvec_wavs.append(sample["dvec_wav"])
            if sample.get("dvec_tensor") is not None:
                dvec_tensors.append(sample["dvec_tensor"])            
            
            mixed_wavs.append(sample["mixed_wav"])
            mixed_stfts.append(sample["mixed_stft"])
            mixed_mags.append(sample["mixed_mag"])
            mixed_phases.append(sample["mixed_phase"])

            target_wavs.append(sample["target_wav"])
            target_stfts.append(sample["target_stft"])
            target_mags.append(sample["target_mag"])
            target_phases.append(sample["target_phase"])

        mixed_wavs = pad_sequence(mixed_wavs, batch_first=True)
        mixed_stfts = pad_sequence(mixed_stfts, batch_first=True)
        mixed_mags = pad_sequence(mixed_mags, batch_first=True)
        mixed_phases = pad_sequence(mixed_phases, batch_first=True)

        features = {
            "dvec": dvecs, 
            "dvec_wav": dvec_wavs,
            "mixed_wav": mixed_wavs,
            "mixed_stft": mixed_stfts,
            "mixed_mag": mixed_mags, 
            "mixed_phase": mixed_phases,
        }

        if len(dvec_tensors) > 0:
            dvec_tensors = torch.stack(dvec_tensors, dim=0)
            features.update({"dvec_tensor": dvec_tensors})
        
        target_stfts = pad_sequence(target_stfts, batch_first=True)
        target_wavs = pad_sequence(target_wavs, batch_first=True)
        target_mags = pad_sequence(target_mags, batch_first=True)
        target_phases = pad_sequence(target_phases, batch_first=True)
        features.update({
            "target_wav": target_wavs,
            "target_stft": target_stfts,
            "target_mag": target_mags, 
            "tagret_phase": target_phases
        })

        return features


    # Genearate dataset
    if features != "all" and scheme != "test":
        print("Warning: customize returned feature will mostly work in test dataloader. If use this for train/eval loader, please be sure to check if collate function have been coded properly.")
    dataset = get_dataset(config, scheme, features)


    if scheme == "train":
        return DataLoader(dataset=dataset,
                          batch_size=config.experiment.train.batch_size,
                          shuffle=True,
                          num_workers=config.experiment.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True if config.experiment.use_cuda else False,
                          drop_last=True,
                          sampler=None)
    elif scheme == "eval":
        return DataLoader(dataset=dataset,
                          batch_size=config.experiment.train.batch_size,
                          shuffle=False,
                          num_workers=config.experiment.train.num_workers,
                          collate_fn=eval_collate_fn,
                          pin_memory=True if config.experiment.use_cuda else False,
                          drop_last=False,
                          sampler=None)
    elif scheme == "test":
        return DataLoader(dataset=dataset,
                          batch_size=config.experiment.train.num_workers//2,
                          shuffle=False,
                          num_workers=config.experiment.train.num_workers//2,
                          collate_fn=test_collate_fn,
                          pin_memory=True if config.experiment.use_cuda else False,
                          drop_last=False,
                          sampler=None)
    else:
        raise NotImplementedError