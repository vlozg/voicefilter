import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .get_dataset import get_dataset

def create_dataloader(config, scheme):

    def train_collate_fn(batch):
        dvecs = list()
        target_stfts = list()
        mixed_stfts = list()
        mixed_mags = list()
        target_mags = list()
        mixed_phases = list()
        target_phases = list()
        
        for sample in batch:
            dvecs.append(sample["dvec_mel"])
            target_stfts.append(sample["target_stft"])
            mixed_stfts.append(sample["mixed_stft"])
            mixed_mags.append(sample["mixed_mag"])
            mixed_phases.append(sample["mixed_phase"])
            target_mags.append(sample["target_mag"])
            target_phases.append(sample["target_phase"])
        
        target_stfts = pad_sequence(target_stfts, batch_first=True)
        mixed_stfts = pad_sequence(mixed_stfts, batch_first=True)
        mixed_mags = pad_sequence(mixed_mags, batch_first=True)
        mixed_phases = pad_sequence(mixed_phases, batch_first=True)
        target_mags = pad_sequence(target_mags, batch_first=True)
        target_phases = pad_sequence(target_phases, batch_first=True)

        return {
            "dvec": dvecs, 
            "target_stft": target_stfts,
            "target_mag": target_mags, 
            "tagret_phase": target_phases,
            "mixed_stft": mixed_stfts,
            "mixed_mag": mixed_mags,
            "mixed_phase": mixed_phases
        }

    def test_collate_fn(batch):
        dvecs = list()
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

        return {
            "dvec": dvecs, 
            "target_wav": target_wavs,
            "mixed_wav": mixed_wavs,
            "target_stft": target_stfts,
            "mixed_stft": mixed_stfts,
            "mixed_mag": mixed_mags, 
            "mixed_phase": mixed_phases,
            "target_mag": target_mags, 
            "tagret_phase": target_phases
        }

    # Genearate dataset
    dataset = get_dataset(config, scheme)


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
                          collate_fn=test_collate_fn,
                          pin_memory=True if config.experiment.use_cuda else False,
                          drop_last=False,
                          sampler=None)
    elif scheme == "test":
        return DataLoader(dataset=dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.experiment.train.num_workers,
                          collate_fn=test_collate_fn,
                          pin_memory=True if config.experiment.use_cuda else False,
                          drop_last=False,
                          sampler=None)
    else:
        raise NotImplementedError