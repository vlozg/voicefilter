from numpy import append
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .GenerateDataset import create_dataset
from .GGSpeakerIDDataset import create_dataset as create_gg_dataset

def create_dataloader(hp, dataset_type, *, dataset_detail=None, scheme, size=None):

# s1_dvec, s1_target, s2, dvec_mel, target_wav, mixed_wav, target_mag, target_phase, mixed_mag, mixed_phase, target_stft, mixed_stft, l1, l2  ==batch
    def train_collate_fn(batch):
        dvec_list = list()
        target_stft_list = list()
        mixed_stft_list = list()
        mixed_mag_list = list()
        target_mag_list = list()
        mixed_phase_list = list()
        target_phase_list = list()
        
        for _, _, _, dvec_mel, _, _, target_mag, target_phase, mixed_mag, mixed_phase, target_stft, mixed_stft, *_ in batch:
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

        return dvec_list, target_mag_list, target_phase_list, \
            mixed_mag_list, mixed_phase_list, \
            target_stft_list, mixed_stft_list

    def train_psedccrn_collate_fn(batch):
        dvec_list = list()
        target_wav_list = list()
        mixed_wav_list = list()
        target_stft_list = list()
        mixed_stft_list = list()
        
        for _, _, _, dvec_mel, w1, mixed, _, _, _, _, target_stft, mixed_stft, *_ in batch:
            dvec_list.append(dvec_mel)
            target_stft_list.append(target_stft)
            mixed_stft_list.append(mixed_stft)
            target_wav_list.append(torch.from_numpy(w1))
            mixed_wav_list.append(torch.from_numpy(mixed))
        target_stft_list = torch.stack(target_stft_list, dim=0)
        mixed_stft_list = torch.stack(mixed_stft_list, dim=0)
        target_wav_list = torch.stack(target_wav_list, dim=0)
        mixed_wav_list = torch.stack(mixed_wav_list, dim=0)

        return dvec_list, target_wav_list, mixed_wav_list, \
            target_stft_list, mixed_stft_list

    def train_raw_collate_fn(batch):
        dvec_list = list()
        target_wav_list = list()
        mixed_wav_list = list()
        
        for _, _, _, dvec_mel, w1, mixed, *_ in batch:
            dvec_list.append(dvec_mel)
            target_wav_list.append(w1)
            mixed_wav_list.append(mixed)
        target_wav_list = torch.stack(target_wav_list, dim=0)
        mixed_wav_list = torch.stack(mixed_wav_list, dim=0)

        return dvec_list, target_wav_list, mixed_wav_list

    def test_collate_fn(batch):
        return batch

    def test_cuda_collate_fn(batch):
        dvec_list = list()
        target_wav_list = list()
        mixed_wav_list = list()
        target_stft_list = list()
        mixed_stft_list = list()
        mixed_mag_list = list()
        target_mag_list = list()
        mixed_phase_list = list()
        target_phase_list = list()
        
        for _, _, _, dvec_mel, target_wav, mixed_wav, target_mag, target_phase, mixed_mag, mixed_phase, target_stft, mixed_stft, *_ in batch:
            dvec_list.append(dvec_mel)
            target_wav_list.append(target_wav)
            mixed_wav_list.append(mixed_wav)
            target_stft_list.append(target_stft)
            mixed_stft_list.append(mixed_stft)
            mixed_mag_list.append(mixed_mag)
            mixed_phase_list.append(mixed_phase)
            target_mag_list.append(target_mag)
            target_phase_list.append(target_phase)
        target_stft_list = pad_sequence(target_stft_list, batch_first=True)
        mixed_stft_list = pad_sequence(mixed_stft_list, batch_first=True)
        mixed_mag_list = pad_sequence(mixed_mag_list, batch_first=True)
        mixed_phase_list = pad_sequence(mixed_phase_list, batch_first=True)
        target_mag_list = pad_sequence(target_mag_list, batch_first=True)
        target_phase_list = pad_sequence(target_phase_list, batch_first=True)

        return dvec_list, target_mag_list, target_phase_list, \
            mixed_mag_list, mixed_phase_list, \
            target_stft_list, mixed_stft_list, \
            target_wav_list, mixed_wav_list

    def test_cuda_debug_collate_fn(batch):
        s1_dvecs = list()
        s1_targets = list()
        s2s = list()
        target_wav_list = list()
        mixed_wav_list = list()
        l1s = list()
        l2s = list()
        
        for s1_dvec, s1_target, s2, _, target_wav, mixed_wav, _, _, _, _, _, _, l1, l2 in batch:
            s1_dvecs.append(s1_dvec)
            s1_targets.append(s1_target)
            s2s.append(s2)
            l1s.append(l1)
            l2s.append(l2)
            target_wav_list.append(target_wav)
            mixed_wav_list.append(mixed_wav)

        return s1_dvecs, s1_targets, s2s, \
            target_wav_list, mixed_wav_list, \
            l1s, l2s

    # Genearate dataset
    if dataset_type == "generate":
        dataset = create_dataset(hp, dataset_detail, train=("train" in scheme), size=size)
    elif dataset_type == "gg":
        dataset = create_gg_dataset(hp, dataset_detail)


    if scheme == "train":
        return DataLoader(dataset=dataset,
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    elif scheme == "train_psedccrn":
        return DataLoader(dataset=dataset,
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_psedccrn_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    elif scheme == "train_raw":
        return DataLoader(dataset=dataset,
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_raw_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    elif scheme == "test":
        return DataLoader(dataset=dataset,
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)
    elif scheme == "test_cuda":
        return DataLoader(dataset=dataset,
                          batch_size=hp.train.batch_size,
                          shuffle=False,
                          num_workers=hp.train.num_workers,
                          collate_fn=test_cuda_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    elif scheme == "test_cuda_debug":
        return DataLoader(dataset=dataset,
                          batch_size=hp.train.batch_size,
                          shuffle=False,
                          num_workers=hp.train.num_workers,
                          collate_fn=test_cuda_debug_collate_fn,
                          drop_last=True,
                          sampler=None)
    elif scheme == "test_cuda_size_1":
        return DataLoader(dataset=dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          collate_fn=test_cuda_collate_fn,
                          pin_memory=True,
                          sampler=None)
    else:
        raise NotImplementedError