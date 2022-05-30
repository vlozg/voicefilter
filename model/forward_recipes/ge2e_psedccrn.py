from contextlib import ExitStack

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


def train_forward(model, embedder, batch, criterion, device):
    dvec_mels = batch["dvec"]
    mixed_wav = batch["mixed_wav"]
    target_stft = batch["target_stft"]
    mixed_stft = batch["mixed_stft"]

    # Move to cuda
    if device == "cuda":
        target_stft = target_stft.cuda(non_blocking=True)
        mixed_stft = mixed_stft.cuda(non_blocking=True)
        mixed_wav = mixed_wav.cuda(non_blocking=True)
        dvec_mels = [mel.cuda(non_blocking=True) for mel in dvec_mels]

    # Get dvec
    dvec_list = list()
    for mel in dvec_mels:
        dvec = embedder(mel)
        dvec_list.append(dvec)
    dvec = torch.stack(dvec_list, dim=0)
    dvec = dvec.detach()

    est_stft, est_wav = model(mixed_wav, dvec)
    est_stft = est_stft.transpose(1,2)
    b, t, _= est_stft.shape
    est_stft = torch.view_as_complex(est_stft.reshape(b, t, 2, -1).transpose(2,3).contiguous())
    est_stft = est_stft[:,:mixed_stft.shape[1], :]
    est_mask = est_stft.abs()/mixed_stft.abs()

    loss = criterion(1, est_stft, target_stft)
    
    return est_stft, est_mask, loss

def inference_forward(model, embedder, batch, device):
    dvec_mels = batch["dvec"]
    mixed_wav = batch["mixed_wav"]
    mixed_stft = batch["mixed_stft"]

    # Move to cuda
    if device == "cuda":
        mixed_wav = mixed_wav.cuda(non_blocking=True)
        dvec_mels = [mel.cuda(non_blocking=True) for mel in dvec_mels]

    # Get dvec
    dvec_list = list()
    for mel in dvec_mels:
        dvec = embedder(mel)
        dvec_list.append(dvec)
    dvec = torch.stack(dvec_list, dim=0)
    dvec = dvec.detach()

    est_stft, est_wav = model(mixed_wav, dvec)
    est_stft = est_stft[:,:mixed_stft.shape[1], :]
    est_mask = est_stft.abs()/mixed_stft.abs()
    
    return est_stft, est_mask