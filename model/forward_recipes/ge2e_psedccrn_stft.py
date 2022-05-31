from contextlib import ExitStack

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


def train_forward(model, embedder, batch, criterion, device):
    dvec_mels = batch["dvec"]
    target_stft = batch["target_stft"]
    mixed_stft = batch["mixed_stft"]

    # Move to cuda
    if device == "cuda":
        target_stft = target_stft.cuda(non_blocking=True)
        mixed_stft = mixed_stft.cuda(non_blocking=True)
        dvec_mels = [mel.cuda(non_blocking=True) for mel in dvec_mels]

    # Get dvec
    dvec_list = list()
    for mel in dvec_mels:
        dvec = embedder(mel)
        dvec_list.append(dvec)
    dvec = torch.stack(dvec_list, dim=0)
    dvec = dvec.detach()

    # Reshape stft to match with DCCRN format
    # (B, T, F_2) -> (B, T, F, 2) -> (B, T, 2, F) -> (B, T, 2*F) -> (B, 2*F, T)
    b, t = mixed_stft.shape[:2]
    mixed_stft_ = torch.view_as_real(mixed_stft).transpose(-2, -1).reshape(b, t, -1).transpose(-1, -2)

    est_stft = model(mixed_stft_, dvec)
    
    # Reshape stft in DCCRN format to complex format
    # (B, 2*F, T) -> (B, T, 2*F)
    est_stft = torch.view_as_complex(est_stft.transpose(-1, -2).reshape(b, t, 2, -1).transpose(-2, -1).contiguous())

    est_mask = est_stft.abs()/mixed_stft.abs()

    loss = criterion(1, est_stft, target_stft)
    
    return est_stft, est_mask, loss

def inference_forward(model, embedder, batch, device):
    dvec_mels = batch["dvec"]
    mixed_stft = batch["mixed_stft"]

    # Move to cuda
    if device == "cuda":
        mixed_stft = mixed_stft.cuda(non_blocking=True)
        dvec_mels = [mel.cuda(non_blocking=True) for mel in dvec_mels]

    # Get dvec
    dvec_list = list()
    for mel in dvec_mels:
        dvec = embedder(mel)
        dvec_list.append(dvec)
    dvec = torch.stack(dvec_list, dim=0)
    dvec = dvec.detach()

    # Reshape stft to match with DCCRN format
    # (B, T, F_2) -> (B, T, F, 2) -> (B, T, 2, F) -> (B, T, 2*F) -> (B, 2*F, T)
    b, t = mixed_stft.shape[:2]
    mixed_stft = torch.view_as_real(mixed_stft).transpose(-2, -1).reshape(b, t, -1).transpose(-2,-1)
    est_stft = model(mixed_stft, dvec)

    # Reshape stft in DCCRN format to complex format
    # (B, 2*F, T) -> (B, T, 2*F)
    est_stft = torch.view_as_complex(est_stft.transpose(-1, -2).reshape(b, t, 2, -1).transpose(-2, -1).contiguous())
    
    est_mask = est_stft.abs()/mixed_stft.abs()
    
    return est_stft, est_mask