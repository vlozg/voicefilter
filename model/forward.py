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

    mask = model(torch.pow(mixed_stft.abs(), 0.3), dvec)
    output = mixed_stft*torch.pow(mask, 10/3)

    loss = criterion(mask, mixed_stft, target_stft)
    
    return output, mask, loss