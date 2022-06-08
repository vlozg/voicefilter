from contextlib import ExitStack

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


def train_forward(model, embedder, batch, criterion, device):
    dvec_wavs = batch["dvec_wav"]
    target_stft = batch["target_stft"]
    mixed_stft = batch["mixed_stft"]

    # Move to cuda
    if device == "cuda":
        target_stft = target_stft.cuda(non_blocking=True)
        mixed_stft = mixed_stft.cuda(non_blocking=True)
        dvec_wavs = [torch.from_numpy(w).unsqueeze(0).cuda(non_blocking=True) for w in dvec_wavs]

    # Get dvec
    dvec_list = list()
    for w in dvec_wavs:
        dvec = embedder(w)
        dvec_list.append(dvec[0])
    dvec = torch.stack(dvec_list, dim=0)
    dvec = dvec.detach()

    mask = model(torch.pow(mixed_stft.abs(), 0.3), dvec)
    output = mixed_stft*torch.pow(mask, 10/3)

    loss = criterion(mask, mixed_stft, target_stft)
    
    return output, mask, loss

def inference_forward(model, embedder, batch, device):
    dvec_wavs = batch["dvec_wav"]
    mixed_stft = batch["mixed_stft"]

    # Move to cuda
    if device == "cuda":
        mixed_stft = mixed_stft.cuda(non_blocking=True)
        dvec_wavs = [torch.from_numpy(w).unsqueeze(0).cuda(non_blocking=True) for w in dvec_wavs]

    # Get dvec
    dvec_list = list()
    for w in dvec_wavs:
        dvec = embedder(w)
        dvec_list.append(dvec[0])
    dvec = torch.stack(dvec_list, dim=0)
    dvec = dvec.detach()

    mask = model(torch.pow(mixed_stft.abs(), 0.3), dvec)
    output = mixed_stft*torch.pow(mask, 10/3)
    
    return output, mask