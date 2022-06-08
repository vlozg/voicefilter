from contextlib import ExitStack

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

def __forward(model, embedder, batch, device):
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

    est_mask = model(torch.pow(mixed_stft.abs(), 0.3), dvec)
    est_stft = mixed_stft*torch.pow(est_mask, 10/3)

    return est_stft, est_mask



def train_forward(model, embedder, batch, criterion, device):
    # Get target for loss calculation
    target_stft = batch["target_stft"]
    if device == "cuda":
        target_stft = target_stft.cuda(non_blocking=True)

    est_stft, est_mask = __forward(model, embedder, batch, device)

    loss = criterion(1, est_stft, target_stft)
    
    return est_stft, est_mask, loss



def inference_forward(model, embedder, batch, device):
    return __forward(model, embedder, batch, device)