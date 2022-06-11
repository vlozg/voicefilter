from contextlib import ExitStack

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

def __forward(model, embedder, batch, device):
    # Get stft feature, then move to cuda
    mixed_stft = batch["mixed_stft"]
    if device == "cuda": mixed_stft = mixed_stft.cuda(non_blocking=True)

    # Get dvec, forward pass to embedder if not precomputed
    if batch.get("dvec_tensor"):
        dvec = batch["dvec_tensor"]
        if device == "cuda": dvec = dvec.cuda(non_blocking=True)
    else:
        dvec_mels = batch["dvec"]
        if device == "cuda": dvec_mels = [mel.cuda(non_blocking=True) for mel in dvec_mels]

        # Get dvec
        dvec_list = list()
        for mel in dvec_mels:
            dvec = embedder(mel)
            dvec_list.append(dvec)
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