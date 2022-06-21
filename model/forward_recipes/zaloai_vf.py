from contextlib import ExitStack

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity



def __get_dvec(embedder, batch, device):
    if batch.get("dvec_tensor") is not None:
        dvec = batch["dvec_tensor"]
        if device == "cuda": dvec = dvec.cuda(non_blocking=True)
    else:
        dvec_wavs = batch["dvec_wav"]
        if device == "cuda": dvec_wavs = [torch.from_numpy(w).unsqueeze(0).cuda(non_blocking=True) for w in dvec_wavs]

        # Get dvec
        dvec_list = list()
        for w in dvec_wavs:
            dvec = embedder(w)
            dvec_list.append(dvec[0])
        dvec = torch.stack(dvec_list, dim=0)
        dvec = dvec.detach()
    
    return dvec



def __forward(model, embedder, batch, device):
    # Get stft feature, then move to cuda
    mixed_stft = batch["mixed_stft"]
    if device == "cuda": mixed_stft = mixed_stft.cuda(non_blocking=True)

    # Get dvec, forward pass to embedder if not precomputed
    dvec = __get_dvec(embedder, batch, device)

    est_mask = model(torch.pow(mixed_stft.abs(), 0.3), dvec)
    with torch.no_grad():
        est_stft = mixed_stft*torch.pow(est_mask, 10/3)

    return est_stft, est_mask



def train_forward(model, embedder, batch, criterion, device):
    # Get target for loss calculation
    target_stft = batch["target_stft"]
    mixed_stft = batch["mixed_stft"]
    if device == "cuda": 
        mixed_stft = mixed_stft.cuda(non_blocking=True)
        target_stft = target_stft.cuda(non_blocking=True)

    est_stft, est_mask = __forward(model, embedder, batch, device)

    loss = criterion(est_mask, mixed_stft, target_stft)
    
    return est_stft, est_mask, loss



def inference_forward(model, embedder, batch, device):
    return __forward(model, embedder, batch, device)