import torch
import torch.nn as nn
import numpy as np

from utils.audio import Audio

from model.get_model import get_vfmodel, get_embedder
from model.forward import train_forward
from loss.get_criterion import get_criterion

from torch_mir_eval import bss_eval_sources

from tqdm import tqdm

def tester(config, testloader, logger):

    # Init model, embedder, optim, criterion
    audio = Audio(config)
    device = "cuda" if config.use_cuda else "cpu"
    embedder = get_embedder(config, train=False, device=device)
    model, chkpt = get_vfmodel(config, train=False, device=device)
    criterion = get_criterion(config, reduction="none")

    if chkpt is None:
        logger.error("There is no pre-trained checkpoint to test, please re-check config file")
        return
    else:
        logger.info(f"Start testing checkpoint: {config.model.pretrained_chkpt}")
    
    with torch.no_grad():
        test_losses = []
        sdrs_before = []
        sdrs_after = []
        for batch in tqdm(testloader):
            est_stft, est_mask, loss = train_forward(model, embedder, batch, criterion, device)
            test_losses += loss.mean((1,2)).cpu().tolist()
            est_stft = est_stft.cpu().detach().numpy()
            for est_stft_, mixed_wav, target_wav in zip(est_stft, batch["mixed_wav"], batch["target_wav"]):
                est_wav = audio._istft(est_stft_.T, length=len(target_wav))
                est_wav = torch.from_numpy(est_wav).to(device=device).reshape(1, -1)
                target_wav = torch.from_numpy(target_wav).to(device=device).reshape(1, -1)
                mixed_wav = torch.from_numpy(mixed_wav).to(device=device).reshape(1, -1)
                
                #print(est_wav.shape, " ", target_wav.shape, " ", mixed_wav.shape)
                
                sdr,sir,sar,perm = bss_eval_sources(target_wav,mixed_wav,compute_permutation=True)
                sdrs_before.append(sdr)
                sdr,sir,sar,perm = bss_eval_sources(target_wav,est_wav,compute_permutation=True)
                sdrs_after.append(sdr)
                #sdrs_after.append(sdr(target_wav, est_wav))
        test_losses = np.array(test_losses)
        sdrs_before = np.array(sdrs_before)
        sdrs_after = np.array(sdrs_after)
                                
        logger.info(f"Complete testing")

    return {
        "loss": test_losses, 
        "sdr_before": sdrs_before, 
        "sdr_after": sdrs_after,
    }