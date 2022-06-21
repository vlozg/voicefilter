import torch
import torch.nn as nn
import numpy as np

from utils.audio import Audio

from model.get_model import get_vfmodel, get_embedder, get_forward
from loss.get_criterion import get_criterion

from torch_mir_eval import bss_eval_sources

from tqdm import tqdm

def tester(config, testloader, logger):

    # Init model, embedder, optim, criterion
    audio = Audio(config)
    device = "cuda" if config.use_cuda else "cpu"
    embedder = get_embedder(config, train=False, device=device)
    model, chkpt = get_vfmodel(config, train=False, device=device)
    train_forward, _ = get_forward(config)
    criterion = get_criterion(config, reduction="none")

    if chkpt is None:
        logger.error("There is no pre-trained checkpoint to test, please re-check config file")
        return
    else:
        logger.info(f"Start testing checkpoint: {config.model.pretrained_chkpt}")
    
    with torch.no_grad():
        test_losses = []
        sdrs_after = []
        for batch in tqdm(testloader):
            est_stft, est_mask, loss = train_forward(model, embedder, batch, criterion, device)
            test_losses += loss.mean((1,2)).cpu().tolist()
            est_stft = est_stft.cpu().detach().numpy()
            for est_stft_, mixed_wav, target_wav in zip(est_stft, batch["mixed_wav"], batch["target_wav"]):
                est_wav = audio._istft(est_stft_.T, length=len(target_wav))

                # Calculate DNSMOS score, STOI, ESTOI,... in future manner
                dnsmos_scores.append(dnsmos(est_wav))

                est_wav = torch.from_numpy(est_wav)
                future_stois[executor.submit(stoi, est_wav, target_wav, 16000, False)] = idx
                future_estois[executor.submit(stoi, est_wav, target_wav, 16000, True)] = idx
                future_pesqs[executor.submit(pesq, est_wav, target_wav, 16000, "wb")] = idx
                future_si_snrs[executor.submit(si_snr, est_wav, target_wav)] = idx


                est_wav = est_wav.to(device=device).reshape(1, -1)
                
                target_wav = target_wav.to(device=device).reshape(1, -1)
                mixed_wav = mixed_wav.to(device=device).reshape(1, -1)
                
                if target_wav.sum() != 0 and mixed_wav.sum() != 0 and est_wav.sum() != 0:
                    sdr,sir,sar,perm = bss_eval_sources(target_wav,est_wav,compute_permutation=False)
                    sdr = sdr.item()
                else: 
                    sdr = None

                sdrs_after.append(sdr)

        test_losses = np.array(test_losses)
        sdrs_after = np.array(sdrs_after)

        ###
        # Get result from all task
        ###

        # List of dict to Dict of list
        dnsmos_scores = {k: np.array([dic[k] for dic in dnsmos_scores]) for k in dnsmos_scores[0].keys()}

        logger.info("Start gathering STOI")
        stois = torch.stack(gather_future(future_stois)).numpy()

        logger.info("Start gathering ESTOI")
        estois = torch.stack(gather_future(future_estois)).numpy()

        logger.info("Start gathering PESQ")
        pesqs = torch.stack(gather_future(future_pesqs)).numpy()

        logger.info("Start gathering SI-SNR")
        si_snrs = torch.stack(gather_future(future_si_snrs)).numpy()
                                
        logger.info(f"Complete testing")

    results = {
        "loss": test_losses, 
        "sdr": sdrs_after,
        "stoi": stois,
        "estoi": estois,
        "pesq": pesqs,
        "si-snr": si_snrs,
        **dnsmos_scores
    }

    return results