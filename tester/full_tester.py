import concurrent.futures
import os

import numpy as np
import soundfile
import torch
import torch.nn as nn
from datasets.dataloader import create_dataloader
from loss.get_criterion import get_criterion
from model.get_model import get_embedder, get_forward, get_vfmodel
from torch_mir_eval import bss_eval_sources
from torchmetrics.functional import \
    scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio.pesq import \
    perceptual_evaluation_speech_quality as pesq
from torchmetrics.functional.audio.stoi import \
    short_time_objective_intelligibility as stoi
from tqdm import tqdm
from utils.audio import Audio
from utils.dnsmos import DNSMOS


def gather_future(futures):
    output_list = [None]*len(futures)
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        idx = futures[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (idx, exc))
        else:
            output_list[idx] = data
    return output_list


def tester(config, logger, out_dir=None):

    logger.info("Start making test set")
    testloader = create_dataloader(config, scheme="test")

    config = config.experiment

    # Init model, embedder, optim, criterion
    audio = Audio(config)
    device = "cuda" if config.use_cuda else "cpu"
    embedder = get_embedder(config, train=False, device=device)
    model, chkpt = get_vfmodel(config, train=False, device=device)
    train_forward, _ = get_forward(config)
    criterion = get_criterion(config, reduction="none")
    dnsmos = DNSMOS("utils", True)

    if chkpt is None:
        logger.error("There is no pre-trained checkpoint to test, please re-check config file")
        return
    else:
        logger.info(f"Start testing checkpoint: {config.model.pretrained_chkpt}")
    
    test_losses = []
    sdrs_after = []
    dnsmos_scores = []
    stois = []
    estois = []
    pesqs = []
    si_snrs = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_stois = {}
        future_estois = {}
        future_pesqs = {}
        future_si_snrs = {}

        for idx, batch in enumerate(tqdm(testloader)):
            with torch.no_grad():
                est_stft, est_mask, loss = train_forward(model, embedder, batch, criterion, device)
            test_losses += loss.mean((1,2)).cpu().tolist()
            est_stft = est_stft.cpu().detach().numpy()
            for est_stft_, mixed_wav, target_wav in zip(est_stft, batch["mixed_wav"], batch["target_wav"]):
                est_wav = audio._istft(est_stft_.T, length=len(target_wav))
                if out_dir is not None:
                    out_file = os.path.join(out_dir, f"{idx}.wav")
                    os.makedirs(out_dir if out_dir != '' else ".", exist_ok=True)
                    soundfile.write(out_file, est_wav, config.audio.sample_rate)

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
                    sdr = float('nan')

                sdrs_after.append(sdr)

        test_losses = np.array(test_losses)
        sdrs_after = np.array(sdrs_after)

        ###
        # Get result from all task
        ###

        # List of dict to Dict of list
        dnsmos_scores = {k: np.array([dic[k] for dic in dnsmos_scores]) for k in dnsmos_scores[0].keys()}

        logger.info("Start gathering STOI")
        stois = torch.stack(gather_future(future_stois), torch.tensor(float('nan'))).numpy()

        logger.info("Start gathering ESTOI")
        estois = torch.stack(gather_future(future_estois), torch.tensor(float('nan'))).numpy()

        logger.info("Start gathering PESQ")
        pesqs = torch.stack(gather_future(future_pesqs), torch.tensor(float('nan'))).numpy()

        logger.info("Start gathering SI-SNR")
        si_snrs = torch.stack(gather_future(future_si_snrs), torch.tensor(float('nan'))).numpy()
                                
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
