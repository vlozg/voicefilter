import os
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)

import librosa
import numpy as np
import torch
from torch_mir_eval import bss_eval_sources
from torchmetrics.functional import \
    scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio import \
    scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio.pesq import \
    perceptual_evaluation_speech_quality as pesq
from torchmetrics.functional.audio.stoi import \
    short_time_objective_intelligibility as stoi
from tqdm import tqdm
from utils.dnsmos import DNSMOS


def gather_future(futures, default_value=None):
    output_list = [default_value]*len(futures)
    for future in tqdm(as_completed(futures), total=len(futures)):
        idx = futures[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (idx, exc))
        else:
            output_list[idx] = data
    return output_list


def _metric_compute_1_sample(est_path, target, target_norm=1, sr=16000, skip_features=[], target_type="path"):
    est_wav, _ = librosa.load(est_path, sr=sr)
    if target_type == "path":
        target_wav, _ = librosa.load(target, sr=sr)

        le, lt = est_wav.shape[0], target_wav.shape[0]

        target_wav = np.pad(target_wav[:le], (0, max(0, le-lt)), constant_values=0)
        target_wav = torch.from_numpy(target_wav).reshape(1, -1)
    elif target_type == "wav":
        if type(target) is np.ndarray:
            target_wav = torch.from_numpy(target).reshape(1, -1)
        else:
            target_wav = target.reshape(1, -1)
        
    est_wav = torch.from_numpy(est_wav).reshape(1, -1)    
    metrics = {}

    if "stoi" not in skip_features:
        try:
            metrics.update({"stoi": stoi(est_wav, target_wav, sr, False),})
        except Exception as exc:
            print("Error while computing stoi: ", exc)
            metrics.update({"stoi": torch.tensor(float('nan'))})

    # if "estoi" not in skip_features:
    #     try:
    #         metrics.update({"estoi": stoi(est_wav, target_wav, sr, True),})
    #     except Exception as exc:
    #         print("Error while computing estoi: ", exc)
    #         metrics.update({"estoi": torch.tensor(float('nan'))})
        
    if "pesq" not in skip_features:
        try:
            metrics.update({"pesq": pesq(est_wav, target_wav, sr, "wb"),})
        except Exception as exc:
            print("Error while computing pesq: ", exc)
            metrics.update({"pesq": torch.tensor(float('nan'))})
        
    # if "si_snr" not in skip_features:
    #     try:
    #         metrics.update({"si_snr": si_snr(est_wav, target_wav)})
    #     except Exception as exc:
    #         print("Error while computing si-snr: ", exc)
    #         metrics.update({"si_snr": torch.tensor(float('nan'))})

    if "si_sdr" not in skip_features:
        try:
            metrics.update({"si_sdr": si_snr(est_wav, target_wav)})
        except Exception as exc:
            print("Error while computing si-sdr: ", exc)
            metrics.update({"si_sdr": torch.tensor(float('nan'))})

    return metrics


def metric_compute(config, testloader, logger, out_dir, skip_features=[]):
    # Init model, embedder, optim, criterion
    device = "cuda" if config.use_cuda else "cpu"
    dnsmos = DNSMOS("utils", True, device == "cuda")

    results = {}

    with ProcessPoolExecutor() as pexecutor, ThreadPoolExecutor() as texecutor:
        future_metrics = {}
        future_dnsmos = {}
        future_sdrs = {}

        for batch in tqdm(testloader):
            if batch.get("target_wav") is None:
                batch["target_wav"] = [None] * len(batch["index"])
            if batch.get("norm") is None:
                batch["norm"] = [1] * len(batch["index"])

            for idx, target_wav, target_path, norm in zip(batch["index"], batch["target_wav"], batch["target_path"], batch["norm"]):
                est_path = os.path.join(out_dir, f"{idx}.wav")

                est_wav, _ = librosa.load(est_path, sr=config.audio.sample_rate)
                target_wav, _ = librosa.load(target_path, sr=config.audio.sample_rate)

                le, lt = est_wav.shape[0], target_wav.shape[0]

                target_wav = np.pad(target_wav[:le], (0, max(0, le-lt)), constant_values=0) / norm
                target_wav = torch.from_numpy(target_wav).reshape(1, -1)

                ###
                # Calculate STOI, ESTOI,... in future manner with multiprocessing
                # Calculate DNSMOS score, SDR using GPU,... in future manner with multithreading
                ###

                if "OVRL_raw" not in skip_features or "OVRL" not in skip_features:
                    future_dnsmos[texecutor.submit(dnsmos, est_wav)] = idx

                if target_wav is not None:
                    est_wav = torch.from_numpy(est_wav).reshape(1, -1)

                    future_metrics[pexecutor.submit(_metric_compute_1_sample, est_path, target_wav, norm,  config.audio.sample_rate, skip_features, "wav")] = idx
                    
                    if "sdr" not in skip_features:
                        est_wav = est_wav.to(device=device)
                        target_wav = target_wav.to(device=device)

                        with torch.no_grad():
                            future_sdrs[texecutor.submit(bss_eval_sources, target_wav, est_wav)] = idx
                        

        ###
        # Get result from all task
        ###

        # List of dict to Dict of list
        if len(future_dnsmos) > 0:
            logger.info("Start gathering DNSMOS")
            dnsmos_scores = gather_future(future_dnsmos, float('nan'))
            dnsmos_scores = {k: np.array([dic[k] for dic in dnsmos_scores]) for k in dnsmos_scores[0].keys()}
            results.update({**dnsmos_scores})

        logger.info("Start gathering STOI, ESTOI, PESQ, SI-SNR")
        metrics = gather_future(future_metrics)
        metrics = {k: torch.cat([dic[k] for dic in metrics]).numpy() for k in metrics[0].keys()}
        results.update({**metrics})

        if len(future_sdrs) > 0:
            logger.info("Start gathering SDR")
            sdrs_after = gather_future(future_sdrs, [torch.tensor(float('nan'))])
            sdrs_after = torch.stack([item[0] for item in sdrs_after]).numpy()
            results.update({"sdr": sdrs_after})


        logger.info(f"Complete testing")

    return results
