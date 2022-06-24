import argparse
from genericpath import exists
import io
import json
import logging
import os
import time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from pathlib import Path

import IPython
import numpy as np
import torch
from google.cloud import speech
from torch_mir_eval import bss_eval_sources
from torchmetrics.functional import \
    scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional import word_error_rate
from torchmetrics.functional.audio import \
    scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_noise_ratio
from torchmetrics.functional.audio.pesq import \
    perceptual_evaluation_speech_quality as pesq
from torchmetrics.functional.audio.stoi import \
    short_time_objective_intelligibility as stoi
from tqdm import tqdm

from datasets.get_dataset import get_dataset
from loss.get_criterion import get_criterion
from utils.audio import Audio
from utils.dnsmos import DNSMOS
from utils.hparams import HParam
import pickle


def transcribe_file(speech_file, label=None, out_file=None, lang="en-US", sr=16000, type="path", norm=False):
    """Transcribe the given audio file."""

    client = speech.SpeechClient()

    if type=="path":
        with io.open(speech_file, "rb") as audio_file:
            content = audio_file.read()
    elif type=="wav":
        content = IPython.display.Audio(speech_file, rate=sr, normalize=norm).data

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
        language_code=lang,
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    asr = {"transcript": [], "confidence": []}
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        asr["confidence"].append(result.alternatives[0].confidence)
        asr["transcript"].append(result.alternatives[0].transcript)

    asr["transcript"] = " ".join(asr["transcript"])
    if len(asr["confidence"]) > 0:
        asr["confidence"] = np.mean(asr["confidence"])
    else:
        asr["confidence"] = float("nan")

    if label is not None:
        wer = word_error_rate(asr["transcript"].upper(), label).item()
    else:
        wer = float("nan")

    asr["wer"] = wer

    if out_file is not None:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(asr, indent = 4, ensure_ascii=False))

    return asr, wer



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
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/default.yaml',
                        help="yaml files for model and env configuration. Default: config/default.yaml")
    parser.add_argument('-d', '--data_config', type=str, required=True,
                        help="yaml files for data used to test and testing configuration")
    parser.add_argument('--use_cuda', type=str, default=None,
                        help="use cuda for testing, overwrite test config. Default: follow test config file")
    parser.add_argument('--skip_exist', type=str, default="False",
                        help="skip testing if result exist. Default: False")
    parser.add_argument('--asr_language', type=str, default=None,
                        help="skip testing asr if parameter is None . Default: None")
    parser.add_argument('--output_audio', type=str, default=None,
                        help="Output infered audio. Default: do not output infered audio")    
    parser.add_argument('--skip_features', type=str, nargs="+", default=None,
                        help="Which type of test to skip, useful for increamental test. Default: do all test")                        
    args = parser.parse_args()

    ###
    # Merge config, only do merge for known key
    ###
    config = HParam(args.config)
    data_config = HParam(args.data_config)

    if args.use_cuda is None:
        config.experiment.use_cuda = data_config.experiment.use_cuda
    else:
        config.experiment.use_cuda = args.use_cuda.upper() in ("Y", "YES", "T", "TRUE")
    config.experiment.dataset = data_config.experiment.dataset
    config.experiment.train.num_workers = 8

    exp = config["experiment"]
    env = config["env"]
    test_exp_name = Path(args.data_config).stem # Ex: ../../libri_eval.yaml -> libri_eval
    exp.name = "dataset_info"

    ###
    # Init logger and create dataset
    ###
    log_dir = os.path.join(env.base_dir, env.log.log_dir, "test", test_exp_name, exp.name)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d-info.log' % (exp.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    ###
    # Load test record and write test config
    ###
    test_record_path = os.path.join(env.base_dir, "test_results", test_exp_name + ".json")
    if os.path.exists(test_record_path):
        with open(test_record_path, "r") as f:
            test_record = json.load(f)
    else:
        test_record = dict()


    if test_record.get(exp.name):
        # Test result on same model exist
        logger.info("Test result exist")
        if not(args.skip_exist.upper() in ("Y", "YES", "T", "TRUE") and test_record.get(exp.name)):
            logger.info(f"Test result for experiment name {exp.name} exits. Change experiment name and re-conduct the test.")
            i = 0
            while test_record.get(f"{exp.name}_{i}"): i+=1
            exp.name = f"{exp.name}_{i}"
            logger.info(f"Result will be stored within {exp.name} instead")
        
    if args.output_audio == "auto":
        output_dir = os.path.join(f"test_results/{test_exp_name}/{exp.name}")
    else:
        output_dir = args.output_audio
    os.makedirs(output_dir, exist_ok=True)

    if test_record.get("data") or test_record.get("data_config"):
        if test_record["data"] != config.experiment.dataset.test.file_path:
            logger.info("Different dataset file path!")
            logger.info(f"Data path in config: {config.experiment.dataset.test.file_path}")
            logger.info(f"Data path in record: {test_record['data']}")
            raise Exception("Different dataset file path")
    else:
        test_record["data"] = config.experiment.dataset.test.file_path
        test_record["config"] = data_config


    lang=args.asr_language
    if lang != None:
        lang = lang.lower()
        if 'en' in lang: lang='en-US'
        elif 'v' in lang: lang='vi-VN'
        else: lang=None


    skip_features = [] if args.skip_features is None else args.skip_features
    if args.skip_exist.upper() in ("Y", "YES", "T", "TRUE") and test_record.get(exp.name):

        if test_record[exp.name].get("asr_lang") is not None:
            skip_features.append("asr")
            logger.info(f"ASR inference for {exp.name} exits. Opt out from conduct computing ASR.")

        if test_record[exp.name].get("metrics"):
            for k in test_record[exp.name]["metrics"].keys():
                skip_features.append(k)
                logger.info(f"{k} metrics for {exp.name} exits. Opt out from conduct testing them.")

    logger.info("Start making test set")
    testset = get_dataset(config, scheme="test")

    ###
    # Conduct testing
    ###
    config = exp
    audio = Audio(config)
    device = "cuda" if config.use_cuda else "cpu"
    criterion = get_criterion(config, reduction="none")
    dnsmos = DNSMOS(os.path.join(env.base_dir, "utils"), True, config.use_cuda)

    if test_record.get(exp.name) is not None:
        test_result = test_record[exp.name]
    else:
        test_result = {}

    test_losses = []
    sdrs = []
    snrs = []
    w1_lens = []
    w2_lens = []
    dvec_lens = []
    w1_silent_ratio_10db = []
    w2_silent_ratio_10db = []
    d_silent_ratio_10db = []

    mixed_dnsmos_scores = []
    target_dnsmos_scores = []
    stois = []
    pesqs = []
    si_sdrs = []
    wers = {}
    clean_wers = {}
    os.makedirs(os.path.join(output_dir,'metrics'), exist_ok=True)

    with ProcessPoolExecutor() as pexecutor, ThreadPoolExecutor() as texecutor:
        future_stois = {}
        # future_estois = {}
        future_pesqs = {}
        # future_si_snrs = {}
        future_mixed_dnsmos = {}
        future_target_dnsmos = {}
        future_sdrs = {}
        future_si_sdrs = {}
        future_snrs = {}
        future_asrs = {}
        future_clean_asrs = {}

        for batch in tqdm(testset):
            idx = batch["index"]

            # Get audio length info
            w1_lens.append(batch["target_len"] / config.audio.sample_rate)
            try:
                w2_lens.append(batch["interf_len"] / config.audio.sample_rate)
            except:
                w2_lens.append(None)
            dvec_lens.append(len(batch["dvec_wav"]) / config.audio.sample_rate)

            # Try to compute silent ratio
            try:    
                w1_silent_ratio_10db.append(audio.silent_len(batch["target_wav"]) / len(batch["target_wav"]))
            except:
                w1_silent_ratio_10db.append(None)
                # logger.info(batch)
            try:
                w2_silent_ratio_10db.append(audio.silent_len(batch["interf_wav"]) / len(batch["interf_wav"]))
            except:
                w2_silent_ratio_10db.append(None)
                # logger.info(batch)
            try:
                d_silent_ratio_10db.append(audio.silent_len(batch["dvec_wav"]) / len(batch["dvec_wav"]))
            except:
                d_silent_ratio_10db.append(None)
                # logger.info(batch)

            out_file = os.path.join(output_dir, f"{idx}_asr.json")
            out_clean_file = os.path.join(output_dir, f"{idx}_clean_asr.json")
            
            target_text = batch["target_text"]
            
            if os.path.isfile(out_file):
                with open(out_file, "r") as f:
                    asr = json.load(f)
                    wers[idx] = asr["wer"]
            else:
                future_asrs[texecutor.submit(transcribe_file, batch["mixed_wav"], target_text, out_file, lang, config.audio.sample_rate, "wav")] = idx
            
            if os.path.isfile(out_clean_file):
                with open(out_clean_file, "r") as f:
                    asr = json.load(f)
                    clean_wers[idx] = asr["wer"]
            else:
                future_clean_asrs[texecutor.submit(transcribe_file, batch["target_wav"], target_text, out_clean_file, lang, config.audio.sample_rate, "wav", True)] = idx

            target_stft = batch["target_stft"]
            mixed_stft = batch["mixed_stft"]

            # Move to cuda
            if device == "cuda":
                target_stft = target_stft.cuda(non_blocking=True)
                mixed_stft = mixed_stft.cuda(non_blocking=True)

            # Calculate loss
            with torch.no_grad():
                loss = criterion(1, mixed_stft.unsqueeze(0), target_stft.unsqueeze(0))
                test_losses += loss.mean((1,2)).cpu().tolist()

            # Calculate STOI, ESTOI,... in future manner with multiprocessing
            target_wav = batch["target_wav"]
            mixed_wav = batch["mixed_wav"]

            if "stoi" not in skip_features:
                future_stois[pexecutor.submit(stoi, mixed_wav, target_wav, 16000, False)] = idx
            
            if "pesq" not in skip_features:
                future_pesqs[pexecutor.submit(pesq, mixed_wav, target_wav, 16000, "wb")] = idx

            # Calculate DNSMOS score, SDR using GPU,... in future manner with multithreading
            if "OVRL" not in skip_features:
                future_mixed_dnsmos[texecutor.submit(dnsmos, mixed_wav.numpy())] = idx
            
            if "target_OVRL" not in skip_features:
                future_target_dnsmos[texecutor.submit(dnsmos, target_wav.numpy())] = idx

            target_wav = target_wav.to(device=device).unsqueeze(0)
            mixed_wav = mixed_wav.to(device=device).unsqueeze(0)
            
            if "snr" not in skip_features:
                future_snrs[texecutor.submit(signal_noise_ratio, mixed_wav, target_wav)]= idx
            
            if "si_sdr" not in skip_features:
                future_snrs[texecutor.submit(si_sdr, mixed_wav, target_wav)]= idx

            with torch.no_grad():
                future_sdrs[texecutor.submit(bss_eval_sources, target_wav,mixed_wav)] = idx

        ###
        # Get result from all task
        ###

        if len(future_asrs) > 0:
            logger.info("Waiting for ASR computing and writing...")
            for f in tqdm(as_completed(future_asrs), total=len(future_asrs)):
                idx = future_asrs[f]
                try:
                    wers[idx] = f.result()[1]
                except Exception as exc:
                    print('%r generated an exception: %s' % (idx, exc))
                    wers[idx] = None

            if wers is not {}:
                wers = [wers[k] for k in sorted(wers.keys())]
                wers = np.array(wers).tolist()
            else:
                wers = []
            test_result.update({"wer": wers})

        # List of dict to Dict of list
        if len(future_mixed_dnsmos) > 0:
            logger.info("Start gathering mixed DNSMOS")
            mixed_dnsmos_scores = gather_future(future_mixed_dnsmos, float('nan'))
            torch.save(mixed_dnsmos_scores, os.path.join(output_dir,'metrics', "dnsmos_bk.pt"))

            mixed_dnsmos_scores = {k: np.array([dic[k] for dic in mixed_dnsmos_scores]).tolist() for k in mixed_dnsmos_scores[0].keys()}

            test_result.update({**mixed_dnsmos_scores})

        if len(future_target_dnsmos) > 0:
            logger.info("Start gathering target DNSMOS")
            target_dnsmos_scores = gather_future(future_target_dnsmos, float('nan'))
            torch.save(target_dnsmos_scores, os.path.join(output_dir,'metrics', "target_dnsmos_bk.pt"))

            target_dnsmos_scores = {"target_"+k: np.array([dic[k] for dic in target_dnsmos_scores]).tolist() for k in target_dnsmos_scores[0].keys()}
            test_result.update({**target_dnsmos_scores})

        if len(future_sdrs) > 0:
            logger.info("Start gathering SDR")
            sdrs_after = gather_future(future_sdrs, [torch.tensor(float('nan'))])
            torch.save(sdrs_after, os.path.join(output_dir,'metrics', "sdr_bk.pt"))
            sdrs_after = torch.stack([item[0].cpu() for item in sdrs_after]).numpy().tolist()
            test_result.update({"sdr": sdrs_after})

        if len(future_snrs) > 0:
            logger.info("Start gathering SNR")
            snrs_after = gather_future(future_snrs, [torch.tensor(float('nan'))])
            torch.save(snrs_after, os.path.join(output_dir,'metrics', "snr_bk.pt"))
            snrs_after = torch.stack([item[0].cpu() for item in snrs_after]).numpy().tolist()
            try:
                test_result.update({"snr": snrs_after})
            except:
                print(snrs_after)

        if len(future_si_sdrs) > 0:
            logger.info("Start gathering SI-SDR")
            si_sdrs_after = gather_future(future_si_sdrs, [torch.tensor(float('nan'))])
            torch.save(si_sdrs_after, os.path.join(output_dir,'metrics', "si_sdr_bk.pt"))
            si_sdrs_after = torch.stack([item[0].cpu() for item in si_sdrs_after]).numpy().tolist()
            try:
                test_result.update({"si_sdr": si_sdrs_after})
            except:
                print(si_sdrs_after)

        if len(future_stois) > 0:
            logger.info("Start gathering STOI")
            stois = torch.stack(gather_future(future_stois)).numpy().tolist()
            test_result.update({"stoi": stois})

        if len(future_pesqs) > 0:
            logger.info("Start gathering PESQ")
            pesqs = torch.stack(gather_future(future_pesqs)).numpy().tolist()
            test_result.update({"pesq": pesqs})

        if len(future_clean_asrs) > 0:
            logger.info("Waiting for ASR computing and writing...")
            for f in tqdm(as_completed(future_clean_asrs), total=len(future_clean_asrs)):
                idx = future_clean_asrs[f]
                try:
                    clean_wers[idx] = f.result()[1]
                except Exception as exc:
                    print('%r generated an exception: %s' % (idx, exc))
                    clean_wers[idx] = None

            if clean_wers is not {}:
                clean_wers = [clean_wers[k] for k in sorted(clean_wers.keys())]
                clean_wers = np.array(clean_wers).tolist()
            else:
                clean_wers = []

            test_result.update({"clean_wer": clean_wers})

        test_losses = np.array(test_losses).tolist()

    logger.info(f"Complete testing")

    test_result.update({
        "loss": test_losses,
        "target_length": w1_lens,
        "interference_length": w2_lens,
        "reference_length": dvec_lens,
        "target_silent_ratio": w1_silent_ratio_10db,
        "interference_silent_ratio": w2_silent_ratio_10db,
        "snr": snrs
    })


    ###
    # Append new test result to file, then rewrite
    ###
    logger.info(f"Appending test result to {test_record_path}")

    if test_record.get(exp.name) is not None:
        if test_record[exp.name].get("metrics") is not None:
            metric_dirs=test_record[exp.name].get("metrics")
    else:
        metric_dirs={}

    os.makedirs(os.path.join(output_dir,'metrics'), exist_ok=True)
    for metric in test_result.keys():
        metrics_path = os.path.join(output_dir,'metrics', metric + '.json')
        json_object = json.dumps(test_result[metric], indent = 4)
        with open(metrics_path, "w") as f:
            f.write(json_object)

        metric_dirs[metric]=metrics_path

    test_result = {
        "config": "raw",
        "asr_lang":args.asr_language,
        "output_dir": output_dir,
        "metrics": metric_dirs
    }

    test_record[exp.name] = test_result

    json_object = json.dumps(test_record, indent = 4)
    with open(test_record_path, "w") as f:
        f.write(json_object)
