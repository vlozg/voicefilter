import os
import time
import logging
import argparse
from pathlib import Path
import json

from utils.hparams import HParam
from datasets.dataloader import get_dataset

import torch
import numpy as np

from utils.audio import Audio

from loss.get_criterion import get_criterion

from torch_mir_eval import bss_eval_sources

from tqdm import tqdm

from torchmetrics.functional.audio import signal_noise_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/default.yaml',
                        help="yaml files for model and env configuration. Default: config/default.yaml")
    parser.add_argument('-d', '--data_config', type=str, required=True,
                        help="yaml files for data used to test and testing configuration")
    parser.add_argument('--use_cuda', type=bool, default=None,
                        help="use cuda for testing, overwrite test config. Default: follow test config file")
    parser.add_argument('--skip_exist', type=bool, default=False,
                        help="skip testing if result exist. Default: False")
    args = parser.parse_args()


    ###
    # Merge config, only do merge for known key
    ###
    config = HParam(args.config)
    data_config = HParam(args.data_config)

    if args.use_cuda is None:
        config.experiment.use_cuda = data_config.experiment.use_cuda
    else:
        config.experiment.use_cuda = args.use_cuda
    config.experiment.dataset = data_config.experiment.dataset
    
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

    logger.info("Start making test set")
    testset = get_dataset(config, scheme="test")
    # testloader = create_dataloader(config, scheme="test")


    ###
    # Load test record and write test config
    ###
    test_record_path = os.path.join(env.base_dir, "test_results", test_exp_name + ".json")
    if os.path.exists(test_record_path):
        with open(test_record_path, "r") as f:
            test_record = json.load(f)
    else:
        test_record = dict()

    if test_record.get("data") or test_record.get("data_config"):
        if test_record["data"] != config.experiment.dataset.test.file_path:
            logger.info("Different dataset file path!")
            logger.info(f"Data path in config: {config.experiment.dataset.test.file_path}")
            logger.info(f"Data path in record: {test_record['data']}")
            raise Exception("Different dataset file path")
    else:
        test_record["data"] = config.experiment.dataset.test.file_path
        test_record["config"] = data_config

    if args.skip_exist and test_record.get(exp.name):
        # Test result exist
        logger.info(f"Test result for {exp.name} exits. Abort this test.")
        exit()

    ###
    # Conduct testing
    ###
    config = exp
    audio = Audio(config)
    device = "cuda" if config.use_cuda else "cpu"
    criterion = get_criterion(config, reduction="none")
    
    with torch.no_grad():
        test_losses = []
        sdrs = []
        snrs = []
        w1_lens = []
        w2_lens = []
        dvec_lens = []
        w1_silent_ratio_10db = []
        w2_silent_ratio_10db = []
        d_silent_ratio_10db = []

        for batch in tqdm(testset):
            w1_lens.append(batch["target_len"])
            w2_lens.append(batch["interf_len"])
            dvec_lens.append(len(batch["dvec_wav"]))

            try:    
                w1_silent_ratio_10db.append(audio.silent_len(batch["target_wav"]) / len(batch["target_wav"]))
            except:
                w1_silent_ratio_10db.append(None)
                logger.info(batch)
            try:
                w2_silent_ratio_10db.append(audio.silent_len(batch["interf_wav"]) / len(batch["interf_wav"]))
            except:
                w2_silent_ratio_10db.append(None)
                logger.info(batch)
            try:
                d_silent_ratio_10db.append(audio.silent_len(batch["dvec_wav"]) / len(batch["dvec_wav"]))
            except:
                d_silent_ratio_10db.append(None)
                logger.info(batch)

            target_stft = batch["target_stft"]
            mixed_stft = batch["mixed_stft"]

            # Move to cuda
            if device == "cuda":
                target_stft = target_stft.cuda(non_blocking=True)
                mixed_stft = mixed_stft.cuda(non_blocking=True)

            loss = criterion(1, mixed_stft.unsqueeze(0), target_stft.unsqueeze(0))
            test_losses += loss.mean((1,2)).cpu().tolist()

            target_wav = batch["target_wav"].to(device=device).unsqueeze(0)
            mixed_wav = batch["mixed_wav"].to(device=device).unsqueeze(0)
            snr = signal_noise_ratio(target_wav, mixed_wav).cpu().item()
            if target_wav.sum() != 0 and mixed_wav.sum() != 0:
                sdr,sir,sar,perm = bss_eval_sources(target_wav,mixed_wav,compute_permutation=False)
                sdr = sdr.item()
            else: 
                sdr = None
            sdrs.append(sdr)
            snrs.append(snr)

        test_losses = np.array(test_losses).tolist()
        sdrs = np.array(sdrs).tolist()
        snrs = np.array(snrs).tolist()
                                
        logger.info(f"Complete testing")

    test_result = {
        "metrics": {
            "loss": test_losses,
            "sdr": sdrs,
        }
    }

    data_info = {
        "target_length": w1_lens,
        "interference_length": w2_lens,
        "reference_length": dvec_lens,
        "target_silent_ratio": w1_silent_ratio_10db,
        "interference_silent_ratio": w2_silent_ratio_10db,
        "snr": snrs
    }


    ###
    # Append new test result to file, then rewrite
    ###
    logger.info(f"Appending test result to {test_record_path}")

    test_record["info"] = data_info

    if test_record.get(exp.name):
        # Test result on same model exist
        if test_record[exp.name] != test_result:
            logger.info("Test result exist, but different result this time!")
            i = 0
            while test_record.get(f"{exp.name}_{i}"): i+=1
            logger.info(f"Result will be stored within {exp.name}_{i}")
            test_record[f"{exp.name}_{i}"] = test_result
    else:
        test_record[exp.name] = test_result

    json_object = json.dumps(test_record, indent = 4)
    with open(test_record_path, "w") as f:
        f.write(json_object)