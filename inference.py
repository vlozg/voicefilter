import os
import glob
import torch
import librosa
import argparse
import numpy as np
from utils.audio import Audio
from utils.hparams import HParam

from model.get_model import get_vfmodel, get_embedder, get_forward
from loss.get_criterion import get_criterion
from datasets.GenerateDataset import vad_merge

from torch_mir_eval import bss_eval_sources

import soundfile
from tqdm import tqdm

import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--chkpt', type=str, required=True,
                        help="path to pre-trained VF model checkpoint")
    parser.add_argument('--use_cuda', type=bool, default=None,
                        help="use cuda for testing, overwrite test config. Default: follow test config file")
    parser.add_argument('-m', '--mixed_path', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-r', '--reference_path', type=str, required=True,
                        help='path of reference wav file')
    parser.add_argument('--ground_truth_path', type=str,
                        help='path of ground truth wav file. Add this to get evaluate metrics result.')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='path of file to output')
    parser.add_argument('--is_directory', type=bool, default=False,
                        help="path above is directory of file. Default: False")
    args = parser.parse_args()

    ###
    # Merge config with CLI parameters
    ###
    config = HParam(args.config)
    
    if args.use_cuda is not None:
        config.experiment.use_cuda = args.use_cuda
    config.experiment.model.pretrained_chkpt = args.chkpt

    config = config["experiment"]

    ###
    # Load model
    ###
    audio = Audio(config)
    device = "cuda" if config.use_cuda else "cpu"
    embedder = get_embedder(config, train=False, device=device)
    model, chkpt = get_vfmodel(config, train=False, device=device)
    _, inference_forward = get_forward(config)
    criterion = get_criterion(config, reduction="none")

    if args.is_directory:
        print("Warning: using directory inference, audio in these folder must have same name as their counterpart")
        file_list = [x.split("/")[-1] for x in glob.glob(os.path.join(args.mixed_path, '**', "*.wav"), recursive=True)]
        mixed_list = [os.path.join(args.mixed_path, x) for x in file_list]
        reference_list = [os.path.join(args.reference_path, x) for x in file_list]
        if args.ground_truth_path:
            groundtruth_list = [os.path.join(args.ground_truth_path, x) for x in file_list]
        else:
            groundtruth_list = [None]*len(file_list)
        out_list = [os.path.join(args.out_path, x) for x in file_list]
    else:
        mixed_list = [args.mixed_path]
        reference_list = [args._reference_path]
        groundtruth_list = [args.ground_truth_path if args.ground_truth_path else None]
        out_list = [args.out_path]


    ###
    # Start infernce
    ###
    sdrs_after = []
    sdrs_before = []
    for mixed_file, reference_file, groundtruth_file, out_file in tqdm(zip(mixed_list, reference_list, groundtruth_list, out_list), total=len(mixed_list)):
        with torch.no_grad():
            d, _ = librosa.load(reference_file, sr=config.audio.sample_rate)
            mixed_wav, _ = librosa.load(mixed_file, sr=config.audio.sample_rate)

            norm = np.max(np.abs(mixed_wav)) * 1.1
            mixed_wav = mixed_wav/norm

            dvec_mel = audio.get_mel(d)
            dvec_mel = torch.from_numpy(dvec_mel).float()
            mixed_stft = audio.stft(mixed_wav).T
            mixed_stft = torch.from_numpy(mixed_stft)

            batch = {
                "dvec": dvec_mel.unsqueeze(0),
                "dvec_wav": [d],
                "mixed_stft": mixed_stft.unsqueeze(0),
                "mixed_wav": torch.from_numpy(mixed_wav).unsqueeze(0)
            }

            est_stft, est_mask = inference_forward(model, embedder, batch, device)
            est_stft = est_stft.cpu().detach().numpy()
            est_wav = audio._istft(est_stft[0].T, length=len(mixed_wav))

            if groundtruth_file:
                target_wav, _ = librosa.load(groundtruth_file, sr=config.audio.sample_rate)
                target_wav = target_wav/norm
                _est_wav = torch.from_numpy(est_wav).to(device=device).reshape(1, -1)
                target_wav = torch.from_numpy(target_wav).to(device=device).reshape(1, -1)
                mixed_wav = torch.from_numpy(mixed_wav).to(device=device).reshape(1, -1)

                if target_wav.sum() != 0 and _est_wav.sum() != 0:
                    sdr,sir,sar,perm = bss_eval_sources(target_wav,_est_wav,compute_permutation=False)
                    sdr = sdr.item()
                else: 
                    sdr = None
                sdrs_after.append(sdr)

                if target_wav.sum() != 0 and mixed_wav.sum() != 0:
                    sdr,sir,sar,perm = bss_eval_sources(target_wav,mixed_wav,compute_permutation=False)
                    sdr = sdr.item()
                else: 
                    sdr = None
                sdrs_before.append(sdr)

        
        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir if out_dir != '' else ".", exist_ok=True)
        soundfile.write(out_file, est_wav, config.audio.sample_rate)

    
    if args.ground_truth_path:
        test_result = {
            "sdrs_before": sdrs_before,
            "sdrs_after": sdrs_after
        }
        json_object = json.dumps(test_result, indent = 4)
        with open(os.path.join(args.out_path, "test_result.json"), "w") as f:
            f.write(json_object)
