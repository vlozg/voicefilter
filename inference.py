import os
import glob
import torch
import librosa
import argparse
import numpy as np
from utils.audio import Audio
from utils.hparams import HParam

from model.get_model import get_vfmodel, get_embedder
from model.forward import inference_forward
from loss.get_criterion import get_criterion
import soundfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-m', '--mixed_file', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-r', '--reference_file', type=str, required=True,
                        help='path of reference wav file')
    parser.add_argument('-o', '--out_file', type=str, required=True,
                        help='path of file to output')
    args = parser.parse_args()

    config = HParam(args.config)["experiment"]

    audio = Audio(config)
    device = "cuda" if config.use_cuda else "cpu"
    embedder = get_embedder(config, train=False, device=device)
    model, chkpt = get_vfmodel(config, train=False, device=device)
    criterion = get_criterion(config, reduction="none")


    with torch.no_grad():
        d, _ = librosa.load(args.reference_file, sr=config.audio.sample_rate)
        mixed_wav, _ = librosa.load(args.mixed_file, sr=config.audio.sample_rate)

        norm = np.max(np.abs(mixed_wav)) * 1.1
        mixed_wav = mixed_wav/norm

        dvec_mel = audio.get_mel(d)
        dvec_mel = torch.from_numpy(dvec_mel).float()
        mixed_stft = audio.stft(mixed_wav).T
        mixed_stft = torch.from_numpy(mixed_stft)

        batch = {
            "dvec": dvec_mel.unsqueeze(0),
            "mixed_stft": mixed_stft.unsqueeze(0)
        }

        est_stft, est_mask = inference_forward(model, embedder, batch, device)
        est_stft = est_stft.cpu().detach().numpy()
        est_wav = audio._istft(est_stft[0].T, length=len(mixed_wav))
    
    print( est_wav.shape)
    print( mixed_wav.shape)
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir if out_dir != '' else ".", exist_ok=True)
    soundfile.write(args.out_file, est_wav, config.audio.sample_rate)
