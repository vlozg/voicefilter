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
import soundfile
from tqdm import tqdm


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
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='path of file to output')
    parser.add_argument('--is_directory', type=bool, default=False,
                        help="path above is directory of file. Default: False")
    args = parser.parse_args()

    config = HParam(args.config)["experiment"]

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
        out_list = [os.path.join(args.out_path, x) for x in file_list]
    else:
        mixed_list = [args.mixed_path]
        reference_list = [args._reference_path]
        out_list = [args.out_path]


    for mixed_file, reference_file, out_file in tqdm(zip(mixed_list, reference_list, out_list)):
        with torch.no_grad():
            d, _ = librosa.load(reference_file, sr=config.audio.sample_rate)
            mixed_wav, _ = librosa.load(mixed_file, sr=config.audio.sample_rate)

            mixed_wav = mixed_wav/(np.max(np.abs(mixed_wav))*1.1)
            d = d/np.max(np.abs(d)) # Normalize volume
            d = vad_merge(d) # Then VAD

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
        
        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir if out_dir != '' else ".", exist_ok=True)
        soundfile.write(out_file, est_wav, config.audio.sample_rate)
        soundfile.write("test_results/vin_asr/dvec/"+os.path.basename(out_file), d, config.audio.sample_rate)
