import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile
import torch
from loss.get_criterion import get_criterion
from model.get_model import get_embedder, get_forward, get_vfmodel
from tqdm import tqdm
from utils.audio import Audio



def inference(config, testloader, logger, out_dir=None):

    # Init model, embedder, optim, criterion
    audio = Audio(config)
    device = "cuda" if config.use_cuda else "cpu"
    embedder = get_embedder(config, train=False, device=device)
    model, chkpt = get_vfmodel(config, train=False, device=device)
    train_forward, inference_forward = get_forward(config)
    criterion = get_criterion(config, reduction="none")

    if chkpt is None:
        logger.error("There is no pre-trained checkpoint to test, please re-check config file")
        return
    else:
        logger.info(f"Start testing checkpoint: {config.model.pretrained_chkpt}")
    
    if out_dir is None:
        logger.error("Cold inference will inference audio first, help reduce overhead caused by testing. Thus you must provide out_dir to write files!")
        return

    test_losses = []
    os.makedirs(out_dir if out_dir != '' else ".", exist_ok=True)

    with ThreadPoolExecutor() as texecutor:
        futu_audio_w = {}

        for batch in tqdm(testloader):
            # Preliminary test if file have already infered
            for i, idx in enumerate(batch["index"]):
                out_file = os.path.join(out_dir, f"{idx}.wav")
                if os.path.isfile(out_file):
                    test_losses += [float('nan')]
                else:
                    sample = {k: batch[k][i] for k in batch.keys()}
                            
                    with torch.no_grad():
                        # If target exist, compute criterion too
                        # if not, then only do forward
                        if sample.get("target_wav") is not None:
                            est_stft, _, loss = train_forward(model, embedder, sample, criterion, device)
                            test_losses += loss.mean((1,2)).cpu().tolist()
                        else:
                            est_stft, _ = inference_forward(model, embedder, sample, device)
                            test_losses += [float('nan')]
                    est_stft = est_stft[0].cpu().detach().numpy()
                    
                    est_wav = audio._istft(est_stft.T, length=sample["mixed_wav"].shape[-1])
                    out_file = os.path.join(out_dir, f"{idx}.wav")
                    
                    futu_audio_w[texecutor.submit(soundfile.write, out_file, est_wav, config.audio.sample_rate)] = idx
            

        logger.info("Waiting for audio writing...")
        for f in tqdm(as_completed(futu_audio_w), total=len(futu_audio_w)):
            idx = futu_audio_w[f]
            try:
                f.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (idx, exc))


    test_losses = np.array(test_losses)

    logger.info(f"Complete inferencing")

    results = {
        "loss": test_losses,
    }

    return results
