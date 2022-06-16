import torch
import torch.nn as nn
import numpy as np
from torch_mir_eval import bss_eval_sources


def validate(model, embedder, testloader, train_forward, criterion, device, audio, writer, logger, step):
    model.eval()
    
    with torch.no_grad():
        test_losses = []
        sdrs = []
        for batch in testloader:
            
            est_stft, est_mask, loss = train_forward(model, embedder, batch, criterion, device)
            
            test_losses.append(loss.item())
           
            est_stft = est_stft.cpu().detach().numpy()

            for est_stft_, target_wav in zip(est_stft, batch["target_wav"]):
                est_wav = audio._istft(est_stft_.T, length=len(target_wav))
                est_wav = torch.from_numpy(est_wav).reshape(1, -1)
                target_wav = target_wav.reshape(1, -1)
                sdr,sir,sar,perm = bss_eval_sources(target_wav,est_wav,compute_permutation=False)
                sdrs.append(sdr.item())
        
        test_loss = np.array(test_losses).mean()
        sdr_mean = np.array(sdrs).mean()
        sdr_med = np.median(np.array(sdrs))


        # Take first sample for visualization
        batch = next(iter(testloader))
        est_stft, est_mask, _ = train_forward(model, embedder, batch, criterion, device)
        est_stft = est_stft[0].cpu().detach().numpy()
        est_mask = est_mask[0].cpu().detach().numpy().T
        
        est_wav = audio._istft(est_stft.T)
        est_mag, _ = audio.stft2spec(est_stft)
        mixed_mag, _ = audio.stft2spec(batch["mixed_stft"][0].numpy())
        target_mag, _ = audio.stft2spec(batch["target_stft"][0].numpy())
        
        mixed_wav, target_wav = batch["mixed_wav"][0], batch["target_wav"][0]
        
        writer.log_evaluation(test_loss, sdr_mean, sdr_med,
                                mixed_wav, target_wav, est_wav,
                                mixed_mag, target_mag, est_mag, est_mask,
                                step)
                                
        logger.info(f"Complete evaluate at step {step}")
        logger.info(f"- Test SDR mean: {sdr_mean}\t Test SDR median: {sdr_med}")

    model.train()
