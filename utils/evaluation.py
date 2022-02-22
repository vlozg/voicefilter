import torch
import torch.nn as nn
import numpy as np
from mir_eval.separation import bss_eval_sources
from .power_law_loss import PowerLawCompLoss


def validate(audio, model, embedder, testloader, writer, logger, step):
    model.eval()
    
    # criterion = nn.MSELoss()
    criterion = PowerLawCompLoss()
    with torch.no_grad():
        first = True
        test_losses = []
        sdrs = []
        saved_sample = None
        for batch in testloader:
            _, _, _, dvec_mel, target_wav, mixed_wav, _, _, mixed_mag, _, target_stft, mixed_stft = batch[0]
            dvec_mel = dvec_mel.cuda()
            target_stft = target_stft.unsqueeze(0).cuda()
            mixed_stft = mixed_stft.unsqueeze(0).cuda()
            # mixed_mag = mixed_mag.unsqueeze(0).cuda()

            dvec = embedder(dvec_mel)
            dvec = dvec.unsqueeze(0)
            # est_mask = model(mixed_mag, dvec)
            est_mask = model(torch.pow(mixed_stft.abs(), 0.3), dvec)
            test_losses.append(criterion(est_mask, mixed_stft, target_stft).item())
            est_mask = torch.pow(est_mask, 10/3)
            est_stft = mixed_stft * est_mask

            mixed_stft = mixed_stft[0].T.cpu().detach().numpy()
            target_stft = target_stft[0].T.cpu().detach().numpy()
            est_stft = est_stft[0].T.cpu().detach().numpy()
            est_wav = audio._istft(est_stft)
            est_mask = est_mask[0].cpu().detach().numpy()

            sdrs.append(bss_eval_sources(target_wav, est_wav, False)[0][0])

            # Take first sample for visualization
            if first:
                mixed_mag, _ = audio.stft2spec(mixed_stft)
                target_mag, _ = audio.stft2spec(target_stft)
                est_mag, _ = audio.stft2spec(est_stft)
                saved_sample = mixed_wav, target_wav, est_wav, mixed_mag.T, target_mag.T, est_mag.T, est_mask.T
                first = False
        
        test_loss = np.array(test_losses).mean()
        sdr_mean = np.array(sdrs).mean()
        sdr_med = np.median(np.array(sdrs))
        mixed_wav, target_wav, est_wav, mixed_mag, target_mag, est_mag, est_mask = saved_sample
        writer.log_evaluation(test_loss, sdr_mean, sdr_med,
                                mixed_wav, target_wav, est_wav,
                                mixed_mag, target_mag, est_mag, est_mask,
                                step)
        logger.info(f"Complete evaluate at step {step}")
        logger.info(f"- Test SDR mean: {sdr_mean}\t Test SDR median: {sdr_med}")

    model.train()
