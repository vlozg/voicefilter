import torch
import torch.nn as nn
import numpy as np
import json

from utils.audio import Audio
from utils.hparams import HParam

from mir_eval.separation import bss_eval_sources
from datasets.dataloader import create_dataloader


from utils.adabound import AdaBound
from utils.power_law_loss import PowerLawCompLoss
from model.voicefilter import VoiceFilter
from model.embedder import SpeechEmbedder

hp = HParam("config.yaml")
testloader_vn = create_dataloader(hp, "generate", dataset_detail=["vin", "zalo-train", "zalo-test"], scheme="test_cuda", size=5000)
testloader_lb = create_dataloader(hp, "generate", dataset_detail=["librispeech-test"], scheme="test_cuda", size=5000)
testloader_gg = create_dataloader(hp, "gg", dataset_detail="test", scheme="test_cuda_size_1")
audio = Audio(hp)
device = "cpu"

embedder_pt = torch.load("embedder.pt", device)
embedder = SpeechEmbedder(hp)
embedder.load_state_dict(embedder_pt)
embedder = embedder.cuda()
embedder.eval()

powlaw = PowerLawCompLoss()
mse = nn.MSELoss()

def powerlaw_forward(model, dvec, target_stft, mixed_stft, target_wavs):
    with torch.no_grad():
        est_mask = model(torch.pow(mixed_stft.abs(), 0.3), dvec)
        loss = powlaw(est_mask, mixed_stft, target_stft).item()
        
        est_mask = torch.pow(est_mask, 10/3)
        est_stft = mixed_stft * est_mask
        est_stft = est_stft.cpu().numpy()

    sdrs = []
    for est_stft_, target_wav in zip(est_stft, target_wavs):
        est_wav = audio._istft(est_stft_.T, length=len(target_wav))
        sdrs.append(bss_eval_sources(target_wav, est_wav, False)[0][0])

    return loss, sdrs

def mse_forward(model, dvec, target_mag, mixed_mag, mixed_phase, target_wavs):
    with torch.no_grad():
        est_mask = model(mixed_mag, dvec)
        est_mag = mixed_mag * est_mask
        
        loss = mse(target_mag, est_mag).item()

        est_mag = est_mag.cpu().numpy()
        mixed_phase = mixed_phase.numpy()
    
    sdrs = []
    for est_mag_, mixed_phase_, target_wav in zip(est_mag, mixed_phase, target_wavs):
        est_wav = audio.spec2wav(est_mag_, mixed_phase_, length=len(target_wav))
        sdrs.append(bss_eval_sources(target_wav, est_wav, False)[0][0])

    return loss, sdrs

# %% [markdown]
# # Final evaluation

# Power-law compressed loss
model = VoiceFilter(hp)
checkpoint = torch.load("chkpt/powlaw_loss/chkpt_168000.pt", device)
model.load_state_dict(checkpoint['model'])
model = model.cuda()
model.eval()

# Power-law compressed loss
model_f = VoiceFilter(hp)
checkpoint = torch.load("chkpt/powlaw_loss_finetune/chkpt_178000.pt", device)
model_f.load_state_dict(checkpoint['model'])
model_f = model_f.cuda()
model_f.eval()


# First try (MSE loss)
model_0 = VoiceFilter(hp)
checkpoint = torch.load("chkpt/new_dataloader/chkpt_108000.pt", device)
model_0.load_state_dict(checkpoint['model'])
model_0 = model_0.cuda()
model_0.eval()

# MSE ver 48k (ms.Tam)
model_t = VoiceFilter(hp)
checkpoint = torch.load("chkpt/mstam_mse/chkpt_48000.pt", device)
model_t.load_state_dict(checkpoint['model'])
model_t = model_t.cuda()
model_t.eval()

# %% [markdown]
# ## GGSpeakerID

# %%
def eval_loop(dataloader, outputfile):
    losses_p = []
    losses_0 = []
    losses_t = []
    losses_f = []
    sdrs_before = []
    sdrs_p = []
    sdrs_0 = []
    sdrs_t = []
    sdrs_f = []
    step = 0

    for batch in dataloader:
        dvec_mels, target_mag, _, mixed_mag, mixed_phase, target_stft, mixed_stft, target_wavs, mixed_wavs = batch
        with torch.no_grad():
            dvec_list = list()
            for mel in dvec_mels:
                dvec = embedder(mel.cuda())
                dvec_list.append(dvec)
            dvec = torch.stack(dvec_list, dim=0)

            mixed_mag = mixed_mag.cuda()
            target_mag = target_mag.cuda()

            target_stft = target_stft.cuda()
            mixed_stft = mixed_stft.cuda()
        
        for target_wav, mixed_wav in zip(target_wavs, mixed_wavs):
            sdrs_before.append(bss_eval_sources(target_wav, mixed_wav, False)[0][0])

        loss, sdrs = powerlaw_forward(model, dvec, target_stft, mixed_stft, target_wavs)
        losses_p.append(loss)
        sdrs_p += sdrs

        loss, sdrs = powerlaw_forward(model_f, dvec, target_stft, mixed_stft, target_wavs)
        losses_f.append(loss)
        sdrs_f += sdrs

        loss, sdrs = mse_forward(model_0, dvec, target_mag, mixed_mag, mixed_phase, target_wavs)
        losses_0.append(loss)
        sdrs_0 += sdrs

        loss, sdrs = mse_forward(model_t, dvec, target_mag, mixed_mag, mixed_phase, target_wavs)
        losses_t.append(loss)
        sdrs_t += sdrs
        
        print(f"Step {step} done")
        step+=1

    with open(outputfile, "w") as f:
        result_json = {
            "losses_p": losses_p,
            "losses_0": losses_0,
            "losses_t": losses_t,
            "losses_f": losses_f,
            "sdrs_before": sdrs_before,
            "sdrs_p": sdrs_p,
            "sdrs_0": sdrs_0,
            "sdrs_t": sdrs_t,
            "sdrs_f": sdrs_f
        }
        json_object = json.dumps(result_json, indent = 4)
        f.write(json_object)



if __name__ == '__main__':
    eval_loop(testloader_vn, "VNGenerate_test.json")
    eval_loop(testloader_lb, "Generate_test.json")
    eval_loop(testloader_gg, "GGSpeaker_test.json")