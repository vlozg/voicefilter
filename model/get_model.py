import torch

# Embedder
from model.ge2e import SpeechEmbedder

# Voice seperation model
from model.voicefilter import VoiceFilter
from model.pse_dccrn import PSE_DCCRN

import importlib
ZaloSolutions = importlib.import_module("model.ZA-Challenge-Voice.embedder")

def get_embedder(exp_config, train, device):

    if exp_config.embedder.name == "GE2E":
        embedder = SpeechEmbedder(exp_config.embedder)

        # Load embedder
        if exp_config.embedder.pretrained_chkpt is not None:
            embedder_pt = torch.load(exp_config.embedder.pretrained_chkpt, "cpu")
            embedder.load_state_dict(embedder_pt)
    elif exp_config.embedder.name == "ZaloTop1": 
        embedder = ZaloSolutions.ZaloSpeechEmbedder()

    if device == "cuda":
        embedder = embedder.cuda()

    if train:
        embedder.train()
    else:
        embedder.eval()

    return embedder



def get_vfmodel(exp_config, train, device):

    if exp_config.model.name == "voicefilter":
        # Check config, compute input dim
        assert exp_config.audio.n_fft // 2 + 1 == exp_config.audio.num_freq == exp_config.model.fc2_dim, \
                "stft-related dimension mismatch"
        exp_config.model.input_dim = 8*exp_config.audio.num_freq + exp_config.embedder.emb_dim
        
        model = VoiceFilter(exp_config.model)
    elif exp_config.model.name == "pse_dccrn":
        model = PSE_DCCRN(exp_config, 
                    fft_len=exp_config.audio.n_fft,
                    win_len=exp_config.audio.win_length,
                    win_inc=exp_config.audio.hop_length,
                    rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256])
    else:
        raise NotImplementedError(f"Please implement {model} model")

    # Load model
    if exp_config.model.pretrained_chkpt is not None:
        checkpoint = torch.load(exp_config.model.pretrained_chkpt, "cpu")
        model.load_state_dict(checkpoint['model'])
    else:
        checkpoint = None

    if device == "cuda":
        model = model.cuda()
    
    if train:
        model.train()
    else:
        model.eval()

    return model, checkpoint

def get_forward(exp_config):
    model = exp_config.model.name
    embedder = exp_config.embedder.name

    if embedder == "ge2e" and model == "pse_dccrn":
        from model.forward_recipes.ge2e_psedccrn import train_forward, inference_forward
        return train_forward, inference_forward
    elif embedder == "ge2e" and model == "voicefilter":
        from model.forward_recipes.ge2e_vf import train_forward, inference_forward
        return train_forward, inference_forward
    else:
        raise NotImplementedError(f"Please implement forward function for {embedder} embedder and {model} model")