import torch

from .model import VoiceFilter
from .embedder import SpeechEmbedder

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

    # Check config config, compute input dim
    assert exp_config.audio.n_fft // 2 + 1 == exp_config.audio.num_freq == exp_config.model.fc2_dim, \
            "stft-related dimension mismatch"
    exp_config.model.input_dim = 8*exp_config.audio.num_freq + exp_config.embedder.emb_dim
    
    model = VoiceFilter(exp_config.model)

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