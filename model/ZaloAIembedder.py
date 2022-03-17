import sys
  
# setting path
sys.path.append('ZaloAITopSolution/top1-nguyenvannha/src')


from ret_benchmark.config import cfg
import torch.nn.functional as F
import torch
from ret_benchmark.modeling import build_model
from ret_benchmark.utils.img_reader import loadWAV
import os
import numpy as np


cfg.merge_from_file("ZaloAITopSolution/top1-nguyenvannha/src/config.yaml")
model = build_model(cfg)
model.load_state_dict(torch.load("checkpoints/model_final.pth",map_location="cpu")["model"])
model.eval()


img1 = loadWAV(os.path.join(SOURCE_DATASET,audio1))
img1 = torch.FloatTensor(img1).to(device)
with torch.no_grad():
    out1 = model.model.__S__(img1).detach().cpu()