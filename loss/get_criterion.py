import torch.nn as nn
from .power_law_loss import PowerLawCompLoss

def get_criterion(exp_config, reduction="mean"):
    if exp_config.loss_function == "mse":
        return nn.MSELoss(reduction=reduction)
    elif exp_config.loss_function == "power_law_compressed":
        return PowerLawCompLoss(reduction=reduction)
    else:
        raise ValueError(f"{exp_config.loss_function} is not supported, please implement and update get_criterion.py")