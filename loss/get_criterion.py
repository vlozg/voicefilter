import torch.nn as nn
from .power_law_loss import PowerLawCompLoss
from .l2_asym_loss import L2AsymLoss
from .power_law_asym_loss import PowerLawCompAsymLoss

def get_criterion(exp_config, reduction="mean"):
    if exp_config.loss_function == "mse":
        return nn.MSELoss(reduction=reduction)
    elif exp_config.loss_function == "power_law_compressed":
        return PowerLawCompLoss(reduction=reduction)
    if exp_config.loss_function == "l2_asym":
        return L2AsymLoss(reduction=reduction)
    elif exp_config.loss_function == "power_law_compressed_asym":
        return PowerLawCompAsymLoss(reduction=reduction)
    else:
        raise ValueError(f"{exp_config.loss_function} is not supported, please implement and update get_criterion.py")