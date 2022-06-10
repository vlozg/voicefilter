# Implement power-law compression loss as in the original paper

import torch
from torch import Tensor, nn
from .l2_asym_loss import L2AsymLoss

class PowerLawCompAsymLoss(nn.Module):
  
  def __init__(self, 
               power: float = 0.3,
               complex_loss_ratio: float = 0.113,
               undersuppress_penalty=1,
               oversuppress_penalty=10,
               l2_asym_loss_ratio: float = 1.0,
               reduction: str = 'mean') -> None:
    super(PowerLawCompAsymLoss, self).__init__()
    self.power = power
    self.alpha = complex_loss_ratio
    self.beta = l2_asym_loss_ratio
    self.l2_norm = nn.MSELoss(reduction=reduction)
    self.l2_asym = L2AsymLoss(undersuppress_penalty, oversuppress_penalty, reduction=reduction)
    
    if reduction == "mean":
      self.reduce = torch.mean
    elif reduction == "sum":
      self.reduce = torch.sum
    elif reduction == "none":
      self.reduce = lambda x: x
    else:
      raise NotImplementedError
    
  def forward(self, mask, input: Tensor, target: Tensor) -> Tensor:    
    input = mask*torch.pow(input, self.power)
    target = torch.pow(target, self.power)
    input_mag = input.abs()
    target_mag = target.abs()
    magnitude_loss = self.l2_norm(input_mag, target_mag)
    
    # MSE doesn't support complex number yet
    complex_loss = self.reduce((input - target).abs()**2)

    # Oversuperession loss
    os_loss = self.l2_asym(input_mag, target_mag)

    return magnitude_loss + self.alpha * complex_loss + self.beta * os_loss
