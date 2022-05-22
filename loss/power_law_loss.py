# Implement power-law compression loss as in the original paper

import torch
from torch import Tensor, nn

class PowerLawCompLoss(nn.Module):
  
  def __init__(self, 
               power: float = 0.3,
               complex_loss_ratio: float = 0.113,
               reduction: str = 'mean') -> None:
    super(PowerLawCompLoss, self).__init__()
    self.power = power
    self.alpha = complex_loss_ratio
    self.l2_norm = nn.MSELoss(reduction=reduction)
    
  def forward(self, mask, input: Tensor, target: Tensor) -> Tensor:    
    input = mask*torch.pow(input, self.power)
    target = torch.pow(target, self.power)
    magnitude_loss = self.l2_norm(input.abs(), target.abs())
    
    # MSE doesn't support complex number yet
    complex_loss = ((input - target).abs()**2).mean()
    return magnitude_loss + self.alpha * complex_loss
