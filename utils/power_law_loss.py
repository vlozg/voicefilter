# Implement power-law compression loss as in the original paper
# Reference: https://github.com/mindslab-ai/voicefilter/pull/18/commits/0acfd8ee94ae875c484745df6fef5d985d195fc2

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
    l2_norm = nn.MSELoss(reduction=reduction)
    
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    # power-law compress
    input = torch.pow(input, self.power)
    target = torch.pow(target, self.power)
    
    magnitude_loss = self.l2_norm(torch.abs(input), torch.abs(target))
    complex_loss = self.l2_norm(input, target)
    
    return magnitude_loss + self.alpha * complex_loss
