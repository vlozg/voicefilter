# Implement power-law compression loss as in the original paper

from functools import reduce
import torch
from torch import Tensor, nn

class L2AsymLoss(nn.Module):
  
  def __init__(self,
               undersuppress_penalty=1,
               oversuppress_penalty=10,
               reduction: str = 'mean') -> None:
    super(L2AsymLoss, self).__init__()
    self.alpha = undersuppress_penalty
    self.beta = oversuppress_penalty
    if reduction == "mean":
      self.reduce = torch.mean
    elif reduction == "sum":
      self.reduce = torch.sum
    elif reduction == "none":
      self.reduce = lambda x: x
    else:
      raise NotImplementedError
    
  def forward(self, input: Tensor, target: Tensor) -> Tensor:    
    delta = target - input
    loss = torch.where(delta <= 0, delta.self.alpha, delta.self.beta)
    loss = torch.pow(loss, 2)
    return self.reduce(loss)
