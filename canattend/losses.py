import numpy
import torch
from torch import Tensor
import torch.nn.functional as F
import pdb

from .utils import pad_col

class _Loss(torch.nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

#### LogisticHazard Loss ####
class NLLLogistiHazardLoss(_Loss):

    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean') -> Tensor:
        
        return nll_logistic_hazard(phi, idx_durations, events, reduction)

def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

def nll_logistic_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                        reduction: str = 'mean') -> Tensor:

    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)

    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)

#### PCH Loss ####
class NLLPCHazardLoss(_Loss):
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, interval_frac: Tensor,
                reduction: str = 'mean') -> Tensor:

        return nll_pc_hazard_loss(phi, idx_durations, events, interval_frac, reduction)

def log_softplus(input, threshold=-15.):

    output = input.clone()
    above = input >= threshold
    output[above] = F.softplus(input[above]).log()
    return output

def nll_pc_hazard_loss(phi: Tensor, idx_durations: Tensor, events: Tensor, interval_frac: Tensor,
                       reduction: str = 'mean') -> Tensor:

    if events.dtype is torch.bool:
        events = events.float()
    # print('loss idx_durations', idx_durations) ##
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1)
    interval_frac = interval_frac.view(-1)

    keep = idx_durations.view(-1) >= 0
    phi = phi[keep, :]
    idx_durations = idx_durations[keep, :]
    events = events[keep]
    interval_frac = interval_frac[keep]

    # print('ST loss phi before gathering: ', phi.shape, phi)
    log_h_e = log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)
    haz = F.softplus(phi)
    scaled_h_e = haz.gather(1, idx_durations).view(-1).mul(interval_frac)
    haz = pad_col(haz, where='start')
    sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1) 
    loss = - log_h_e.sub(scaled_h_e).sub(sum_haz)
    return _reduction(loss, reduction)