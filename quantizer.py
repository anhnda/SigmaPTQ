"""
quantizer.py
============
Group-wise symmetric RTN quantizer accepting a PER-GROUP clip of shape
(d_out, n_groups, 1). Carried over verbatim from base_cr.py
groupwise_rtn_symmetric (which was already correct) so the rest of the
refactor depends only on this signature.

The asymmetric (alpha, beta) quantizer from Sec. 3 of the paper is stubbed at
the bottom for later (TODO C6); the metric machinery above is identical for
both, only e(alpha,beta) changes.
"""

from __future__ import annotations

import torch
from typing import Callable

# clip shape (d_out, n_groups, 1)
QuantFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def groupwise_rtn_symmetric(W: torch.Tensor, clip: torch.Tensor,
                            bits: int, group_size: int) -> torch.Tensor:
    d_out, d_in = W.shape
    qmax = 2 ** (bits - 1) - 1
    n_groups = (d_in + group_size - 1) // group_size
    pad = n_groups * group_size - d_in
    if pad > 0:
        Wp = torch.zeros(d_out, n_groups * group_size, device=W.device, dtype=W.dtype)
        Wp[:, :d_in] = W
    else:
        Wp = W
    Wg = Wp.reshape(d_out, n_groups, group_size)
    g_maxabs = Wg.abs().amax(dim=2, keepdim=True)
    eff = torch.minimum(g_maxabs, clip).clamp(min=1e-8)
    delta = eff / qmax
    q = torch.round(Wg / delta).clamp(-qmax, qmax)
    Wdq = (q * delta).reshape(d_out, n_groups * group_size)
    if pad > 0:
        Wdq = Wdq[:, :d_in]
    return Wdq


def make_quant_fn(bits: int, group_size: int) -> QuantFn:
    def quant_fn(W, clip):
        return groupwise_rtn_symmetric(W, clip, bits, group_size)
    return quant_fn