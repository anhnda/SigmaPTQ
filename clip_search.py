"""
clip_search.py
==============
Group-wise clip selection (paper Eq. 4, symmetric case beta=-alpha=c):

    c*_{r,gp} = argmin_c  score_{r,gp}(Q_c(W) - W)

The score is supplied as a callable  score(E) -> (d_out, n_groups), built once
per layer (the metric / sensitivity are fixed; only the candidate clip varies
across the grid). This decouples "what metric" (sensitivity.py + metric.py)
from "how to search" (here).

Carried over from ClipRange.select_clip in base_cr.py, generalized to take an
arbitrary score callable instead of subclassing.
"""

from __future__ import annotations

import torch
from typing import Callable

from quantizer import QuantFn

ScoreFn = Callable[[torch.Tensor], torch.Tensor]  # E (d_out,d_in) -> (d_out,n_groups)


@torch.no_grad()
def select_clip(W: torch.Tensor, quant_fn: QuantFn, score_fn: ScoreFn, *,
                group_size: int, n_grid: int = 20,
                grid_min: float = 0.5, grid_max: float = 1.0) -> torch.Tensor:
    """Returns per-group clip (d_out, n_groups, 1)."""
    d_out, d_in = W.shape
    gs = group_size
    n_groups = (d_in + gs - 1) // gs
    pad = n_groups * gs - d_in

    if pad > 0:
        Wp = torch.zeros(d_out, n_groups * gs, device=W.device, dtype=W.dtype)
        Wp[:, :d_in] = W
    else:
        Wp = W
    g_maxabs = Wp.reshape(d_out, n_groups, gs).abs().amax(dim=2, keepdim=True).clamp(min=1e-8)

    best_score = torch.full((d_out, n_groups), float("inf"), device=W.device)
    best_clip = g_maxabs.clone()

    for g in range(n_grid + 1):
        frac = grid_min + (grid_max - grid_min) * g / n_grid
        clip = g_maxabs * frac
        Wq = quant_fn(W, clip)
        E = Wq - W
        s = score_fn(E)                      # (d_out, n_groups)
        improve = s < best_score
        best_score = torch.where(improve, s, best_score)
        best_clip = torch.where(improve.unsqueeze(2), clip, best_clip)

    return best_clip


@torch.no_grad()
def rtn_clip(W: torch.Tensor, group_size: int) -> torch.Tensor:
    """Plain RTN: clip = per-group max-abs (cf = 1.0), no search."""
    d_out, d_in = W.shape
    gs = group_size
    n_groups = (d_in + gs - 1) // gs
    pad = n_groups * gs - d_in
    if pad > 0:
        Wp = torch.zeros(d_out, n_groups * gs, device=W.device, dtype=W.dtype)
        Wp[:, :d_in] = W
    else:
        Wp = W
    return Wp.reshape(d_out, n_groups, gs).abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
