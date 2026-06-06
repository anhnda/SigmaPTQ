"""
metric.py
=========
The ONE generic reducer. Given the full-row error E and a sensitivity s^2,
compute the per-group quadratic score

    score_{r, gp} = (1/N) sum_t  s^2_{r,t} ( sum_{j in gp} E_{r,j} x_{j,t} )^2

which is the per-group restriction of  e^T M_i e  with M_i = (1/N) X diag(s^2) X^T.

This replaces the duplicated group loops that lived in LinearResponse /
SigmaAware / Mixed: every sensitivity kind now flows through here.

MEMORY: per group we form U_g = E[:, gp] @ X[gp, :] of shape (d_out, n_tok) and
reduce it immediately, never materializing (d_out, n_groups, n_tok). This is
the same n_groups-fold peak reduction as the original base_cr.py.

Sensitivity shapes (from sensitivity.py):
    1.0                  -> linear response, no token reweighting
    (n_tok,)             -> row-shared reweighting
    (d_out, n_tok)       -> per (row, token) reweighting  (gate/up/pointwise)
"""

from __future__ import annotations

import torch


def group_index(d_in: int, group_size: int, device):
    n_groups = (d_in + group_size - 1) // group_size
    pad = n_groups * group_size - d_in
    gidx = torch.arange(d_in, device=device) // group_size  # (d_in,)
    return gidx, n_groups, pad


def scatter_group_sum(per_channel: torch.Tensor, gidx: torch.Tensor,
                      n_groups: int) -> torch.Tensor:
    """per_channel (d_out, d_in) -> (d_out, n_groups) summed within groups."""
    d_out = per_channel.shape[0]
    out = torch.zeros(d_out, n_groups, device=per_channel.device,
                      dtype=per_channel.dtype)
    idx = gidx.unsqueeze(0).expand(d_out, -1)
    out.scatter_add_(1, idx, per_channel)
    return out


def weight_score(E: torch.Tensor, gidx: torch.Tensor,
                 n_groups: int) -> torch.Tensor:
    """Identity-metric term  sum_{j in gp} E_{r,j}^2 . Used as the lam floor."""
    return scatter_group_sum(E.pow(2), gidx, n_groups)


def activation_score(E: torch.Tensor, X: torch.Tensor, s2,
                     gidx: torch.Tensor, n_groups: int, ntok: int):
    """
    Per-group activation-aware score under sensitivity s2.

    E    : (d_out, d_in)   full-row error
    X    : (d_in, n_tok)   stored activations
    s2   : 1.0 | (n_tok,) | (d_out, n_tok)
    gidx : (d_in,)         channel -> group id
    returns (d_out, n_groups)
    """
    d_out = E.shape[0]
    out = torch.empty(d_out, n_groups, device=E.device, dtype=E.dtype)

    # Resolve the reweighting form once.
    if isinstance(s2, float):
        mode = "none"
    elif s2.dim() == 1:
        mode = "token"          # (n_tok,)
    else:
        mode = "rowtoken"       # (d_out, n_tok)

    for gp in range(n_groups):
        mask = (gidx == gp)
        Ug = E[:, mask] @ X[mask, :]            # (d_out, n_tok)
        Ug2 = Ug.pow(2)
        if mode == "none":
            w = Ug2
        elif mode == "token":
            w = Ug2 * s2.unsqueeze(0)           # broadcast (1, n_tok)
        else:  # rowtoken
            w = Ug2 * s2                        # (d_out, n_tok) elementwise
        out[:, gp] = w.sum(dim=1) / ntok
    return out