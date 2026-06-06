"""
base_cr.py
====================
Group-wise clip-range selection for weight-only group-wise PTQ.

FIX over base_cr.py: the clip candidate is now PER GROUP, matching the
group-wise quantizer. Previously a single per-row scalar c was searched while
groupwise_rtn_symmetric quantized per group of `group_size` channels; via
eff = min(group_maxabs, c) the clip could only bite the largest group per row.
Now c has shape (d_out, n_groups, 1) and each group's clip is chosen
independently.

MEMORY FIX: the activation-based scores (LinearResponse, Mixed, SigmaAware)
no longer materialize the dense partial-products tensor
U of shape (d_out, n_groups, n_tok). For Qwen2.5-7B down_proj
(d_out=18944, n_groups=148, n_tok=4096) that single fp32 tensor is ~46 GiB.
Instead each group's (d_out, n_tok) partial is reduced to (d_out,) inside the
group loop and discarded, dropping peak from d_out*n_groups*n_tok to
d_out*n_tok (an n_groups-fold reduction).

Clip selection is still METRIC SELECTION (Sigma-Aware PTQ paper, Eq. 5):

    c* = argmin_c  e(c)^T M e(c),     e(c) = Q_c(w) - w

but the argmin is now per (row, group): the score is reduced over the
group_size channels of each group, leaving the other groups' channels at zero
contribution for that group's decision.

Quadratic forms, per row r, per group gp (channels j in group gp):
  WeightMSE       : sum_{j in gp} e_{r,j}^2
  LinearResponse  : (1/N) sum_t ( sum_{j in gp} e_{r,j} x_{j,t} )^2
  Mixed(linear)   : (1-lam)*weight_gp + lam*linear_gp
  SigmaAware      : (1/N) sum_t s_{r,t}^2 ( sum_{j in gp} e_{r,j} x_{j,t} )^2
  Mixed(sigma)    : (1-lam)*weight_gp + lam*sigma_gp (shared mean slope)

Each method's score(E) returns (d_out, n_groups): the per-group penalty of the
FULL-row error E restricted to each group's channels. Cross-group coupling is
ignored by construction, matching how the quantizer applies an independent
scale per group.
"""

from __future__ import annotations

import torch
from typing import Callable


# QuantFn now takes a PER-GROUP clip of shape (d_out, n_groups, 1).
QuantFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _group_index(d_in: int, group_size: int, device) -> tuple[torch.Tensor, int, int]:
    """
    Returns (gidx, n_groups, pad) where gidx maps each input channel j in
    [0, d_in) to its group id in [0, n_groups). Padding channels (if d_in is
    not a multiple of group_size) are NOT included; callers pad separately.
    """
    n_groups = (d_in + group_size - 1) // group_size
    pad = n_groups * group_size - d_in
    gidx = torch.arange(d_in, device=device) // group_size  # (d_in,)
    return gidx, n_groups, pad


def _scatter_group_sum(per_channel: torch.Tensor, gidx: torch.Tensor,
                       n_groups: int) -> torch.Tensor:
    """
    per_channel : (d_out, d_in) nonneg per-channel contributions.
    Returns (d_out, n_groups): sum of contributions within each group.
    """
    d_out = per_channel.shape[0]
    out = torch.zeros(d_out, n_groups, device=per_channel.device,
                      dtype=per_channel.dtype)
    idx = gidx.unsqueeze(0).expand(d_out, -1)  # (d_out, d_in)
    out.scatter_add_(1, idx, per_channel)
    return out


class ClipRange:
    """Base group-wise clip-range selector. Subclasses implement score()."""

    name = "base"

    def __init__(self, n_grid: int = 20, grid_min: float = 0.5, grid_max: float = 1.0):
        self.n_grid = n_grid
        self.grid_min = grid_min
        self.grid_max = grid_max

    def prepare(self, W: torch.Tensor, *, group_size: int,
                G=None, X=None, pre_act_fn=None):
        self.W = W
        self.group_size = group_size
        d_out, d_in = W.shape
        self.gidx, self.n_groups, self.pad = _group_index(d_in, group_size, W.device)

    def score(self, E: torch.Tensor) -> torch.Tensor:
        """E: (d_out, d_in) full-row error. Return (d_out, n_groups)."""
        raise NotImplementedError

    @torch.no_grad()
    def select_clip(self, W: torch.Tensor, quant_fn: QuantFn) -> torch.Tensor:
        """
        Returns per-group clip of shape (d_out, n_groups, 1), suitable for the
        group-wise quantizer below.
        """
        d_out, d_in = W.shape
        gs = self.group_size
        n_groups = self.n_groups

        # Per-group max-abs of W -> per-group clip ceiling.
        if self.pad > 0:
            Wp = torch.zeros(d_out, n_groups * gs, device=W.device, dtype=W.dtype)
            Wp[:, :d_in] = W
        else:
            Wp = W
        Wg = Wp.reshape(d_out, n_groups, gs)
        g_maxabs = Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)  # (d_out, G, 1)

        # Standard per-group MSE clip search: grid the clip ratio over
        # [grid_min, grid_max], score per group, keep the per-group minimum.
        best_score = torch.full((d_out, n_groups), float("inf"), device=W.device)
        best_clip = g_maxabs.clone()  # (d_out, G, 1)

        for g in range(self.n_grid + 1):
            frac = self.grid_min + (self.grid_max - self.grid_min) * g / self.n_grid
            clip = g_maxabs * frac                       # (d_out, G, 1)
            Wq = quant_fn(W, clip)                        # (d_out, d_in)
            E = Wq - W
            s = self.score(E)                            # (d_out, G)
            improve = s < best_score                     # (d_out, G)
            best_score = torch.where(improve, s, best_score)
            best_clip = torch.where(improve.unsqueeze(2), clip, best_clip)

        return best_clip                                 # (d_out, G, 1)


# ==========================================================================
# Tier 0 : plain RTN — NO clip search. Clip = per-group max-abs (cf = 1.0).
# ==========================================================================
class PlainRTN(ClipRange):
    name = "rtn"

    def prepare(self, W, *, group_size, G=None, X=None, pre_act_fn=None):
        super().prepare(W, group_size=group_size)

    @torch.no_grad()
    def select_clip(self, W, quant_fn):
        d_out, d_in = W.shape
        gs = self.group_size
        n_groups = self.n_groups
        if self.pad > 0:
            Wp = torch.zeros(d_out, n_groups * gs, device=W.device, dtype=W.dtype)
            Wp[:, :d_in] = W
        else:
            Wp = W
        g_maxabs = Wp.reshape(d_out, n_groups, gs).abs().amax(dim=2, keepdim=True)
        return g_maxabs.clamp(min=1e-8)                  # (d_out, G, 1), cf = 1.0

    def score(self, E):                                  # never used
        return _scatter_group_sum(E.pow(2), self.gidx, self.n_groups)


# ==========================================================================
# Tier 1 : weight-MSE   M = I   (per-group)
# ==========================================================================
class WeightMSE(ClipRange):
    name = "weight_mse"

    def prepare(self, W, *, group_size, G=None, X=None, pre_act_fn=None):
        super().prepare(W, group_size=group_size)

    def score(self, E):
        per_ch = E.pow(2)                                # (d_out, d_in)
        return _scatter_group_sum(per_ch, self.gidx, self.n_groups)


# ==========================================================================
# Tier 2 : linear-response   M = XX^T/N   (per-group, X backend)
#   group penalty = (1/N) sum_t ( sum_{j in gp} e_{r,j} x_{j,t} )^2
#
# MEMORY FIX: reduce per group inside the loop; never store all groups' tokens.
# Peak: (d_out, n_tok) per group instead of (d_out, n_groups, n_tok).
# ==========================================================================
class LinearResponse(ClipRange):
    name = "linear_response"

    def prepare(self, W, *, group_size, G=None, X=None, pre_act_fn=None):
        assert X is not None, "LinearResponse needs stored X"
        super().prepare(W, group_size=group_size)
        self.X = X                                       # (d_in, n_tok)
        self.ntok = X.shape[1]

    def score(self, E):
        d_out = E.shape[0]
        out = torch.empty(d_out, self.n_groups, device=E.device, dtype=E.dtype)
        for gp in range(self.n_groups):
            mask = (self.gidx == gp)                     # (d_in,)
            Ug = E[:, mask] @ self.X[mask, :]            # (d_out, n_tok)
            out[:, gp] = Ug.pow(2).sum(dim=1) / self.ntok
        return out                                       # (d_out, n_groups)


# ==========================================================================
# Tier 3 : mixed  M = (1-lam) I + lam M_inner   (per-group)
# ==========================================================================
class Mixed(ClipRange):
    name = "mixed"

    def __init__(self, lam=0.5, inner="linear", n_grid=20, grid_min=0.5, grid_max=1.0):
        super().__init__(n_grid, grid_min, grid_max)
        self.lam = float(lam)
        self.inner = inner

    def prepare(self, W, *, group_size, G=None, X=None, pre_act_fn=None):
        super().prepare(W, group_size=group_size)
        if self.inner == "linear":
            assert X is not None, "Mixed(linear) needs stored X"
            self.X = X
            self.ntok = X.shape[1]
        else:
            assert X is not None and pre_act_fn is not None, \
                "Mixed(sigma) needs stored X and pre_act_fn"
            A = W @ X
            S = pre_act_fn(A)
            s = S.mean(dim=0).clamp(min=0.0)             # (n_tok,)
            self.XS = X * s.unsqueeze(0)                 # (d_in, n_tok)
            self.ntok = X.shape[1]

    def score(self, E):
        weight_term = _scatter_group_sum(E.pow(2), self.gidx, self.n_groups)
        Xmat = self.X if self.inner == "linear" else self.XS
        d_out = E.shape[0]
        inner_term = torch.empty(d_out, self.n_groups, device=E.device,
                                 dtype=E.dtype)
        for gp in range(self.n_groups):
            mask = (self.gidx == gp)
            Ug = E[:, mask] @ Xmat[mask, :]              # (d_out, n_tok)
            inner_term[:, gp] = Ug.pow(2).sum(dim=1) / self.ntok
        return (1.0 - self.lam) * weight_term + self.lam * inner_term


# ==========================================================================
# Paper method : sigma-aware   M_sigma = X S^2 X^T / N   (per-group, per-row S)
#   group penalty = (1/N) sum_t s_{r,t}^2 ( sum_{j in gp} e_{r,j} x_{j,t} )^2
#
# MEMORY FIX: fold S^2 weighting into the per-group reduction; never store the
# dense U or the S2*U^2 temporary across all groups.
# ==========================================================================
class SigmaAware(ClipRange):
    name = "sigma_aware"

    def __init__(self, lam=0.0, n_grid=20, grid_min=0.5, grid_max=1.0):
        super().__init__(n_grid, grid_min, grid_max)
        self.lam = float(lam)

    def prepare(self, W, *, group_size, G=None, X=None, pre_act_fn=None):
        assert X is not None and pre_act_fn is not None, \
            "SigmaAware needs stored X and pre_act_fn"
        super().prepare(W, group_size=group_size)
        self.X = X                                       # (d_in, n_tok)
        A = W @ X                                        # (d_out, n_tok)
        self.S = pre_act_fn(A)                            # (d_out, n_tok)
        self.ntok = X.shape[1]

    def score(self, E):
        d_out = E.shape[0]
        S2 = self.S.pow(2)                               # (d_out, n_tok)
        sigma_term = torch.empty(d_out, self.n_groups, device=E.device,
                                 dtype=E.dtype)
        for gp in range(self.n_groups):
            mask = (self.gidx == gp)
            Ug = E[:, mask] @ self.X[mask, :]            # (d_out, n_tok)
            sigma_term[:, gp] = (S2 * Ug.pow(2)).sum(dim=1) / self.ntok
        if self.lam > 0.0:
            weight_term = _scatter_group_sum(E.pow(2), self.gidx, self.n_groups)
            return (1.0 - self.lam) * weight_term + self.lam * sigma_term
        return sigma_term


def build_clip_range(kind, lam=0.5, inner="linear", n_grid=20,
                     grid_min=0.5, grid_max=1.0):
    kind = kind.lower()
    if kind == "rtn":
        return PlainRTN(n_grid, grid_min, grid_max)
    if kind == "weight_mse":
        return WeightMSE(n_grid, grid_min, grid_max)
    if kind == "linear_response":
        return LinearResponse(n_grid, grid_min, grid_max)
    if kind == "mixed":
        return Mixed(lam=lam, inner=inner, n_grid=n_grid,
                     grid_min=grid_min, grid_max=grid_max)
    if kind == "sigma_aware":
        return SigmaAware(lam=lam, n_grid=n_grid, grid_min=grid_min, grid_max=grid_max)
    raise ValueError(f"Unknown clip-range kind: {kind!r}")


# ==========================================================================
# Group-wise quantizer accepting a PER-GROUP clip (d_out, n_groups, 1).
# ==========================================================================
@torch.no_grad()
def groupwise_rtn_symmetric(W: torch.Tensor, clip: torch.Tensor,
                            bits: int, group_size: int) -> torch.Tensor:
    """
    W    : (d_out, d_in)
    clip : (d_out, n_groups, 1) per-GROUP clip magnitude.
    """
    d_out, d_in = W.shape
    qmax = 2 ** (bits - 1) - 1
    n_groups = (d_in + group_size - 1) // group_size
    pad = n_groups * group_size - d_in
    if pad > 0:
        Wp = torch.zeros(d_out, n_groups * group_size, device=W.device, dtype=W.dtype)
        Wp[:, :d_in] = W
    else:
        Wp = W
    Wg = Wp.reshape(d_out, n_groups, group_size)         # (d_out, G, gs)
    g_maxabs = Wg.abs().amax(dim=2, keepdim=True)         # (d_out, G, 1)
    eff = torch.minimum(g_maxabs, clip).clamp(min=1e-8)   # (d_out, G, 1)
    delta = eff / qmax
    q = torch.round(Wg / delta).clamp(-qmax, qmax)
    Wdq = (q * delta).reshape(d_out, n_groups * group_size)
    if pad > 0:
        Wdq = Wdq[:, :d_in]
    return Wdq


def make_quant_fn(bits: int, group_size: int):
    def quant_fn(W, clip):
        return groupwise_rtn_symmetric(W, clip, bits, group_size)
    return quant_fn