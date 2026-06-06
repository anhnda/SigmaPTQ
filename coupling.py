"""
coupling.py
===========
Builds the per-layer score callable, and — crucially — resolves the SwiGLU
gate/up projections as a COUPLED PAIR so each gets the sibling response it needs.

Why coupling is required (verified against Llama-3.1, Mistral-v0.3, Qwen2.5):
    h_t = SiLU(g_t) ⊙ u_t,   g_t = W_g x_t,   u_t = W_u x_t
    gate metric (Eq.11): s^2_{k,t} = rho_k u_{k,t}^2 SiLU'(g_{k,t})^2  -> needs u
    up   metric (Eq.13): s^2_{k,t} = rho_k SiLU(g_{k,t})^2            -> needs g
A per-layer prepare() that sees only its own W and X CANNOT form these. The
coupler computes g = W_g@X and u = W_u@X once per block and hands the correct
combination to each layer's sensitivity.

The down projection's column norms rho_k = ||W_d^{(:,k)}||^2 are the optional
downstream gain (Eq.14); pass use_rho=False to disable (rho_k ≡ 1).

Metrics offered:
    weight_mse        M = I                                  (no activations)
    linear_response   M = XX^T/N         s^2 = 1
    pointwise         M = XS^2X^T/N       s^2 = sigma'(Wx)^2  (incomplete gate)
    gate              Eq.11 (+Eq.14)      coupled, needs sibling up
    up                Eq.13 (+Eq.14)      coupled, needs sibling gate
    mixed             (1-lam) I + lam * <inner>
"""

from __future__ import annotations

import torch
from typing import Callable

import metric
import sensitivity as sens

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------
# Generic score builder: given a Sensitivity object (or None for weight_mse),
# return score(E) -> (d_out, n_groups), with optional identity floor lam.
# --------------------------------------------------------------------------
def build_score_fn(W: torch.Tensor, *, group_size: int,
                   X: torch.Tensor | None = None,
                   sensitivity: sens.Sensitivity | None = None,
                   lam: float = 0.0) -> ScoreFn:
    """
    sensitivity is None  -> pure weight_mse (M=I).
    sensitivity present  -> activation score; if 0<lam<1 mix with identity:
                            (1-lam)*weight + lam*activation.
    lam semantics match the old SigmaAware/Mixed: lam=0 => activation only when
    a sensitivity is given (we treat None separately as weight_mse).
    """
    d_out, d_in = W.shape
    gidx, n_groups, _ = metric.group_index(d_in, group_size, W.device)

    if sensitivity is None:
        def score_fn(E):
            return metric.weight_score(E, gidx, n_groups)
        return score_fn

    assert X is not None, "activation-based sensitivity needs stored X"
    ntok = X.shape[1]
    s2 = sensitivity.value()

    def score_fn(E):
        act = metric.activation_score(E, X, s2, gidx, n_groups, ntok)
        if lam > 0.0:
            w = metric.weight_score(E, gidx, n_groups)
            return (1.0 - lam) * w + lam * act
        return act
    return score_fn


# --------------------------------------------------------------------------
# SwiGLU block coupler
# --------------------------------------------------------------------------
class SwiGLUBlock:
    """
    Holds the gate/up/down weights and stored X for one MLP block, computes the
    shared linear responses g, u once, and builds the correct score_fn for the
    gate and up layers respectively.

    Usage (driver side):
        blk = SwiGLUBlock(Wg, Wu, Wd, Xg, Xu, act="silu", use_rho=True)
        gate_score = blk.gate_score_fn(group_size, lam)
        up_score   = blk.up_score_fn(group_size, lam)
    Xg and Xu are the inputs feeding gate and up. In SwiGLU they are the SAME
    tensor (both read the post-attention layernorm output), so a single X is
    fine; we keep them separate only to be defensive about hook capture.
    """

    def __init__(self, Wg: torch.Tensor, Wu: torch.Tensor,
                 Wd: torch.Tensor, X: torch.Tensor,
                 act: str = "silu", use_rho: bool = True):
        self.Wg = Wg.float()
        self.Wu = Wu.float()
        self.X = X                              # (d_model, n_tok)
        self.act = act
        # Shared linear responses on the intermediate axis.
        self.g = self.Wg @ self.X               # (d_inter, n_tok)
        self.u = self.Wu @ self.X               # (d_inter, n_tok)
        self.rho = sens.downstream_gain(Wd) if use_rho else None  # (d_inter,)

    def gate_score_fn(self, group_size: int, lam: float = 0.0) -> ScoreFn:
        s = sens.GateSensitivity(self.g, self.u, rho=self.rho, act=self.act)
        return build_score_fn(self.Wg, group_size=group_size, X=self.X,
                              sensitivity=s, lam=lam)

    def up_score_fn(self, group_size: int, lam: float = 0.0) -> ScoreFn:
        s = sens.UpSensitivity(self.g, rho=self.rho, act=self.act)
        return build_score_fn(self.Wu, group_size=group_size, X=self.X,
                              sensitivity=s, lam=lam)
