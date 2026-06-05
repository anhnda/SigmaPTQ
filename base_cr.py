"""
base_cr.py
==========
Clip-range selection interface for weight-only group-wise PTQ.

Clip selection is cast as METRIC SELECTION (Sigma-Aware PTQ paper, Eq. 5):

    c* = argmin_c  e(c)^T M e(c),     e(c) = Q_c(w) - w

The entire design space is the PSD metric M. This module provides:

  Tier 1  WeightMSE       M = I                       (penalize ||e||^2)
  Tier 2  LinearResponse  M = (1/N) X X^T             (penalize delta of W x)
  Tier 3  Mixed           M = (1-lam) I + lam (XX^T)  (weight-space floor)
  Paper   SigmaAware      M = (1/N) X S^2 X^T         (penalize delta of sigma(Wx))
          FlooredSigma    M = (1-lam) I + lam (X S^2 X^T)   [SigmaAware, lam>0]

TWO METRIC BACKENDS
-------------------
Tier 1/2/3 use a precomputed GRAM matrix  G = (1/N) X X^T  of shape
(d_in, d_in), accumulated incrementally during one calibration pass. This is
token-count-independent (d_in^2 floats regardless of n_tok) and needs no
stored activations. Scoring is the dense quadratic form e^T G e.

Sigma-aware needs per-output-row slopes S = sigma'(W x), so its metric
X S^2 X^T differs per row and cannot collapse to a single shared Gram. It
therefore keeps the stored activations X (layer-batched upstream to cap RAM).

Quadratic forms, per row r of E (d_out x d_in):
  WeightMSE       : ||e_r||^2
  LinearResponse  : e_r^T G e_r              with G = XX^T/N      (Gram)
  Mixed(linear)   : (1-lam)||e_r||^2 + lam e_r^T G e_r           (Gram)
  Mixed(sigma)    : (1-lam)||e_r||^2 + lam e_r^T (X Sbar^2 X^T/N) e_r  (X)
  SigmaAware      : (1-lam)||e_r||^2 + lam (1/N) sum_t s_{r,t}^2 (e_r.x_t)^2  (X)
"""

from __future__ import annotations

import torch
from typing import Callable, Optional


QuantFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ==========================================================================
# Gram accumulator for Tier 1/2/3.
#   Accumulate G = sum_t x_t x_t^T over calibration tokens, then normalize by
#   N. Token-count-independent: only d_in x d_in floats are ever held.
#   Update in chunks to bound the transient (X_chunk X_chunk^T).
# ==========================================================================
class GramAccumulator:
    """
    Incremental accumulation of G = (1/N) sum_t x_t x_t^T  (d_in x d_in).

    Usage:
        acc = GramAccumulator(d_in, device, dtype=torch.float32)
        for X_chunk in chunks:            # X_chunk: (d_in, n_chunk)
            acc.update(X_chunk)
        G = acc.finalize()                # (d_in, d_in)
    """

    def __init__(self, d_in: int, device, dtype=torch.float32):
        self.d_in = d_in
        self.device = device
        self.dtype = dtype
        self.G = torch.zeros(d_in, d_in, device=device, dtype=dtype)
        self.n = 0

    @torch.no_grad()
    def update(self, X_chunk: torch.Tensor):
        """X_chunk: (d_in, n_chunk) on any device; moved to accumulator device."""
        Xc = X_chunk.to(self.device, self.dtype)
        self.G += Xc @ Xc.t()
        self.n += Xc.shape[1]

    @torch.no_grad()
    def finalize(self) -> torch.Tensor:
        if self.n == 0:
            return self.G
        return self.G / self.n


class ClipRange:
    """Base clip-range selector. Subclasses implement score(E)."""

    name = "base"

    def __init__(self, n_grid: int = 20, grid_min: float = 0.5, grid_max: float = 1.0):
        self.n_grid = n_grid
        self.grid_min = grid_min
        self.grid_max = grid_max

    # uses_gram: True -> driver supplies a precomputed Gram via prepare(G=...)
    #            False -> driver supplies stored activations X via prepare(X=...)
    uses_gram = False

    def prepare(self, W: torch.Tensor, *, G=None, X=None, pre_act_fn=None):
        """
        Cache metric machinery. Called once per layer.

        W : (d_out, d_in) weight (float).
        G : (d_in, d_in) precomputed Gram XX^T/N  (Gram-backend methods).
        X : (d_in, n_tok) stored activations      (X-backend methods).
        pre_act_fn : callable A=W@X -> slopes S    (sigma methods).
        """
        self.W = W

    def score(self, E: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def select_clip(self, W: torch.Tensor, quant_fn: QuantFn) -> torch.Tensor:
        d_out = W.shape[0]
        max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        best_score = torch.full((d_out,), float("inf"), device=W.device)
        best_clip = max_abs.clone()
        for g in range(self.n_grid + 1):
            frac = self.grid_min + (self.grid_max - self.grid_min) * g / self.n_grid
            clip = max_abs * frac
            Wq = quant_fn(W, clip)
            E = Wq - W
            s = self.score(E)
            improve = s < best_score
            best_score = torch.where(improve, s, best_score)
            best_clip = torch.where(improve.unsqueeze(1), clip, best_clip)
        return best_clip


# ==========================================================================
# Tier 1 : weight-MSE   M = I   (no metric data needed)
# ==========================================================================
class WeightMSE(ClipRange):
    name = "weight_mse"
    uses_gram = False  # needs neither G nor X

    def prepare(self, W, *, G=None, X=None, pre_act_fn=None):
        self.W = W

    def score(self, E):
        return E.pow(2).sum(dim=1)


# ==========================================================================
# Tier 2 : linear-response   M = G = XX^T/N    (GRAM backend)
#   e^T G e = sum_j (E G)_{rj} E_{rj}  =  (E @ G * E).sum(dim=1)
# ==========================================================================
class LinearResponse(ClipRange):
    name = "linear_response"
    uses_gram = True

    def prepare(self, W, *, G=None, X=None, pre_act_fn=None):
        assert G is not None, "LinearResponse needs precomputed Gram G"
        self.W = W
        self.G = G

    def score(self, E):
        return (E @ self.G * E).sum(dim=1)


# ==========================================================================
# Tier 3 : mixed  M = (1-lam) I + lam M_inner
#   inner="linear" : GRAM backend (G = XX^T/N)
#   inner="sigma"  : X backend, single shared mean-slope weighting
# ==========================================================================
class Mixed(ClipRange):
    name = "mixed"

    def __init__(self, lam=0.5, inner="linear", n_grid=20, grid_min=0.5, grid_max=1.0):
        super().__init__(n_grid, grid_min, grid_max)
        self.lam = float(lam)
        self.inner = inner

    @property
    def uses_gram(self):
        return self.inner == "linear"

    def prepare(self, W, *, G=None, X=None, pre_act_fn=None):
        self.W = W
        if self.inner == "linear":
            assert G is not None, "Mixed(linear) needs precomputed Gram G"
            self.G = G
        else:  # sigma inner: shared mean-slope weighting on stored X
            assert X is not None and pre_act_fn is not None, \
                "Mixed(sigma) needs stored X and pre_act_fn"
            A = W @ X
            S = pre_act_fn(A)
            s = S.mean(dim=0).clamp(min=0.0)        # (n_tok,)
            self.XS = X * s.unsqueeze(0)            # (d_in, n_tok)
            self.ntok = X.shape[1]

    def score(self, E):
        weight_term = E.pow(2).sum(dim=1)
        if self.inner == "linear":
            inner_term = (E @ self.G * E).sum(dim=1)
        else:
            EX = E @ self.XS
            inner_term = EX.pow(2).sum(dim=1) / self.ntok
        return (1.0 - self.lam) * weight_term + self.lam * inner_term


# ==========================================================================
# Paper method : sigma-aware   M_sigma = (1/N) X S^2 X^T   (X backend, per-row)
#   lam = 0   -> unfloored NAC
#   lam > 0   -> Floored-NAC : (1-lam) I + lam M_sigma
# ==========================================================================
class SigmaAware(ClipRange):
    name = "sigma_aware"
    uses_gram = False

    def __init__(self, lam=0.0, n_grid=20, grid_min=0.5, grid_max=1.0):
        super().__init__(n_grid, grid_min, grid_max)
        self.lam = float(lam)

    def prepare(self, W, *, G=None, X=None, pre_act_fn=None):
        assert X is not None and pre_act_fn is not None, \
            "SigmaAware needs stored X and pre_act_fn"
        self.W = W
        self.X = X                          # (d_in, n_tok)
        A = W @ X                           # (d_out, n_tok)
        self.S = pre_act_fn(A)              # (d_out, n_tok)
        self.ntok = X.shape[1]

    def score(self, E):
        U = E @ self.X                       # (d_out, n_tok) ; u_{r,t}=e_r.x_t
        sigma_term = (self.S.pow(2) * U.pow(2)).sum(dim=1) / self.ntok
        if self.lam > 0.0:
            weight_term = E.pow(2).sum(dim=1)
            return (1.0 - self.lam) * weight_term + self.lam * sigma_term
        return sigma_term


def build_clip_range(kind, lam=0.5, inner="linear", n_grid=20,
                     grid_min=0.5, grid_max=1.0):
    kind = kind.lower()
    if kind == "weight_mse":
        return WeightMSE(n_grid, grid_min, grid_max)
    if kind == "linear_response":
        return LinearResponse(n_grid, grid_min, grid_max)
    if kind == "mixed":
        return Mixed(lam=lam, inner=inner, n_grid=n_grid,
                     grid_min=grid_min, grid_max=grid_max)
    if kind == "sigma_aware":
        return SigmaAware(lam=lam, n_grid=n_grid, grid_min=grid_min, grid_max=grid_max)
    raise ValueError(f"Unknown clip-range kind: {kind!r}. "
                     "weight_mse | linear_response | mixed | sigma_aware")